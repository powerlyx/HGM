"""Memory agent wrapping the hierarchical memory system for QA."""

import logging
import copy
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from memory.memory_system import HGMMemorySystem
from memory.prompts import (
    parse_plain_text_answer,
    parse_keywords_response,
    _extract_section,
)
from memory.anchors import anchor_overlap_score
from llm.controller import RobustLLMController
from dataset.locomo import Turn
from eval.longmemeval_prompts import (
    build_cot_answer_prompt,
    render_answer_check_context,
    extract_cot_final_answer,
)

logger = logging.getLogger("amem_robust")


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    unique_values = []
    for value in values:
        if not value:
            continue
        value = re.sub(r"\s+", " ", str(value)).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_values.append(value)
    return unique_values


def _parse_items(text: str) -> List[str]:
    if not text:
        return []

    items = []
    for line in text.splitlines():
        line = re.sub(r'^\s*[-*•]\s*', '', line).strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(',')] if ',' in line else [line]
        for part in parts:
            part = part.strip().strip('"').strip("'")
            if part and part.upper() not in {"NONE", "N/A", "NULL"}:
                items.append(part)
    return _unique_preserve_order(items)


class RobustAdvancedMemAgent:
    """Agent using the robust memory system with plain-text LLM calls."""

    def __init__(self, model, backend, retrieve_k,
                 sglang_host="http://localhost", sglang_port=30000,
                 memory_config=None,
                 enable_hierarchy_memory: bool = True,
                 enable_evidence_packet_loop: bool = True,
                 answer_max_tokens: int = 32768,
                 enable_answer_driven_verification: bool = False,
                 answer_verification_gate: str = "low_evidence"):
        self.answer_max_tokens = max(1, int(answer_max_tokens))
        self.memory_system = HGMMemorySystem(
            model_name='text-embedding-3-small',
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            hierarchy_config=memory_config,
        )
        self.retriever_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.enable_hierarchy_memory = bool(enable_hierarchy_memory)
        self.enable_evidence_packet_loop = bool(enable_evidence_packet_loop)
        self.enable_answer_driven_verification = bool(enable_answer_driven_verification)
        self.answer_verification_gate = str(answer_verification_gate or "low_evidence").lower()
        if self.answer_verification_gate not in ("all", "low_evidence"):
            self.answer_verification_gate = "low_evidence"

    def add_memory(self, content, time=None, **metadata):
        self.memory_system.add_note(content, time=time, **metadata)

    def add_session_summary(self, session_id: int, date_time: str, turns: List[Turn]):
        """Finalize a session into the episode and topic layers."""
        del turns
        if not self.enable_hierarchy_memory:
            return None
        return self.memory_system.finalize_session(
            session_id=session_id, date_time=date_time
        )

    def finalize_topics(self):
        if not self.enable_hierarchy_memory:
            return None
        return self.memory_system.finalize_topics()

    def generate_query_llm(self, question):
        """Fallback keyword generation — plain text, no JSON schema."""
        prompt = f"""Given the following question, generate several retrieval keywords separated by commas.

Question: {question}

Keywords:"""

        response = self.retriever_llm.llm.get_completion(prompt)
        result = parse_keywords_response(response)
        logger.debug("generate_query_llm response: %s", result)
        return result

    @staticmethod
    def _has_explicit_temporal_signal(source: str) -> bool:
        source_text = str(source or "")
        if re.search(r"(?:19|20)\d{2}", source_text):
            return True
        if re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", source_text):
            return True
        if re.search(r"\b\d{1,2}:\d{2}\b", source_text):
            return True
        return False

    @staticmethod
    def _extract_deterministic_time_hints(question: str) -> List[str]:
        """Detect temporal cues the LLM plan frequently omits.

        Elapsed-time / ordering / relative-window questions (e.g. "how many days
        ago ...", "most recent ...", "first time ...", "last month ...") rarely
        contain explicit numeric dates, so ``_has_explicit_temporal_signal``
        misses them and the retriever stays in a non-temporal mode that surfaces
        the wrong occurrence. Returning concrete cue phrases populates
        ``time_hints`` so routing marks the question ``temporal_required`` and
        the retriever switches to its temporal/recency-weighted profile to
        locate the right time point. Conservative: only clearly temporal cues,
        never bare before/after/latest which over-fire on enumeration questions.
        """
        q = str(question or "")
        ql = q.lower()
        hints: List[str] = []

        elapsed = re.search(
            r"how (?:many|long)\s+(?:days?|weeks?|months?|years?|hours?)?\s*"
            r"(?:ago|(?:have\s+)?passed\s*(?:since|between)?|since|between)",
            ql,
        )
        if elapsed:
            hints.append(elapsed.group(0).strip())

        for pat in (
            r"most recent(?:ly)?",
            r"first time",
            r"last time",
            r"\bearliest\b",
        ):
            m = re.search(pat, ql)
            if m:
                hints.append(m.group(0).strip())

        for pat in (
            r"last (?:week|weekend|month|year|day)\b",
            r"this (?:week|month|year)\b",
            r"in the (?:last|past) \w+",
            r"since the start of",
            r"a few (?:days|weeks|months) ago",
            r"\brecently\b",
        ):
            m = re.search(pat, ql)
            if m:
                hints.append(m.group(0).strip())

        cleaned = _unique_preserve_order(hints)
        return [h for h in cleaned if len(h) >= 3]

    @staticmethod
    def _build_routing_context(question: str, plan: Dict[str, List[str]]) -> Dict[str, object]:
        answer_style = str(plan.get("answer_style", "PHRASE") or "PHRASE").strip().upper()
        time_hints = _unique_preserve_order(list(plan.get("time_hints", []) or []))
        temporal_sources: List[str] = []

        if answer_style == "DATE":
            temporal_sources.append("answer_style")
        if time_hints:
            temporal_sources.append("time_hints")
        if RobustAdvancedMemAgent._has_explicit_temporal_signal(question):
            temporal_sources.append("question_expression")

        return {
            "question": str(question or ""),
            "answer_style": answer_style,
            "time_hints": time_hints,
            "temporal_required": bool(temporal_sources),
            "temporal_sources": _unique_preserve_order(temporal_sources),
        }

    @staticmethod
    def _contains_any(source: str, candidates: List[str]) -> bool:
        source_text = str(source or "").lower()
        for candidate in candidates:
            token = str(candidate or "").strip().lower()
            if token and token in source_text:
                return True
        return False

    def build_query_plan(self, question: str, category: int) -> Dict[str, List[str]]:
        del category
        prompt = f"""You are preparing search queries for long-term conversational question answering.

Question: {question}

Guidelines:
- If the question refers to an event or activity, identify event phrases plus participants, objects, locations, and ordering cues that distinguish it from similar events.
- If the question contains relative or absolute time references, list them in TIME_HINTS exactly as they appear without converting them yourself.
- Generate query variants for both narrow evidence lookup and broad context recall.
- Avoid sensitive demographic inference. Use only explicit lexical cues from the question.

Return EXACTLY this format:
FOCUS: one short sentence
ANSWER_STYLE: DATE or PHRASE or BINARY
EVENT_HINTS: comma-separated event or activity phrases or NONE
TIME_HINTS: comma-separated time constraints or NONE
ENTITIES: comma-separated names, places, objects, or NONE
KEYWORDS: comma-separated retrieval keywords
SEARCH_QUERIES:
- query 1
- query 2
- query 3
ZOOM_IN_QUERIES:
- query 1
- query 2
ZOOM_OUT_QUERIES:
- query 1
- query 2
VERIFICATION_QUERIES:
- query 1
- query 2"""

        try:
            response = self.retriever_llm.llm.get_completion(prompt, temperature=0.2)
        except Exception as e:
            logger.warning("build_query_plan failed: %s", e)
            response = ""

        focus = _extract_section(response, "FOCUS", ["ANSWER_STYLE", "EVENT_HINTS", "TIME_HINTS", "ENTITIES", "KEYWORDS", "SEARCH_QUERIES"])
        answer_style = _extract_section(response, "ANSWER_STYLE", ["EVENT_HINTS", "TIME_HINTS", "ENTITIES", "KEYWORDS", "SEARCH_QUERIES"])
        event_hints = _parse_items(_extract_section(response, "EVENT_HINTS", ["TIME_HINTS", "ENTITIES", "KEYWORDS", "SEARCH_QUERIES"]))
        llm_time_hints = _parse_items(_extract_section(response, "TIME_HINTS", ["EVENT_HINTS", "ENTITIES", "KEYWORDS", "SEARCH_QUERIES"]))
        det_time_hints = RobustAdvancedMemAgent._extract_deterministic_time_hints(question)
        time_hints = _unique_preserve_order(llm_time_hints + det_time_hints)
        entities = _parse_items(_extract_section(response, "ENTITIES", ["KEYWORDS", "SEARCH_QUERIES"]))
        keywords = _parse_items(_extract_section(response, "KEYWORDS", ["SEARCH_QUERIES", "ZOOM_IN_QUERIES", "ZOOM_OUT_QUERIES", "VERIFICATION_QUERIES"]))
        search_queries = _parse_items(_extract_section(response, "SEARCH_QUERIES", ["ZOOM_IN_QUERIES", "ZOOM_OUT_QUERIES", "VERIFICATION_QUERIES"]))
        zoom_in_queries = _parse_items(_extract_section(response, "ZOOM_IN_QUERIES", ["ZOOM_OUT_QUERIES", "VERIFICATION_QUERIES"]))
        zoom_out_queries = _parse_items(_extract_section(response, "ZOOM_OUT_QUERIES", ["VERIFICATION_QUERIES"]))
        verification_queries = _parse_items(_extract_section(response, "VERIFICATION_QUERIES"))

        if not keywords:
            keywords = _parse_items(self.generate_query_llm(question))

        default_style = "DATE" if bool(time_hints) else "PHRASE"
        normalized_answer_style = (answer_style or default_style).upper()
        if normalized_answer_style not in {"DATE", "PHRASE", "BINARY"}:
            normalized_answer_style = default_style

        fallback_queries = [
            question,
            " ".join(keywords),
            f"{question} {' '.join(entities)}".strip(),
            f"{focus} {' '.join(event_hints)}".strip(),
            f"{focus} {' '.join(time_hints)}".strip(),
        ]

        fallback_zoom_in = [
            f"{question} {' '.join(event_hints)}".strip(),
            f"{question} {' '.join(entities)} timeline".strip(),
            f"{question} exact evidence".strip(),
        ]
        fallback_zoom_out = [
            focus,
            " ".join(keywords),
            f"{focus} related background".strip(),
        ]
        fallback_verification = [
            f"{question} contradiction latest update".strip(),
            f"{question} source quote".strip(),
            f"{question} {' '.join(time_hints)}".strip(),
        ]

        plan = {
            "focus": focus or question,
            "answer_style": normalized_answer_style,
            "event_hints": event_hints,
            "time_hints": time_hints,
            "entities": entities,
            "keywords": keywords,
            "search_queries": _unique_preserve_order(search_queries + fallback_queries),
            "zoom_in_queries": _unique_preserve_order(zoom_in_queries + fallback_zoom_in),
            "zoom_out_queries": _unique_preserve_order(zoom_out_queries + fallback_zoom_out),
            "verification_queries": _unique_preserve_order(verification_queries + fallback_verification),
        }
        return plan

    def rank_memories(
        self,
        queries: List[str],
        question: str,
        category: int,
        routing_context: Optional[Dict[str, object]] = None,
    ) -> Dict[str, object]:
        del category
        combined_queries = _unique_preserve_order(list(queries) + [question])
        return self.memory_system.search_hierarchy(
            combined_queries,
            retrieve_k=self.retrieve_k,
            routing_context=routing_context,
        )

    def _estimate_evidence_sufficiency(
        self,
        question: str,
        plan: Dict[str, List[str]],
        ranked_memories: Dict[str, object],
    ) -> Dict[str, object]:
        topics = ranked_memories.get("topics", []) or []
        episodes = ranked_memories.get("episodes", []) or []
        turns = ranked_memories.get("turns", []) or []
        routing_info = ranked_memories.get("routing", {}) if isinstance(ranked_memories, dict) else {}

        top_topic_text = "\n".join(item["note"].content for item in topics[:2])
        top_episode_text = "\n".join(item["note"].content for item in episodes[:3])
        top_turn_text = "\n".join(item["note"].content for item in turns[:6])
        evidence_text = "\n".join([top_topic_text, top_episode_text, top_turn_text]).lower()

        routing_context = plan.get("routing_context")
        if not isinstance(routing_context, dict):
            routing_context = self._build_routing_context(question, plan)
        answer_style = str(routing_context.get("answer_style", "PHRASE")).upper()
        time_hints = list(routing_context.get("time_hints", []) or [])
        temporal_required = bool(routing_context.get("temporal_required", False))
        temporal_supported = not temporal_required
        if temporal_required:
            if time_hints:
                temporal_supported = self._contains_any(evidence_text, time_hints)
            else:
                temporal_supported = self._has_explicit_temporal_signal(evidence_text)

        entity_hints = plan.get("entities", []) or []
        event_hints = plan.get("event_hints", []) or []
        lexical_supported = True
        if entity_hints or event_hints:
            lexical_supported = self._contains_any(
                evidence_text,
                entity_hints + event_hints,
            )

        turn_support = len(turns)
        episode_support = len(episodes)
        topic_support = len(topics)

        sufficiency_score = 0.0
        sufficiency_score += min(1.0, turn_support / max(3, self.retrieve_k / 2)) * 0.45
        sufficiency_score += min(1.0, episode_support / 2) * 0.20
        sufficiency_score += min(1.0, topic_support / 1) * 0.10
        sufficiency_score += (0.15 if temporal_supported else 0.0)
        sufficiency_score += (0.10 if lexical_supported else 0.0)

        need_escalation = (
            turn_support < max(2, min(4, self.retrieve_k // 2))
            or not temporal_supported
            or not lexical_supported
            or sufficiency_score < 0.66
        )

        return {
            "turn_support": turn_support,
            "episode_support": episode_support,
            "topic_support": topic_support,
            "temporal_required": temporal_required,
            "temporal_supported": temporal_supported,
            "temporal_sources": list(routing_context.get("temporal_sources", []) or []),
            "time_hints": time_hints,
            "lexical_supported": lexical_supported,
            "sufficiency_score": round(float(sufficiency_score), 4),
            "need_escalation": bool(need_escalation),
        }

    def _merge_ranked_memories(
        self,
        primary: Dict[str, object],
        secondary: Dict[str, object],
    ) -> Dict[str, object]:
        routing = secondary.get("routing") or primary.get("routing") or {}
        limits = self._layer_output_limits(routing)
        merged: Dict[str, object] = {
            "topics": [],
            "episodes": [],
            "turns": [],
            "routing": routing,
        }
        for layer in ("topics", "episodes", "turns"):
            by_id: Dict[str, Dict[str, object]] = {}
            for item in (primary.get(layer, []) or []) + (secondary.get(layer, []) or []):
                note_id = str(item.get("id", ""))
                if not note_id:
                    continue
                current = by_id.get(note_id)
                if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
                    by_id[note_id] = item
            merged[layer] = sorted(
                by_id.values(),
                key=lambda item: float(item.get("score", 0.0)),
                reverse=True,
            )[: limits.get(layer, 0)]
        return merged

    def _layer_output_limits(self, routing: Optional[Dict[str, object]] = None) -> Dict[str, int]:
        config = self.memory_system.hierarchy_config
        defaults = {
            "topics": max(0, int(getattr(config, "topic_result_limit", 1))),
            "episodes": max(0, int(getattr(config, "episode_result_limit", 3))),
            "turns": max(0, int(getattr(config, "turn_result_limit", 5))),
        }
        if not isinstance(routing, dict):
            return defaults
        final_limits = routing.get("final_limits")
        if not isinstance(final_limits, dict):
            return defaults
        return {
            layer: max(0, int(final_limits.get(layer, defaults[layer])))
            for layer in ("topics", "episodes", "turns")
        }

    def _truncate_ranked_memories(self, ranked_memories: Dict[str, object]) -> Dict[str, object]:
        routing = ranked_memories.get("routing", {}) if isinstance(ranked_memories, dict) else {}
        limits = self._layer_output_limits(routing)
        truncated = dict(ranked_memories or {})
        for layer, limit in limits.items():
            layer_items = truncated.get(layer, []) or []
            truncated[layer] = list(layer_items)[:limit]
        if "routing" not in truncated:
            truncated["routing"] = {}
        return truncated

    @staticmethod
    def _token_set(text: str) -> set:
        return {
            token
            for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
            if len(token) > 1
        }

    @staticmethod
    def _jaccard(left: set, right: set) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _anchors_enabled(self) -> bool:
        hierarchy_config = getattr(self.memory_system, "hierarchy_config", None)
        return bool(getattr(hierarchy_config, "enable_anchors", False))

    def _build_evidence_candidates(self, ranked_memories: Dict[str, object]) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        routing = ranked_memories.get("routing", {}) if isinstance(ranked_memories, dict) else {}
        layer_limits = self._layer_output_limits(routing)
        for layer in ("topics", "episodes", "turns"):
            if layer_limits.get(layer, 0) <= 0:
                continue
            for item in ranked_memories.get(layer, []) or []:
                note = item.get("note")
                if note is None:
                    continue
                content = str(getattr(note, "content", "") or "")
                context = str(getattr(note, "context", "") or "")
                keywords = list(getattr(note, "keywords", []) or [])
                tags = list(getattr(note, "tags", []) or [])
                anchors = (
                    dict(getattr(note, "anchors", {}) or {})
                    if self._anchors_enabled()
                    else {}
                )
                speaker = str(getattr(note, "speaker", "mixed") or "mixed")
                timestamp = str(getattr(note, "timestamp", "") or "")
                note_id = str(item.get("id") or getattr(note, "id", ""))
                anchor_text = ""
                if self._anchors_enabled():
                    anchor_text = " ".join(
                        list(anchors.get("entities", []) or [])
                        + list(anchors.get("times", []) or [])
                        + list(anchors.get("events", []) or [])
                    )
                search_text = " ".join(
                    [content, context, " ".join(keywords), " ".join(tags), anchor_text]
                ).strip()
                candidates.append(
                    {
                        "id": note_id,
                        "layer": layer[:-1] if layer.endswith("s") else layer,
                        "speaker": speaker,
                        "timestamp": timestamp,
                        "content": content,
                        "context": context,
                        "keywords": keywords,
                        "tags": tags,
                        "anchors": anchors,
                        "score": float(item.get("score", 0.0)),
                        "search_text": search_text,
                    }
                )
        return candidates

    def _score_evidence_candidate(
        self,
        candidate: Dict[str, Any],
        question: str,
        plan: Dict[str, List[str]],
    ) -> float:
        search_text = candidate.get("search_text", "")
        candidate_tokens = self._token_set(search_text)
        question_tokens = self._token_set(question)
        focus_tokens = self._token_set(plan.get("focus", ""))
        hint_tokens = self._token_set(
            " ".join(
                list(plan.get("event_hints", []) or [])
                + list(plan.get("time_hints", []) or [])
                + list(plan.get("entities", []) or [])
                + list(plan.get("keywords", []) or [])
            )
        )

        focus_union = question_tokens | focus_tokens
        lexical_focus = len(candidate_tokens & focus_union) / max(1, len(focus_union))
        lexical_hint = len(candidate_tokens & hint_tokens) / max(1, len(hint_tokens)) if hint_tokens else 0.0

        layer_prior = {
            "turn": 1.00,
            "episode": 0.82,
            "topic": 0.66,
        }.get(candidate.get("layer", "turn"), 0.75)

        temporal_bonus = 0.0
        if self._has_explicit_temporal_signal(question):
            if self._has_explicit_temporal_signal(search_text):
                temporal_bonus += 0.12

        anchor_score = 0.0
        if self._anchors_enabled():
            anchor_score = anchor_overlap_score(question, candidate.get("anchors", {}) or {})
            candidate["anchor_score"] = float(anchor_score)
        base_score = float(candidate.get("score", 0.0))
        final_score = (
            0.36 * base_score
            + 0.26 * lexical_focus
            + 0.09 * lexical_hint
            + 0.12 * layer_prior
            + temporal_bonus
        )
        if self._anchors_enabled():
            final_score += 0.17 * anchor_score
        return float(final_score)

    def _select_evidence_candidates(
        self,
        ranked_memories: Dict[str, object],
        question: str,
        plan: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        candidates = self._build_evidence_candidates(ranked_memories)
        if not candidates:
            return []

        for candidate in candidates:
            candidate["evidence_score"] = self._score_evidence_candidate(candidate, question, plan)

        ranked = sorted(candidates, key=lambda item: float(item.get("evidence_score", 0.0)), reverse=True)

        routing = ranked_memories.get("routing", {}) if isinstance(ranked_memories, dict) else {}
        base_limits = self._layer_output_limits(routing)
        has_dynamic_final_limits = isinstance(routing, dict) and isinstance(
            routing.get("final_limits"), dict
        )
        topic_limit = base_limits.get("topics", 1)
        episode_limit = base_limits.get("episodes", 3)
        turn_limit = base_limits.get("turns", 5)
        if not has_dynamic_final_limits:
            # Preserve the legacy evidence-selection headroom for callers that
            # do not yet provide routing metadata. Routed retrievals use the
            # calibrated final limits exactly.
            if topic_limit > 0:
                topic_limit += 1
            if episode_limit > 0:
                episode_limit += 2
            if turn_limit > 0:
                turn_limit += 3

        layer_limits = {
            "topic": topic_limit,
            "episode": episode_limit,
            "turn": turn_limit,
        }
        total_limit = topic_limit + episode_limit + turn_limit

        selected: List[Dict[str, Any]] = []
        layer_counts = defaultdict(int)

        for candidate in ranked:
            layer = str(candidate.get("layer", "turn"))
            if layer_counts[layer] >= layer_limits.get(layer, 0):
                continue

            token_set = self._token_set(candidate.get("search_text", ""))
            is_duplicate = False
            for existing in selected:
                existing_set = self._token_set(existing.get("search_text", ""))
                if self._jaccard(token_set, existing_set) > 0.85:
                    is_duplicate = True
                    break
            if is_duplicate:
                continue

            selected.append(candidate)
            layer_counts[layer] += 1

            if len(selected) >= total_limit:
                break

        for idx, item in enumerate(selected, start=1):
            item["evidence_id"] = f"E{idx:02d}"
        return selected

    def _format_evidence_context(self, selected: List[Dict[str, Any]]) -> str:
        blocks = []
        for item in selected:
            evidence_id = item.get("evidence_id", "E00")
            block_lines = [
                f"[{evidence_id}] layer={item.get('layer', 'turn')} id={item.get('id', '')}",
                f"time={item.get('timestamp', 'unknown')} speaker={item.get('speaker', 'mixed')}",
                f"content={item.get('content', '')}",
            ]
            context = str(item.get("context", "") or "").strip()
            if context:
                block_lines.append(f"context={context}")
            anchors = item.get("anchors", {}) or {}
            if self._anchors_enabled() and isinstance(anchors, dict):
                anchor_parts = []
                for label, key in (
                    ("entities", "entities"),
                    ("times", "times"),
                    ("events", "events"),
                ):
                    values = anchors.get(key, []) or []
                    if values:
                        anchor_parts.append(f"{label}={', '.join(map(str, values[:8]))}")
                if anchor_parts:
                    block_lines.append("anchors=" + "; ".join(anchor_parts))
            blocks.append("\n".join(block_lines))
        return "\n\n".join(blocks)

    def render_memories(self, ranked_memories: Dict[str, List[Dict[str, object]]]) -> str:
        return self.memory_system.render_hierarchy_search_results(ranked_memories)

    def retrieve_context(
        self,
        question: str,
        category: int,
        query_plan: Optional[Dict[str, object]] = None,
    ):
        if isinstance(query_plan, dict):
            reusable_keys = (
                "focus",
                "answer_style",
                "event_hints",
                "time_hints",
                "entities",
                "keywords",
                "search_queries",
                "zoom_in_queries",
                "zoom_out_queries",
                "verification_queries",
            )
            plan = {
                key: copy.deepcopy(query_plan[key])
                for key in reusable_keys
                if key in query_plan
            }
            plan["query_plan_source"] = "precomputed"
        else:
            plan = self.build_query_plan(question, category)
            plan["query_plan_source"] = "generated"
        routing_context = self._build_routing_context(question, plan)
        plan["routing_context"] = routing_context
        primary_queries = _unique_preserve_order(
            list(plan.get("search_queries", []) or [])
            + list(plan.get("zoom_in_queries", []) or [])
        )
        ranked_memories = self.rank_memories(
            primary_queries,
            question,
            category,
            routing_context=routing_context,
        )
        ranked_memories = self._truncate_ranked_memories(ranked_memories)
        diagnostics = self._estimate_evidence_sufficiency(question, plan, ranked_memories)

        if not self.enable_evidence_packet_loop:
            plan["retrieval_escalated"] = False
            plan["verification_pass"] = False
            plan["retrieval_routing"] = ranked_memories.get("routing", {})
            plan["evidence_diagnostics"] = {
                **diagnostics,
                "loop_enabled": False,
            }
            raw_retrieval_context = self.render_memories(ranked_memories)
            plan["raw_retrieval_context"] = raw_retrieval_context
            plan["selected_evidence"] = []
            return plan, raw_retrieval_context

        if diagnostics["need_escalation"]:
            expanded_queries = _unique_preserve_order(
                list(plan.get("search_queries", []) or [])
                + list(plan.get("zoom_in_queries", []) or [])
                + list(plan.get("zoom_out_queries", []) or [])
                + list(plan.get("verification_queries", []) or [])
                + [
                    question,
                    plan.get("focus", ""),
                    " ".join(plan.get("entities", []) or []),
                    " ".join(plan.get("event_hints", []) or []),
                    " ".join(plan.get("time_hints", []) or []),
                    f"{question} evidence timeline source",
                ]
            )
            expanded_k = max(self.retrieve_k + 4, int(self.retrieve_k * 1.8))
            expanded_ranked = self.memory_system.search_hierarchy(
                expanded_queries,
                retrieve_k=expanded_k,
                routing_context=routing_context,
            )
            expanded_ranked = self._truncate_ranked_memories(expanded_ranked)
            ranked_memories = self._merge_ranked_memories(ranked_memories, expanded_ranked)
            ranked_memories = self._truncate_ranked_memories(ranked_memories)
            diagnostics = self._estimate_evidence_sufficiency(question, plan, ranked_memories)
            plan["retrieval_escalated"] = True
            plan["retrieve_k_expanded"] = expanded_k

            if diagnostics["need_escalation"]:
                verification_queries = _unique_preserve_order(
                    list(plan.get("verification_queries", []) or [])
                    + [
                        f"{question} latest update contradiction",
                        f"{question} direct quote source",
                    ]
                )
                verification_ranked = self.memory_system.search_hierarchy(
                    verification_queries,
                    retrieve_k=max(expanded_k, self.retrieve_k + 6),
                    routing_context=routing_context,
                )
                verification_ranked = self._truncate_ranked_memories(verification_ranked)
                ranked_memories = self._merge_ranked_memories(ranked_memories, verification_ranked)
                ranked_memories = self._truncate_ranked_memories(ranked_memories)
                diagnostics = self._estimate_evidence_sufficiency(question, plan, ranked_memories)
                plan["verification_pass"] = True
            else:
                plan["verification_pass"] = False
        else:
            plan["retrieval_escalated"] = False
            plan["verification_pass"] = False

        plan["retrieval_routing"] = ranked_memories.get("routing", {})
        plan["evidence_diagnostics"] = diagnostics
        raw_retrieval_context = self.render_memories(ranked_memories)
        selected_evidence = self._select_evidence_candidates(ranked_memories, question, plan)
        plan["_evidence_candidates_full"] = selected_evidence
        evidence_context = self._format_evidence_context(selected_evidence)
        plan["selected_evidence"] = []
        for item in selected_evidence:
            evidence = {
                "evidence_id": item.get("evidence_id", ""),
                "layer": item.get("layer", ""),
                "id": item.get("id", ""),
                "timestamp": item.get("timestamp", ""),
                "speaker": item.get("speaker", ""),
                "score": round(float(item.get("evidence_score", item.get("score", 0.0))), 6),
            }
            if self._anchors_enabled():
                evidence["anchor_score"] = round(float(item.get("anchor_score", 0.0)), 6)
            plan["selected_evidence"].append(evidence)
        plan["raw_retrieval_context"] = raw_retrieval_context
        if not evidence_context:
            evidence_context = raw_retrieval_context
        if not evidence_context:
            return plan, ""
        return plan, evidence_context

    def answer_question(
        self,
        question: str,
        category: int,
        query_plan: Optional[Dict[str, object]] = None,
        question_date: Optional[str] = None,
        question_type: Optional[str] = None,
        answer_prompt_mode: str = "locomo",
    ) -> tuple:
        """Generate answer for a question — plain text, no JSON schema."""
        # The numeric `category` (1-4) is a LoCoMo-only concept. LongMemEval uses
        # string `question_type` and passes category=0, so the answer_check mode
        # must skip this assertion.
        if answer_prompt_mode != "answer_check":
            assert category in [1, 2, 3, 4]

        retrieval_start = time.perf_counter()
        plan, context = self.retrieve_context(question, category, query_plan=query_plan)

        retrieval_time = time.perf_counter() - retrieval_start

        base_context = context

        final_answer, user_prompt, context, _raw_response, answer_time = self._generate_answer(
            question, base_context, plan, answer_prompt_mode, question_date
        )

        if self.enable_answer_driven_verification and answer_prompt_mode == "locomo":
            final_answer, user_prompt, context, answer_time, verification_diag = self._maybe_answer_driven_verify(
                question, final_answer, user_prompt, context, base_context,
                plan, answer_prompt_mode, question_date, answer_time
            )
            plan["answer_verification"] = verification_diag
        elif self.enable_answer_driven_verification:
            plan["answer_verification"] = {
                "enabled": True,
                "gate": self.answer_verification_gate,
                "applied": False,
                "skip_reason": f"answer_prompt_mode={answer_prompt_mode!r} not supported by answer-driven verification",
            }

        return final_answer, user_prompt, context, plan, retrieval_time, answer_time

    def _generate_answer(self, question, context, plan, answer_prompt_mode,
                         question_date):
        answer_start = time.perf_counter()
        if answer_prompt_mode == "answer_check":
            # Answer-check pure path: fixed answer template + rendered context.
            # question_date is injected as the Current Date anchor (and evidence
            # timestamps are surfaced in the context) so elapsed-time / ordering
            # questions can be computed. Answer extracted from the FINAL ANSWER section.
            # max_tokens is lifted to answer_max_tokens (CoT needs room); the
            # controller's default 1000 would truncate STEP 7: FINAL ANSWER.
            evidence_items = list(plan.get("_evidence_candidates_full") or [])
            # Source the conversation speakers from the evidence items themselves
            # (each item carries its real `speaker` field) so the context header
            # shows the real speaker names instead of placeholders.
            ev_speakers = []
            for _ev in evidence_items:
                _sp = str(_ev.get("speaker") or "").strip()
                if _sp and _sp not in ev_speakers:
                    ev_speakers.append(_sp)
            speaker_a = ev_speakers[0] if len(ev_speakers) >= 1 else "speaker_a"
            if len(ev_speakers) >= 2:
                speaker_b = ev_speakers[1]
            elif len(ev_speakers) == 1:
                speaker_b = ev_speakers[0]
            else:
                speaker_b = "speaker_b"
            answer_check_context = render_answer_check_context(evidence_items, speaker_a=speaker_a, speaker_b=speaker_b)
            user_prompt = build_cot_answer_prompt(
                question=question,
                context=answer_check_context,
                current_date=question_date or "",
            )
            self.memory_system.llm_controller.set_usage_phase("answer_generation")
            try:
                response = self.memory_system.llm_controller.llm.get_completion(
                    user_prompt, temperature=0.0, max_tokens=self.answer_max_tokens
                )
            except Exception as e:
                logger.warning("answer_question (answer_check) failed: %s — returning empty", e)
                response = ""
            final_answer = extract_cot_final_answer(response)
            answer_time = time.perf_counter() - answer_start
            return final_answer, user_prompt, context, response, answer_time

        user_prompt = f"""
You are an intelligent memory assistant tasked with retrieving accurate information from conversation memories.

# CONTEXT:
You have access to memories from two speakers in a conversation. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
1. Carefully analyze all provided memories from both speakers
2. Pay special attention to the timestamps to determine the answer
3. If the question asks about a specific event or fact, look for direct evidence in the memories
4. If the memories contain contradictory information, prioritize the most recent memory
5. If there is a question about time references (like "last year", "two months ago", etc.), calculate the actual date based on the memory timestamp. For example, if a memory from 4 May 2022 mentions "went to India last year," then the trip occurred in 2021.
6. Always convert relative time references to specific dates, months, or years. For example, convert "last year" to "2022" or "two months ago" to "March 2023" based on the memory timestamp. Ignore the reference while answering the question.
7. Focus only on the content of the memories from both speakers. Do not confuse character names mentioned in memories with the actual users who created those memories.
8. The answer should be less than 5-6 words.


# ANSWER REQUIREMENTS:
1. First, examine all memories that contain information related to the question
2. Examine the timestamps and content of these memories carefully
3. Look for explicit mentions of dates, times, locations, or events that answer the question
4. If the answer requires calculation, perform it internally and do not include reasoning or calculations in the response
5. Formulate a precise, concise answer based solely on the evidence in the memories
6. Double-check that your answer directly addresses the question asked
7. Ensure your final answer is specific and avoids vague time references
8. Output only the final answer. Do not include reasoning, explanations, prefixes, or supporting text.

Memories:
{context}

Question: {question}

Answer:
"""
        self.memory_system.llm_controller.set_usage_phase("answer_generation")
        try:
            response = self.memory_system.llm_controller.llm.get_completion(
                user_prompt,
                temperature=0.2,
            )
        except Exception as e:
            logger.warning("answer_question failed: %s — returning empty", e)
            response = ""

        final_answer = parse_plain_text_answer(response)
        answer_time = time.perf_counter() - answer_start

        return final_answer, user_prompt, context, response, answer_time

    def _maybe_answer_driven_verify(self, question, answer, user_prompt, context,
                                    base_context, plan, answer_prompt_mode,
                                    question_date, answer_time):
        """Answer-driven verification link (switchable via ablation token
        `answer_driven`). Probes answer confidence; on low confidence runs a
        verification retrieval and regenerates the answer. Returns
        (answer, user_prompt, context, answer_time, diag). On any failure the
        initial answer is kept and the error is recorded in diag."""
        diag = {
            "enabled": True,
            "gate": self.answer_verification_gate,
            "applied": True,
            "initial_answer": answer,
            "probe_label": None,
            "probe_reason": "",
            "probe_skipped": False,
            "triggered": False,
            "retrieval_seconds": 0.0,
            "turn_support_before": plan.get("evidence_diagnostics", {}).get("turn_support"),
            "turn_support_after": None,
            "regenerated_answer": None,
        }
        try:
            sufficiency = float(plan.get("evidence_diagnostics", {}).get("sufficiency_score", 1.0) or 1.0)
            if self.answer_verification_gate == "low_evidence" and sufficiency >= 0.66:
                diag["probe_skipped"] = True
                return answer, user_prompt, context, answer_time, diag

            probe = self._probe_answer_confidence(question, answer, context)
            diag["probe_label"] = probe.get("label")
            diag["probe_reason"] = probe.get("reason", "")
            if probe.get("confident", True):
                return answer, user_prompt, context, answer_time, diag

            v_start = time.perf_counter()
            v_render, turns_after = self._answer_verification_retrieval(question, answer, plan)
            v_retr = time.perf_counter() - v_start
            aug_context = f"{base_context}\n\n# VERIFICATION EVIDENCE (answer-driven):\n{v_render}"
            new_answer, new_prompt, new_context, _raw2, new_ans_time = self._generate_answer(
                question, aug_context, plan, answer_prompt_mode, question_date
            )
            diag["triggered"] = True
            diag["retrieval_seconds"] = v_retr
            diag["turn_support_after"] = turns_after
            diag["regenerated_answer"] = new_answer
            answer_time += v_retr + new_ans_time
            return new_answer, new_prompt, new_context, answer_time, diag
        except Exception as exc:
            logger.warning("answer-driven verification failed: %s — keeping initial answer", exc)
            diag["error"] = repr(exc)
            return answer, user_prompt, context, answer_time, diag

    def _probe_answer_confidence(self, question, answer, evidence_context):
        """Lightweight LLM probe: is `answer` directly supported by the evidence?
        Returns {confident, label, reason, raw}. Parse failure -> confident."""
        prompt = (
            "You are verifying whether a proposed answer is supported by the provided memory evidence.\n"
            f"Evidence:\n{evidence_context}\n\n"
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n\n"
            "Is the proposed answer directly and specifically supported by the evidence above "
            "(i.e. the evidence contains the fact/date/entity/name needed to justify it)?\n"
            "Reply in exactly two lines:\n"
            "LABEL: YES or NO\n"
            "REASON: <one short sentence>\n"
        )
        self.retriever_llm.set_usage_phase("answer_verification")
        try:
            resp = self.retriever_llm.llm.get_completion(prompt, temperature=0.0)
        except Exception as e:
            logger.warning("answer confidence probe failed: %s — assuming confident", e)
            return {"confident": True, "label": "YES", "reason": "probe-error", "raw": ""}
        text = str(resp or "")
        label = None
        reason = ""
        for line in text.splitlines():
            stripped = line.strip().upper()
            if stripped.startswith("LABEL"):
                val = line.split(":", 1)[-1].strip().upper()
                if val.startswith("YES"):
                    label = "YES"
                elif val.startswith("NO"):
                    label = "NO"
            elif stripped.startswith("REASON"):
                reason = line.split(":", 1)[-1].strip()
        confident = True if label is None else (label == "YES")
        return {"confident": confident, "label": label, "reason": reason, "raw": text}

    def _answer_verification_retrieval(self, question, answer, plan):
        """Verification retrieval driven by the answer hypothesis: verification
        + zoom_out queries plus the question and the proposed answer. Returns
        (rendered_context, turns_after)."""
        expanded_k = max(self.retrieve_k + 4, int(self.retrieve_k * 1.8))
        queries = _unique_preserve_order(
            list(plan.get("verification_queries", []) or [])
            + list(plan.get("zoom_out_queries", []) or [])
            + [question, str(answer or ""), str(plan.get("focus", "") or "")]
        )
        routing_context = plan.get("routing_context")
        ranked = self.memory_system.search_hierarchy(
            queries, retrieve_k=expanded_k, routing_context=routing_context
        )
        rendered = self.memory_system.render_hierarchy_search_results(ranked)
        turns_after = len(ranked.get("turns", []) or [])
        return rendered, turns_after

# 8. The answer should be less than 5-6 words.
# 5. Formulate a precise, concise answer based solely on the evidence in the memories
# 5. Formulate a specific answer based solely on the evidence in the memories
