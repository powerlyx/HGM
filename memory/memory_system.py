"""Three-layer hierarchical memory system with coarse-to-fine retrieval."""

from collections import OrderedDict, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional
import logging
import os
import re
import time
import json

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity

from memory.retriever import SimpleEmbeddingRetriever
from memory.memory_note import RobustMemoryNote
from memory.anchors import (
    extract_structured_anchors,
    merge_structured_anchors,
)
from memory.prompts import (
    parse_relevant_parts,
)
from llm.controller import (
    RobustLLMController,
)

logger = logging.getLogger("amem_robust")

SESSION_EPISODE_SUMMARY_PROMPT = """You are building long-term memory for an AI assistant.

Summarize the following dialogue session into {min_bullets} to {max_bullets} concise bullet points.

Requirements:
- Use only facts explicitly supported by the dialogue.
- Preserve named entities, concrete events, preferences, plans, updates, and temporal cues.
- If speakers disagree or correct earlier information, keep both the change and the latest state explicit.
- Do not infer demographics, motives, personality traits, or unstated facts.
- Keep wording close to the dialogue when possible.
- Output only bullet points.

Dialogue date: {date_time}

Dialogue:
{dialogue}
"""

TOPIC_CLUSTER_SUMMARY_PROMPT = """You are consolidating long-term memory across multiple sessions.

Write {min_bullets} to {max_bullets} bullet points describing the recurring theme shared by the episode summaries below.

Requirements:
- Use only facts explicitly supported by the episode summaries.
- Focus on stable cross-session patterns, recurring entities, preferences, projects, and evolving facts.
- Preserve uncertainty or updates instead of collapsing conflicting evidence into one invented claim.
- Do not add information that is not present in the episode summaries.
- Output only bullet points.

Episode summaries:
{episodes}
"""



SESSION_SEGMENTATION_PROMPT = """You are segmenting one dialogue session into coherent episodes.

Task:
- Split the turn sequence into contiguous topical episodes.
- Start a new episode when there is a clear topic/activity/entity shift.

Safety and grounding constraints:
- Use only the provided dialogue turns.
- Do not infer demographics, traits, motives, or unstated facts.
- Do not use any external knowledge.

Output format (strict):
BOUNDARIES: comma-separated start indices of each episode (0-based), must include 0.

Example:
BOUNDARIES: 0, 5, 11


Dialogue turns:
{dialogue}
"""



TOPIC_TITLE_PROMPT = """You are consolidating long-term memory across multiple sessions into a topic title.

Write ONE concise representative title (a short noun phrase, NOT a full sentence, no trailing period) that captures the recurring theme shared by the episode summaries below.

Requirements:
- Use only facts explicitly supported by the episode summaries.
- Do not include any time information in the title.
- Summarize the shared theme of the episodes; do not just copy one episode\'s wording.
- Output ONLY the title text on a single line, nothing else.

Episode summaries:
{episodes}
"""

EPISODE_TITLE_PROMPT = """You are building long-term memory for an AI assistant.

Write ONE concise representative title (a short noun phrase, NOT a full sentence, no trailing period) that captures the main topic of the dialogue below.

Requirements:
- Use only facts explicitly supported by the dialogue.
- Do not include any time information in the title.
- Summarize the shared topic of the dialogue; do not just copy a single turn or speaker wording.
- Output ONLY the title text on a single line, nothing else.

Dialogue:
{dialogue}
"""
from memory.config import (
    HierarchyMemoryConfig,
    _RETRIEVAL_ONLY_CONFIG_KEYS,
    _config_without_retrieval_only_keys,
    build_hierarchy_config,
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _safe_session_key(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def _unique_preserve_order(values: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        cleaned = _normalize_text(value)
        if not cleaned:
            continue
        lowered = cleaned.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(cleaned)
    return result


def _token_set(text: str) -> set:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
        if len(token) > 2
    }


def _extract_anchor_terms(text: str, max_terms: int = 16) -> List[str]:
    """Extract high-signal temporal/entity anchors without external knowledge."""
    raw_text = str(text or "")
    if not raw_text:
        return []

    anchors: List[str] = []
    # Temporal anchors
    anchors.extend(re.findall(r"(?:19|20)\d{2}", raw_text))
    anchors.extend(re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", raw_text))
    anchors.extend(re.findall(r"\b\d{1,2}:\d{2}\b", raw_text))

    # Entity-like anchors: capitalized words and multiword title-case phrases.
    anchors.extend(re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", raw_text))
    anchors.extend(re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", raw_text))

    # Number anchors can be useful for ordinal/quantity questions.
    anchors.extend(re.findall(r"\b\d+(?:st|nd|rd|th)?\b", raw_text.lower()))

    cleaned = _unique_preserve_order(anchors)
    return cleaned[:max_terms]


def _tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+", str(text or "").lower())


def _parse_int_list(text: str) -> List[int]:
    values = []
    for token in re.findall(r"\d+", str(text or "")):
        try:
            values.append(int(token))
        except ValueError:
            continue
    return values


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


def _estimate_token_count(text: Any) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0

    # Keep the estimator lightweight and backend-agnostic for environments
    # where native token usage is unavailable.
    cjk_units = len(re.findall(r"[\u4e00-\u9fff]", normalized))
    non_cjk_text = re.sub(r"[\u4e00-\u9fff]", " ", normalized)
    word_like_units = len(re.findall(r"[A-Za-z0-9_]+|[^\s]", non_cjk_text))
    return cjk_units + word_like_units


class HGMMemorySystem:
    """Three-layer memory system with coarse-to-fine hierarchical retrieval."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        llm_backend: str = "sglang",
        llm_model: str = "gpt-4.1-mini-2025-04-14",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        check_connection: bool = False,
        hierarchy_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_name = model_name
        self.hierarchy_config = build_hierarchy_config(hierarchy_config)
        self.hierarchy_config_signature = self.hierarchy_config.signature()
        self.llm_controller = RobustLLMController(
            llm_backend,
            llm_model,
            api_key,
            api_base,
            sglang_host,
            sglang_port,
            check_connection,
        )

        self.turn_memories: "OrderedDict[str, RobustMemoryNote]" = OrderedDict()
        self.episode_memories: "OrderedDict[str, RobustMemoryNote]" = OrderedDict()
        self.topic_memories: "OrderedDict[str, RobustMemoryNote]" = OrderedDict()

        self.turn_order: List[str] = []
        self.episode_order: List[str] = []
        self.topic_order: List[str] = []

        self.session_turns: Dict[str, List[str]] = defaultdict(list)
        self.session_episode_ids_map: Dict[str, List[str]] = defaultdict(list)
        self.session_episode_map: Dict[str, str] = {}
        self.topic_episode_map: Dict[str, List[str]] = defaultdict(list)

        self.turn_retriever = SimpleEmbeddingRetriever(model_name)
        self.episode_retriever = SimpleEmbeddingRetriever(model_name)
        self.topic_retriever = SimpleEmbeddingRetriever(model_name)

        self.topic_match_threshold = self.hierarchy_config.topic_match_threshold
        self._episodes_since_recluster = 0
        self._topics_dirty = False
        self.topic_rebuild_count = 0

        self._layer_bm25_cache: Dict[str, Dict[str, Any]] = {}
        self._layer_recency_cache: Dict[str, Dict[str, Any]] = {}
        self._note_temporal_token_cache: Dict[str, set] = {}
        # Deferred turn analysis queue for batched turn analysis
        self._pending_turn_ids: List[str] = []

    def _invalidate_layer_runtime_cache(self, layer: Optional[str] = None):
        """Invalidate per-layer retrieval caches after corpus/order changes."""
        if layer is None:
            self._layer_bm25_cache.clear()
            self._layer_recency_cache.clear()
            self._note_temporal_token_cache.clear()
            return
        self._layer_bm25_cache.pop(layer, None)
        self._layer_recency_cache.pop(layer, None)
        self._note_temporal_token_cache.clear()

    @staticmethod
    def _layer_cache_signature(order: List[str], corpus: Optional[List[str]] = None) -> tuple:
        corpus_length = len(corpus) if corpus is not None else 0
        return (tuple(order), corpus_length)

    def _anchors_enabled(self) -> bool:
        return bool(getattr(self.hierarchy_config, "enable_anchors", False))

    def _build_retrieval_document(self, note: RobustMemoryNote) -> str:
        document = (
            f"layer:{note.layer} "
            f"type:{note.memory_type} "
            f"speaker:{note.speaker} "
            f"time:{note.timestamp} "
            f"title:{note.title} "
            f"content:{note.content} "
            f"context:{note.context} "
            f"keywords:{', '.join(note.keywords)} "
            f"tags:{', '.join(note.tags)}"
        )
        if not self._anchors_enabled():
            return document

        anchors_payload = getattr(note, "anchors", {}) or {}
        anchor_entities = anchors_payload.get("entities", []) if isinstance(anchors_payload, dict) else []
        anchor_times = anchors_payload.get("times", []) if isinstance(anchors_payload, dict) else []
        anchor_events = anchors_payload.get("events", []) if isinstance(anchors_payload, dict) else []
        anchors = _extract_anchor_terms(
            " ".join(
                [
                    str(getattr(note, "title", "") or ""),
                    str(getattr(note, "content", "") or ""),
                    str(getattr(note, "context", "") or ""),
                    str(getattr(note, "timestamp", "") or ""),
                ]
            )
        )
        return (
            f"{document} "
            f"anchors:{', '.join(anchors)} "
            f"anchor_entities:{', '.join(anchor_entities)} "
            f"anchor_times:{', '.join(anchor_times)} "
            f"anchor_events:{', '.join(anchor_events)}"
        )

    def _rebuild_turn_retriever(self):
        self.turn_order = list(self.turn_memories.keys())
        self.turn_retriever = SimpleEmbeddingRetriever(self.model_name)
        if self.turn_order:
            docs = [self._build_retrieval_document(self.turn_memories[note_id]) for note_id in self.turn_order]
            self.turn_retriever.add_documents(docs)
        self._invalidate_layer_runtime_cache("turn")

    def _rebuild_episode_retriever(self):
        self.episode_order = list(self.episode_memories.keys())
        self.episode_retriever = SimpleEmbeddingRetriever(self.model_name)
        if self.episode_order:
            docs = [self._build_retrieval_document(self.episode_memories[note_id]) for note_id in self.episode_order]
            self.episode_retriever.add_documents(docs)
        self._invalidate_layer_runtime_cache("episode")

    def _rebuild_topic_retriever(self):
        self.topic_order = list(self.topic_memories.keys())
        self.topic_retriever = SimpleEmbeddingRetriever(self.model_name)
        if self.topic_order:
            docs = [self._build_retrieval_document(self.topic_memories[note_id]) for note_id in self.topic_order]
            self.topic_retriever.add_documents(docs)
        self._invalidate_layer_runtime_cache("topic")

    def consolidate_memories(self):
        self._rebuild_turn_retriever()
        self._rebuild_episode_retriever()
        self._rebuild_topic_retriever()

    def _refresh_note_metadata(self, note: RobustMemoryNote):
        analysis = RobustMemoryNote.analyze_content(note.content, self.llm_controller)
        note.keywords = analysis["keywords"]
        note.context = analysis["context"]
        note.tags = analysis["tags"]
        note.last_accessed = datetime.now().strftime("%Y%m%d%H%M")

    def _select_evenly_spaced_notes(
        self,
        notes: List[RobustMemoryNote],
        limit: int,
    ) -> List[RobustMemoryNote]:
        if limit <= 0 or len(notes) <= limit:
            return list(notes)

        selected: List[RobustMemoryNote] = []
        seen_indices = set()
        for raw_idx in np.linspace(0, len(notes) - 1, num=limit, dtype=int):
            idx = int(raw_idx)
            if idx in seen_indices:
                continue
            seen_indices.add(idx)
            selected.append(notes[idx])
        return selected

    def _select_turn_notes_for_episode_summary(
        self,
        turn_notes: List[RobustMemoryNote],
    ) -> List[RobustMemoryNote]:
        return self._select_evenly_spaced_notes(
            turn_notes,
            self.hierarchy_config.episode_summary_turn_window,
        )

    def _select_episode_notes_for_topic_summary(
        self,
        episode_notes: List[RobustMemoryNote],
    ) -> List[RobustMemoryNote]:
        limit = self.hierarchy_config.topic_summary_episode_window
        if limit <= 0 or len(episode_notes) <= limit:
            return list(episode_notes)
        if self.hierarchy_config.topic_update_strategy == "recent":
            return list(episode_notes[-limit:])
        return self._select_evenly_spaced_notes(episode_notes, limit)

    def _episode_ids_in_order(self) -> List[str]:
        return [note_id for note_id in self.episode_order if note_id in self.episode_memories]

    def _build_episode_topic_document(self, note: RobustMemoryNote) -> str:
        """Build a semantic-only document for episode-level topic clustering."""
        raw_content = str(note.content or "")
        lines = [line.strip() for line in raw_content.splitlines() if line.strip()]
        if lines and lines[0].lower().startswith("session episode summary."):
            lines = lines[1:]

        cleaned_content = "\n".join(lines).strip() or _normalize_text(raw_content)
        cleaned_context = _normalize_text(note.context)
        if cleaned_context.lower() == "general":
            cleaned_context = ""

        parts = [cleaned_content]
        if cleaned_context:
            parts.append(f"context: {cleaned_context}")
        if note.keywords:
            parts.append("keywords: " + ", ".join(_unique_preserve_order(note.keywords)))
        if note.tags:
            parts.append("tags: " + ", ".join(_unique_preserve_order(note.tags)))
        return "\n".join(parts).strip()

    def _build_episode_topic_documents(self, episode_ids: List[str]) -> List[str]:
        documents = []
        for episode_id in episode_ids:
            note = self.episode_memories[episode_id]
            documents.append(self._build_episode_topic_document(note))
        return documents

    def _cluster_episode_ids_with_threshold(
        self,
        episode_ids: List[str],
        distance_matrix: np.ndarray,
        distance_threshold: float,
    ) -> List[List[str]]:
        try:
            model = AgglomerativeClustering(
                n_clusters=None,
                metric="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
            )
        except TypeError:
            model = AgglomerativeClustering(
                n_clusters=None,
                affinity="precomputed",
                linkage="average",
                distance_threshold=distance_threshold,
            )

        labels = model.fit_predict(distance_matrix)
        grouped_indices: Dict[int, List[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            grouped_indices[int(label)].append(idx)

        groups: List[List[str]] = []
        for _, indices in sorted(grouped_indices.items(), key=lambda item: min(item[1])):
            groups.append([episode_ids[idx] for idx in indices])
        return groups

    def _cluster_episode_ids(self, episode_ids: List[str]) -> List[List[str]]:
        if not episode_ids:
            return []
        if len(episode_ids) == 1:
            return [[episode_ids[0]]]

        documents = self._build_episode_topic_documents(episode_ids)
        embeddings = np.asarray(self.episode_retriever.model.encode(documents))
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        similarity_matrix = cosine_similarity(embeddings)
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)

        distance_threshold = self.hierarchy_config.topic_cluster_distance_threshold
        groups = self._cluster_episode_ids_with_threshold(episode_ids, distance_matrix, distance_threshold)

        min_size = max(1, self.hierarchy_config.topic_cluster_min_size)
        if min_size <= 1:
            return groups

        large_groups = [list(group) for group in groups if len(group) >= min_size]
        small_groups = [list(group) for group in groups if len(group) < min_size]

        if not large_groups:
            return groups

        id_to_index = {episode_id: idx for idx, episode_id in enumerate(episode_ids)}
        merge_threshold = self.hierarchy_config.topic_small_cluster_merge_threshold

        for group in small_groups:
            candidate_indices = [id_to_index[item] for item in group if item in id_to_index]
            if not candidate_indices:
                continue

            best_group_idx = None
            best_score = -1.0
            for lg_idx, large_group in enumerate(large_groups):
                large_indices = [id_to_index[item] for item in large_group if item in id_to_index]
                if not large_indices:
                    continue
                block = similarity_matrix[np.ix_(candidate_indices, large_indices)]
                score = float(np.mean(block)) if block.size else -1.0
                if score > best_score:
                    best_score = score
                    best_group_idx = lg_idx

            if best_group_idx is not None and best_score >= merge_threshold:
                large_groups[best_group_idx].extend(group)
            else:
                large_groups.append(group)

        normalized_groups = []
        for group in large_groups:
            deduped = _unique_preserve_order(group)
            if deduped:
                normalized_groups.append(deduped)
        return normalized_groups

    def _fallback_topic_title(self, episode_notes: List[RobustMemoryNote]) -> str:
        """Descriptive fallback title when LLM title generation is unavailable.

        Derives a short title from the episodes\' own content (context or first
        summary line) instead of copying a single episode\'s context verbatim.
        """
        candidates: List[str] = []
        for note in episode_notes:
            candidate = _normalize_text(note.context)
            if candidate and candidate.lower() != "general":
                candidates.append(candidate)
                continue
            for raw_line in str(note.content or "").splitlines():
                line = _normalize_text(raw_line)
                if not line:
                    continue
                if line.lower().startswith(("episode summary", "cross-session topic")):
                    continue
                line = re.sub(r"^[-*\u2022\s]+", "", line).strip()
                if line:
                    candidates.append(line)
                    break
        if candidates:
            return _unique_preserve_order(candidates)[0][:96]
        return f"Topic cluster of {len(episode_notes)} episodes"

    def _generate_topic_title(self, episode_notes: List[RobustMemoryNote]) -> str:
        """Generate a topic title by summarizing the corresponding episodes.

        The title is produced by the LLM from the episode summaries, rather than
        taken from the first non-"General" episode context. Falls back to a
        descriptive title when the LLM call fails or returns nothing useful.
        """
        if not episode_notes:
            return "Topic cluster of 0 episodes"

        selected_episode_notes = self._select_episode_notes_for_topic_summary(episode_notes)
        episodes_text = "\n\n".join(
            f"[Episode {idx + 1}]\n{note.content}"
            for idx, note in enumerate(selected_episode_notes)
        )
        prompt = TOPIC_TITLE_PROMPT.format(episodes=episodes_text)
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
        except Exception as exc:
            logger.warning("Topic title generation failed: %s", exc)
            return self._fallback_topic_title(episode_notes)

        title = _normalize_text(response)
        # Keep only the first non-empty line; strip common list/quote decorations.
        for raw_line in str(response or "").splitlines():
            line = _normalize_text(raw_line)
            if not line:
                continue
            line = re.sub(r"^(?:title|topic title|topic)\s*[:：-]?\s*", "", line, flags=re.IGNORECASE).strip()
            line = re.sub(r'^[\-\*\u2022\d.)\s]+', "", line).strip().strip("\"\'“”‘’").strip()
            if line and line.lower() not in {"none", "n/a", "null"}:
                title = line
                break
        if not title:
            return self._fallback_topic_title(episode_notes)
        return title[:96]

    def _derive_topic_title(self, episode_notes: List[RobustMemoryNote]) -> str:
        """Public topic-title entry point (kept for callers/tests).

        Title is summarized from the corresponding episodes, not copied from an
        episode context field.
        """
        return self._generate_topic_title(episode_notes)

    def _rebuild_topics_from_episode_clusters(self):
        episode_ids = self._episode_ids_in_order()
        new_topic_memories: "OrderedDict[str, RobustMemoryNote]" = OrderedDict()
        new_topic_episode_map: Dict[str, List[str]] = defaultdict(list)
        proposed_parent_ids: Dict[str, str] = {}

        if episode_ids:
            raw_groups = self._cluster_episode_ids(episode_ids)
            groups: List[List[str]] = []
            assigned_episode_ids = set()
            for raw_group in raw_groups or []:
                group = []
                for episode_id in raw_group or []:
                    if episode_id in self.episode_memories and episode_id not in assigned_episode_ids:
                        group.append(episode_id)
                        assigned_episode_ids.add(episode_id)
                if group:
                    groups.append(group)
            for episode_id in episode_ids:
                if episode_id not in assigned_episode_ids:
                    groups.append([episode_id])

            # Compute topic summary+title per cluster serially, then commit into
            # new_topic_memories.
            valid_groups: List[tuple] = []
            for group in groups:
                episode_notes = [
                    self.episode_memories[item]
                    for item in group
                    if item in self.episode_memories
                ]
                if not episode_notes:
                    continue
                valid_groups.append((group, episode_notes))

            def _compute_topic_llm(gpair):
                grp, enotes = gpair
                summary = self._generate_topic_summary(enotes)
                title = self._derive_topic_title(enotes)
                return (grp, enotes, summary, title)

            computed_topics: List[tuple] = []
            for gpair in valid_groups:
                computed_topics.append(_compute_topic_llm(gpair))

            for group, episode_notes, topic_summary, topic_title in computed_topics:
                topic_note = RobustMemoryNote(
                    content=topic_summary,
                    llm_controller=self.llm_controller,
                    timestamp=episode_notes[-1].timestamp,
                    memory_type="topic_cluster",
                    layer="topic",
                    speaker="mixed",
                    child_ids=list(group),
                    source_ids=list(group),
                    title=topic_title,
                    anchors=(
                        merge_structured_anchors(
                            [getattr(note, "anchors", {}) for note in episode_notes],
                            source_ids=list(group),
                        )
                        if self._anchors_enabled()
                        else {}
                    ),
                )
                new_topic_memories[topic_note.id] = topic_note
                new_topic_episode_map[topic_note.id] = list(group)
                for episode_note in episode_notes:
                    proposed_parent_ids[episode_note.id] = topic_note.id

        new_topic_order = list(new_topic_memories.keys())
        new_topic_retriever = SimpleEmbeddingRetriever(self.model_name)
        if new_topic_order:
            documents = [
                self._build_retrieval_document(new_topic_memories[note_id])
                for note_id in new_topic_order
            ]
            new_topic_retriever.add_documents(documents)

        self.topic_memories = new_topic_memories
        self.topic_episode_map = defaultdict(list, new_topic_episode_map)
        for episode_id in episode_ids:
            self.episode_memories[episode_id].parent_id = proposed_parent_ids.get(episode_id)
        self.topic_order = new_topic_order
        self.topic_retriever = new_topic_retriever
        self._invalidate_layer_runtime_cache("topic")

    def _refresh_topics_after_episode_changes(self, new_episode_count: int, force: bool = False) -> bool:
        strategy = self.hierarchy_config.topic_assignment_strategy
        if strategy != "clustered":
            return False

        self._episodes_since_recluster += max(0, int(new_episode_count))
        interval = max(1, self.hierarchy_config.topic_recluster_interval)

        if force or not self.topic_memories or self._episodes_since_recluster >= interval:
            self._rebuild_topics_from_episode_clusters()
            self._episodes_since_recluster = 0
            self._topics_dirty = False
            self.topic_rebuild_count += 1
            return True
        return False

    def _lexical_overlap(self, left: str, right: str) -> float:
        left_tokens = _token_set(left)
        right_tokens = _token_set(right)
        if not left_tokens or not right_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))

    def _split_indices_to_groups(self, n_turns: int, boundaries: List[int]) -> List[List[int]]:
        if n_turns <= 0:
            return []
        boundary_set = sorted({idx for idx in boundaries if 0 <= idx < n_turns})
        if not boundary_set or boundary_set[0] != 0:
            boundary_set = [0] + boundary_set

        groups: List[List[int]] = []
        for pos, start_idx in enumerate(boundary_set):
            end_idx = boundary_set[pos + 1] if pos + 1 < len(boundary_set) else n_turns
            if end_idx <= start_idx:
                continue
            groups.append(list(range(start_idx, end_idx)))
        return groups

    def _normalize_segment_groups(
        self,
        groups: List[List[int]],
        n_turns: int,
    ) -> List[List[int]]:
        # Episode segmentation trusts the produced boundaries (LLM or fallback).
        # No preset min/max turn bounds are enforced: episodes are neither merged
        # for being too short nor split for being too long. We only sanitize so the
        # output stays contiguous and covers the full session.
        if n_turns <= 0:
            return []
        if not groups:
            groups = [list(range(n_turns))]

        boundary_starts = [group[0] for group in groups if group and 0 <= group[0] < n_turns]
        groups = self._split_indices_to_groups(n_turns, boundary_starts)
        return [group for group in groups if group]

    def _segment_session_turn_notes_llm(
        self,
        turn_notes: List[RobustMemoryNote],
    ) -> List[List[int]]:
        n_turns = len(turn_notes)
        if n_turns <= 1:
            return [list(range(n_turns))] if n_turns else []

        dialogue_lines = []
        for idx, note in enumerate(turn_notes):
            content = _normalize_text(note.content)
            if len(content) > 220:
                content = content[:220].rstrip() + " ..."
            dialogue_lines.append(f"[{idx}] {note.speaker}: {content}")

        prompt = SESSION_SEGMENTATION_PROMPT.format(
            dialogue="\n".join(dialogue_lines),
        )

        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.0)
        except Exception as exc:
            logger.warning("Session segmentation LLM call failed: %s", exc)
            return []

        boundary_section = ""
        marker_match = re.search(r"BOUNDARIES\s*:\s*(.*)", response or "", flags=re.IGNORECASE)
        if marker_match:
            boundary_section = marker_match.group(1)
        else:
            boundary_section = response or ""

        candidate_boundaries = _parse_int_list(boundary_section)
        groups = self._split_indices_to_groups(n_turns, candidate_boundaries)
        return self._normalize_segment_groups(groups, n_turns)

    def _segment_session_turn_notes(self, turn_notes: List[RobustMemoryNote]) -> List[List[int]]:
        n_turns = len(turn_notes)
        if n_turns <= 1:
            return [list(range(n_turns))] if n_turns else []

        llm_groups = self._segment_session_turn_notes_llm(turn_notes)
        if llm_groups:
            return llm_groups
        return [list(range(n_turns))]

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        session_id = kwargs.get("session_id")
        session_key = _safe_session_key(session_id)
        structured_anchors = {}
        if self._anchors_enabled():
            structured_anchors = extract_structured_anchors(
                content,
                timestamp=time,
                speaker=kwargs.get("speaker"),
            )
        batch_mode = getattr(self.hierarchy_config, "analyze_turn_mode", "single") == "batch"
        note = RobustMemoryNote(
            content=content,
            llm_controller=None if batch_mode else self.llm_controller,
            timestamp=time,
            memory_type=kwargs.get("memory_type", "dialogue_turn"),
            speaker=kwargs.get("speaker"),
            layer="turn",
            anchors=structured_anchors,
        )
        if self._anchors_enabled():
            note.anchors.setdefault("source", {})
            note.anchors["source"].update(
                {
                    "id": note.id,
                    "speaker": note.speaker,
                    "timestamp": note.timestamp,
                    "session_id": session_key,
                }
            )
        self.turn_memories[note.id] = note
        self.turn_order.append(note.id)
        if batch_mode and hasattr(self, "_pending_turn_ids"):
            # Defer keyword/context/tag extraction and turn-embedding until
            # flush_pending_turn_analysis(), so each turn no longer costs one LLM call.
            self._pending_turn_ids.append(note.id)
        else:
            self.turn_retriever.add_documents([self._build_retrieval_document(note)])
            self._invalidate_layer_runtime_cache("turn")

        if session_key is not None:
            self.session_turns[session_key].append(note.id)
        return note.id

    def flush_pending_turn_analysis(self) -> None:
        """Batch-analyze all pending turn notes and index them.

        Called by finalize_session before any episode segmentation, so turn
        notes carry keywords/context/tags (which feed _build_retrieval_document)
        and get embedded exactly once.
        """
        pending = getattr(self, "_pending_turn_ids", None)
        if not pending:
            return
        pending_ids = list(pending)
        pending.clear()
        notes = [self.turn_memories[nid] for nid in pending_ids if nid in self.turn_memories]
        if not notes:
            return
        contents = [n.content for n in notes]
        try:
            analyses = RobustMemoryNote.analyze_contents_batch(contents, self.llm_controller)
        except Exception as exc:
            logger.error("flush_pending_turn_analysis batch failed: %s; degrading all to heuristics", exc)
            analyses = None
        for idx, n in enumerate(notes):
            if analyses is not None and idx < len(analyses):
                a = analyses[idx]
            else:
                from memory.prompts import _heuristic_keywords, _heuristic_context
                kw = _heuristic_keywords(n.content)
                a = {"keywords": kw, "context": _heuristic_context(n.content), "tags": kw[:3]}
            n.keywords = list(a.get("keywords", []))
            n.context = a.get("context", "") or n.context
            n.tags = list(a.get("tags", []))
        docs = [self._build_retrieval_document(n) for n in notes]
        self.turn_retriever.add_documents(docs)
        self._invalidate_layer_runtime_cache("turn")

    def _fallback_episode_summary(
        self,
        date_time: Optional[str],
        turn_notes: List[RobustMemoryNote],
    ) -> str:
        bullets = [f"- {note.speaker}: {note.content}" for note in turn_notes[:8]]
        header = "Episode summary"
        if date_time:
            header += f" on {date_time}"
        return header + "\n" + "\n".join(bullets)

    def _generate_episode_summary(
        self,
        date_time: Optional[str],
        turn_notes: List[RobustMemoryNote],
        episode_index: Optional[int] = None,
        total_episodes: Optional[int] = None,
    ) -> str:
        selected_turn_notes = self._select_turn_notes_for_episode_summary(turn_notes)
        dialogue = "\n".join(f"- {note.speaker}: {note.content}" for note in selected_turn_notes)
        prompt = SESSION_EPISODE_SUMMARY_PROMPT.format(
            min_bullets=self.hierarchy_config.episode_summary_min_bullets,
            max_bullets=self.hierarchy_config.episode_summary_max_bullets,
            date_time=date_time or "unknown",
            dialogue=dialogue,
        )
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            summary = parse_relevant_parts(response)
            summary = summary or self._fallback_episode_summary(date_time, selected_turn_notes)
        except Exception as exc:
            logger.warning("Episode summary generation failed: %s", exc)
            summary = self._fallback_episode_summary(date_time, selected_turn_notes)
        date_prefix = f"Date: {date_time}. " if date_time else ""
        segment_prefix = ""
        if episode_index is not None and total_episodes is not None:
            segment_prefix = f"Episode: {episode_index}/{total_episodes}. "
        return f"Episode summary. {segment_prefix}{date_prefix}\n{summary}".strip()

    def _fallback_topic_summary(self, episode_notes: List[RobustMemoryNote]) -> str:
        bullets = []
        for note in episode_notes[:6]:
            summary_line = note.context or note.content.splitlines()[0]
            bullets.append(f"- {summary_line}")
        return "Cross-session topic cluster.\n" + "\n".join(bullets)

    def _generate_topic_summary(self, episode_notes: List[RobustMemoryNote]) -> str:
        selected_episode_notes = self._select_episode_notes_for_topic_summary(episode_notes)
        episodes_text = "\n\n".join(
            f"[Episode {idx + 1}]\n{note.content}"
            for idx, note in enumerate(selected_episode_notes)
        )
        prompt = TOPIC_CLUSTER_SUMMARY_PROMPT.format(
            min_bullets=self.hierarchy_config.topic_summary_min_bullets,
            max_bullets=self.hierarchy_config.topic_summary_max_bullets,
            episodes=episodes_text,
        )
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
            summary = parse_relevant_parts(response)
            summary = summary or self._fallback_topic_summary(selected_episode_notes)
        except Exception as exc:
            logger.warning("Topic summary generation failed: %s", exc)
            summary = self._fallback_topic_summary(selected_episode_notes)
        return f"Cross-session topic summary.\n{summary}".strip()

    def _query_similarity_scores(
        self,
        retriever: SimpleEmbeddingRetriever,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if retriever.embeddings is None or not retriever.corpus:
            return np.array([])

        embeddings = np.asarray(retriever.embeddings)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        query_vector = None
        if query_embedding is not None:
            query_vector = np.asarray(query_embedding)
            if query_vector.ndim > 1:
                query_vector = query_vector.reshape(-1)

            # Defensive fallback: if dimensions mismatch, re-encode with the
            # retriever's model to keep similarity computation valid.
            if query_vector.shape[0] != embeddings.shape[1]:
                query_vector = None

        if query_vector is None:
            query_vector = np.asarray(retriever.model.encode([query])[0])

        return cosine_similarity([query_vector], embeddings)[0]

    def _best_topic_match(self, episode_note: RobustMemoryNote) -> Optional[str]:
        if not self.topic_order:
            return None
        query = self._build_retrieval_document(episode_note)
        similarities = self._query_similarity_scores(self.topic_retriever, query)
        if similarities.size == 0:
            return None

        episode_tokens = _token_set(query)
        min_overlap = self.hierarchy_config.topic_min_token_overlap
        best_topic_id = None
        best_score = -1.0
        for idx, topic_id in enumerate(self.topic_order):
            topic_note = self.topic_memories[topic_id]
            topic_doc = self._build_retrieval_document(topic_note)
            overlap = len(episode_tokens & _token_set(topic_doc))
            if overlap < min_overlap:
                continue
            lexical_overlap = self._lexical_overlap(query, topic_doc)
            score = float(similarities[idx]) + 0.03 * overlap + 0.08 * lexical_overlap
            if score > best_score:
                best_score = score
                best_topic_id = topic_id

        if best_score >= self.hierarchy_config.topic_match_threshold:
            return best_topic_id
        return None

    def _upsert_topic_for_episode(self, episode_note: RobustMemoryNote):
        matched_topic_id = self._best_topic_match(episode_note)
        if matched_topic_id is None:
            topic_note = RobustMemoryNote(
                content=self._generate_topic_summary([episode_note]),
                llm_controller=self.llm_controller,
                timestamp=episode_note.timestamp,
                memory_type="topic_cluster",
                layer="topic",
                speaker="mixed",
                child_ids=[episode_note.id],
                source_ids=[episode_note.id],
                title=self._derive_topic_title([episode_note]),
                anchors=(
                    merge_structured_anchors(
                        [getattr(episode_note, "anchors", {})],
                        source_ids=[episode_note.id],
                    )
                    if self._anchors_enabled()
                    else {}
                ),
            )
            self.topic_memories[topic_note.id] = topic_note
            self.topic_episode_map[topic_note.id] = [episode_note.id]
            episode_note.parent_id = topic_note.id
        else:
            topic_note = self.topic_memories[matched_topic_id]
            if episode_note.id not in topic_note.child_ids:
                topic_note.child_ids.append(episode_note.id)
            topic_note.source_ids = _unique_preserve_order(topic_note.source_ids + [episode_note.id])
            episode_note.parent_id = topic_note.id
            self.topic_episode_map[topic_note.id] = _unique_preserve_order(
                self.topic_episode_map.get(topic_note.id, []) + [episode_note.id]
            )
            child_notes = [
                self.episode_memories[child_id]
                for child_id in topic_note.child_ids
                if child_id in self.episode_memories
            ]
            topic_note.content = self._generate_topic_summary(child_notes)
            topic_note.anchors = (
                merge_structured_anchors(
                    [getattr(note, "anchors", {}) for note in child_notes],
                    source_ids=list(topic_note.child_ids),
                )
                if self._anchors_enabled()
                else {}
            )
            self._refresh_note_metadata(topic_note)

        self._rebuild_topic_retriever()

    def _update_topics_for_new_episodes(self, episode_ids: List[str]):
        if not episode_ids:
            return

        self._topics_dirty = True
        if self.hierarchy_config.topic_build_mode == "after_sample":
            return

        if self.hierarchy_config.topic_assignment_strategy == "clustered":
            self._refresh_topics_after_episode_changes(new_episode_count=len(episode_ids))
            return

        for episode_id in episode_ids:
            episode_note = self.episode_memories.get(episode_id)
            if episode_note is not None:
                self._upsert_topic_for_episode(episode_note)

    def finalize_topics(self) -> bool:
        """Build a complete Topic layer once all pending Episodes are available."""
        if not self._topics_dirty:
            return False

        if not self._episode_ids_in_order():
            self._topics_dirty = False
            return False

        self._rebuild_topics_from_episode_clusters()
        self._episodes_since_recluster = 0
        self._topics_dirty = False
        self.topic_rebuild_count += 1
        return True

    def _fallback_episode_title(self, episode_note: RobustMemoryNote, fallback_index: int) -> str:
        """Descriptive fallback title when LLM title generation is unavailable.

        Derives a short title from the episode\'s own context or first summary
        line instead of copying a single turn verbatim.
        """
        candidate = _normalize_text(episode_note.context)
        if candidate and candidate.lower() != "general":
            return candidate[:96]

        for raw_line in str(episode_note.content or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("episode summary."):
                continue
            line = re.sub(r"^[-*\u2022\s]+", "", line).strip()
            if line:
                return line[:96]

        return f"Episode {fallback_index}"

    def _generate_episode_title(
        self,
        turn_notes: List["RobustMemoryNote"],
        fallback_index: int,
        episode_note: Optional["RobustMemoryNote"] = None,
    ) -> str:
        """Generate an episode title by summarizing the dialogue turns.

        The title is produced by the LLM directly from the raw turn content of
        the episode (the same source used for the episode summary), rather than
        taken from the episode context field or re-summarizing the summary.
        Falls back to a descriptive title when the LLM call fails or returns
        nothing useful.
        """
        if not turn_notes:
            if episode_note is not None:
                return self._fallback_episode_title(episode_note, fallback_index)
            return f"Episode {fallback_index}"

        selected_turn_notes = self._select_turn_notes_for_episode_summary(turn_notes)
        dialogue = "\n".join(f"- {note.speaker}: {note.content}" for note in selected_turn_notes)
        if not _normalize_text(dialogue):
            note_for_fallback = episode_note if episode_note is not None else None
            if note_for_fallback is not None:
                return self._fallback_episode_title(note_for_fallback, fallback_index)
            return f"Episode {fallback_index}"

        prompt = EPISODE_TITLE_PROMPT.format(dialogue=dialogue)
        try:
            response = self.llm_controller.llm.get_completion(prompt, temperature=0.1)
        except Exception as exc:
            logger.warning("Episode title generation failed: %s", exc)
            note_for_fallback = episode_note if episode_note is not None else None
            if note_for_fallback is not None:
                return self._fallback_episode_title(note_for_fallback, fallback_index)
            return f"Episode {fallback_index}"

        for raw_line in str(response or "").splitlines():
            line = _normalize_text(raw_line)
            if not line:
                continue
            line = re.sub(r"^(?:title|episode title|episode)\s*[:：\-]?\s*", "", line, flags=re.IGNORECASE).strip()
            line = re.sub(r"^[\-\*\u2022\d.)\s]+", "", line).strip().strip("\"\'“”‘’").strip()
            if line and line.lower() not in {"none", "n/a", "null"}:
                return line[:96]
        note_for_fallback = episode_note if episode_note is not None else None
        if note_for_fallback is not None:
            return self._fallback_episode_title(note_for_fallback, fallback_index)
        return f"Episode {fallback_index}"

    def _derive_episode_title(self, episode_note: RobustMemoryNote, fallback_index: int) -> str:
        """Public episode-title entry point (kept for callers/tests).

        Title is summarized from the episode\'s raw dialogue turns, not copied
        from an episode context field or summary line.
        """
        turn_notes = [
            self.turn_memories[turn_id]
            for turn_id in episode_note.child_ids
            if turn_id in self.turn_memories
        ]
        return self._generate_episode_title(turn_notes, fallback_index, episode_note=episode_note)

    def finalize_session(self, session_id: Any, date_time: Optional[str] = None) -> Optional[str]:
        # Ensure this session's turns are analyzed+indexed before segmentation.
        self.flush_pending_turn_analysis()
        session_key = _safe_session_key(session_id)
        if session_key is None:
            return None
        turn_ids = self.session_turns.get(session_key, [])
        if not turn_ids:
            return None

        if session_key in self.session_episode_ids_map and self.session_episode_ids_map[session_key]:
            logger.info("Session %s already finalized; returning existing episode node", session_key)
            return self.session_episode_ids_map[session_key][0]

        turn_notes = [self.turn_memories[note_id] for note_id in turn_ids if note_id in self.turn_memories]
        if not turn_notes:
            return None

        segment_groups = self._segment_session_turn_notes(turn_notes)
        if not segment_groups:
            segment_groups = [list(range(len(turn_notes)))]

        created_episode_ids: List[str] = []
        next_episode_index = len(self.episode_memories) + 1
        total_segments = len(segment_groups)

        # Pre-compute per-episode summary serially.
        # Shared structures are mutated only in the serial commit phase below.
        prepared: List[tuple] = []
        pending_segments: List[tuple] = []
        for episode_pos, group_indices in enumerate(segment_groups, start=1):
            grouped_turn_ids = [turn_ids[idx] for idx in group_indices if 0 <= idx < len(turn_ids)]
            grouped_turn_notes = [self.turn_memories[note_id] for note_id in grouped_turn_ids if note_id in self.turn_memories]
            if not grouped_turn_notes:
                continue
            pending_segments.append((episode_pos, grouped_turn_ids, grouped_turn_notes))

        def _compute_episode_llm(seg_tuple):
            pos, gturn_ids, gturn_notes = seg_tuple
            summary = self._generate_episode_summary(
                date_time, gturn_notes,
                episode_index=pos, total_episodes=total_segments,
            )
            return (pos, gturn_ids, gturn_notes, summary)

        for seg_tuple in pending_segments:
            prepared.append(_compute_episode_llm(seg_tuple))

        # serial commit phase (writes shared structures); episode title uses the
        # public _derive_episode_title entry point to stay compatible with callers.
        for pos, gturn_ids, gturn_notes, summary_content in prepared:
            episode_note = RobustMemoryNote(
                content=summary_content,
                llm_controller=self.llm_controller,
                timestamp=date_time or gturn_notes[0].timestamp,
                memory_type="session_episode",
                speaker="mixed",
                layer="episode",
                child_ids=gturn_ids,
                source_ids=gturn_ids,
                title="",
                anchors=(
                    merge_structured_anchors(
                        [getattr(note, "anchors", {}) for note in gturn_notes],
                        source_ids=gturn_ids,
                    )
                    if self._anchors_enabled()
                    else {}
                ),
            )
            episode_note.title = self._derive_episode_title(episode_note, next_episode_index)
            next_episode_index += 1
            for turn_note in gturn_notes:
                turn_note.parent_id = episode_note.id
            self.episode_memories[episode_note.id] = episode_note
            created_episode_ids.append(episode_note.id)

        if not created_episode_ids:
            return None

        self.session_episode_ids_map[session_key] = created_episode_ids
        self.session_episode_map[session_key] = created_episode_ids[0]
        self._rebuild_episode_retriever()
        self._update_topics_for_new_episodes(created_episode_ids)
        return created_episode_ids[0]

    def _resolve_layer_state(self, layer: str):
        if layer == "turn":
            return self.turn_memories, self.turn_order, self.turn_retriever
        if layer == "episode":
            return self.episode_memories, self.episode_order, self.episode_retriever
        if layer == "topic":
            return self.topic_memories, self.topic_order, self.topic_retriever
        return OrderedDict(), [], None

    @staticmethod
    def _normalize_scores(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return np.array([], dtype=np.float32)
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper - lower <= 1e-8:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lower) / (upper - lower)).astype(np.float32)

    def _extract_temporal_tokens(self, text: str) -> List[str]:
        normalized = str(text or "").lower()
        candidates: List[str] = re.findall(r"(?:19|20)\d{2}", normalized)
        candidates.extend(re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", normalized))
        candidates.extend(re.findall(r"\b\d{1,2}:\d{2}\b", normalized))
        return _unique_preserve_order(candidates)

    @staticmethod
    def _score_distribution_uncertainty(
        scores: np.ndarray,
        temperature: float = 0.08,
        margin_scale: float = 0.08,
    ) -> Dict[str, float]:
        values = np.asarray(scores, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return {
                "uncertainty": 1.0,
                "normalized_margin": 0.0,
                "entropy": 1.0,
                "concentration": 0.0,
            }

        ordered = np.sort(values)[::-1]
        if ordered.size >= 2:
            margin_raw = float(ordered[0] - ordered[1])
            normalized_margin = float(
                np.clip(margin_raw / max(float(margin_scale), 1e-8), 0.0, 1.0)
            )
        else:
            normalized_margin = 1.0

        safe_temperature = max(float(temperature), 1e-6)
        shifted = (ordered - np.max(ordered)) / safe_temperature
        clipped = np.clip(shifted, -30.0, 30.0)
        probs = np.exp(clipped)
        probs = probs / max(float(np.sum(probs)), 1e-8)

        if probs.size > 1:
            entropy = float(-np.sum(probs * np.log(probs + 1e-12)) / np.log(float(probs.size)))
        else:
            entropy = 0.0

        concentration = float(probs[0]) if probs.size else 0.0
        uncertainty = (
            0.45 * (1.0 - normalized_margin)
            + 0.35 * entropy
            + 0.20 * (1.0 - concentration)
        )
        return {
            "uncertainty": float(np.clip(uncertainty, 0.0, 1.0)),
            "normalized_margin": normalized_margin,
            "entropy": float(np.clip(entropy, 0.0, 1.0)),
            "concentration": float(np.clip(concentration, 0.0, 1.0)),
        }

    @staticmethod
    def _classify_uncertainty_tier(
        score: float,
        low_threshold: float,
        high_threshold: float,
    ) -> str:
        normalized = float(np.clip(float(score), 0.0, 1.0))
        if normalized < float(low_threshold):
            return "low"
        if normalized >= float(high_threshold):
            return "high"
        return "mid"

    @staticmethod
    def _resolve_final_limits(
        base_limits: Dict[str, int],
        multiplier: float,
        max_items: Optional[int],
    ) -> Dict[str, int]:
        normalized_multiplier = max(0.0, float(multiplier))
        limits = {}
        for layer in ("topics", "episodes", "turns"):
            base_limit = int(base_limits.get(layer, 0))
            limits[layer] = max(0, int(np.ceil(base_limit * normalized_multiplier)))

        if max_items is None:
            return limits

        global_limit = max(0, int(max_items))
        priority = {"topics": 0, "episodes": 1, "turns": 2}
        while sum(limits.values()) > global_limit:
            reducible = [layer for layer, value in limits.items() if value > 0]
            if not reducible:
                break
            layer = max(reducible, key=lambda name: (limits[name], priority[name]))
            limits[layer] -= 1
        return limits

    @staticmethod
    def _set_jaccard(left: set, right: set) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 0.0
        return float(len(left & right) / len(union))

    def _routing_layer_statistics(
        self,
        layer: str,
        queries: List[str],
        query_embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        similarity_lookup: Optional[Dict[tuple, np.ndarray]] = None,
        probe_k: int = 8,
    ) -> Dict[str, float]:
        notes, order, retriever = self._resolve_layer_state(layer)
        del notes
        if retriever is None or not order or retriever.embeddings is None:
            return {
                "mean_uncertainty": 1.0,
                "max_uncertainty": 1.0,
                "mean_margin": 0.0,
                "mean_entropy": 1.0,
                "mean_concentration": 0.0,
                "agreement": 0.0,
            }

        uncertainties: List[float] = []
        margins: List[float] = []
        entropies: List[float] = []
        concentrations: List[float] = []
        top_id_sets: List[set] = []

        for query in queries:
            query_embedding = (
                query_embedding_lookup.get(query) if query_embedding_lookup else None
            )
            cache_key = (layer, query)
            similarities = None
            if similarity_lookup is not None:
                similarities = similarity_lookup.get(cache_key)
            if similarities is None:
                similarities = self._query_similarity_scores(
                    retriever,
                    query,
                    query_embedding=query_embedding,
                )
                if similarity_lookup is not None:
                    similarity_lookup[cache_key] = similarities
            if similarities.size == 0:
                continue

            top_count = min(len(order), max(2, probe_k))
            top_indices = np.argsort(similarities)[-top_count:][::-1]
            top_scores = np.asarray(similarities[top_indices], dtype=np.float32)

            distribution = self._score_distribution_uncertainty(
                top_scores,
                temperature=self.hierarchy_config.retrieval_uncertainty_temperature,
                margin_scale=self.hierarchy_config.retrieval_uncertainty_margin_scale,
            )
            uncertainties.append(distribution["uncertainty"])
            margins.append(distribution["normalized_margin"])
            entropies.append(distribution["entropy"])
            concentrations.append(distribution["concentration"])

            id_set = {
                order[int(idx)]
                for idx in top_indices[: min(5, len(top_indices))]
                if 0 <= int(idx) < len(order)
            }
            if id_set:
                top_id_sets.append(id_set)

        if not uncertainties:
            return {
                "mean_uncertainty": 1.0,
                "max_uncertainty": 1.0,
                "mean_margin": 0.0,
                "mean_entropy": 1.0,
                "mean_concentration": 0.0,
                "agreement": 0.0,
            }

        if len(top_id_sets) <= 1:
            agreement = 1.0 if top_id_sets else 0.0
        else:
            pair_scores: List[float] = []
            for left_idx in range(len(top_id_sets)):
                for right_idx in range(left_idx + 1, len(top_id_sets)):
                    pair_scores.append(
                        self._set_jaccard(top_id_sets[left_idx], top_id_sets[right_idx])
                    )
            agreement = float(np.mean(pair_scores)) if pair_scores else 1.0

        return {
            "mean_uncertainty": float(np.mean(uncertainties)),
            "max_uncertainty": float(np.max(uncertainties)),
            "mean_margin": float(np.mean(margins)),
            "mean_entropy": float(np.mean(entropies)),
            "mean_concentration": float(np.mean(concentrations)),
            "agreement": float(np.clip(agreement, 0.0, 1.0)),
        }

    def _estimate_budget_profile(
        self,
        queries: List[str],
        query_embedding_lookup: Optional[Dict[str, np.ndarray]] = None,
        similarity_lookup: Optional[Dict[tuple, np.ndarray]] = None,
        routing_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        layer_stats = {
            "topic": self._routing_layer_statistics(
                "topic",
                queries,
                query_embedding_lookup=query_embedding_lookup,
                similarity_lookup=similarity_lookup,
            ),
            "episode": self._routing_layer_statistics(
                "episode",
                queries,
                query_embedding_lookup=query_embedding_lookup,
                similarity_lookup=similarity_lookup,
            ),
            "turn": self._routing_layer_statistics(
                "turn",
                queries,
                query_embedding_lookup=query_embedding_lookup,
                similarity_lookup=similarity_lookup,
            ),
        }

        topic_uncertainty = layer_stats["topic"]["mean_uncertainty"]
        episode_uncertainty = layer_stats["episode"]["mean_uncertainty"]
        turn_uncertainty = layer_stats["turn"]["mean_uncertainty"]

        mean_uncertainty = float(
            0.25 * topic_uncertainty
            + 0.30 * episode_uncertainty
            + 0.45 * turn_uncertainty
        )
        mean_agreement = float(
            np.mean(
                [
                    layer_stats["topic"]["agreement"],
                    layer_stats["episode"]["agreement"],
                    layer_stats["turn"]["agreement"],
                ]
            )
        )
        disagreement = float(np.clip(1.0 - mean_agreement, 0.0, 1.0))
        turn_focus_gap = float(
            turn_uncertainty - 0.5 * (topic_uncertainty + episode_uncertainty)
        )

        routing_score = float(np.clip(0.80 * mean_uncertainty + 0.20 * disagreement, 0.0, 1.0))
        tier = self._classify_uncertainty_tier(
            routing_score,
            low_threshold=self.hierarchy_config.retrieval_uncertainty_low_threshold,
            high_threshold=self.hierarchy_config.retrieval_uncertainty_high_threshold,
        )
        complexity = {"low": 0, "mid": 3, "high": 6}[tier]

        routing_context = routing_context if isinstance(routing_context, dict) else {}
        temporal_sources = list(routing_context.get("temporal_sources", []) or [])
        temporal_sensitive = bool(routing_context.get("temporal_required", False))

        multiplier_map = {
            "low": self.hierarchy_config.retrieval_budget_low_multiplier,
            "mid": self.hierarchy_config.retrieval_budget_mid_multiplier,
            "high": self.hierarchy_config.retrieval_budget_high_multiplier,
        }
        multiplier = float(multiplier_map.get(tier, self.hierarchy_config.retrieval_budget_mid_multiplier))
        return {
            "tier": tier,
            "complexity": complexity,
            "routing_score": round(routing_score, 6),
            "routing_reasons": {
                "weighted_uncertainty": round(mean_uncertainty, 6),
                "query_disagreement": round(disagreement, 6),
                "low_threshold": float(self.hierarchy_config.retrieval_uncertainty_low_threshold),
                "high_threshold": float(self.hierarchy_config.retrieval_uncertainty_high_threshold),
            },
            "multiplier": multiplier,
            "temporal_sensitive": temporal_sensitive,
            "temporal_sources": temporal_sources,
            "query_count": len(queries),
            "uncertainty": {
                "weighted": round(mean_uncertainty, 6),
                "disagreement": round(disagreement, 6),
                "turn_focus_gap": round(turn_focus_gap, 6),
                "topic": layer_stats["topic"],
                "episode": layer_stats["episode"],
                "turn": layer_stats["turn"],
            },
        }

    def _layer_recency_scores(self, layer: str, order: List[str], notes: OrderedDict) -> np.ndarray:
        if not order:
            return np.array([], dtype=np.float32)
        signature = self._layer_cache_signature(order)
        cached = self._layer_recency_cache.get(layer)
        if cached and cached.get("signature") == signature:
            return np.asarray(cached.get("scores", []), dtype=np.float32)

        raw_values = []
        for index, note_id in enumerate(order):
            note = notes.get(note_id)
            ts = "" if note is None else "".join(re.findall(r"\d+", str(note.timestamp or "")))
            if ts:
                try:
                    raw_values.append(float(ts[:14]))
                    continue
                except ValueError:
                    pass
            raw_values.append(float(index + 1))
        scores = self._normalize_scores(np.asarray(raw_values, dtype=np.float32))
        self._layer_recency_cache[layer] = {
            "signature": signature,
            "scores": scores,
        }
        return scores

    def _layer_bm25_index(self, layer: str, corpus: List[str]) -> Optional[BM25Okapi]:
        if not corpus:
            return None
        signature = self._layer_cache_signature([], corpus)
        cached = self._layer_bm25_cache.get(layer)
        if cached and cached.get("signature") == signature:
            return cached.get("bm25")

        tokenized_docs = [_tokenize_for_bm25(doc) for doc in corpus]
        bm25 = BM25Okapi(tokenized_docs)
        self._layer_bm25_cache[layer] = {
            "signature": signature,
            "bm25": bm25,
        }
        return bm25

    def _layer_lexical_scores(self, layer: str, corpus: List[str], query_tokens: List[str]) -> np.ndarray:
        if not corpus:
            return np.array([], dtype=np.float32)
        if not query_tokens:
            return np.zeros(len(corpus), dtype=np.float32)

        try:
            bm25 = self._layer_bm25_index(layer, corpus)
            if bm25 is None:
                return np.zeros(len(corpus), dtype=np.float32)
            scores = np.asarray(bm25.get_scores(query_tokens), dtype=np.float32)
        except Exception as exc:
            logger.warning("BM25 scoring failed; falling back to zeros: %s", exc)
            scores = np.zeros(len(corpus), dtype=np.float32)
        return self._normalize_scores(scores)

    def _note_temporal_tokens(self, note: RobustMemoryNote) -> set:
        note_id = str(getattr(note, "id", "") or "")
        cached = self._note_temporal_token_cache.get(note_id)
        if cached is not None:
            return cached

        tokens = set(self._extract_temporal_tokens(self._build_retrieval_document(note)))
        if note_id:
            self._note_temporal_token_cache[note_id] = tokens
        return tokens

    def _temporal_alignment_score(
        self,
        query: str,
        note: RobustMemoryNote,
        temporal_tokens: Optional[List[str]] = None,
    ) -> float:
        if temporal_tokens is None:
            temporal_tokens = self._extract_temporal_tokens(query)
        if not temporal_tokens:
            return 0.0

        note_tokens = self._note_temporal_tokens(note)
        if not note_tokens:
            return 0.0

        matches = sum(1 for token in temporal_tokens if token in note_tokens)
        if matches > 0:
            return min(1.0, matches / max(1, len(temporal_tokens)))
        return 0.15

    def _resolve_retrieval_fusion_weights(self, budget_profile: Dict[str, Any]) -> Dict[str, float]:
        base = np.asarray(
            [
                self.hierarchy_config.retrieval_semantic_weight,
                self.hierarchy_config.retrieval_lexical_weight,
                self.hierarchy_config.retrieval_recency_weight,
            ],
            dtype=np.float32,
        )
        base = np.clip(base, 0.0, None)
        if not np.isfinite(base).all() or float(base.sum()) <= 0.0:
            base = np.asarray([0.68, 0.24, 0.08], dtype=np.float32)
        base = base / max(float(base.sum()), 1e-6)

        mode = str(getattr(self.hierarchy_config, "retrieval_weight_mode", "static") or "static").strip().lower()
        strength = float(getattr(self.hierarchy_config, "retrieval_adaptive_strength", 0.35))
        strength = float(np.clip(strength, 0.0, 1.0))

        if mode != "adaptive" or strength <= 0.0:
            return {
                "mode": "static",
                "semantic": float(base[0]),
                "lexical": float(base[1]),
                "recency": float(base[2]),
                "strength": 0.0,
            }

        uncertainty_payload = budget_profile.get("uncertainty", {}) if isinstance(budget_profile, dict) else {}
        if not isinstance(uncertainty_payload, dict):
            uncertainty_payload = {}

        uncertainty = float(np.clip(float(uncertainty_payload.get("weighted", 0.0)), 0.0, 1.0))
        disagreement = float(np.clip(float(uncertainty_payload.get("disagreement", 0.0)), 0.0, 1.0))
        turn_focus_gap = float(uncertainty_payload.get("turn_focus_gap", 0.0))
        turn_focus = float(np.clip(max(0.0, turn_focus_gap) / 0.30, 0.0, 1.0))
        complexity = float(np.clip(float(budget_profile.get("complexity", 0.0)) / 6.0, 0.0, 1.0))
        temporal_signal = 1.0 if bool(budget_profile.get("temporal_sensitive", False)) else 0.0

        # Soft adaptation: blend toward a query-conditioned target without hard tier switching.
        shift = np.asarray(
            [
                0.12 * complexity + 0.10 * disagreement + 0.06 * uncertainty - 0.08 * temporal_signal,
                -0.04 * complexity + 0.07 * (1.0 - disagreement) + 0.04 * temporal_signal,
                -0.03 * complexity + 0.04 * uncertainty + 0.14 * temporal_signal + 0.10 * turn_focus,
            ],
            dtype=np.float32,
        )
        target = np.clip(base + shift, 0.01, None)
        target = target / max(float(target.sum()), 1e-6)

        adapted = np.clip((1.0 - strength) * base + strength * target, 0.01, None)
        adapted = adapted / max(float(adapted.sum()), 1e-6)

        return {
            "mode": "adaptive",
            "semantic": float(adapted[0]),
            "lexical": float(adapted[1]),
            "recency": float(adapted[2]),
            "base_semantic": float(base[0]),
            "base_lexical": float(base[1]),
            "base_recency": float(base[2]),
            "strength": float(strength),
            "features": {
                "complexity": round(complexity, 6),
                "uncertainty": round(uncertainty, 6),
                "disagreement": round(disagreement, 6),
                "turn_focus": round(turn_focus, 6),
                "temporal_signal": round(temporal_signal, 6),
            },
        }

    def _apply_provenance_escalation(
        self,
        topic_scores: Dict[str, float],
        episode_scores: Dict[str, float],
        turn_scores: Dict[str, float],
        retrieve_k: int,
        active_layers: Optional[set] = None,
    ):
        active_layers = set(active_layers or {"topic", "episode", "turn"})
        support_target = max(1, int(self.hierarchy_config.retrieval_min_turn_support))
        provenance_boost = float(self.hierarchy_config.retrieval_provenance_boost)
        if provenance_boost <= 0.0:
            return

        if {"topic", "episode"}.issubset(active_layers):
            top_topics = sorted(topic_scores.items(), key=lambda item: item[1], reverse=True)[: max(1, retrieve_k)]
            for topic_id, topic_score in top_topics:
                topic_note = self.topic_memories.get(topic_id)
                if topic_note is None:
                    continue
                for position, episode_id in enumerate(topic_note.child_ids):
                    if episode_id not in self.episode_memories:
                        continue
                    episode_scores[episode_id] += (
                        topic_score
                        * self.hierarchy_config.topic_to_episode_boost
                        * provenance_boost
                        / max(1, position + 1)
                    )

        if not {"episode", "turn"}.issubset(active_layers):
            return

        ranked_episode_ids = [
            episode_id
            for episode_id, _ in sorted(
                episode_scores.items(),
                key=lambda item: item[1],
                reverse=True,
            )[: max(2, retrieve_k * 2)]
        ]
        for episode_id in ranked_episode_ids:
            episode_note = self.episode_memories.get(episode_id)
            if episode_note is None:
                continue

            child_turn_ids = [turn_id for turn_id in episode_note.child_ids if turn_id in self.turn_memories]
            if not child_turn_ids:
                continue

            support_count = sum(1 for turn_id in child_turn_ids if turn_scores.get(turn_id, 0.0) > 0.0)
            if support_count >= support_target:
                continue

            missing_support = support_target - support_count
            candidate_count = min(len(child_turn_ids), max(missing_support * 4, missing_support))
            for position, turn_id in enumerate(child_turn_ids[:candidate_count]):
                turn_scores[turn_id] += (
                    episode_scores.get(episode_id, 0.0)
                    * self.hierarchy_config.episode_to_turn_boost
                    * provenance_boost
                    / max(1, position + 1)
                )

    def _layer_items(
        self,
        layer: str,
        query: str,
        top_k: int,
        query_embedding: Optional[np.ndarray] = None,
        similarities: Optional[np.ndarray] = None,
        budget_multiplier: float = 1.0,
        temporal_sensitive: bool = False,
        fusion_weights: Optional[Dict[str, float]] = None,
    ) -> List[Dict[str, Any]]:
        if top_k <= 0:
            return []
        # Lazy-flush pending turn analysis on direct turn-layer queries too.
        if layer == "turn":
            self.flush_pending_turn_analysis()
        notes, order, retriever = self._resolve_layer_state(layer)
        if retriever is None or not order or retriever.embeddings is None:
            return []

        if similarities is None:
            similarities = self._query_similarity_scores(
                retriever,
                query,
                query_embedding=query_embedding,
            )
        if similarities.size == 0:
            return []

        semantic_scores = self._normalize_scores(np.asarray(similarities, dtype=np.float32))
        query_tokens = sorted(_token_set(query))
        lexical_scores = self._layer_lexical_scores(layer, retriever.corpus, query_tokens)
        recency_scores = self._layer_recency_scores(layer, order, notes)

        if lexical_scores.size != semantic_scores.size:
            lexical_scores = np.zeros_like(semantic_scores, dtype=np.float32)
        if recency_scores.size != semantic_scores.size:
            recency_scores = np.zeros_like(semantic_scores, dtype=np.float32)

        semantic_weight = float(self.hierarchy_config.retrieval_semantic_weight)
        lexical_weight = float(self.hierarchy_config.retrieval_lexical_weight)
        recency_weight = float(self.hierarchy_config.retrieval_recency_weight)
        if isinstance(fusion_weights, dict):
            semantic_weight = float(fusion_weights.get("semantic", semantic_weight))
            lexical_weight = float(fusion_weights.get("lexical", lexical_weight))
            recency_weight = float(fusion_weights.get("recency", recency_weight))

        normalized_weight_array = np.asarray(
            [semantic_weight, lexical_weight, recency_weight],
            dtype=np.float32,
        )
        normalized_weight_array = np.clip(normalized_weight_array, 0.0, None)
        if float(normalized_weight_array.sum()) <= 0.0:
            normalized_weight_array = np.asarray([0.68, 0.24, 0.08], dtype=np.float32)
        normalized_weight_array = normalized_weight_array / max(
            float(normalized_weight_array.sum()),
            1e-6,
        )

        combined_scores = (
            float(normalized_weight_array[0]) * semantic_scores
            + float(normalized_weight_array[1]) * lexical_scores
            + float(normalized_weight_array[2]) * recency_scores
        )

        if temporal_sensitive:
            temporal_bonus = float(self.hierarchy_config.retrieval_temporal_bonus)
            temporal_tokens = self._extract_temporal_tokens(query)
            for idx, note_id in enumerate(order):
                note = notes.get(note_id)
                if note is None:
                    continue
                combined_scores[idx] += temporal_bonus * self._temporal_alignment_score(
                    query,
                    note,
                    temporal_tokens=temporal_tokens,
                )

        candidate_k = max(top_k, int(round(max(0.25, budget_multiplier) * top_k)))
        candidate_k = min(len(order), max(1, candidate_k))
        top_indices = np.argsort(combined_scores)[-candidate_k:][::-1]

        results = []
        for rank, raw_idx in enumerate(top_indices):
            idx = int(raw_idx)
            note_id = order[idx]
            note = notes.get(note_id)
            if note is None:
                continue
            results.append(
                {
                    "id": note_id,
                    "layer": layer,
                    "note": note,
                    "score": float(combined_scores[idx]) + (0.01 / (rank + 1)),
                    "semantic_score": float(semantic_scores[idx]),
                    "lexical_score": float(lexical_scores[idx]),
                    "recency_score": float(recency_scores[idx]),
                }
            )
        return results

    def search_hierarchy(
        self,
        queries: List[str],
        retrieve_k: int = 10,
        routing_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Lazy-flush pending turn analysis so the turn layer is indexed even
        # when callers add_note() without finalize_session() (e.g. unit tests).
        self.flush_pending_turn_analysis()
        unique_queries = _unique_preserve_order(list(queries))
        if not unique_queries:
            return {"topics": [], "episodes": [], "turns": [], "routing": {}}

        query_embedding_lookup: Dict[str, np.ndarray] = {}
        similarity_lookup: Dict[tuple, np.ndarray] = {}
        try:
            batched_embeddings = np.asarray(self.turn_retriever.model.encode(unique_queries))
            if batched_embeddings.ndim == 1:
                batched_embeddings = batched_embeddings.reshape(1, -1)
            if batched_embeddings.shape[0] == len(unique_queries):
                query_embedding_lookup = {
                    query: batched_embeddings[idx]
                    for idx, query in enumerate(unique_queries)
                }
        except Exception as exc:
            logger.warning(
                "Batch query embedding failed; falling back to per-query encoding: %s",
                exc,
            )

        uncertainty_routing_enabled = bool(
            getattr(self.hierarchy_config, "enable_uncertainty_routing", True)
        )
        if uncertainty_routing_enabled:
            budget_profile = self._estimate_budget_profile(
                unique_queries,
                query_embedding_lookup=query_embedding_lookup,
                similarity_lookup=similarity_lookup,
                routing_context=routing_context,
            )
            budget_multiplier = max(0.0, float(budget_profile.get("multiplier", 1.0)))
            temporal_sensitive = bool(budget_profile.get("temporal_sensitive", False))
            fusion_weights = self._resolve_retrieval_fusion_weights(budget_profile)
        else:
            budget_profile = {
                "tier": "fixed",
                "complexity": 0,
                "multiplier": 1.0,
                "temporal_sensitive": False,
                "query_count": len(unique_queries),
                "uncertainty": {
                    "weighted": 0.0,
                    "disagreement": 0.0,
                    "turn_focus_gap": 0.0,
                },
            }
            budget_multiplier = 1.0
            temporal_sensitive = False

            static_weights = np.asarray(
                [
                    self.hierarchy_config.retrieval_semantic_weight,
                    self.hierarchy_config.retrieval_lexical_weight,
                    self.hierarchy_config.retrieval_recency_weight,
                ],
                dtype=np.float32,
            )
            static_weights = np.clip(static_weights, 0.0, None)
            if float(static_weights.sum()) <= 0.0:
                static_weights = np.asarray([0.68, 0.24, 0.08], dtype=np.float32)
            static_weights = static_weights / max(float(static_weights.sum()), 1e-6)
            fusion_weights = {
                "mode": "static",
                "semantic": float(static_weights[0]),
                "lexical": float(static_weights[1]),
                "recency": float(static_weights[2]),
                "strength": 0.0,
            }

        requested_limits = {
            "topics": max(0, int(self.hierarchy_config.topic_result_limit)),
            "episodes": max(0, int(self.hierarchy_config.episode_result_limit)),
            "turns": max(0, int(self.hierarchy_config.turn_result_limit)),
        }
        disabled_layers = {
            layer: requested_limits[layer] == 0
            for layer in ("topics", "episodes", "turns")
        }
        active_layers = {
            layer[:-1]
            for layer in ("topics", "episodes", "turns")
            if not disabled_layers[layer]
        }

        topic_scores: Dict[str, float] = defaultdict(float)
        episode_scores: Dict[str, float] = defaultdict(float)
        turn_scores: Dict[str, float] = defaultdict(float)

        def _candidate_limit(layer: str, extra_items: int, capacity: int) -> int:
            if disabled_layers[layer]:
                return 0
            requested = requested_limits[layer]
            expanded = max(0, int(np.ceil((requested + extra_items) * budget_multiplier)))
            return min(expanded, max(0, capacity))

        topic_limit = _candidate_limit("topics", 1, len(self.topic_order) or retrieve_k)
        episode_limit = _candidate_limit("episodes", 2, len(self.episode_order) or retrieve_k)
        turn_limit = _candidate_limit("turns", requested_limits["turns"], len(self.turn_order) or (retrieve_k * 2))

        for query in unique_queries:
            query_embedding = query_embedding_lookup.get(query)
            topic_hits = [] if disabled_layers["topics"] else self._layer_items(
                "topic", query, topic_limit, query_embedding=query_embedding,
                similarities=similarity_lookup.get(("topic", query)),
                budget_multiplier=budget_multiplier, temporal_sensitive=temporal_sensitive,
                fusion_weights=fusion_weights,
            )
            episode_hits = [] if disabled_layers["episodes"] else self._layer_items(
                "episode", query, episode_limit, query_embedding=query_embedding,
                similarities=similarity_lookup.get(("episode", query)),
                budget_multiplier=budget_multiplier, temporal_sensitive=temporal_sensitive,
                fusion_weights=fusion_weights,
            )
            turn_hits = [] if disabled_layers["turns"] else self._layer_items(
                "turn", query, turn_limit, query_embedding=query_embedding,
                similarities=similarity_lookup.get(("turn", query)),
                budget_multiplier=budget_multiplier, temporal_sensitive=temporal_sensitive,
                fusion_weights=fusion_weights,
            )

            for hit in topic_hits:
                topic_scores[hit["id"]] += hit["score"]
            for hit in episode_hits:
                episode_scores[hit["id"]] += hit["score"]
            for hit in turn_hits:
                turn_scores[hit["id"]] += hit["score"]

            for hit in topic_hits:
                for child_id in hit["note"].child_ids:
                    episode_scores[child_id] += hit["score"] * self.hierarchy_config.topic_to_episode_boost

            for hit in episode_hits if {"topic", "turn"} & active_layers else []:
                episode_note = hit["note"]
                if "topic" in active_layers and episode_note.parent_id:
                    topic_scores[episode_note.parent_id] += (
                        hit["score"] * self.hierarchy_config.episode_to_topic_boost
                    )
                if "turn" in active_layers:
                    child_boost = hit["score"] * self.hierarchy_config.episode_to_turn_boost
                    for position, child_id in enumerate(episode_note.child_ids):
                        turn_scores[child_id] += child_boost / max(position + 1, 1)

            for hit in turn_hits if {"topic", "episode"} & active_layers else []:
                turn_note = hit["note"]
                if "episode" in active_layers and turn_note.parent_id:
                    episode_scores[turn_note.parent_id] += (
                        hit["score"] * self.hierarchy_config.turn_to_episode_boost
                    )
                    episode_note = self.episode_memories.get(turn_note.parent_id)
                    if "topic" in active_layers and episode_note and episode_note.parent_id:
                        topic_scores[episode_note.parent_id] += (
                            hit["score"] * self.hierarchy_config.turn_to_topic_boost
                        )

        self._apply_provenance_escalation(
            topic_scores, episode_scores, turn_scores, retrieve_k, active_layers=active_layers
        )

        def _rank(
            score_map: Dict[str, float],
            memory_map: OrderedDict,
            limit: int,
            layer: str,
        ) -> List[Dict[str, Any]]:
            if limit <= 0:
                return []
            ranked = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
            results = []
            for note_id, score in ranked[:limit]:
                note = memory_map.get(note_id)
                if note is None:
                    continue
                results.append(
                    {
                        "id": note_id,
                        "layer": layer,
                        "note": note,
                        "score": float(score),
                    }
                )
            return results

        base_final_limits = dict(requested_limits)
        final_limits = self._resolve_final_limits(
            base_final_limits,
            multiplier=budget_multiplier,
            max_items=self.hierarchy_config.retrieval_final_max_items,
        )

        final_results = {
            "topics": _rank(topic_scores, self.topic_memories, final_limits["topics"], "topic"),
            "episodes": _rank(episode_scores, self.episode_memories, final_limits["episodes"], "episode"),
            "turns": _rank(turn_scores, self.turn_memories, final_limits["turns"], "turn"),
        }
        return {
            **final_results,
            "routing": {
                **budget_profile,
                "fusion_weights": fusion_weights,
                "uncertainty_routing_enabled": uncertainty_routing_enabled,
                "candidate_limits": {
                    "topics": topic_limit,
                    "episodes": episode_limit,
                    "turns": turn_limit,
                },
                "requested_limits": requested_limits,
                "final_limits": final_limits,
                "final_counts": {
                    layer: len(final_results[layer])
                    for layer in ("topics", "episodes", "turns")
                },
                "disabled_layers": disabled_layers,
            },
        }

    def render_hierarchy_search_results(self, search_results: Dict[str, List[Dict[str, Any]]]) -> str:
        def _display_id(note_id: str, layer: str) -> str:
            if layer == "topic":
                prefix = "T"
                order = self.topic_order
            elif layer == "episode":
                prefix = "E"
                order = self.episode_order
            else:
                prefix = "R"
                order = self.turn_order

            try:
                idx = order.index(note_id) + 1
                return f"{prefix}{idx:03d}"
            except ValueError:
                return note_id

        sections: List[str] = []
        for label, items in (
            ("Topic", search_results.get("topics", [])),
            ("Episode", search_results.get("episodes", [])),
            ("Turn", search_results.get("turns", [])),
        ):
            for order, item in enumerate(items, start=1):
                note = item["note"]
                note_id = item.get("id") or getattr(note, "id", "unknown")
                lines = [
                    f"[{label} {order} | score={item['score']:.4f}]",
                    f"ID: {_display_id(note_id, note.layer)}",
                    f"Layer: {note.layer}",
                    f"Time: {getattr(note, 'timestamp', 'unknown')}",
                ]
                if label == "Topic":
                    lines.append(
                        "Episode IDs: "
                        + (
                            ", ".join(_display_id(child_id, "episode") for child_id in note.child_ids)
                            if note.child_ids
                            else "none"
                        )
                    )
                elif label == "Episode":
                    lines.append(
                        f"Topic ID: {_display_id(note.parent_id, 'topic') if note.parent_id else 'none'}"
                    )
                elif label == "Turn":
                    lines.append(
                        f"Episode ID: {_display_id(note.parent_id, 'episode') if note.parent_id else 'none'}"
                    )
                if label == "Turn":
                    lines.append(f"Speaker: {note.speaker}")
                if note.title:
                    lines.append(f"Title: {note.title}")
                lines.extend(
                    [
                        f"Content: {note.content}",
                        f"Context: {note.context}",
                        f"Keywords: {', '.join(note.keywords)}",
                        f"Tags: {', '.join(note.tags)}",
                    ]
                )
                sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def export_state(self) -> Dict[str, Any]:
        return {
            "schema_version": "hierarchical_memory_v5",
            "model_name": self.model_name,
            "hierarchy_config": self.hierarchy_config.to_dict(),
            "hierarchy_config_signature": self.hierarchy_config_signature,
            "turn_memories": {note_id: note.to_dict() for note_id, note in self.turn_memories.items()},
            "episode_memories": {note_id: note.to_dict() for note_id, note in self.episode_memories.items()},
            "topic_memories": {note_id: note.to_dict() for note_id, note in self.topic_memories.items()},
            "session_turns": {key: list(value) for key, value in self.session_turns.items()},
            "session_episode_ids_map": {
                key: list(value) for key, value in self.session_episode_ids_map.items()
            },
            "session_episode_map": dict(self.session_episode_map),
            "topic_episode_map": {key: list(value) for key, value in self.topic_episode_map.items()},
            "episodes_since_recluster": self._episodes_since_recluster,
            "topics_dirty": self._topics_dirty,
            "topic_rebuild_count": self.topic_rebuild_count,
        }

    def load_state(self, state: Dict[str, Any]):
        schema_version = state.get("schema_version")
        if schema_version not in {"hierarchical_memory_v2", "hierarchical_memory_v3", "hierarchical_memory_v4", "hierarchical_memory_v5"}:
            raise ValueError(
                f"Unsupported memory cache schema: {schema_version}. "
                "Please clear the cache directory and rebuild memories."
            )

        # A loaded cache never has pending (un-analyzed) turns.
        self._pending_turn_ids = []
        cached_signature = state.get("hierarchy_config_signature")
        runtime_config = self.hierarchy_config.to_dict()
        runtime_non_retrieval_config = _config_without_retrieval_only_keys(runtime_config)

        cached_config_raw = state.get("hierarchy_config") or {}
        required_construction_fields = {
            "topic_build_mode",
            "enable_anchors",
            "topic_recluster_interval",
        }
        if not required_construction_fields.issubset(cached_config_raw):
            raise ValueError(
                "Cached hierarchy config does not match the requested runtime config. "
                "Please rebuild the cache for this configuration."
            )
        cached_config = (
            HierarchyMemoryConfig.from_dict(cached_config_raw).to_dict()
            if cached_config_raw
            else {}
        )
        cached_non_retrieval_config = _config_without_retrieval_only_keys(cached_config)

        if cached_signature and cached_signature != self.hierarchy_config_signature:
            if not cached_config or cached_non_retrieval_config != runtime_non_retrieval_config:
                raise ValueError(
                    "Cached hierarchy config does not match the requested runtime config. "
                    "Please rebuild the cache for this configuration."
                )
            logger.info(
                "Hierarchy retrieval-only config changed (%s); reusing cached memory state.",
                ", ".join(sorted(_RETRIEVAL_ONLY_CONFIG_KEYS)),
            )

        if cached_config:
            if cached_non_retrieval_config != runtime_non_retrieval_config:
                raise ValueError(
                    "Cached hierarchy config payload does not match the requested runtime config. "
                    "Please rebuild the cache for this configuration."
                )

            changed_retrieval_keys = [
                key
                for key in sorted(_RETRIEVAL_ONLY_CONFIG_KEYS)
                if cached_config.get(key) != runtime_config.get(key)
            ]
            if changed_retrieval_keys:
                logger.info(
                    "Applying runtime retrieval-only config without cache rebuild: %s",
                    ", ".join(changed_retrieval_keys),
                )

        self.turn_memories = OrderedDict(
            (note_id, RobustMemoryNote.from_dict(note_dict))
            for note_id, note_dict in state.get("turn_memories", {}).items()
        )
        self.episode_memories = OrderedDict(
            (note_id, RobustMemoryNote.from_dict(note_dict))
            for note_id, note_dict in state.get("episode_memories", {}).items()
        )
        self.topic_memories = OrderedDict(
            (note_id, RobustMemoryNote.from_dict(note_dict))
            for note_id, note_dict in state.get("topic_memories", {}).items()
        )
        self.session_turns = defaultdict(
            list,
            {str(key): list(value) for key, value in state.get("session_turns", {}).items()},
        )
        loaded_episode_ids_map = state.get("session_episode_ids_map") or {}
        if not loaded_episode_ids_map:
            legacy_map = state.get("session_episode_map", {})
            loaded_episode_ids_map = {
                str(key): ([value] if value is not None else [])
                for key, value in legacy_map.items()
            }
        self.session_episode_ids_map = defaultdict(
            list,
            {str(key): [str(item) for item in value] for key, value in loaded_episode_ids_map.items()},
        )
        self.session_episode_map = {
            str(key): value for key, value in state.get("session_episode_map", {}).items()
        }
        if not self.session_episode_map:
            self.session_episode_map = {
                key: value_list[0]
                for key, value_list in self.session_episode_ids_map.items()
                if value_list
            }
        self.topic_episode_map = defaultdict(
            list,
            {str(key): list(value) for key, value in state.get("topic_episode_map", {}).items()},
        )
        self._episodes_since_recluster = int(state.get("episodes_since_recluster", 0))
        self._topics_dirty = bool(
            state.get("topics_dirty", bool(self.episode_memories))
        )
        self.topic_rebuild_count = int(state.get("topic_rebuild_count", 0))


        self.consolidate_memories()
