"""Lightweight memory note used across turn, episode, and topic layers."""

import logging
import re
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("amem_robust")


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


class RobustMemoryNote:
    """Memory node used across turn, episode, and topic layers."""

    def __init__(
        self,
        content: str,
        id: Optional[str] = None,
        keywords: Optional[List[str]] = None,
        links: Optional[List[Any]] = None,
        importance_score: Optional[float] = None,
        retrieval_count: Optional[int] = None,
        timestamp: Optional[str] = None,
        last_accessed: Optional[str] = None,
        context: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        speaker: Optional[str] = None,
        memory_type: Optional[str] = None,
        llm_controller: Any = None,
        layer: str = "turn",
        child_ids: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
        source_ids: Optional[List[str]] = None,
        title: Optional[str] = None,
        anchors: Optional[Dict[str, Any]] = None,
    ):
        self.content = str(content or "")

        if llm_controller and any(param is None for param in [keywords, context, category, tags]):
            analysis = self.analyze_content(self.content, llm_controller)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = list(keywords or [])
        self.links = list(links or [])
        self.importance_score = importance_score if importance_score is not None else 1.0
        self.retrieval_count = retrieval_count if retrieval_count is not None else 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time
        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)
        self.category = category or "Uncategorized"
        self.tags = list(tags or [])
        self.speaker = speaker or "unknown"
        self.memory_type = memory_type or "dialogue_turn"
        self.layer = layer
        self.child_ids = list(child_ids or [])
        self.parent_id = parent_id
        self.source_ids = list(source_ids or [])
        self.title = _normalize_text(title)
        self.anchors = dict(anchors or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "id": self.id,
            "keywords": self.keywords,
            "links": self.links,
            "importance_score": self.importance_score,
            "retrieval_count": self.retrieval_count,
            "timestamp": self.timestamp,
            "last_accessed": self.last_accessed,
            "context": self.context,
            "category": self.category,
            "tags": self.tags,
            "speaker": self.speaker,
            "memory_type": self.memory_type,
            "layer": self.layer,
            "child_ids": self.child_ids,
            "parent_id": self.parent_id,
            "source_ids": self.source_ids,
            "title": self.title,
            "anchors": self.anchors,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RobustMemoryNote":
        return cls(
            content=data.get("content", ""),
            id=data.get("id"),
            keywords=data.get("keywords"),
            links=data.get("links"),
            importance_score=data.get("importance_score"),
            retrieval_count=data.get("retrieval_count"),
            timestamp=data.get("timestamp"),
            last_accessed=data.get("last_accessed"),
            context=data.get("context"),
            category=data.get("category"),
            tags=data.get("tags"),
            speaker=data.get("speaker"),
            memory_type=data.get("memory_type"),
            layer=data.get("layer", "turn"),
            child_ids=data.get("child_ids"),
            parent_id=data.get("parent_id"),
            source_ids=data.get("source_ids"),
            title=data.get("title"),
            anchors=data.get("anchors"),
        )

    @staticmethod
    def analyze_content(content: str, llm_controller: Any) -> Dict[str, Any]:
        from memory.prompts import ANALYZE_CONTENT_PROMPT, FOCUSED_KEYWORDS_PROMPT, parse_analyze_content, validate_analysis_result, _parse_list_items, _heuristic_context, _heuristic_keywords

        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse; retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                analysis["keywords"] = _parse_list_items(retry_response)
            return validate_analysis_result(analysis, content)
        except Exception as exc:
            logger.error("Error analyzing content: %s", exc)
            keywords = _heuristic_keywords(content)
            return {
                "keywords": keywords,
                "context": _heuristic_context(content),
                "tags": keywords[:3],
            }

    @staticmethod
    def analyze_contents_batch(
        contents: List[str],
        llm_controller: Any,
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Analyze many turn contents in one LLM call per batch.

        Returns a list aligned with ``contents`` (same length, same order), each
        element being ``{keywords, context, tags}``. Single-element failure does
        NOT retry with FOCUSED_KEYWORDS_PROMPT; it degrades to heuristics for
        that element only. A whole-batch failure degrades all to heuristics.
        """
        if not contents:
            return []
        from memory.prompts import (
            ANALYZE_CONTENT_BATCH_PROMPT,
            parse_analyze_content_batch,
        )

        bs = max(1, int(batch_size or int(__import__("os").getenv("OPENAI_ANALYZE_BATCH_SIZE", "16"))))
        results: List[Dict[str, Any]] = []
        for start in range(0, len(contents), bs):
            chunk = contents[start:start + bs]
            snippets = chr(10).join(
                f"[{i + 1}] {chunk[i]}" for i in range(len(chunk))
            )
            prompt = ANALYZE_CONTENT_BATCH_PROMPT.format(n=len(chunk), snippets=snippets)
            try:
                response = llm_controller.llm.get_completion(prompt, temperature=0.1)
                results.extend(parse_analyze_content_batch(response, len(chunk), chunk))
            except Exception as exc:
                logger.error("Batch analyze failed (offset %d): %s; degrading chunk to heuristics", start, exc)
                from memory.prompts import _heuristic_keywords, _heuristic_context
                for c in chunk:
                    kw = _heuristic_keywords(c)
                    results.append({"keywords": kw, "context": _heuristic_context(c), "tags": kw[:3]})
        return results
