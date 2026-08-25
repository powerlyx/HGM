"""Lightweight structured anchors for memory retrieval and evidence checks."""

import re
from typing import Any, Dict, Iterable, List, Optional


_MONTH_PATTERN = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)

_EVENT_HEADS = (
    "support group",
    "workshop",
    "conference",
    "meeting",
    "event",
    "race",
    "trip",
    "hike",
    "class",
    "course",
    "job",
    "promotion",
    "accident",
    "party",
    "program",
    "test",
    "group",
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique(values: Iterable[Any], limit: Optional[int] = None) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        cleaned = _normalize_text(value).strip(" .,;:!?()[]{}\"'")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
        if limit is not None and len(result) >= limit:
            break
    return result


def _token_set(text: str) -> set:
    return {
        token
        for token in re.findall(r"[A-Za-z0-9]+", str(text or "").lower())
        if len(token) > 1
    }


def extract_time_anchors(text: str) -> List[str]:
    raw = str(text or "")
    candidates: List[str] = []
    candidates.extend(re.findall(r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b", raw))
    candidates.extend(
        re.findall(
            rf"\b\d{{1,2}}\s+(?:{_MONTH_PATTERN}),?\s+(?:19|20)\d{{2}}\b",
            raw,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(
        re.findall(
            rf"\b(?:{_MONTH_PATTERN})\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}}\b",
            raw,
            flags=re.IGNORECASE,
        )
    )
    candidates.extend(re.findall(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b", raw))
    candidates.extend(re.findall(r"\b(?:19|20)\d{2}\b", raw))
    return _unique(candidates, limit=12)


def extract_entity_anchors(text: str, speaker: Optional[str] = None) -> List[str]:
    raw = str(text or "")
    candidates: List[str] = []
    if speaker:
        candidates.append(str(speaker))
    candidates.extend(
        re.findall(r"\b[A-Z][a-zA-Z0-9_-]{2,}(?:\s+[A-Z][a-zA-Z0-9_-]{2,})?\b", raw)
    )
    stop = {"Speaker", "Image", "Date", "Episode", "The", "This", "That"}
    return [item for item in _unique(candidates, limit=16) if item not in stop]


def extract_event_anchors(text: str) -> List[str]:
    raw = _normalize_text(text)
    candidates: List[str] = []
    for head in _EVENT_HEADS:
        pattern = rf"\b((?:[A-Za-z0-9+'-]+\s+){{0,4}}{re.escape(head)})\b"
        for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
            phrase = match.group(1)
            phrase = re.sub(
                r"^(?:i|we|he|she|they|you)\s+"
                r"(?:attended|joined|went to|visited|started|took|had|saw|did)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            )
            phrase = re.sub(
                r"^(?:attended|joined|went to|visited|started|took|had|saw|did)\s+",
                "",
                phrase,
                flags=re.IGNORECASE,
            )
            phrase = re.sub(r"^the\s+", "", phrase, flags=re.IGNORECASE)
            candidates.append(phrase)
    return _unique(candidates, limit=12)


def extract_structured_anchors(
    text: str,
    timestamp: Optional[str] = None,
    speaker: Optional[str] = None,
    source_id: Optional[str] = None,
) -> Dict[str, Any]:
    joined = " ".join(part for part in [str(text or ""), str(timestamp or "")] if part)
    return {
        "entities": extract_entity_anchors(joined, speaker=speaker),
        "times": extract_time_anchors(joined),
        "events": extract_event_anchors(text),
        "source": {
            "id": source_id,
            "speaker": speaker,
            "timestamp": timestamp,
        },
    }


def merge_structured_anchors(
    anchors_list: Iterable[Dict[str, Any]],
    source_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    entities: List[str] = []
    times: List[str] = []
    events: List[str] = []
    for anchors in anchors_list:
        if not isinstance(anchors, dict):
            continue
        entities.extend(anchors.get("entities", []) or [])
        times.extend(anchors.get("times", []) or [])
        events.extend(anchors.get("events", []) or [])
    return {
        "entities": _unique(entities, limit=24),
        "times": _unique(times, limit=24),
        "events": _unique(events, limit=24),
        "source": {
            "ids": list(source_ids or []),
        },
    }


def anchor_overlap_score(query: str, anchors: Dict[str, Any]) -> float:
    if not isinstance(anchors, dict):
        return 0.0

    query_tokens = _token_set(query)
    if not query_tokens:
        return 0.0

    entity_tokens = _token_set(" ".join(anchors.get("entities", []) or []))
    event_tokens = _token_set(" ".join(anchors.get("events", []) or []))
    time_text = " ".join(anchors.get("times", []) or [])

    entity_score = 1.0 if entity_tokens and entity_tokens & query_tokens else 0.0
    event_score = 0.0
    if event_tokens:
        event_score = len(event_tokens & query_tokens) / max(1, len(event_tokens))
    time_score = 0.0
    for time_anchor in anchors.get("times", []) or []:
        if str(time_anchor).lower() in str(query or "").lower():
            time_score = 1.0
            break
    if time_score == 0.0 and extract_time_anchors(query) and time_text:
        time_score = 0.35

    return min(1.0, 0.35 * entity_score + 0.45 * event_score + 0.20 * time_score)
