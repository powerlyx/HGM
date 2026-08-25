"""LongMemEval dataset adapter — minimal, single-purpose.

Parses the official LongMemEval JSON into a data model that reuses
``dataset.locomo.Turn`` so the existing hierarchical-memory build path
(``add_memory`` / ``add_session_summary`` / ``finalize_session``) can consume
each session without modification.

Official field reference (per github.com/xiaowu0162/longmemeval README):
    question_id, question_type, question, question_date, answer,
    answer_session_ids, haystack_dates, haystack_session_ids, haystack_sessions
A session is a list of turns; a turn is {"role", "content", "has_answer"?}.
Abstention questions have a ``question_id`` ending with ``_abs``.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union

from dataset.locomo import Turn


def is_abstention(question_id) -> bool:
    """An abstention question's id ends with ``_abs`` (official convention)."""
    return str(question_id or "").endswith("_abs")


@dataclass
class LongMemEvalSession:
    session_id: str
    date_time: str
    turns: List[Turn] = field(default_factory=list)


@dataclass
class LongMemEvalEntry:
    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    sessions: List[LongMemEvalSession] = field(default_factory=list)
    answer_session_ids: List[str] = field(default_factory=list)

    @property
    def is_abstention(self) -> bool:
        return is_abstention(self.question_id)


def _parse_turns(session_id: str, turn_entries: list) -> List[Turn]:
    turns: List[Turn] = []
    for idx, turn_entry in enumerate(turn_entries):
        # Build a traceable dia_id aligned with the official corpus_id scheme
        # (session_id + 1-based turn index).
        dia_id = f"{session_id}_{idx + 1}"
        turns.append(
            Turn(
                speaker=str(turn_entry["role"]),
                dia_id=dia_id,
                text=str(turn_entry.get("content", "")),
            )
        )
    return turns


def load_longmemeval_dataset(file_path: Union[str, Path]) -> List[LongMemEvalEntry]:
    if isinstance(file_path, str):
        file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"LongMemEval dataset not found at {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("LongMemEval dataset must be a JSON list of entries")

    entries: List[LongMemEvalEntry] = []
    required_fields = (
        "question_id",
        "question_type",
        "question",
        "question_date",
        "answer",
        "haystack_dates",
        "haystack_session_ids",
        "haystack_sessions",
    )

    for entry_idx, raw in enumerate(data):
        missing = [key for key in required_fields if key not in raw]
        if missing:
            raise KeyError(
                f"LongMemEval entry {entry_idx} missing required fields: {missing}"
            )

        haystack_dates = list(raw["haystack_dates"])
        haystack_session_ids = list(raw["haystack_session_ids"])
        haystack_sessions = list(raw["haystack_sessions"])
        if not (len(haystack_dates) == len(haystack_session_ids) == len(haystack_sessions)):
            raise ValueError(
                f"LongMemEval entry {entry_idx} ({raw['question_id']}): "
                f"haystack_dates/session_ids/sessions length mismatch "
                f"({len(haystack_dates)}/{len(haystack_session_ids)}/{len(haystack_sessions)})"
            )

        sessions: List[LongMemEvalSession] = []
        for session_id, date_time, session_turns in zip(
            haystack_session_ids, haystack_dates, haystack_sessions
        ):
            sessions.append(
                LongMemEvalSession(
                    session_id=str(session_id),
                    date_time=str(date_time),
                    turns=_parse_turns(str(session_id), list(session_turns or [])),
                )
            )

        entries.append(
            LongMemEvalEntry(
                question_id=str(raw["question_id"]),
                question_type=str(raw["question_type"]),
                question=str(raw["question"]),
                question_date=str(raw["question_date"]),
                answer=str(raw["answer"]),
                sessions=sessions,
                answer_session_ids=[str(x) for x in (raw.get("answer_session_ids") or [])],
            )
        )

    return entries
