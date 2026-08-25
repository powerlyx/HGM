"""LoCoMo dataset loader — minimal, single-purpose."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None

    @property
    def final_answer(self) -> Optional[str]:
        return self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: dict  # Dict[int, Session]


@dataclass
class EventSummary:
    events: dict  # session -> speaker -> events


@dataclass
class Observation:
    observations: dict  # session -> speaker -> [observation, evidence]


@dataclass
class LoCoMoSample:
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: dict


def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    turns = []
    for turn in session_data:
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            text = f"{caption_text} {text}" if text else caption_text

        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                if session.turns:
                    sessions[session_id] = session

    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    total_qa = 0
    total_image_qa = 0
    qa_counts_per_sample = []

    for sample_idx, sample in enumerate(data):
        try:
            qa_list = []
            sample_qa_count = 0
            sample_image_qa_count = 0

            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    has_image_evidence = False
                    for evidence_id in qa.get("evidence", []):
                        if ":" not in evidence_id:
                            continue
                        turn_id = evidence_id.split(":")[1]
                        for session in sample["conversation"].values():
                            if isinstance(session, list):
                                for turn in session:
                                    if turn.get("dia_id", "").endswith(turn_id):
                                        if "img_url" in turn or "blip_caption" in turn:
                                            has_image_evidence = True
                                            break

                    if has_image_evidence:
                        sample_image_qa_count += 1

                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1

                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e
                except Exception as e:
                    print(f"Unexpected error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e

            conversation = parse_conversation(sample["conversation"])
            event_summary = EventSummary(events=sample["event_summary"])
            observation = Observation(observations=sample["observation"])
            session_summary = sample.get("session_summary", {})

            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)

            total_qa += sample_qa_count
            total_image_qa += sample_image_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            print(f"\nSample {sample_idx}:")
            print(f"  Total QAs: {sample_qa_count}")
            print(f"  QAs with image evidence: {sample_image_qa_count}")

        except Exception as e:
            print(f"Error processing sample {sample_idx}:")
            print(str(e))
            raise e

    print("\nOverall Statistics:")
    print(f"Total QAs: {total_qa}")
    print(f"Total QAs with image evidence: {total_image_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")
    print(f"Min QAs in a sample: {min(qa_counts_per_sample)}")
    print(f"Max QAs in a sample: {max(qa_counts_per_sample)}")

    return samples
