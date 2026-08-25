"""Answer-generation and judge prompts used on the LongMemEval path
(answer-check CoT template, official answer-check judge, context rendering,
and answer/judge parsers)."""

from typing import Any, Optional

# ---------------------------------------------------------------------------
# Answer-check answer + judge prompts (fixed path for LongMemEval).
# Kept pure: no question_date injection, no question_type dispatch.
# ---------------------------------------------------------------------------


COT_ANSWER_PROMPT_TEMPLATE = """You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[If the question asks about elapsed time, ordering, or relative dates, use the Current Date below together with each memory's timestamp to compute the answer. The Current Date is the date the question was asked - treat it as "today".]
- Current Date: {current_date}
- Original reference: [e.g., "last year" from May 2022]
- Calculation: [Show logic, e.g. event timestamp 2023/04/06 vs Current Date 2023/04/10 => 4 days ago]
- Actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

Current Date: {current_date}
Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""


ANSWER_CHECK_JUDGE_SYSTEM_PROMPT = "You are an expert grader that determines if answers to questions match a gold standard answer"


ANSWER_CHECK_JUDGE_USER_PROMPT_TEMPLATE = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


# Context template used to build the answer {context}.
ANSWER_CHECK_CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""


def build_cot_answer_prompt(question: str, context: str, current_date: str = "") -> str:
    """Fill the CoT answer prompt with context, current date and question.

    ``current_date`` (the question_date) is injected as the today anchor so elapsed-time / ordering
    questions can convert memory timestamps into X-days-ago answers. Empty string keeps old behavior.
    """
    return COT_ANSWER_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
        current_date=str(current_date or ""),
    )


def build_answer_check_judge_prompt(question: str, golden_answer: str, generated_answer: str) -> str:
    """Fill the answer-check judge user prompt. Labels are CORRECT/WRONG (JSON)."""
    return ANSWER_CHECK_JUDGE_USER_PROMPT_TEMPLATE.format(
        question=question,
        golden_answer=golden_answer,
        generated_answer=generated_answer,
    )


def render_answer_check_context(selected_evidence, speaker_a: str = "speaker_a", speaker_b: str = "speaker_b") -> str:
    """Render evidence packets into the answer-check context format.

    Maps each evidence item (timestamp/speaker/content) to an episode line
    ``{subject}: {episode_text}\n---`` joined by double newlines, wrapped in
    the context-template header.

    Evidence ordering follows retrieval order (no forced sort), which is
    consistent with how the agent's selected_evidence is already ranked.
    """
    episode_lines = []
    for ev in (selected_evidence or []):
        subject = str(ev.get("speaker") or ev.get("subject") or "N/A")
        text = str(
            ev.get("content")
            or ev.get("episode")
            or ev.get("summary")
            or "N/A"
        )
        # Surface the memory timestamp so the model can perform the STEP 4
        # time-reference calculation (elapsed time / ordering) against the
        # Current Date anchor provided in the answer prompt.
        ts = str(ev.get("timestamp") or ev.get("time") or "").strip()
        ts_tag = f" [timestamp: {ts}]" if ts else ""
        episode_lines.append(f"{subject}{ts_tag}: {text}\n---")
    episodes = "\n\n".join(episode_lines)
    return ANSWER_CHECK_CONTEXT_TEMPLATE.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        episodes=episodes,
    )

def extract_cot_final_answer(text: Any) -> str:
    """Extract the final answer from a CoT response. 3-marker priority chain.

    1. ``## STEP 7: FINAL ANSWER``
    2. ``FINAL ANSWER:``
    3. ``FINAL ANSWER`` (strip leading colon)
    Uses rsplit to take the LAST occurrence. Falls back to full text strip.
    """
    result = str(text or "").strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            answer = result.rsplit(marker, 1)[1].strip()
            if marker == "FINAL ANSWER" and answer.startswith(":"):
                answer = answer[1:].strip()
            return answer
    return result


def extract_answer_check_judge_json(content: Any) -> Optional[str]:
    """Robustly extract JSON from a judge response."""
    import re as _re
    c = str(content or "")
    m = _re.search(r"```(?:json)?\s*(\{[^`]*\})\s*```", c, _re.DOTALL)
    if m:
        return m.group(1).strip()
    m = _re.search(r'\{[^{}]*"label"\s*:\s*"[^"]*"[^{}]*\}', c)
    if m:
        return m.group(0)
    return c.strip()


def parse_answer_check_judge_label(content: Any) -> Optional[bool]:
    """Parse a CORRECT/WRONG JSON label. Returns True/False or None on failure.

    Accepts a judge response (json with key "label"): label must be
    exactly 'CORRECT' or 'WRONG'.
    """
    json_str = extract_answer_check_judge_json(content)
    if not json_str:
        return None
    try:
        import json as _json
        result = _json.loads(json_str)
        if isinstance(result, dict):
            label = str(result.get("label", "") or "").strip().upper()
            if label == "CORRECT":
                return True
            if label == "WRONG":
                return False
    except (ValueError, TypeError):
        pass
    return None
