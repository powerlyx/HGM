"""LLM-as-Judge evaluator for LoCoMo and LongMemEval QA."""

import json
import logging
import os
import re
import time
from typing import Dict, List, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger("locomo_judge")

try:
    from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

    RETRYABLE_OPENAI_ERRORS = (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        TimeoutError,
    )
except Exception:
    RETRYABLE_OPENAI_ERRORS = (TimeoutError,)


PROMPT_SOURCES: Dict[str, Dict[str, str]] = {
    "locomo": {
        "label": "LoCoMo category judge",
        "kind": "code",
        "notes": (
            "Grades LoCoMo categories 1-4 with a generous CORRECT/WRONG "
            "LLM-as-Judge prompt and skips category 5."
        ),
    },
    "answer_check": {
        "label": "Answer-check judge (CORRECT/WRONG + JSON, majority vote)",
        "kind": "code",
        "notes": (
            "A single CORRECT/WRONG template for all questions (per question_type "
            "dispatch and abstention handling intentionally ignored), system+user, "
            "generous grading, JSON label, judge_runs majority vote. On the "
            "LongMemEval path abstention/per-type accuracy is therefore NOT "
            "comparable to the official yes/no protocol."
        ),
    },
}


ACCURACY_PROMPT = """
Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
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
Gold answer: {gold_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
""".strip()


LOCOMO_JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions match a gold standard answer."
)


def list_prompt_sources() -> Dict[str, Dict[str, str]]:
    return PROMPT_SOURCES


def get_prompt_source(source_name: str) -> Dict[str, str]:
    if source_name not in PROMPT_SOURCES:
        raise ValueError(
            f"Unknown prompt source '{source_name}'. "
            f"Available sources: {', '.join(sorted(PROMPT_SOURCES))}"
        )
    return PROMPT_SOURCES[source_name]


def build_judge_prompt(
    source_name: str,
    question: str,
    gold_answer: str,
    generated_answer: str,
) -> str:
    get_prompt_source(source_name)
    return ACCURACY_PROMPT.format(
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )


def parse_judge_label(response: str) -> bool:
    cleaned = str(response or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE).strip()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "label" in parsed:
            cleaned = str(parsed["label"]).strip()
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    upper = cleaned.upper()
    if "CORRECT" in upper and "WRONG" not in upper:
        return True
    if "WRONG" in upper and "CORRECT" not in upper:
        return False

    normalized = re.sub(r"[^A-Z]+", "", upper)
    return normalized == "CORRECT"


def evaluate_llm_judge(
    question: str,
    gold_answer: str,
    generated_answer: str,
    model: str = "gpt-4.1-mini-2025-04-14",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    source_name: str = "locomo",
    backend: str = "openai",
    host: str = "http://localhost",
    port: int = 30000,
    judge_runs: int = 1,
) -> Dict[str, object]:
    normalized_backend = str(backend or "").strip().lower()

    # ---- Answer-check judge. Single CORRECT/WRONG template for ALL questions.
    # Majority vote over judge_runs. system+user, temperature=0, max_tokens NOT
    # capped so JSON+explanation fit.
    if source_name == "answer_check":
        from eval.longmemeval_prompts import (
            build_answer_check_judge_prompt,
            parse_answer_check_judge_label,
            ANSWER_CHECK_JUDGE_SYSTEM_PROMPT,
        )
        user_prompt_ac = build_answer_check_judge_prompt(question, gold_answer, generated_answer)
        normalized_backend_ac = str(backend or "").strip().lower()
        effective_config_ac = {
            "backend": normalized_backend_ac,
            "model": model,
            "base_url": base_url,
            "host": host,
            "port": int(port),
            "judge_runs": int(max(1, judge_runs)),
            "label_scheme": "CORRECT_WRONG",
        }

        judgments: List[bool] = []
        raw_responses: List[str] = []
        tokens_total = 0
        runs = int(max(1, judge_runs))

        if normalized_backend_ac == "openai":
            if api_key is None:
                api_key = os.getenv("OPENAI_API_KEY")
            if base_url is None:
                base_url = os.getenv("OPENAI_BASE_URL")
            effective_config_ac["base_url"] = base_url
            openai_factory_ac = OpenAI
            if openai_factory_ac is None:
                from openai import OpenAI as openai_factory_ac
            client_ac = openai_factory_ac(base_url=base_url, api_key=api_key)
            request_timeout_ac = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
            max_retries_ac = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
            retry_base_delay_ac = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_DELAY", "1.0")))
            for _run in range(runs):
                last_exc_ac = None
                for attempt in range(max_retries_ac + 1):
                    try:
                        r = client_ac.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": ANSWER_CHECK_JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": user_prompt_ac},
                            ],
                            temperature=0.0,
                            timeout=request_timeout_ac,
                        )
                        raw_j = r.choices[0].message.content if r.choices else ""
                        tokens_total += getattr(getattr(r, "usage", None), "total_tokens", None) or 0
                        lbl = parse_answer_check_judge_label(raw_j)
                        if lbl is None:
                            raise ValueError("Could not parse CORRECT/WRONG label from judge response")
                        judgments.append(bool(lbl))
                        raw_responses.append(raw_j)
                        break
                    except RETRYABLE_OPENAI_ERRORS as exc:
                        last_exc_ac = exc
                        if attempt < max_retries_ac:
                            time.sleep(retry_base_delay_ac * (2 ** attempt))
                            continue
                        raise
        else:
            from llm.controller import RobustLLMController
            ctrl_ac = RobustLLMController(
                backend=normalized_backend_ac,
                model=model,
                api_base=base_url,
                sglang_host=host,
                sglang_port=int(port),
                check_connection=False,
            )
            for _run in range(runs):
                combined = ANSWER_CHECK_JUDGE_SYSTEM_PROMPT + "\n\n" + user_prompt_ac
                raw_j = ctrl_ac.llm.get_completion(combined, temperature=0.0)
                lbl = parse_answer_check_judge_label(raw_j)
                if lbl is None:
                    raise ValueError("Could not parse CORRECT/WRONG label from judge response")
                judgments.append(bool(lbl))
                raw_responses.append(raw_j)

        correct = sum(judgments) > (runs / 2)
        return {
            "label": correct,
            "raw_response": raw_responses[0] if raw_responses else "",
            "all_raw_responses": raw_responses,
            "judgments": judgments,
            "prompt": user_prompt_ac,
            "system_prompt": ANSWER_CHECK_JUDGE_SYSTEM_PROMPT,
            "effective_config": effective_config_ac,
            "judge_runs": runs,
            "label_scheme": "CORRECT_WRONG",
        }

    prompt = build_judge_prompt(
        source_name=source_name,
        question=question,
        gold_answer=gold_answer,
        generated_answer=generated_answer,
    )

    normalized_backend = str(backend or "").strip().lower()
    if normalized_backend not in {"openai", "ollama", "sglang", "vllm"}:
        raise ValueError(
            "Unsupported judge backend. Expected one of: openai, ollama, sglang, vllm"
        )

    effective_config = {
        "backend": normalized_backend,
        "model": model,
        "base_url": base_url,
        "host": host,
        "port": int(port),
    }

    if normalized_backend != "openai":
        from llm.controller import RobustLLMController

        controller = RobustLLMController(
            backend=normalized_backend,
            model=model,
            api_base=base_url,
            sglang_host=host,
            sglang_port=int(port),
            check_connection=False,
        )
        combined_prompt = f"{LOCOMO_JUDGE_SYSTEM_PROMPT}\n\n{prompt}"
        raw_response = controller.llm.get_completion(combined_prompt, temperature=0.0)
        return {
            "label": parse_judge_label(raw_response),
            "raw_response": raw_response,
            "prompt": prompt,
            "system_prompt": LOCOMO_JUDGE_SYSTEM_PROMPT,
            "effective_config": effective_config,
        }

    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
    if base_url is None:
        base_url = os.getenv("OPENAI_BASE_URL")
    effective_config["base_url"] = base_url

    request_timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
    max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
    retry_base_delay = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_DELAY", "1.0")))

    openai_factory = OpenAI
    if openai_factory is None:
        try:
            from openai import OpenAI as openai_factory
        except ImportError as exc:
            raise ImportError(
                "OpenAI judge backend requires the openai package"
            ) from exc
    client = openai_factory(base_url=base_url, api_key=api_key)

    last_exc = None
    response = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": LOCOMO_JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                timeout=request_timeout,
            )
            break
        except RETRYABLE_OPENAI_ERRORS as exc:
            last_exc = exc
            if attempt < max_retries:
                delay = retry_base_delay * (2 ** attempt)
                logger.warning(
                    "Judge OpenAI call failed (attempt %d/%d): %s; retrying in %.1fs",
                    attempt + 1, max_retries + 1, exc, delay,
                )
                time.sleep(delay)
                continue
            logger.error("Judge OpenAI call failed after %d attempts: %s", max_retries + 1, exc)
            raise

    if response is None:
        raise RuntimeError(f"Judge OpenAI call did not return a response: {last_exc}")

    raw_response = response.choices[0].message.content if response.choices else ""
    label = parse_judge_label(raw_response)

    return {
        "label": label,
        "raw_response": raw_response,
        "prompt": prompt,
        "system_prompt": LOCOMO_JUDGE_SYSTEM_PROMPT,
        "effective_config": effective_config,
    }
