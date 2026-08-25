"""Unified LLM controllers with retry, token tracking, and multi-backend support."""

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any, Dict, Literal, Optional
import functools
import logging
import os
import re
import threading
import time

logger = logging.getLogger("amem_robust")


# ---------------------------------------------------------------------------
# Token counting helpers
# ---------------------------------------------------------------------------

def _estimate_token_count(text: Any) -> int:
    normalized = str(text or "")
    if not normalized:
        return 0
    cjk_units = len(re.findall(r"[一-鿿]", normalized))
    non_cjk_text = re.sub(r"[一-鿿]", " ", normalized)
    word_like_units = len(re.findall(r"[A-Za-z0-9_]+|[^\s]", non_cjk_text))
    return cjk_units + word_like_units


def _coerce_non_negative_int(value: Any) -> Optional[int]:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < 0:
        return None
    return parsed


# ---------------------------------------------------------------------------
# Token usage tracker
# ---------------------------------------------------------------------------

class LLMTokenUsageTracker:
    """Phase-aware token accounting shared across LLM backends."""

    def __init__(self):
        self._lock = threading.Lock()
        self._phase = "unspecified"
        self._stats = defaultdict(
            lambda: {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "calls": 0,
            }
        )

    @staticmethod
    def _normalize_phase(phase: Optional[str]) -> str:
        normalized = re.sub(r"\s+", "_", str(phase or "").strip().lower())
        return normalized or "unspecified"

    def set_phase(self, phase: Optional[str]):
        with self._lock:
            self._phase = self._normalize_phase(phase)

    def record(
        self,
        prompt: Any,
        completion: Any,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ):
        prompt_count = _coerce_non_negative_int(prompt_tokens)
        completion_count = _coerce_non_negative_int(completion_tokens)
        total_count = _coerce_non_negative_int(total_tokens)

        if prompt_count is None:
            prompt_count = _estimate_token_count(prompt)
        if completion_count is None:
            completion_count = _estimate_token_count(completion)
        if total_count is None:
            total_count = prompt_count + completion_count

        with self._lock:
            phase = self._phase
            phase_stats = self._stats[phase]
            phase_stats["prompt_tokens"] += prompt_count
            phase_stats["completion_tokens"] += completion_count
            phase_stats["total_tokens"] += total_count
            phase_stats["calls"] += 1

    def snapshot(self) -> Dict[str, Dict[str, int]]:
        with self._lock:
            return {
                phase: {
                    "prompt_tokens": int(values["prompt_tokens"]),
                    "completion_tokens": int(values["completion_tokens"]),
                    "total_tokens": int(values["total_tokens"]),
                    "calls": int(values["calls"]),
                }
                for phase, values in self._stats.items()
            }


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s; retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, exc, delay,
                        )
                        time.sleep(delay)
            logger.error(
                "LLM call %s failed after %d attempts: %s",
                func.__name__, max_retries + 1, last_exc,
            )
            raise last_exc

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    def set_token_tracker(self, tracker: Optional[LLMTokenUsageTracker]):
        self._token_tracker = tracker

    def _record_token_usage(
        self,
        prompt: str,
        completion: str,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
        total_tokens: Optional[int] = None,
    ):
        tracker = getattr(self, "_token_tracker", None)
        if tracker is None:
            return
        tracker.record(
            prompt=prompt,
            completion=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        """Get a plain-text completion from the LLM.

        ``max_tokens`` overrides the controller default output cap when provided
        (needed for long structured-CoT prompts). When ``None`` the
        controller keeps its existing default so legacy callers are unchanged.
        """

    def check_connectivity(self):
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as exc:
            raise ConnectionError(
                f"Cannot reach LLM backend: {exc}. Check that the server is running and accessible."
            ) from exc


# ---------------------------------------------------------------------------
# OpenAI controller
# ---------------------------------------------------------------------------

class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
            try:
                from openai import APITimeoutError, APIConnectionError, InternalServerError, RateLimitError

                self._retryable_errors = (
                    APITimeoutError,
                    APIConnectionError,
                    RateLimitError,
                    InternalServerError,
                    TimeoutError,
                )
            except Exception:
                self._retryable_errors = (TimeoutError,)
        except ImportError as exc:
            raise ImportError("OpenAI package not found. Install it with: pip install openai") from exc

        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.request_timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
        self.max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
        self.retry_base_delay = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_DELAY", "1.0")))
        self.client = OpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=api_key,
        )

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        effective_max_tokens = 1000 if max_tokens is None else max(1, int(max_tokens))
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": self.SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=temperature,
                    max_tokens=effective_max_tokens,
                    timeout=self.request_timeout,
                )
                content = response.choices[0].message.content or ""
                usage = getattr(response, "usage", None)
                prompt_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
                completion_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
                total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
                self._record_token_usage(
                    prompt, content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
                return content
            except self._retryable_errors as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    # Rate-limit (429) under burst load needs a much longer backoff
                    # than transient timeouts/5xx, otherwise retries also hit 429.
                    try:
                        is_rate_limit = isinstance(exc, RateLimitError)
                    except NameError:
                        is_rate_limit = "429" in str(exc) or "rate" in str(exc).lower()
                    if is_rate_limit:
                        rate_limit_min_wait = float(os.getenv("OPENAI_RATE_LIMIT_BASE_DELAY", "15.0"))
                        delay = max(delay, rate_limit_min_wait * (attempt + 1))
                    logger.warning(
                        "OpenAI call failed (attempt %d/%d): %s; retrying in %.1fs",
                        attempt + 1, self.max_retries + 1, exc, delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error("OpenAI call failed after %d attempts: %s", self.max_retries + 1, exc)
                raise
        raise last_exc


# ---------------------------------------------------------------------------
# Ollama controller
# ---------------------------------------------------------------------------

class RobustOllamaController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "llama2",
        api_base: Optional[str] = None,
        request_timeout: float = 10.0,
    ):
        import requests as _requests

        self._requests = _requests
        self.model = model
        self.api_base = (api_base or os.getenv("OLLAMA_API_BASE") or "http://127.0.0.1:11434").rstrip("/")
        env_timeout = os.getenv("OLLAMA_TIMEOUT_SECONDS")
        if env_timeout is not None:
            try:
                request_timeout = float(env_timeout)
            except ValueError:
                logger.warning(
                    "Invalid OLLAMA_TIMEOUT_SECONDS=%r, falling back to %.1fs",
                    env_timeout, request_timeout,
                )
        self.request_timeout = request_timeout

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        options = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max(1, int(max_tokens))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": options,
        }
        try:
            response = self._requests.post(
                f"{self.api_base}/api/chat",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
        except self._requests.Timeout as exc:
            raise TimeoutError(
                f"Ollama request exceeded {self.request_timeout:.1f}s; retrying with a fresh request"
            ) from exc
        except self._requests.RequestException as exc:
            raise RuntimeError(f"Ollama request failed: {exc}") from exc

        try:
            result = response.json()
        except ValueError as exc:
            raise RuntimeError(f"Ollama returned non-JSON response: {response.text[:200]}") from exc

        content = result.get("message", {}).get("content", "")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"Ollama returned empty content: {result}")
        self._record_token_usage(
            prompt, content,
            prompt_tokens=result.get("prompt_eval_count"),
            completion_tokens=result.get("eval_count"),
        )
        return content


# ---------------------------------------------------------------------------
# SGLang controller
# ---------------------------------------------------------------------------

class RobustSGLangController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "llama2",
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
    ):
        import requests as _requests

        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        effective_max_tokens = 1000 if max_tokens is None else max(1, int(max_tokens))
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": effective_max_tokens,
            },
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            result = response.json()
            content = result.get("text", "")
            if not isinstance(content, str):
                content = str(content)
            meta = result.get("meta_info") if isinstance(result, dict) else None
            prompt_tokens = None
            completion_tokens = None
            if isinstance(meta, dict):
                prompt_tokens = meta.get("prompt_tokens")
                completion_tokens = (
                    meta.get("completion_tokens")
                    or meta.get("output_tokens")
                    or meta.get("generated_tokens")
                )
            self._record_token_usage(
                prompt, content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
            return content
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


# ---------------------------------------------------------------------------
# vLLM controller
# ---------------------------------------------------------------------------

class RobustVLLMController(RobustBaseLLMController):
    def __init__(
        self,
        model: str = "llama2",
        vllm_host: str = "http://localhost",
        vllm_port: int = 30000,
    ):
        import requests as _requests

        self._requests = _requests
        self.model = model
        self.base_url = f"{vllm_host}:{vllm_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        effective_max_tokens = 1000 if max_tokens is None else max(1, int(max_tokens))
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": effective_max_tokens,
        }
        response = self._requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage") if isinstance(result, dict) else None
            prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
            completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
            total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
            self._record_token_usage(
                prompt, content,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )
            return content
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")

# ---------------------------------------------------------------------------
# Controller factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(
        self,
        backend: Literal["openai", "ollama", "sglang", "vllm"] = "sglang",
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        check_connection: bool = False,
    ):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = RobustOllamaController(model, api_base=api_base)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'ollama', 'sglang', or 'vllm'")

        self.token_usage_tracker = LLMTokenUsageTracker()
        self.llm.set_token_tracker(self.token_usage_tracker)

        if check_connection:
            self.llm.check_connectivity()

    def set_usage_phase(self, phase: str):
        self.token_usage_tracker.set_phase(phase)

    def get_token_usage_summary(self) -> Dict[str, Dict[str, int]]:
        return self.token_usage_tracker.snapshot()
