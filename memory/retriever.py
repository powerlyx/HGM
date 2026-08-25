"""Embedding-based retriever with OpenAI and sentence-transformer backends."""

from __future__ import annotations

import os
import time
from typing import List, Optional

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("amem_robust")


class OpenAIEmbeddingModel:
    """OpenAI embedding client with retry and timeout guards."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: Optional[str] = None):
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

        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.request_timeout = float(os.getenv("OPENAI_TIMEOUT_SECONDS", "120"))
        self.max_retries = max(0, int(os.getenv("OPENAI_MAX_RETRIES", "4")))
        self.retry_base_delay = max(0.1, float(os.getenv("OPENAI_RETRY_BASE_DELAY", "1.0")))
        self.batch_size = max(1, int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64")))
        # Per-input truncation to avoid 8192-token 400 BadRequest. Prefer an
        # accurate tokenizer when available; fall back to a conservative char cap.
        self.max_tokens = max(64, int(os.getenv("OPENAI_EMBEDDING_MAX_TOKENS", "8000")))
        self.max_chars = max(256, int(os.getenv("OPENAI_EMBEDDING_MAX_CHARS", "24000")))
        self._tokenizer = None
        self._tokenizer_mode = "chars"
        try:
            import tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(model_name)
            except Exception:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            self._tokenizer_mode = "tokens"
        except Exception:
            pass
        # truncation stats for diagnostics (reset per encode() call)
        self.last_truncation = {"count": 0, "max_chars": 0, "mode": self._tokenizer_mode}

    def _truncate_one(self, text: str) -> str:
        """Truncate a single text so its embedding input stays under the model cap.

        Token-accurate when a tokenizer is available; otherwise a conservative
        char-based truncation. Only the text fed to the embedding API is affected;
        the note's stored content is never modified here.
        """
        if not text:
            return text
        original_chars = len(text)
        if self._tokenizer_mode == "tokens" and self._tokenizer is not None:
            try:
                tokens = self._tokenizer.encode(text)
                if len(tokens) <= self.max_tokens:
                    return text
                truncated = self._tokenizer.decode(tokens[: self.max_tokens])
                logger.warning(
                    "Embedding input truncated: %d->%d tokens", len(tokens), self.max_tokens
                )
                return truncated
            except Exception:
                # fall through to char-based
                pass
        if original_chars <= self.max_chars:
            return text
        logger.warning(
            "Embedding input truncated (chars fallback): %d->%d chars",
            original_chars, self.max_chars,
        )
        return text[: self.max_chars]

    def _embed_batch(self, batch_texts: List[str]) -> List[List[float]]:
        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts,
                    timeout=self.request_timeout,
                )
                return [item.embedding for item in response.data]
            except self._retryable_errors as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    time.sleep(delay)
                    continue
                raise
        raise last_exc

    def encode(self, texts: List[str]) -> np.ndarray:
        normalized = [str(text or "") for text in texts]
        if not normalized:
            self.last_truncation = {"count": 0, "max_chars": 0, "mode": self._tokenizer_mode}
            return np.empty((0, 0), dtype=np.float32)

        # Truncate each input independently so one over-long text never fails
        # the whole batch (text-embedding-3-small caps a single input at 8192 tokens).
        trunc_count = 0
        max_chars = 0
        truncated_inputs: List[str] = []
        for t in normalized:
            max_chars = max(max_chars, len(t))
            if len(t) > (self.max_chars if self._tokenizer_mode == "chars" else self.max_tokens * 4):
                # quick pre-check before invoking tokenizer/char-slice
                trunc_count += 1
            truncated_inputs.append(self._truncate_one(t))
        self.last_truncation = {
            "count": trunc_count,
            "max_chars": max_chars,
            "mode": self._tokenizer_mode,
        }

        vectors: List[List[float]] = []
        for start in range(0, len(truncated_inputs), self.batch_size):
            batch = truncated_inputs[start:start + self.batch_size]
            vectors.extend(self._embed_batch(batch))

        return np.asarray(vectors, dtype=np.float32)


class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        if str(model_name).startswith("text-embedding-"):
            self.model = OpenAIEmbeddingModel(model_name=model_name)
        else:
            self.model = SentenceTransformer(model_name)
        self.corpus: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.document_ids = {}

    def add_documents(self, documents: List[str]):
        if not documents:
            return

        if not self.corpus:
            self.corpus = list(documents)
            self.embeddings = self.model.encode(self.corpus)
            self.document_ids = {doc: idx for idx, doc in enumerate(self.corpus)}
            return

        start_idx = len(self.corpus)
        self.corpus.extend(documents)
        new_embeddings = self.model.encode(documents)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        for idx, doc in enumerate(documents):
            self.document_ids[doc] = start_idx + idx
