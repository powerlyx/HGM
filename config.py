"""Global configuration for HGM."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Embedding
    embedding_model: str = "text-embedding-3-small"

    # LLM for memory operations
    llm_backend: str = "openai"
    llm_model: str = "gpt-4.1-mini-2025-04-14"
    llm_base_url: Optional[str] = None

    # LLM for judging
    judge_model: str = "gpt-4.1-mini-2025-04-14"
    judge_backend: str = "openai"
    judge_base_url: Optional[str] = None

    # Memory hierarchy
    topic_build_mode: str = "after_sample"
    enable_anchors: bool = False

    # Retrieval
    retrieve_k: int = 10
    topic_result_limit: Optional[int] = None
    episode_result_limit: Optional[int] = None
    turn_result_limit: Optional[int] = None
    retrieval_semantic_weight: float = 0.68
    retrieval_lexical_weight: float = 0.24
    retrieval_recency_weight: float = 0.08
    retrieval_weight_mode: str = "adaptive"
    retrieval_adaptive_strength: float = 0.35
    retrieval_temporal_bonus: float = 0.10
    retrieval_budget_low_multiplier: float = 0.80
    retrieval_budget_mid_multiplier: float = 1.00
    retrieval_budget_high_multiplier: float = 1.55
    retrieval_provenance_boost: float = 0.55
    retrieval_min_turn_support: int = 2
    retrieval_uncertainty_temperature: float = 0.08
    retrieval_uncertainty_margin_scale: float = 0.08
    retrieval_uncertainty_low_threshold: float = 0.40
    retrieval_uncertainty_high_threshold: float = 0.70
    retrieval_final_max_items: Optional[int] = None

    # SGLang / vLLM hosts
    sglang_host: str = "http://localhost"
    sglang_port: int = 30000
    judge_sglang_host: str = "http://localhost"
    judge_sglang_port: int = 30000

    # Ablation
    ablation_mode: str = "main"


    # Evaluation
    ratio: float = 1.0
    seed: int = 0
    query_plan_input: Optional[str] = None

    # Paths
    dataset_path: str = "data/locomo10.json"
    output_dir: str = "outputs"
    cache_dir: str = "cached_memories"

    # API key loaded from env
    openai_api_key: Optional[str] = None

    def __post_init__(self):
        import os
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        except ImportError:
            pass
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")


def detect_dataset_type(file_path, hint: str = "auto") -> str:
    """Inspect a dataset file and return 'locomo' or 'longmemeval'.

    'locomo' must be a JSON list whose first element has a 'conversation' and 'qa' key.
    'longmemeval' must be a JSON list whose first element has 'haystack_sessions'.
    """
    import json as _json
    import os as _os
    normalized = str(hint or "auto").strip().lower()
    if normalized in ("locomo", "longmemeval"):
        return normalized
    if not file_path or not _os.path.exists(file_path):
        # fall back to locomo for backward compatibility
        return "locomo"
    with open(file_path, "r", encoding="utf-8") as _f:
        data = _json.load(_f)
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return "locomo"
    first = data[0]
    if "haystack_sessions" in first and "question_id" in first:
        return "longmemeval"
    if "conversation" in first and "qa" in first:
        return "locomo"
    return "locomo"

