"""Hierarchy memory configuration with validation."""

from dataclasses import asdict, dataclass
from numbers import Integral, Real
from typing import Any, Dict, Optional
import hashlib
import json
import re


@dataclass
class HierarchyMemoryConfig:
    topic_match_threshold: float = 0.62
    topic_min_token_overlap: int = 1
    episode_summary_turn_window: int = 24
    topic_summary_episode_window: int = 8
    topic_update_strategy: str = "full"
    episode_summary_min_bullets: int = 4
    episode_summary_max_bullets: int = 8
    topic_summary_min_bullets: int = 4
    topic_summary_max_bullets: int = 8
    topic_result_limit: int = 1
    episode_result_limit: int = 3
    turn_result_limit: int = 5
    topic_to_episode_boost: float = 0.60
    episode_to_topic_boost: float = 0.20
    episode_to_turn_boost: float = 0.50
    turn_to_episode_boost: float = 0.30
    turn_to_topic_boost: float = 0.12
    retrieval_semantic_weight: float = 0.68
    retrieval_lexical_weight: float = 0.24
    retrieval_recency_weight: float = 0.08
    retrieval_weight_mode: str = "adaptive"
    retrieval_adaptive_strength: float = 0.35
    retrieval_temporal_bonus: float = 0.10
    retrieval_budget_low_multiplier: float = 0.80
    retrieval_budget_mid_multiplier: float = 1.00
    retrieval_budget_high_multiplier: float = 1.55
    retrieval_uncertainty_temperature: float = 0.08
    retrieval_uncertainty_margin_scale: float = 0.08
    retrieval_uncertainty_low_threshold: float = 0.40
    retrieval_uncertainty_high_threshold: float = 0.70
    retrieval_final_max_items: Optional[int] = None
    enable_uncertainty_routing: bool = True
    retrieval_provenance_boost: float = 0.55
    retrieval_min_turn_support: int = 2
    episode_min_turns: int = 4
    episode_max_turns: int = 18
    episode_shift_similarity_threshold: float = 0.32
    episode_shift_lexical_overlap_threshold: float = 0.08
    episode_max_llm_turns: int = 60
    topic_assignment_strategy: str = "clustered"
    topic_cluster_distance_threshold: float = 0.30
    topic_cluster_min_size: int = 2
    topic_recluster_interval: int = 16
    topic_small_cluster_merge_threshold: float = 0.64
    topic_build_mode: str = "after_sample"
    enable_anchors: bool = False
    # Turn-level keyword/context/tag extraction mode. "batch" uses one LLM call
    # per batch (fast); "single" uses the legacy per-turn path. Build-acceleration
    # only; excluded from the construction cache signature (reuses cache regardless).
    analyze_turn_mode: str = "single"


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HierarchyMemoryConfig":
        valid_fields = set(cls.__dataclass_fields__.keys())
        filtered = {key: value for key, value in (data or {}).items() if key in valid_fields}
        return cls(**filtered)

    def signature(self, include_retrieval_only: bool = True) -> str:
        config_payload = self.to_dict()
        if not include_retrieval_only:
            config_payload = _config_without_retrieval_only_keys(config_payload)
        serialized = json.dumps(
            config_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
        return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:12]


# These keys only affect retrieval-stage scoring/breadth and do not change
# memory construction outputs.
_RETRIEVAL_ONLY_CONFIG_KEYS = frozenset(
    {
        "topic_result_limit",
        "episode_result_limit",
        "turn_result_limit",
        "topic_to_episode_boost",
        "episode_to_topic_boost",
        "episode_to_turn_boost",
        "turn_to_episode_boost",
        "turn_to_topic_boost",
        "retrieval_semantic_weight",
        "retrieval_lexical_weight",
        "retrieval_recency_weight",
        "retrieval_weight_mode",
        "retrieval_adaptive_strength",
        "retrieval_temporal_bonus",
        "retrieval_budget_low_multiplier",
        "retrieval_budget_mid_multiplier",
        "retrieval_budget_high_multiplier",
        "retrieval_uncertainty_temperature",
        "retrieval_uncertainty_margin_scale",
        "retrieval_uncertainty_low_threshold",
        "retrieval_uncertainty_high_threshold",
        "retrieval_final_max_items",
        "enable_uncertainty_routing",
        "retrieval_provenance_boost",
        "retrieval_min_turn_support",
        # build-acceleration knobs (do not affect cached-sample reuse)
        "analyze_turn_mode",
    }
)


def _config_without_retrieval_only_keys(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in (config or {}).items()
        if key not in _RETRIEVAL_ONLY_CONFIG_KEYS
    }


def _coerce_bool(value: Any, default: bool = True) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return default


_RESULT_LIMIT_FIELDS = (
    "topic_result_limit",
    "episode_result_limit",
    "turn_result_limit",
)


def _parse_result_limit(field_name: str, value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a non-negative integer")
    if isinstance(value, Integral):
        parsed = int(value)
    elif isinstance(value, Real):
        if not float(value).is_integer():
            raise ValueError(f"{field_name} must be a non-negative integer")
        parsed = int(value)
    elif isinstance(value, str):
        normalized = value.strip()
        if not re.fullmatch(r"\+?\d+", normalized):
            raise ValueError(f"{field_name} must be a non-negative integer")
        parsed = int(normalized)
    else:
        raise ValueError(f"{field_name} must be a non-negative integer")

    if parsed < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return parsed


def build_hierarchy_config(
    overrides: Optional[Dict[str, Any]] = None,
) -> HierarchyMemoryConfig:
    overrides = dict(overrides or {})

    config_data = HierarchyMemoryConfig().to_dict()

    valid_override_keys = set(config_data.keys())
    unknown_keys = sorted(set(overrides.keys()) - valid_override_keys)
    if unknown_keys:
        raise ValueError(f"Unknown hierarchy config keys: {unknown_keys}")

    for key, value in overrides.items():
        if value is not None:
            config_data[key] = value

    non_negative_integer_fields = (
        "topic_min_token_overlap",
        "episode_summary_turn_window",
        "topic_summary_episode_window",
        "episode_max_llm_turns",
        "topic_recluster_interval",
    )
    positive_integer_fields = (
        "episode_summary_min_bullets",
        "episode_summary_max_bullets",
        "topic_summary_min_bullets",
        "topic_summary_max_bullets",
        "episode_min_turns",
        "episode_max_turns",
        "topic_cluster_min_size",
        "retrieval_min_turn_support",
    )
    non_negative_float_fields = (
        "topic_to_episode_boost",
        "episode_to_topic_boost",
        "episode_to_turn_boost",
        "turn_to_episode_boost",
        "turn_to_topic_boost",
        "retrieval_semantic_weight",
        "retrieval_lexical_weight",
        "retrieval_recency_weight",
        "retrieval_adaptive_strength",
        "retrieval_temporal_bonus",
        "retrieval_budget_low_multiplier",
        "retrieval_budget_mid_multiplier",
        "retrieval_budget_high_multiplier",
        "retrieval_uncertainty_temperature",
        "retrieval_uncertainty_margin_scale",
        "retrieval_uncertainty_low_threshold",
        "retrieval_uncertainty_high_threshold",
        "retrieval_provenance_boost",
        "episode_shift_similarity_threshold",
        "episode_shift_lexical_overlap_threshold",
        "topic_cluster_distance_threshold",
        "topic_small_cluster_merge_threshold",
    )

    for key in non_negative_integer_fields:
        config_data[key] = max(0, int(config_data[key]))
    for key in positive_integer_fields:
        config_data[key] = max(1, int(config_data[key]))
    for key in _RESULT_LIMIT_FIELDS:
        config_data[key] = _parse_result_limit(key, config_data[key])
    for key in non_negative_float_fields:
        config_data[key] = max(0.0, float(config_data[key]))

    for min_key, max_key in (
        ("episode_summary_min_bullets", "episode_summary_max_bullets"),
        ("topic_summary_min_bullets", "topic_summary_max_bullets"),
    ):
        lower = min(config_data[min_key], config_data[max_key])
        upper = max(config_data[min_key], config_data[max_key])
        config_data[min_key] = lower
        config_data[max_key] = upper

    if config_data["topic_update_strategy"] not in {"full", "recent"}:
        raise ValueError("topic_update_strategy must be either 'full' or 'recent'")

    if config_data["topic_assignment_strategy"] not in {"incremental", "clustered"}:
        raise ValueError("topic_assignment_strategy must be either 'incremental' or 'clustered'")

    topic_build_mode = str(config_data.get("topic_build_mode", "after_sample")).strip().lower()
    if topic_build_mode not in {"after_sample", "incremental"}:
        raise ValueError(
            "topic_build_mode must be either 'after_sample' or 'incremental'"
        )
    config_data["topic_build_mode"] = topic_build_mode
    config_data["enable_anchors"] = _coerce_bool(
        config_data.get("enable_anchors", False),
        default=False,
    )
    analyze_turn_mode = str(config_data.get("analyze_turn_mode", "single")).strip().lower()
    if analyze_turn_mode not in {"batch", "single"}:
        raise ValueError("analyze_turn_mode must be either 'batch' or 'single'")
    config_data["analyze_turn_mode"] = analyze_turn_mode

    retrieval_weight_mode = str(config_data.get("retrieval_weight_mode", "static")).strip().lower()
    if retrieval_weight_mode not in {"static", "adaptive"}:
        raise ValueError("retrieval_weight_mode must be either 'static' or 'adaptive'")
    config_data["retrieval_weight_mode"] = retrieval_weight_mode
    config_data["retrieval_adaptive_strength"] = min(
        1.0,
        max(0.0, float(config_data.get("retrieval_adaptive_strength", 0.35))),
    )
    config_data["enable_uncertainty_routing"] = _coerce_bool(
        config_data.get("enable_uncertainty_routing", True),
        default=True,
    )

    if config_data["retrieval_uncertainty_temperature"] <= 0.0:
        raise ValueError("retrieval_uncertainty_temperature must be greater than 0")
    if config_data["retrieval_uncertainty_margin_scale"] <= 0.0:
        raise ValueError("retrieval_uncertainty_margin_scale must be greater than 0")
    low_threshold = float(config_data["retrieval_uncertainty_low_threshold"])
    high_threshold = float(config_data["retrieval_uncertainty_high_threshold"])
    if not 0.0 <= low_threshold < high_threshold <= 1.0:
        raise ValueError(
            "retrieval uncertainty thresholds must satisfy 0 <= low < high <= 1"
        )
    final_max_items = config_data.get("retrieval_final_max_items")
    config_data["retrieval_final_max_items"] = (
        None if final_max_items is None else max(0, int(final_max_items))
    )

    if config_data["episode_min_turns"] > config_data["episode_max_turns"]:
        lower = min(config_data["episode_min_turns"], config_data["episode_max_turns"])
        upper = max(config_data["episode_min_turns"], config_data["episode_max_turns"])
        config_data["episode_min_turns"] = lower
        config_data["episode_max_turns"] = upper

    config_data["episode_shift_similarity_threshold"] = min(
        1.0,
        max(-1.0, float(config_data["episode_shift_similarity_threshold"])),
    )
    config_data["episode_shift_lexical_overlap_threshold"] = min(
        1.0,
        max(0.0, float(config_data["episode_shift_lexical_overlap_threshold"])),
    )

    config_data["topic_cluster_distance_threshold"] = min(
        2.0,
        max(0.0, float(config_data["topic_cluster_distance_threshold"])),
    )
    config_data["topic_small_cluster_merge_threshold"] = min(
        1.0,
        max(-1.0, float(config_data["topic_small_cluster_merge_threshold"])),
    )

    for key in (
        "retrieval_budget_low_multiplier",
        "retrieval_budget_mid_multiplier",
        "retrieval_budget_high_multiplier",
    ):
        config_data[key] = min(3.0, max(0.25, float(config_data[key])))

    retrieval_weight_keys = (
        "retrieval_semantic_weight",
        "retrieval_lexical_weight",
        "retrieval_recency_weight",
    )
    normalized_weights = [max(0.0, float(config_data[key])) for key in retrieval_weight_keys]
    total_weight = sum(normalized_weights)
    if total_weight <= 0.0:
        normalized_weights = [0.68, 0.24, 0.08]
        total_weight = 1.0
    for key, value in zip(retrieval_weight_keys, normalized_weights):
        config_data[key] = value / total_weight

    threshold = float(config_data["topic_match_threshold"])
    if threshold < 0.0 or threshold > 2.0:
        raise ValueError("topic_match_threshold must be between 0.0 and 2.0")
    config_data["topic_match_threshold"] = threshold

    return HierarchyMemoryConfig.from_dict(config_data)
