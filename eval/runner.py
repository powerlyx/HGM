"""
Evaluation harness using the robust memory layer (no JSON schema dependency).
Drop-in replacement for test_advanced.py with optional LoCoMo LLM-as-Judge output.

Usage:
    python test_advanced_robust.py --backend openai --model gpt-4.1-mini-2025-04-14 --dataset data/locomo10.json
    python test_advanced_robust.py --backend ollama --model qwen2.5:3b --dataset data/locomo10.json
    python test_advanced_robust.py --backend openai --model gpt-4.1-mini-2025-04-14 \
        --judge-backend openai --judge-model gpt-4.1-mini-2025-04-14 \
        --dataset data/locomo10.json --output outputs/locomo_results.json
"""

from memory.config import (
    build_hierarchy_config,
)
from memory.prompts import (
    parse_plain_text_answer,
)
import hashlib
import os
import json
import argparse
import logging
import datetime as _dt_module
import re
from typing import Any, List, Dict, Optional
import numpy as np
from dataset.locomo import load_locomo_dataset
import statistics
from collections import defaultdict
import pickle
import random
import tempfile
import time
from eval.metrics import calculate_metrics, aggregate_metrics
from datetime import datetime
from eval.llm_judge import evaluate_llm_judge, get_prompt_source, list_prompt_sources
from eval.agent import RobustAdvancedMemAgent
import threading
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger("amem_robust")


def _atomic_write_json(path, payload):
    """Atomically write JSON (tmp + os.replace) for incremental robustness."""
    import tempfile as _tf
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = _tf.mkstemp(prefix=".locomo_inc_", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _summarize_failures(failed_entries):
    """Bucket failed questions by error category for quick triage."""
    from collections import defaultdict as _dd
    buckets = _dd(int)
    for f in failed_entries or []:
        err = str(f.get("error", ""))
        if "8192" in err or "maximum input length" in err.lower():
            cat = "embedding_too_long"
        elif "429" in err or "rate" in err.lower():
            cat = "rate_limit"
        elif "timeout" in err.lower() or "timed out" in err.lower():
            cat = "timeout"
        elif "connection" in err.lower():
            cat = "connection"
        else:
            cat = err.split(":", 1)[0] or "other"
        buckets[cat] += 1
    return {"count": len(failed_entries or []), "by_error": dict(sorted(buckets.items()))}
CACHE_VERSION = "hierarchical_v9_evidence_packet"
CONTEXT_STRATEGY = "hierarchical_multi_stage_evidence_packet"
ABLATION_MODES = (
    "main",
    "no_hierarchy",
    "no_uncertainty",
    "no_evidence_loop",
    "no_evidence_loop,no_hierarchy",
    "no_evidence_loop,no_uncertainty",
    "no_hierarchy,no_uncertainty",
    "no_evidence_loop,no_hierarchy,no_uncertainty",
)


def _safe_path_component(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)


def _fingerprint_file(file_path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha1()
    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _resolve_cache_family(ablation_flags: Dict[str, bool]) -> str:
    return (
        "hierarchical"
        if bool(ablation_flags.get("enable_hierarchy_memory", True))
        else "flat"
    )


def _build_cache_identity(
    cache_version: str,
    backend: str,
    model: str,
    dataset_fingerprint: str,
    memory_cache_signature: str,
    cache_family: str,
) -> str:
    identity_payload = {
        "cache_version": str(cache_version),
        "backend": str(backend),
        "model": str(model),
        "dataset_fingerprint": str(dataset_fingerprint),
        "memory_cache_signature": str(memory_cache_signature),
        "cache_family": str(cache_family),
    }
    identity_digest = hashlib.sha1(
        json.dumps(
            identity_payload,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:12]
    readable_components = (
        cache_version,
        backend,
        model,
        dataset_fingerprint,
        memory_cache_signature,
        cache_family,
        identity_digest,
    )
    return "_".join(
        _safe_path_component(str(component))
        for component in readable_components
    )


def _atomic_pickle_dump(payload: Any, target_path) -> None:
    target_path = os.fspath(target_path)
    target_directory = os.path.dirname(os.path.abspath(target_path))
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=target_directory,
            prefix=f"{os.path.basename(target_path)}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = handle.name
            pickle.dump(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, target_path)
        temp_path = None
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def _empty_token_stats() -> Dict[str, int]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }


def _accumulate_token_stats(target: Dict[str, int], source: Dict[str, int]):
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        try:
            value = int(source.get(key, 0))
        except (TypeError, ValueError):
            value = 0
        target[key] += max(0, value)


def _combined_token_stats(*sources: Dict[str, int]) -> Dict[str, int]:
    combined = _empty_token_stats()
    for source in sources:
        if isinstance(source, dict):
            _accumulate_token_stats(combined, source)
    return combined


def _accumulate_phase_usage(
    totals: Dict[str, Dict[str, int]],
    usage_summary: Dict[str, Dict[str, int]],
):
    for phase in ("memory_build", "qa", "answer_generation", "answer_verification"):
        phase_stats = usage_summary.get(phase, {}) if isinstance(usage_summary, dict) else {}
        if isinstance(phase_stats, dict):
            totals.setdefault(phase, _empty_token_stats())
            _accumulate_token_stats(totals[phase], phase_stats)


def _summarize_routing_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tier_counts = defaultdict(int)
    temporal_trigger_count = 0
    routed_count = 0
    evidence_totals = defaultdict(int)

    for result in results:
        query_plan = result.get("query_plan", {}) if isinstance(result, dict) else {}
        routing = query_plan.get("retrieval_routing", {}) if isinstance(query_plan, dict) else {}
        if not isinstance(routing, dict) or not routing:
            continue
        routed_count += 1
        tier_counts[str(routing.get("tier", "unknown"))] += 1
        temporal_trigger_count += int(bool(routing.get("temporal_sensitive", False)))
        final_counts = routing.get("final_counts", {})
        if isinstance(final_counts, dict):
            for layer in ("topics", "episodes", "turns"):
                try:
                    evidence_totals[layer] += max(0, int(final_counts.get(layer, 0)))
                except (TypeError, ValueError):
                    continue

    total_evidence = sum(evidence_totals.values())
    divisor = routed_count or 1
    return {
        "questions_with_routing": routed_count,
        "tier_distribution": dict(sorted(tier_counts.items())),
        "temporal_trigger_count": temporal_trigger_count,
        "temporal_trigger_rate": temporal_trigger_count / divisor if routed_count else 0.0,
        "evidence_count": {
            "average_total": total_evidence / divisor if routed_count else 0.0,
            "average_by_layer": {
                layer: evidence_totals[layer] / divisor if routed_count else 0.0
                for layer in ("topics", "episodes", "turns")
            },
        },
    }


def _summarize_answer_lengths(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    prediction_lengths = [len(str(item.get("prediction", "")).strip()) for item in results]
    reference_lengths = [len(str(item.get("reference", "")).strip()) for item in results]
    count = len(results)
    prediction_total = sum(prediction_lengths)
    reference_total = sum(reference_lengths)
    return {
        "count": count,
        "prediction_total_chars": prediction_total,
        "reference_total_chars": reference_total,
        "prediction_average_chars": prediction_total / count if count else 0.0,
        "reference_average_chars": reference_total / count if count else 0.0,
        "prediction_reference_char_ratio": (
            prediction_total / reference_total if reference_total else None
        ),
    }




def _build_memory_config(
    topic_match_threshold: Optional[float] = None,
    topic_min_token_overlap: Optional[int] = None,
    episode_summary_turn_window: Optional[int] = None,
    topic_summary_episode_window: Optional[int] = None,
    topic_update_strategy: Optional[str] = None,
    topic_result_limit: Optional[int] = None,
    episode_result_limit: Optional[int] = None,
    turn_result_limit: Optional[int] = None,
    episode_min_turns: Optional[int] = None,
    episode_max_turns: Optional[int] = None,
    episode_shift_similarity_threshold: Optional[float] = None,
    episode_shift_lexical_overlap_threshold: Optional[float] = None,
    episode_max_llm_turns: Optional[int] = None,
    topic_assignment_strategy: Optional[str] = None,
    topic_cluster_distance_threshold: Optional[float] = None,
    topic_cluster_min_size: Optional[int] = None,
    topic_recluster_interval: Optional[int] = None,
    topic_small_cluster_merge_threshold: Optional[float] = None,
    topic_build_mode: Optional[str] = None,
    enable_anchors: Optional[bool] = None,
    retrieval_semantic_weight: Optional[float] = None,
    retrieval_lexical_weight: Optional[float] = None,
    retrieval_recency_weight: Optional[float] = None,
    retrieval_temporal_bonus: Optional[float] = None,
    retrieval_budget_low_multiplier: Optional[float] = None,
    retrieval_budget_mid_multiplier: Optional[float] = None,
    retrieval_budget_high_multiplier: Optional[float] = None,
    retrieval_provenance_boost: Optional[float] = None,
    retrieval_min_turn_support: Optional[int] = None,
    retrieval_weight_mode: Optional[str] = None,
    retrieval_adaptive_strength: Optional[float] = None,
    retrieval_uncertainty_temperature: Optional[float] = None,
    retrieval_uncertainty_margin_scale: Optional[float] = None,
    retrieval_uncertainty_low_threshold: Optional[float] = None,
    retrieval_uncertainty_high_threshold: Optional[float] = None,
    retrieval_final_max_items: Optional[int] = None,
):
    overrides = {
        "topic_match_threshold": topic_match_threshold,
        "topic_min_token_overlap": topic_min_token_overlap,
        "episode_summary_turn_window": episode_summary_turn_window,
        "topic_summary_episode_window": topic_summary_episode_window,
        "topic_update_strategy": topic_update_strategy,
        "topic_result_limit": topic_result_limit,
        "episode_result_limit": episode_result_limit,
        "turn_result_limit": turn_result_limit,
        "episode_min_turns": episode_min_turns,
        "episode_max_turns": episode_max_turns,
        "episode_shift_similarity_threshold": episode_shift_similarity_threshold,
        "episode_shift_lexical_overlap_threshold": episode_shift_lexical_overlap_threshold,
        "episode_max_llm_turns": episode_max_llm_turns,
        "topic_assignment_strategy": topic_assignment_strategy,
        "topic_cluster_distance_threshold": topic_cluster_distance_threshold,
        "topic_cluster_min_size": topic_cluster_min_size,
        "topic_recluster_interval": topic_recluster_interval,
        "topic_small_cluster_merge_threshold": topic_small_cluster_merge_threshold,
        "topic_build_mode": topic_build_mode,
        "enable_anchors": enable_anchors,
        "retrieval_semantic_weight": retrieval_semantic_weight,
        "retrieval_lexical_weight": retrieval_lexical_weight,
        "retrieval_recency_weight": retrieval_recency_weight,
        "retrieval_temporal_bonus": retrieval_temporal_bonus,
        "retrieval_budget_low_multiplier": retrieval_budget_low_multiplier,
        "retrieval_budget_mid_multiplier": retrieval_budget_mid_multiplier,
        "retrieval_budget_high_multiplier": retrieval_budget_high_multiplier,
        "retrieval_provenance_boost": retrieval_provenance_boost,
        "retrieval_min_turn_support": retrieval_min_turn_support,
        "retrieval_weight_mode": retrieval_weight_mode,
        "retrieval_adaptive_strength": retrieval_adaptive_strength,
        "retrieval_uncertainty_temperature": retrieval_uncertainty_temperature,
        "retrieval_uncertainty_margin_scale": retrieval_uncertainty_margin_scale,
        "retrieval_uncertainty_low_threshold": retrieval_uncertainty_low_threshold,
        "retrieval_uncertainty_high_threshold": retrieval_uncertainty_high_threshold,
        "retrieval_final_max_items": retrieval_final_max_items,
    }
    return build_hierarchy_config(overrides)


def _memory_layer_counts(memory_system) -> Dict[str, int]:
    return {
        "episode_count": len(getattr(memory_system, "episode_memories", {}) or {}),
        "topic_count": len(getattr(memory_system, "topic_memories", {}) or {}),
    }


def _unassigned_episode_ids(memory_system) -> List[str]:
    episode_ids = {
        str(episode_id)
        for episode_id in (getattr(memory_system, "episode_memories", {}) or {})
    }
    referenced_episode_ids = set()
    for child_ids in (getattr(memory_system, "topic_episode_map", {}) or {}).values():
        referenced_episode_ids.update(str(episode_id) for episode_id in (child_ids or []))
    for topic in (getattr(memory_system, "topic_memories", {}) or {}).values():
        referenced_episode_ids.update(
            str(episode_id) for episode_id in (getattr(topic, "child_ids", []) or [])
        )
    return sorted(episode_ids - referenced_episode_ids)


def _normalize_ablation_mode(ablation_mode: str) -> str:
    raw = re.sub(r"\s+", "", str(ablation_mode or "main").strip().lower())
    aliases = {
        "baseline": "main",
        "full": "main",
        "w_o_hierarchy": "no_hierarchy",
        "without_hierarchy": "no_hierarchy",
        "w_o_uncertainty": "no_uncertainty",
        "without_uncertainty": "no_uncertainty",
        "w_o_evidence_loop": "no_evidence_loop",
        "without_evidence_loop": "no_evidence_loop",
    }
    tokens = raw.split(",")
    normalized_tokens = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        # `answer_driven` is an additive toggle (enables the answer-driven
        # verification link), orthogonal to the ablation axes. It is captured
        # into ablation_flags separately and must NOT enter the component
        # allowlist or the canonical membership check; skip it here so the raw
        # mode string can be passed directly to this function.
        if token == "answer_driven":
            continue
        token = aliases.get(token, token)
        if token not in ("main", "no_hierarchy", "no_uncertainty", "no_evidence_loop"):
            raise ValueError(
                f"Unknown ablation component: {token}. Allowed: main, no_hierarchy, no_uncertainty, no_evidence_loop"
            )
        normalized_tokens.append(token)
    if not normalized_tokens:
        normalized_tokens = ["main"]
    # "main" overrides everything else
    if "main" in normalized_tokens:
        return "main"
    # deduplicate and sort alphabetically for canonical naming
    result = ",".join(sorted(set(normalized_tokens)))
    if result not in ABLATION_MODES:
        raise ValueError(
            f"Unknown ablation_mode: {ablation_mode}. Available modes: {', '.join(ABLATION_MODES)}"
        )
    return result


def _resolve_ablation_flags(ablation_mode: str) -> Dict[str, bool]:
    # `answer_driven` is an additive toggle (enables the answer-driven verification
    # link) that is orthogonal to the ablation axes: it does NOT affect memory
    # construction, cache_family, or cache_identity, so strip it before
    # normalization. This keeps it out of the component allowlist and the
    # "main overrides everything" reduction, while still being recorded here.
    answer_driven_active = "answer_driven" in {
        t.strip().lower() for t in str(ablation_mode or "").split(",")
    }
    stripped = ",".join(
        t.strip() for t in str(ablation_mode or "").split(",")
        if t.strip() and t.strip().lower() != "answer_driven"
    )
    mode = _normalize_ablation_mode(stripped or "main")
    tokens = set(mode.split(","))
    return {
        "enable_hierarchy_memory": "no_hierarchy" not in tokens,
        "enable_uncertainty_routing": "no_uncertainty" not in tokens,
        "enable_evidence_packet_loop": "no_evidence_loop" not in tokens,
        "enable_answer_driven_verification": answer_driven_active,
    }


def _apply_ablation_to_memory_config(memory_config_obj, ablation_flags: Dict[str, bool]):
    if bool(ablation_flags.get("enable_uncertainty_routing", True)):
        return memory_config_obj

    merged = memory_config_obj.to_dict()
    merged.update(
        {
            "enable_uncertainty_routing": False,
            "retrieval_weight_mode": "static",
            "retrieval_adaptive_strength": 0.0,
            "retrieval_budget_low_multiplier": 1.0,
            "retrieval_budget_mid_multiplier": 1.0,
            "retrieval_budget_high_multiplier": 1.0,
        }
    )
    return build_hierarchy_config(merged)


def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """Set up logging configuration."""
    eval_logger = logging.getLogger('locomo_eval_robust')
    eval_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def evaluate_dataset(dataset_path: str, model: str, output_path: Optional[str] = None,
                     ratio: float = 1.0, backend: str = "sglang",
                     retrieve_k: int = 10,
                     sglang_host: str = "http://localhost", sglang_port: int = 30000,
                     judge_model: str = "gpt-4.1-mini-2025-04-14", judge_backend: str = "openai",
                     judge_base_url: Optional[str] = None,
                     judge_sglang_host: str = "http://localhost",
                     judge_sglang_port: int = 30000,
                     topic_match_threshold: Optional[float] = None,
                     topic_min_token_overlap: Optional[int] = None,
                     episode_summary_turn_window: Optional[int] = None,
                     topic_summary_episode_window: Optional[int] = None,
                     topic_update_strategy: Optional[str] = None,
                     topic_result_limit: Optional[int] = None,
                     episode_result_limit: Optional[int] = None,
                     turn_result_limit: Optional[int] = None,
                     episode_min_turns: Optional[int] = None,
                     episode_max_turns: Optional[int] = None,
                     episode_shift_similarity_threshold: Optional[float] = None,
                     episode_shift_lexical_overlap_threshold: Optional[float] = None,
                     episode_max_llm_turns: Optional[int] = None,
                     topic_assignment_strategy: Optional[str] = None,
                     topic_cluster_distance_threshold: Optional[float] = None,
                     topic_cluster_min_size: Optional[int] = None,
                     topic_recluster_interval: Optional[int] = None,
                     topic_small_cluster_merge_threshold: Optional[float] = None,
                     topic_build_mode: str = "after_sample",
                     enable_anchors: bool = False,
                     retrieval_semantic_weight: Optional[float] = None,
                     retrieval_lexical_weight: Optional[float] = None,
                     retrieval_recency_weight: Optional[float] = None,
                     retrieval_temporal_bonus: Optional[float] = None,
                     retrieval_budget_low_multiplier: Optional[float] = None,
                     retrieval_budget_mid_multiplier: Optional[float] = None,
                     retrieval_budget_high_multiplier: Optional[float] = None,
                     retrieval_provenance_boost: Optional[float] = None,
                     retrieval_min_turn_support: Optional[int] = None,
                     retrieval_weight_mode: Optional[str] = None,
                     retrieval_adaptive_strength: Optional[float] = None,
                     retrieval_uncertainty_temperature: Optional[float] = None,
                     retrieval_uncertainty_margin_scale: Optional[float] = None,
                     retrieval_uncertainty_low_threshold: Optional[float] = None,
                     retrieval_uncertainty_high_threshold: Optional[float] = None,
                     retrieval_final_max_items: Optional[int] = None,
                     ablation_mode: str = "main",
                     seed: int = 0,
                     answer_max_tokens: int = 32768,
                     judge_runs: int = 1,
                     num_workers: int = 1,
                     query_plan_input: Optional[str] = None,
                     answer_verification_gate: str = "low_evidence"):
    """Evaluate the robust agent on the LoComo dataset."""
    random.seed(seed)
    np.random.seed(seed)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"eval_robust_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_logger = setup_logger(log_path)
    eval_logger.info(f"Loading dataset from {dataset_path}")
    eval_logger.info(f"Using ROBUST hierarchical memory layer (3 levels, configurable, no JSON schema dependency)")

    # `answer_driven` is stripped by _normalize_ablation_mode (it is not part of
    # the canonical mode), so capture it from the raw input here to preserve it
    # through normalization and inject it into the flags used downstream.
    answer_driven_active = "answer_driven" in {
        t.strip().lower() for t in str(ablation_mode or "").split(",")
    }
    ablation_mode = _normalize_ablation_mode(ablation_mode)
    ablation_flags = _resolve_ablation_flags(ablation_mode)
    ablation_flags["enable_answer_driven_verification"] = answer_driven_active
    eval_logger.info(
        f"Ablation mode: {ablation_mode}" + (",answer_driven" if answer_driven_active else "")
    )
    eval_logger.info(f"Ablation flags: {ablation_flags}")

    memory_config_obj = _build_memory_config(
        topic_match_threshold=topic_match_threshold,
        topic_min_token_overlap=topic_min_token_overlap,
        episode_summary_turn_window=episode_summary_turn_window,
        topic_summary_episode_window=topic_summary_episode_window,
        topic_update_strategy=topic_update_strategy,
        topic_result_limit=topic_result_limit,
        episode_result_limit=episode_result_limit,
        turn_result_limit=turn_result_limit,
        episode_min_turns=episode_min_turns,
        episode_max_turns=episode_max_turns,
        episode_shift_similarity_threshold=episode_shift_similarity_threshold,
        episode_shift_lexical_overlap_threshold=episode_shift_lexical_overlap_threshold,
        episode_max_llm_turns=episode_max_llm_turns,
        topic_assignment_strategy=topic_assignment_strategy,
        topic_cluster_distance_threshold=topic_cluster_distance_threshold,
        topic_cluster_min_size=topic_cluster_min_size,
        topic_recluster_interval=topic_recluster_interval,
        topic_small_cluster_merge_threshold=topic_small_cluster_merge_threshold,
        topic_build_mode=topic_build_mode,
        enable_anchors=enable_anchors,
        retrieval_semantic_weight=retrieval_semantic_weight,
        retrieval_lexical_weight=retrieval_lexical_weight,
        retrieval_recency_weight=retrieval_recency_weight,
        retrieval_temporal_bonus=retrieval_temporal_bonus,
        retrieval_budget_low_multiplier=retrieval_budget_low_multiplier,
        retrieval_budget_mid_multiplier=retrieval_budget_mid_multiplier,
        retrieval_budget_high_multiplier=retrieval_budget_high_multiplier,
        retrieval_provenance_boost=retrieval_provenance_boost,
        retrieval_min_turn_support=retrieval_min_turn_support,
        retrieval_weight_mode=retrieval_weight_mode,
        retrieval_adaptive_strength=retrieval_adaptive_strength,
        retrieval_uncertainty_temperature=retrieval_uncertainty_temperature,
        retrieval_uncertainty_margin_scale=retrieval_uncertainty_margin_scale,
        retrieval_uncertainty_low_threshold=retrieval_uncertainty_low_threshold,
        retrieval_uncertainty_high_threshold=retrieval_uncertainty_high_threshold,
        retrieval_final_max_items=retrieval_final_max_items,
    )
    memory_config_obj = _apply_ablation_to_memory_config(memory_config_obj, ablation_flags)
    memory_config = memory_config_obj.to_dict()
    memory_config_signature = memory_config_obj.signature()
    memory_cache_signature = memory_config_obj.signature(include_retrieval_only=False)
    dataset_fingerprint = _fingerprint_file(dataset_path)
    cache_family = _resolve_cache_family(ablation_flags)
    cache_identity = _build_cache_identity(
        cache_version=CACHE_VERSION,
        backend=backend,
        model=model,
        dataset_fingerprint=dataset_fingerprint,
        memory_cache_signature=memory_cache_signature,
        cache_family=cache_family,
    )
    context_strategy = f"{CONTEXT_STRATEGY}:{ablation_mode}"
    primary_weight_mode = str(memory_config_obj.retrieval_weight_mode).strip().lower()

    eval_logger.info(f"Hierarchy config signature: {memory_config_signature}")
    eval_logger.info(f"Hierarchy cache signature: {memory_cache_signature}")
    eval_logger.info(f"Cache family: {cache_family}")
    eval_logger.info(f"Dataset fingerprint: {dataset_fingerprint}")
    eval_logger.info(f"Cache identity: {cache_identity}")
    eval_logger.info(f"Hierarchy config: {memory_config}")
    eval_logger.info(
        "Uncertainty routing enabled: %s",
        bool(getattr(memory_config_obj, "enable_uncertainty_routing", True)),
    )
    eval_logger.info(
        "Retrieval fusion mode: %s (adaptive_strength=%.3f)",
        primary_weight_mode,
        float(memory_config_obj.retrieval_adaptive_strength),
    )

    samples = load_locomo_dataset(dataset_path)
    eval_logger.info(f"Loaded {len(samples)} samples")

    precomputed_query_results = []
    if query_plan_input:
        with open(query_plan_input, "r", encoding="utf-8") as handle:
            query_plan_payload = json.load(handle)
        precomputed_query_results = list(query_plan_payload.get("individual_results", []) or [])
        if not precomputed_query_results:
            raise ValueError("Query Plan input contains no individual_results")
        eval_logger.info(
            "Loaded %d precomputed Query Plans from %s",
            len(precomputed_query_results),
            query_plan_input,
        )

    prompt_source_meta = get_prompt_source("locomo")

    if ratio < 1.0:
        num_samples = max(1, int(len(samples) * ratio))
        samples = samples[:num_samples]
        eval_logger.info(f"Using {num_samples} samples ({ratio*100:.1f}% of dataset)")

    results = []
    failed_entries: List[Dict[str, Any]] = []
    all_metrics = []
    all_categories = []
    total_questions = 0
    category_counts = defaultdict(int)
    judge_scores = []
    judge_scores_by_category = defaultdict(list)
    token_usage_totals = {
        "memory_build": _empty_token_stats(),
        "qa": _empty_token_stats(),
        "answer_generation": _empty_token_stats(),
        "answer_verification": _empty_token_stats(),
    }
    time_cost_totals = {
        "memory_build_seconds": 0.0,
        "retrieval_seconds": 0.0,
        "answer_seconds": 0.0,
    }
    memory_build_samples = []

    error_num = 0
    state_lock = threading.Lock()
    memories_dir = os.path.join(
        os.path.dirname(__file__),
        f"cached_memories_robust_{cache_identity}",
    )
    os.makedirs(memories_dir, exist_ok=True)
    # LoCoMo LLM-as-Judge evaluation skips category 5.
    allow_categories = [1, 2, 3, 4]

    for sample_idx, sample in enumerate(samples):
        sample_memory_build_start = time.perf_counter()
        agent = RobustAdvancedMemAgent(
            model,
            backend,
            retrieve_k,
            sglang_host,
            sglang_port,
            memory_config=memory_config,
            enable_hierarchy_memory=ablation_flags["enable_hierarchy_memory"],
            enable_evidence_packet_loop=ablation_flags["enable_evidence_packet_loop"],
            answer_max_tokens=answer_max_tokens,
            enable_answer_driven_verification=ablation_flags.get("enable_answer_driven_verification", False),
            answer_verification_gate=answer_verification_gate,
        )
        agent.memory_system.llm_controller.set_usage_phase("memory_build")
        agent.retriever_llm.set_usage_phase("memory_build")

        memory_cache_file = os.path.join(memories_dir, f"memory_cache_sample_{sample_idx}.pkl")
        retriever_cache_file = os.path.join(memories_dir, f"retriever_cache_sample_{sample_idx}.pkl")
        retriever_cache_embeddings_file = os.path.join(
            memories_dir, f"retriever_cache_embeddings_sample_{sample_idx}.npy"
        )

        rebuild_cache = True
        loaded_from_cache = False
        cached_topic_rebuild_count = 0
        if os.path.exists(memory_cache_file):
            eval_logger.info(f"Loading cached memories for sample {sample_idx}")
            try:
                with open(memory_cache_file, 'rb') as f:
                    cached_state = pickle.load(f)
                agent.memory_system.load_state(cached_state)
                eval_logger.info(
                    "Successfully loaded hierarchical memories: %d turns, %d episodes, %d topics",
                    len(agent.memory_system.turn_memories),
                    len(agent.memory_system.episode_memories),
                    len(agent.memory_system.topic_memories),
                )
                rebuild_cache = False
                loaded_from_cache = True
                cached_topic_rebuild_count = int(
                    getattr(agent.memory_system, "topic_rebuild_count", 0) or 0
                )
            except Exception as e:
                eval_logger.warning(
                    "Failed to load cache for sample %s: %s. Rebuilding hierarchical memory.",
                    sample_idx,
                    e,
                )

        if rebuild_cache:
            eval_logger.info(f"No cached memories found for sample {sample_idx}. Creating new hierarchical memories.")

            for session_id, turns in sample.conversation.sessions.items():
                turn_datatime = turns.date_time
                for turn in turns.turns:
                    conversation_tmp = f"Speaker {turn.speaker} says: {turn.text}"
                    agent.add_memory(
                        conversation_tmp,
                        time=turn_datatime,
                        memory_type="dialogue_turn",
                        session_id=session_id,
                        speaker=turn.speaker,
                    )

                agent.add_session_summary(session_id, turn_datatime, turns.turns)

            finalization_before = _memory_layer_counts(agent.memory_system)
            finalize_topics_called = False
            if ablation_flags["enable_hierarchy_memory"]:
                agent.finalize_topics()
                finalize_topics_called = True
            finalization_after = _memory_layer_counts(agent.memory_system)

            state_to_cache = agent.memory_system.export_state()
            _atomic_pickle_dump(state_to_cache, memory_cache_file)
            eval_logger.info(
                "Successfully cached hierarchical memories: %d turns, %d episodes, %d topics",
                len(agent.memory_system.turn_memories),
                len(agent.memory_system.episode_memories),
                len(agent.memory_system.topic_memories),
            )
        else:
            finalization_before = _memory_layer_counts(agent.memory_system)
            finalization_after = dict(finalization_before)
            finalize_topics_called = False

        topic_rebuild_count = 0
        if not loaded_from_cache:
            topic_rebuild_count = int(
                getattr(agent.memory_system, "topic_rebuild_count", 0) or 0
            )
        unassigned_episode_ids = _unassigned_episode_ids(agent.memory_system)
        memory_build_samples.append(
            {
                "sample_id": sample_idx,
                "loaded_from_cache": loaded_from_cache,
                "topic_rebuild_count": topic_rebuild_count,
                "cached_topic_rebuild_count": cached_topic_rebuild_count,
                "topic_finalization": {
                    "called": finalize_topics_called,
                    "before": finalization_before,
                    "after": finalization_after,
                },
                "unassigned_episode_count": len(unassigned_episode_ids),
                "unassigned_episode_ids": unassigned_episode_ids,
            }
        )

        time_cost_totals["memory_build_seconds"] += time.perf_counter() - sample_memory_build_start

        eval_logger.info(f"Processing sample {sample_idx + 1}/{len(samples)}")

        # ---- QA processing: serial or concurrent within this sample ----
        # retrieval+answer+judge are IO-bound (LLM API calls). The shared agent's
        # retrieval stack is concurrency-safe here: embedding encode uses the
        # OpenAI API (text-embedding-3-small, IO-bound), flush_pending_turn_analysis
        # is a no-op after memory build, the bm25 lazy cache races are benign under
        # the GIL, and turn_retriever state is read-only at answer time.
        sample_allowed = [qa for qa in sample.qa if int(qa.category) in allow_categories]
        base = total_questions
        tasks = []  # list of (plan_index, qa, precomputed_plan) in original order
        for _j, _qa in enumerate(sample_allowed):
            _plan_index = base + _j
            _precomputed_plan = None
            if precomputed_query_results:
                if _plan_index >= len(precomputed_query_results):
                    raise ValueError("Query Plan input has fewer questions than this run")
                _source_result = precomputed_query_results[_plan_index]
                if (
                    str(_source_result.get("question", "")) != str(_qa.question)
                    or str(_source_result.get("category", "")) != str(_qa.category)
                ):
                    raise ValueError(
                        "Query Plan input question alignment mismatch at index {}".format(_plan_index)
                    )
                _precomputed_plan = _source_result.get("query_plan")
                if not isinstance(_precomputed_plan, dict):
                    raise ValueError(
                        "Query Plan input is missing query_plan at index {}".format(_plan_index)
                    )
            tasks.append((_plan_index, _qa, _precomputed_plan))
        # Count allowed QAs up-front (matches prior behavior: counted before their answer).
        total_questions = base + len(sample_allowed)
        for _qa in sample_allowed:
            category_counts[_qa.category] += 1

        def _process_one_qa(_agent, plan_index, qa, precomputed_plan):
            """IO-bound per-QA work (retrieval+answer+judge). Runs in a worker
            thread when num_workers>1. Returns a structured dict; shared
            accumulators are NOT touched here -- the main thread updates them
            under state_lock, in original QA order, to preserve ordering."""
            _agent.memory_system.llm_controller.set_usage_phase("qa")
            _agent.retriever_llm.set_usage_phase("qa")
            try:
                prediction, user_prompt, ranked_context, query_plan, retrieval_time, answer_time = _agent.answer_question(
                    qa.question, qa.category, query_plan=precomputed_plan,
                    answer_prompt_mode="locomo",
                )
                prediction = parse_plain_text_answer(prediction)
                metrics = calculate_metrics(prediction, qa.final_answer) if qa.final_answer else {
                    "exact_match": 0, "f1": 0.0, "rouge1_f": 0.0, "rouge2_f": 0.0,
                    "rougeL_f": 0.0, "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0,
                    "bleu4": 0.0, "bert_f1": 0.0, "meteor": 0.0, "sbert_similarity": 0.0
                }
                judge_result = evaluate_llm_judge(
                    question=qa.question,
                    gold_answer=str(qa.final_answer),
                    generated_answer=prediction,
                    model=judge_model,
                    backend=judge_backend,
                    base_url=judge_base_url,
                    host=judge_sglang_host,
                    port=judge_sglang_port,
                    source_name="locomo",
                    judge_runs=judge_runs,
                )
                return {
                    "ok": True, "plan_index": plan_index, "qa": qa,
                    "prediction": prediction, "user_prompt": user_prompt,
                    "ranked_context": ranked_context, "query_plan": query_plan,
                    "retrieval_time": retrieval_time, "answer_time": answer_time,
                    "metrics": metrics, "judge_result": judge_result,
                }
            except Exception as e:
                import traceback as _tb
                return {
                    "ok": False, "plan_index": plan_index, "qa": qa,
                    "error": repr(e), "traceback": _tb.format_exc(),
                }

        if num_workers and num_workers > 1 and len(tasks) > 1:
            ordered = [None] * len(tasks)
            with ThreadPoolExecutor(max_workers=min(num_workers, len(tasks))) as _ex:
                _futs = {
                    _j: _ex.submit(_process_one_qa, agent, pi, qa, pp)
                    for _j, (pi, qa, pp) in enumerate(tasks)
                }
                for _j, _fut in _futs.items():
                    ordered[_j] = _fut.result()
        else:
            ordered = [_process_one_qa(agent, pi, qa, pp) for (pi, qa, pp) in tasks]

        for _res in ordered:
            with state_lock:
                _qa = _res["qa"]
                _plan_index = _res["plan_index"]
                _global_idx = _plan_index + 1
                if not _res.get("ok"):
                    failed_entries.append({
                        "sample_id": sample_idx,
                        "plan_index": _plan_index,
                        "question": getattr(_qa, "question", ""),
                        "category": getattr(_qa, "category", None),
                        "error": _res.get("error", ""),
                        "traceback": _res.get("traceback", ""),
                    })
                    error_num += 1
                    eval_logger.error("QA failed (sample %s, question %d): %s",
                                      sample_idx, _global_idx, _res.get("error", ""))
                    continue
                prediction = _res["prediction"]
                user_prompt = _res["user_prompt"]
                ranked_context = _res["ranked_context"]
                query_plan = _res["query_plan"]
                judge_result = _res["judge_result"]
                metrics = _res["metrics"]
                time_cost_totals["retrieval_seconds"] += _res["retrieval_time"]
                time_cost_totals["answer_seconds"] += _res["answer_time"]

                judge_prompt = str(judge_result["prompt"])
                judge_response = str(judge_result["raw_response"])
                judge_label = bool(judge_result["label"])
                llm_score = 1 if judge_label else 0
                metrics["llm_score"] = llm_score
                judge_scores.append(llm_score)
                judge_scores_by_category[_qa.category].append(llm_score)

                all_metrics.append(metrics)
                all_categories.append(_qa.category)

                result = {
                    "sample_id": sample_idx,
                    "ablation_mode": ablation_mode,
                    "answer_prompt_mode": "locomo",
                    "answer_max_tokens": answer_max_tokens,
                    "question": _qa.question,
                    "prediction": prediction,
                    "reference": _qa.final_answer,
                    "category": _qa.category,
                    "query_plan": query_plan,
                    "context_strategy": context_strategy,
                    "retrieved_context": query_plan.get("raw_retrieval_context", ranked_context),
                    "answer_context": ranked_context,
                    "raw_context": query_plan.get("raw_retrieval_context", ranked_context),
                    "focused_context": ranked_context,
                    "evidence_context": ranked_context,
                    "selected_evidence": query_plan.get("selected_evidence", []),
                    "metrics": metrics,
                    "llm_score": llm_score,
                    "judge": {
                        "model": judge_model,
                        "backend": judge_backend,
                        "prompt_source": "locomo",
                        "prompt_source_meta": prompt_source_meta,
                        "effective_config": judge_result.get("effective_config", {}),
                        "judge_runs": judge_result.get("judge_runs", judge_runs),
                        "judgments": judge_result.get("judgments"),
                        "all_raw_responses": judge_result.get("all_raw_responses"),
                        "system_prompt": str(judge_result["system_prompt"]),
                        "prompt": judge_prompt,
                        "response": judge_response,
                        "label": judge_label,
                    },
                }
                results.append(result)
                if output_path:
                    try:
                        _atomic_write_json(output_path, {
                            "benchmark": "locomo",
                            "dataset": dataset_path,
                            "model": model,
                            "ablation_mode": ablation_mode,
                            "answer_prompt_mode": "locomo",
                            "num_workers": num_workers,
                            "runtime_status": {"completed": len(results), "failed": len(failed_entries), "finalized": False, "total": total_questions, "last_updated": _dt_module.datetime.now().isoformat(timespec="seconds")},
                            "failed_entries": failed_entries,
                            "individual_results": results,
                        })
                    except Exception as _exc:
                        eval_logger.warning("Incremental write failed: %s", _exc)
                eval_logger.info(f"Question {_global_idx}: {_qa.question}")
                eval_logger.info(f"Prediction: {prediction}")
                eval_logger.info(f"Reference: {_qa.final_answer}")
                eval_logger.info(f"Judge label: {judge_label}")
                eval_logger.info(f"User Prompt: {user_prompt}")
                eval_logger.info(f"Category: {_qa.category}")
                eval_logger.info(f"Query Plan: {query_plan}")
                eval_logger.info(f"Context Strategy: {context_strategy}")
                if _global_idx % 10 == 0:
                    eval_logger.info(f"Processed {_global_idx} questions")

        _accumulate_phase_usage(
            token_usage_totals,
            agent.memory_system.llm_controller.get_token_usage_summary(),
        )
        _accumulate_phase_usage(
            token_usage_totals,
            agent.retriever_llm.get_token_usage_summary(),
        )

    aggregate_results = aggregate_metrics(all_metrics, all_categories)
    aggregate_judge_metrics = {
        "enabled": True,
        "overall_accuracy": statistics.mean(judge_scores) if judge_scores else None,
        "per_category": {
            str(category): {
                "accuracy": statistics.mean(scores),
                "count": len(scores),
            }
            for category, scores in sorted(judge_scores_by_category.items())
            if scores
        },
        "count": len(judge_scores),
    }
    sample_count = len(samples)
    time_cost_summary = {
        "memory_build_seconds": time_cost_totals["memory_build_seconds"],
        "retrieval_seconds": time_cost_totals["retrieval_seconds"],
        "answer_seconds": time_cost_totals["answer_seconds"],
        "memory_build_avg_seconds_per_sample": (
            time_cost_totals["memory_build_seconds"] / sample_count if sample_count else 0.0
        ),
        "retrieval_avg_seconds_per_question": (
            time_cost_totals["retrieval_seconds"] / total_questions if total_questions else 0.0
        ),
        "answer_avg_seconds_per_question": (
            time_cost_totals["answer_seconds"] / total_questions if total_questions else 0.0
        ),
    }
    question_answering_usage = _combined_token_stats(
        token_usage_totals["qa"],
        token_usage_totals["answer_generation"],
    )
    requested_config = {
        "ablation_mode": ablation_mode,
        "seed": seed,
        "query_plan_input": query_plan_input,
        "topic_recluster_interval": topic_recluster_interval,
        "topic_result_limit": topic_result_limit,
        "episode_result_limit": episode_result_limit,
        "turn_result_limit": turn_result_limit,
        "retrieval_semantic_weight": retrieval_semantic_weight,
        "retrieval_lexical_weight": retrieval_lexical_weight,
        "retrieval_recency_weight": retrieval_recency_weight,
        "retrieval_weight_mode": retrieval_weight_mode,
        "retrieval_adaptive_strength": retrieval_adaptive_strength,
        "retrieval_temporal_bonus": retrieval_temporal_bonus,
        "retrieval_budget_low_multiplier": retrieval_budget_low_multiplier,
        "retrieval_budget_mid_multiplier": retrieval_budget_mid_multiplier,
        "retrieval_budget_high_multiplier": retrieval_budget_high_multiplier,
        "retrieval_provenance_boost": retrieval_provenance_boost,
        "retrieval_min_turn_support": retrieval_min_turn_support,
        "retrieval_uncertainty_temperature": retrieval_uncertainty_temperature,
        "retrieval_uncertainty_margin_scale": retrieval_uncertainty_margin_scale,
        "retrieval_uncertainty_low_threshold": retrieval_uncertainty_low_threshold,
        "retrieval_uncertainty_high_threshold": retrieval_uncertainty_high_threshold,
        "retrieval_final_max_items": retrieval_final_max_items,
        "judge_backend": judge_backend,
        "judge_base_url": judge_base_url,
        "judge_host": judge_sglang_host,
        "judge_port": judge_sglang_port,
        "topic_build_mode": topic_build_mode,
        "enable_anchors": enable_anchors,
    }
    effective_config = {
        "memory": memory_config,
        "seed": seed,
        "query_plan_reused": bool(query_plan_input),
        "judge": {
            "backend": judge_backend,
            "model": judge_model,
            "base_url": judge_base_url,
            "host": judge_sglang_host,
            "port": judge_sglang_port,
        },
    }
    memory_build_diagnostics = {
        "topic_build_mode": memory_config_obj.topic_build_mode,
        "topic_recluster_interval": memory_config_obj.topic_recluster_interval,
        "enable_anchors": memory_config_obj.enable_anchors,
        "samples": memory_build_samples,
    }

    final_results = {
        "model": model,
        "backend": backend,
        "dataset": dataset_path,
        "ablation_mode": ablation_mode,
        "answer_prompt_mode": "locomo",
        "answer_max_tokens": answer_max_tokens,
        "judge_runs": judge_runs,
        "num_workers": num_workers,
        "ablation_flags": ablation_flags,
        "answer_verification_config": {
            "enabled": bool(ablation_flags.get("enable_answer_driven_verification", False)),
            "gate": answer_verification_gate,
        },
        "memory_layer": "robust_hierarchical",
        "memory_hierarchy_levels": 3,
        "memory_config": memory_config,
        "memory_config_signature": memory_config_signature,
        "memory_cache_signature": memory_cache_signature,
        "cache_family": cache_family,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_identity": cache_identity,
        "retrieval_weight_mode": primary_weight_mode,
        "retrieval_adaptive_strength": float(memory_config_obj.retrieval_adaptive_strength),
        "context_strategy": context_strategy,
        "cache_version": CACHE_VERSION,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "judge_base_url": judge_base_url,
        "judge_host": judge_sglang_host,
        "judge_port": judge_sglang_port,
        "judge_prompt_source": "locomo",
        "judge_enabled": True,
        "judge_prompt_sources_available": list_prompt_sources(),
        "total_questions": total_questions,
        "category_distribution": {
            str(cat): count for cat, count in category_counts.items()
        },
        "aggregate_metrics": aggregate_results,
        "aggregate_judge_metrics": aggregate_judge_metrics,
        "routing_summary": _summarize_routing_results(results),
        "answer_length_summary": _summarize_answer_lengths(results),
        "requested_config": requested_config,
        "effective_config": effective_config,
        "memory_build_diagnostics": memory_build_diagnostics,
        "generation_uses_reference_answer": False,
        "token_usage": {
            "memory_build": token_usage_totals["memory_build"],
            "qa": token_usage_totals["qa"],
            "answer_generation": token_usage_totals["answer_generation"],
            "question_answering": question_answering_usage,
            "memory_build_total_tokens": token_usage_totals["memory_build"]["total_tokens"],
            "qa_total_tokens": token_usage_totals["qa"]["total_tokens"],
            "answer_generation_total_tokens": token_usage_totals["answer_generation"]["total_tokens"],
            "question_answering_total_tokens": question_answering_usage["total_tokens"],
        },
        "time_cost": time_cost_summary,
        "individual_results": results,
    }
    eval_logger.info(f"Error number: {error_num}")

    # Attach runtime status & failure accounting, write atomically.
    final_results["runtime_status"] = {
        "total": total_questions,
        "completed": len(results),
        "failed": len(failed_entries),
        "pending": 0,
        "finalized": True,
        "last_updated": _dt_module.datetime.now().isoformat(timespec="seconds"),
    }
    final_results["failed_entries"] = failed_entries
    final_results["failed_summary"] = _summarize_failures(failed_entries)
    if output_path:
        _atomic_write_json(output_path, final_results)
        eval_logger.info(f"Results saved to {output_path}")

    eval_logger.info("Evaluation Summary:")
    eval_logger.info(f"Total questions evaluated: {total_questions}")
    eval_logger.info("Category Distribution:")
    for category, count in sorted(category_counts.items()):
        eval_logger.info(f"Category {category}: {count} questions ({count/total_questions*100:.1f}%)")

    eval_logger.info("Aggregate Metrics:")
    for split_name, metrics in aggregate_results.items():
        eval_logger.info(f"{split_name.replace('_', ' ').title()}:")
        for metric_name, stats in metrics.items():
            eval_logger.info(f"  {metric_name}:")
            for stat_name, value in stats.items():
                eval_logger.info(f"    {stat_name}: {value:.4f}")

    eval_logger.info("Token Usage Summary:")
    eval_logger.info(
        "Memory build tokens: total=%d (prompt=%d, completion=%d, calls=%d)",
        token_usage_totals["memory_build"]["total_tokens"],
        token_usage_totals["memory_build"]["prompt_tokens"],
        token_usage_totals["memory_build"]["completion_tokens"],
        token_usage_totals["memory_build"]["calls"],
    )
    eval_logger.info(
        "QA tokens (retrieval/planning/question pipeline): total=%d (prompt=%d, completion=%d, calls=%d)",
        token_usage_totals["qa"]["total_tokens"],
        token_usage_totals["qa"]["prompt_tokens"],
        token_usage_totals["qa"]["completion_tokens"],
        token_usage_totals["qa"]["calls"],
    )
    eval_logger.info(
        "Answer generation tokens (final answer only): total=%d (prompt=%d, completion=%d, calls=%d)",
        token_usage_totals["answer_generation"]["total_tokens"],
        token_usage_totals["answer_generation"]["prompt_tokens"],
        token_usage_totals["answer_generation"]["completion_tokens"],
        token_usage_totals["answer_generation"]["calls"],
    )
    eval_logger.info(
        "Answer verification tokens (confidence probe): total=%d (prompt=%d, completion=%d, calls=%d)",
        token_usage_totals.get("answer_verification", {}).get("total_tokens", 0),
        token_usage_totals.get("answer_verification", {}).get("prompt_tokens", 0),
        token_usage_totals.get("answer_verification", {}).get("completion_tokens", 0),
        token_usage_totals.get("answer_verification", {}).get("calls", 0),
    )
    eval_logger.info(
        "Question answering tokens (qa + answer_generation): total=%d (prompt=%d, completion=%d, calls=%d)",
        question_answering_usage["total_tokens"],
        question_answering_usage["prompt_tokens"],
        question_answering_usage["completion_tokens"],
        question_answering_usage["calls"],
    )
    eval_logger.info("Time Cost Summary:")
    eval_logger.info(
        "Memory build time: total=%.4fs, avg_per_sample=%.4fs",
        time_cost_summary["memory_build_seconds"],
        time_cost_summary["memory_build_avg_seconds_per_sample"],
    )
    eval_logger.info(
        "Memory retrieval time: total=%.4fs, avg_per_question=%.4fs",
        time_cost_summary["retrieval_seconds"],
        time_cost_summary["retrieval_avg_seconds_per_question"],
    )
    eval_logger.info(
        "Answer generation time: total=%.4fs, avg_per_question=%.4fs",
        time_cost_summary["answer_seconds"],
        time_cost_summary["answer_avg_seconds_per_question"],
    )

    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate robust text-only agent on LoComo dataset (no JSON schema dependency)"
    )
    parser.add_argument("--dataset", type=str, default="data/locomo10.json",
                        help="Path to the dataset file")
    parser.add_argument("--model", type=str, default="gpt-4.1-mini-2025-04-14",
                        help="Model to use")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Ratio of dataset to evaluate (0.0 to 1.0)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used for reproducible evaluation")
    parser.add_argument("--query-plan-input", default=None,
                        help="Prior runner JSON whose Query Plans must be reused")
    parser.add_argument("--backend", type=str, default="openai",
                        help="Backend to use (openai, ollama, sglang, or vllm)")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-mini-2025-04-14",
                        help="Model to use for LoCoMo LLM-as-Judge")
    parser.add_argument("--judge-backend", type=str, default="openai",
                        help="Backend to use for LoCoMo LLM-as-Judge")
    parser.add_argument("--judge-base-url", type=str, default=None,
                        help="OpenAI-compatible base URL for the judge backend")
    parser.add_argument("--retrieve_k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--ablation-mode", type=str, default="main",
                        choices=ABLATION_MODES,
                        help="Ablation setting: main | no_hierarchy | no_uncertainty | no_evidence_loop | "
                             "no_evidence_loop,no_hierarchy | no_evidence_loop,no_uncertainty | "
                             "no_hierarchy,no_uncertainty | no_evidence_loop,no_hierarchy,no_uncertainty")
    parser.add_argument("--topic-match-threshold", type=float, default=None,
                        help="Override the topic merge threshold used for cross-session clustering")
    parser.add_argument("--topic-min-token-overlap", type=int, default=None,
                        help="Override the minimum lexical overlap required before merging into a topic")
    parser.add_argument("--episode-summary-turn-window", type=int, default=None,
                        help="Maximum number of turns sampled into each episode summary; 0 means use all turns")
    parser.add_argument("--topic-summary-episode-window", type=int, default=None,
                        help="Maximum number of episode summaries used when refreshing a topic; 0 means use all")
    parser.add_argument("--topic-update-strategy", type=str, default=None, choices=["full", "recent"],
                        help="Cross-session topic refresh strategy")
    parser.add_argument("--topic-result-limit", type=int, default=None,
                        help="Maximum number of topic-layer memories returned in retrieval")
    parser.add_argument("--episode-result-limit", type=int, default=None,
                        help="Maximum number of episode-layer memories returned in retrieval")
    parser.add_argument("--turn-result-limit", type=int, default=None,
                        help="Maximum number of turn-layer memories returned in retrieval")
    parser.add_argument("--episode-min-turns", type=int, default=None,
                        help="Minimum turns per episode segment")
    parser.add_argument("--episode-max-turns", type=int, default=None,
                        help="Maximum turns per episode segment")
    parser.add_argument("--episode-shift-similarity-threshold", type=float, default=None,
                        help="Embedding similarity threshold below which a topic shift boundary is allowed")
    parser.add_argument("--episode-shift-lexical-overlap-threshold", type=float, default=None,
                        help="Lexical overlap threshold below which a topic shift boundary is allowed")
    parser.add_argument("--episode-max-llm-turns", type=int, default=None,
                        help="Maximum turns to send to the LLM segmenter; longer sessions stay as one episode")
    parser.add_argument("--topic-assignment-strategy", type=str, default="clustered",
                        choices=["incremental", "clustered"],
                        help="Topic construction strategy")
    parser.add_argument("--topic-cluster-distance-threshold", type=float, default=0.30,
                        help="Distance threshold used by agglomerative clustering over episode embeddings")
    parser.add_argument("--topic-cluster-min-size", type=int, default=2,
                        help="Minimum preferred cluster size before small-cluster post-merge")
    parser.add_argument("--topic-recluster-interval", type=int, default=16,
                        help="How many new episodes to accumulate before reclustering topics")
    parser.add_argument("--topic-small-cluster-merge-threshold", type=float, default=0.64,
                        help="Similarity threshold for merging small clusters into larger topic clusters")
    parser.add_argument("--retrieval-semantic-weight", type=float, default=None,
                        help="Semantic score weight in hybrid retrieval")
    parser.add_argument("--retrieval-lexical-weight", type=float, default=None,
                        help="Lexical BM25 score weight in hybrid retrieval")
    parser.add_argument("--retrieval-recency-weight", type=float, default=None,
                        help="Recency score weight in hybrid retrieval")
    parser.add_argument("--retrieval-weight-mode", type=str, default="adaptive",
                        choices=["static", "adaptive"],
                        help="Fusion mode for retrieval weights (static fallback or non-LLM soft adaptive)")
    parser.add_argument("--retrieval-adaptive-strength", type=float, default=0.35,
                        help="Blend strength for adaptive retrieval weights (0.0 uses static fallback)")
    parser.add_argument("--retrieval-temporal-bonus", type=float, default=0.10,
                        help="Bonus multiplier for temporal cue alignment")
    parser.add_argument("--retrieval-budget-low-multiplier", type=float, default=None,
                        help="Candidate expansion multiplier for low-complexity queries")
    parser.add_argument("--retrieval-budget-mid-multiplier", type=float, default=None,
                        help="Candidate expansion multiplier for medium-complexity queries")
    parser.add_argument("--retrieval-budget-high-multiplier", type=float, default=None,
                        help="Candidate expansion multiplier for high-complexity queries")
    parser.add_argument("--retrieval-provenance-boost", type=float, default=0.55,
                        help="Boost factor for provenance-based evidence escalation")
    parser.add_argument("--retrieval-min-turn-support", type=int, default=2,
                        help="Minimum number of supporting turn memories before skipping escalation")
    parser.add_argument("--retrieval-uncertainty-temperature", type=float, default=0.08,
                        help="Softmax temperature used by score-distribution uncertainty")
    parser.add_argument("--retrieval-uncertainty-margin-scale", type=float, default=0.08,
                        help="Scale used to normalize the top-score margin")
    parser.add_argument("--retrieval-uncertainty-low-threshold", type=float, default=0.40,
                        help="Upper routing-score boundary for the low tier")
    parser.add_argument("--retrieval-uncertainty-high-threshold", type=float, default=0.70,
                        help="Lower routing-score boundary for the high tier")
    parser.add_argument("--retrieval-final-max-items", type=int, default=24,
                        help="Global upper bound for final routed evidence items")
    parser.add_argument("--sglang_host", type=str, default="http://localhost",
                        help="SGLang server host (for sglang backend)")
    parser.add_argument("--sglang_port", type=int, default=30000,
                        help="SGLang server port (for sglang backend)")
    parser.add_argument("--judge-sglang-host", type=str, default="http://localhost",
                        help="SGLang/vLLM host for the judge backend")
    parser.add_argument("--judge-sglang-port", type=int, default=30000,
                        help="SGLang/vLLM port for the judge backend")
    args = parser.parse_args()

    if args.ratio <= 0.0 or args.ratio > 1.0:
        raise ValueError("Ratio must be between 0.0 and 1.0")

    dataset_path = os.path.join(os.path.dirname(__file__), args.dataset)
    output_path = os.path.join(os.path.dirname(__file__), args.output) if args.output else None

    evaluate_dataset(
        dataset_path=dataset_path,
        model=args.model,
        output_path=output_path,
        ratio=args.ratio,
        seed=args.seed,
        query_plan_input=args.query_plan_input,
        backend=args.backend,
        retrieve_k=args.retrieve_k,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        judge_base_url=args.judge_base_url,
        judge_sglang_host=args.judge_sglang_host,
        judge_sglang_port=args.judge_sglang_port,
        topic_match_threshold=args.topic_match_threshold,
        topic_min_token_overlap=args.topic_min_token_overlap,
        episode_summary_turn_window=args.episode_summary_turn_window,
        topic_summary_episode_window=args.topic_summary_episode_window,
        topic_update_strategy=args.topic_update_strategy,
        topic_result_limit=args.topic_result_limit,
        episode_result_limit=args.episode_result_limit,
        turn_result_limit=args.turn_result_limit,
        episode_min_turns=args.episode_min_turns,
        episode_max_turns=args.episode_max_turns,
        episode_shift_similarity_threshold=args.episode_shift_similarity_threshold,
        episode_shift_lexical_overlap_threshold=args.episode_shift_lexical_overlap_threshold,
        episode_max_llm_turns=args.episode_max_llm_turns,
        topic_assignment_strategy=args.topic_assignment_strategy,
        topic_cluster_distance_threshold=args.topic_cluster_distance_threshold,
        topic_cluster_min_size=args.topic_cluster_min_size,
        topic_recluster_interval=args.topic_recluster_interval,
        topic_small_cluster_merge_threshold=args.topic_small_cluster_merge_threshold,
        retrieval_semantic_weight=args.retrieval_semantic_weight,
        retrieval_lexical_weight=args.retrieval_lexical_weight,
        retrieval_recency_weight=args.retrieval_recency_weight,
        retrieval_temporal_bonus=args.retrieval_temporal_bonus,
        retrieval_budget_low_multiplier=args.retrieval_budget_low_multiplier,
        retrieval_budget_mid_multiplier=args.retrieval_budget_mid_multiplier,
        retrieval_budget_high_multiplier=args.retrieval_budget_high_multiplier,
        retrieval_provenance_boost=args.retrieval_provenance_boost,
        retrieval_min_turn_support=args.retrieval_min_turn_support,
        retrieval_weight_mode=args.retrieval_weight_mode,
        retrieval_adaptive_strength=args.retrieval_adaptive_strength,
        retrieval_uncertainty_temperature=args.retrieval_uncertainty_temperature,
        retrieval_uncertainty_margin_scale=args.retrieval_uncertainty_margin_scale,
        retrieval_uncertainty_low_threshold=args.retrieval_uncertainty_low_threshold,
        retrieval_uncertainty_high_threshold=args.retrieval_uncertainty_high_threshold,
        retrieval_final_max_items=args.retrieval_final_max_items,
        ablation_mode=args.ablation_mode,
    )


if __name__ == "__main__":
    main()
