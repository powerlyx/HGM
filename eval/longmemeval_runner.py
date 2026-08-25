"""LongMemEval evaluation runner.

Builds a hierarchical memory per LongMemEval entry (cached by question_id),
answers with the official LongMemEval answer prompt, judges with the official
answer-check prompt, and aggregates the official accuracy metrics.

Reuses the cache-identity, ablation, token and timeline helpers from
``eval.runner`` so ablations share the same construction-aware cache semantics
as the LoCoMo path.
"""

import copy
import json
import logging
import os
import pickle
import random
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from dataset.longmemeval import (
    LongMemEvalEntry,
    is_abstention,
    load_longmemeval_dataset,
)
from eval.llm_judge import evaluate_llm_judge
from eval.metrics import calculate_metrics
from eval.runner import (
    CACHE_VERSION,
    _accumulate_phase_usage,
    _apply_ablation_to_memory_config,
    _atomic_pickle_dump,
    _build_cache_identity,
    _build_memory_config,
    _combined_token_stats,
    _empty_token_stats,
    _fingerprint_file,
    _normalize_ablation_mode,
    _resolve_ablation_flags,
    _resolve_cache_family,
    _summarize_answer_lengths,
    _summarize_routing_results,
    setup_logger,
)
from eval.agent import RobustAdvancedMemAgent

logger = logging.getLogger("amem_robust")


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    """Atomically write JSON (tmp file + os.replace) so a crash never leaves a
    half-written result file."""
    import os as _os, json as _json, tempfile as _tf
    directory = _os.path.dirname(path) or "."
    _os.makedirs(directory, exist_ok=True)
    fd, tmp_path = _tf.mkstemp(prefix=".lme_inc_", dir=directory)
    try:
        with _os.fdopen(fd, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
            f.flush()
            _os.fsync(f.fileno())
        _os.replace(tmp_path, path)
    finally:
        if _os.path.exists(tmp_path):
            try:
                _os.remove(tmp_path)
            except OSError:
                pass


def _summarize_failures(failed_entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Bucket failed entries by error category for quick triage."""
    buckets: Dict[str, int] = defaultdict(int)
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
            cat = type_name = err.split(":", 1)[0] or "other"
        buckets[cat] += 1
    return {"count": len(failed_entries or []), "by_error": dict(sorted(buckets.items()))}

def _reuse_norm_eq(a, b) -> bool:
    """Normalized equality used by the reuse guards."""
    if a is None and b is None:
        return True
    if isinstance(a, bool) or isinstance(b, bool):
        return bool(a) == bool(b)
    if isinstance(a, str) or isinstance(b, str):
        return str(a).strip() == str(b).strip()
    try:
        return a == b
    except Exception:
        return False


def _build_run_meta_entry(entry, top_meta: Dict[str, Any]) -> Dict[str, Any]:
    """Per-entry run metadata combined with the run-level config snapshot."""
    meta = dict(top_meta)
    meta.update({
        "question_id": entry.question_id,
        "question": entry.question,
        "question_date": entry.question_date,
        "question_type": entry.question_type,
        "reference": entry.answer,
    })
    return meta


def _check_reuse_eligibility(source_record, source_meta, run_meta):
    """Return ``(answer_ok, judge_ok, reason)`` for reusing a prior result row.

    Pure function: no side effects, all inputs are plain dicts. ``reason`` is a
    short stable code (``ok`` / ``ok_judge_disabled`` / ``<field>_mismatch`` /
    ``incomplete_*`` / ``no_source_record``). When ``answer_ok`` is True and
    ``judge_ok`` is False the caller may re-invoke only the judge.
    """
    if not source_record or not isinstance(source_record, dict):
        return False, False, "no_source_record"
    # ---- per-entry alignment ----
    for fld, reason in (
        ("question_id", "question_id_mismatch"),
        ("question", "question_mismatch"),
        ("question_date", "question_date_mismatch"),
        ("question_type", "question_type_mismatch"),
        ("reference", "reference_mismatch"),
        ("answer_prompt_mode", "answer_prompt_mode_mismatch"),
    ):
        src_val = source_record.get(fld)
        # Older result files did not record `answer_prompt_mode` (None). A
        # missing value means "not recorded" rather than "a different mode".
        # Skip the guard in that case, otherwise every row is rejected and
        # --reuse-results becomes a silent no-op (all entries full-rerun).
        if fld == "answer_prompt_mode" and src_val is None:
            continue
        if not _reuse_norm_eq(src_val, run_meta.get(fld)):
            return False, False, reason
    # ---- run-level config equality ----
    src_meta = source_meta or {}
    for fld, reason in (
        ("model", "model_mismatch"),
        ("backend", "backend_mismatch"),
        ("cache_identity", "cache_identity_mismatch"),
        ("memory_config_signature", "memory_config_signature_mismatch"),
        ("ablation_mode", "ablation_mismatch"),
        ("retrieve_k", "retrieve_k_mismatch"),
    ):
        if not _reuse_norm_eq(src_meta.get(fld), run_meta.get(fld)):
            return False, False, reason
    # ---- completeness ----
    prediction = source_record.get("prediction")
    if prediction is None or str(prediction).strip() == "":
        return False, False, "incomplete_prediction"
    if not source_record.get("query_plan"):
        return False, False, "incomplete_query_plan"
    if "selected_evidence" not in source_record:
        return False, False, "incomplete_evidence"
    answer_ok = True
    # ---- judge guard ----
    if not run_meta.get("judge_enabled"):
        return answer_ok, False, "ok_judge_disabled"
    if not src_meta.get("judge_enabled"):
        return answer_ok, False, "source_judge_disabled"
    for fld, reason in (
        ("judge_model", "judge_model_mismatch"),
        ("judge_backend", "judge_backend_mismatch"),
        ("judge_prompt_source", "judge_prompt_source_mismatch"),
    ):
        if not _reuse_norm_eq(src_meta.get(fld), run_meta.get(fld)):
            return answer_ok, False, reason
    src_url = src_meta.get("judge_base_url")
    run_url = run_meta.get("judge_base_url")
    if src_url and run_url and not _reuse_norm_eq(src_url, run_url):
        return answer_ok, False, "judge_base_url_mismatch"
    judge_block = source_record.get("judge") or {}
    if judge_block.get("label") is None:
        return answer_ok, False, "incomplete_judge"
    if not _reuse_norm_eq(judge_block.get("model"), run_meta.get("judge_model")):
        return answer_ok, False, "judge_block_model_mismatch"
    return answer_ok, True, "ok"

_LONGMEMEVAL_QUESTION_TYPES = (
    "single-session-user",
    "single-session-preference",
    "single-session-assistant",
    "multi-session",
    "temporal-reasoning",
    "knowledge-update",
)


def _aggregate_longmemeval_metrics(
    entries: List[LongMemEvalEntry],
    labels: List[bool],
) -> Dict[str, Any]:
    """Official-style accuracy aggregation (evaluate_qa.py + print_qa_metrics.py)."""
    type2labels: Dict[str, List[int]] = {t: [] for t in _LONGMEMEVAL_QUESTION_TYPES}
    abstention_labels: List[int] = []
    overall: List[int] = []

    for entry, label in zip(entries, labels):
        bit = 1 if bool(label) else 0
        qtype = entry.question_type
        if qtype in type2labels:
            type2labels[qtype].append(bit)
        overall.append(bit)
        if entry.is_abstention:
            abstention_labels.append(bit)

    per_type = {
        qtype: {
            "accuracy": round(statistics.mean(scores), 4) if scores else None,
            "count": len(scores),
        }
        for qtype, scores in type2labels.items()
    }
    task_accs = [statistics.mean(scores) for scores in type2labels.values() if scores]
    return {
        "overall_accuracy": round(statistics.mean(overall), 4) if overall else None,
        "per_question_type": per_type,
        "task_averaged_accuracy": round(statistics.mean(task_accs), 4) if task_accs else None,
        "abstention_accuracy": (
            round(statistics.mean(abstention_labels), 4) if abstention_labels else None
        ),
        "abstention_count": len(abstention_labels),
        "total_count": len(overall),
    }


def _aggregate_longmemeval_text_metrics(
    entries: List[LongMemEvalEntry],
    per_question_metrics: List[Dict[str, float]],
) -> Dict[str, Dict[str, Any]]:
    """Per-question-type averages of traditional text metrics + accuracy.

    Also folds in abstention as its own pseudo-bucket when present.
    Used to produce the extra per-type-summary file requested by the user.
    """
    type_buckets: Dict[str, List[Dict[str, float]]] = {t: [] for t in _LONGMEMEVAL_QUESTION_TYPES}
    abstention_bucket: List[Dict[str, float]] = []
    for entry, m in zip(entries, per_question_metrics):
        if entry.is_abstention:
            abstention_bucket.append(m)
        if entry.question_type in type_buckets:
            type_buckets[entry.question_type].append(m)

    metric_keys = ("exact_match", "f1", "rouge1_f", "rouge2_f", "rougeL_f",
                   "bleu1", "bleu2", "bleu3", "bleu4", "meteor", "sbert_similarity", "llm_score")

    def _avg(rows, keys=metric_keys):
        out = {}
        for k in keys:
            vals = [float(r.get(k, 0.0) or 0.0) for r in rows if r.get(k) is not None]
            out[k] = round(sum(vals) / len(vals), 6) if vals else None
        return out

    summary = {}
    for qtype, rows in type_buckets.items():
        agg = _avg(rows) if rows else {k: None for k in metric_keys}
        agg["count"] = len(rows)
        summary[qtype] = agg
    # abstention bucket
    summary["abstention"] = {**(_avg(abstention_bucket) if abstention_bucket else {k: None for k in metric_keys}),
                             "count": len(abstention_bucket)}
    # overall row (all entries)
    all_rows = per_question_metrics
    summary["overall"] = {**_avg(all_rows), "count": len(all_rows)}
    return summary


def _write_per_type_summary(
    path: str,
    entries: List[LongMemEvalEntry],
    per_question_metrics: List[Dict[str, float]],
    judge_labels: List[bool],
    judge_enabled: bool,
    meta: Dict[str, Any],
) -> None:
    """Write the per-question-type averages summary JSON file."""
    text_summary = _aggregate_longmemeval_text_metrics(entries, per_question_metrics)
    # attach judge accuracy per type
    type2labels = {t: [] for t in _LONGMEMEVAL_QUESTION_TYPES}
    abst_labels = []
    for entry, lab in zip(entries, judge_labels if judge_labels else [None] * len(entries)):
        if lab is None:
            continue
        if entry.question_type in type2labels:
            type2labels[entry.question_type].append(1 if lab else 0)
        if entry.is_abstention:
            abst_labels.append(1 if lab else 0)
    for qtype, v in type2labels.items():
        text_summary[qtype]["judge_accuracy"] = (
            round(sum(v) / len(v), 6) if v else None
        ) if judge_enabled else None
        text_summary[qtype]["judge_count"] = len(v)
    abst_acc = (round(sum(abst_labels) / len(abst_labels), 6) if abst_labels else None) if judge_enabled else None
    text_summary["abstention"]["judge_accuracy"] = abst_acc
    text_summary["abstention"]["judge_count"] = len(abst_labels)

    payload = {
        "benchmark": "longmemeval",
        "description": "Per-question-type averages (accuracy + text metrics). "
                       "llm_score/judge_accuracy are judge-based accuracy; others are traditional text metrics.",
        "judge_enabled": judge_enabled,
        **meta,
        "per_type": text_summary,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _build_entry_memory(agent, entry: LongMemEvalEntry, enable_hierarchy: bool):
    """Add all sessions of one entry into the agent's memory system."""
    for session in entry.sessions:
        for turn in session.turns:
            conversation_tmp = f"Speaker {turn.speaker} says: {turn.text}"
            agent.add_memory(
                conversation_tmp,
                time=session.date_time,
                memory_type="dialogue_turn",
                session_id=session.session_id,
                speaker=turn.speaker,
            )
        agent.add_session_summary(session.session_id, session.date_time, session.turns)

    if enable_hierarchy:
        agent.finalize_topics()


def evaluate_longmemeval_dataset(
    dataset_path: str,
    model: str,
    output_path: Optional[str] = None,
    ratio: float = 1.0,
    backend: str = "sglang",
    retrieve_k: int = 10,
    sglang_host: str = "http://localhost",
    sglang_port: int = 30000,
    judge_model: str = "gpt-4.1-mini-2025-04-14",
    judge_backend: str = "openai",
    judge_base_url: Optional[str] = None,
    judge_sglang_host: str = "http://localhost",
    judge_sglang_port: int = 30000,
    answer_max_tokens: int = 32768,
    judge_runs: int = 1,
    ablation_mode: str = "main",
    seed: int = 0,
    query_plan_input: Optional[str] = None,
    num_workers: int = 1,
    per_type_summary: Optional[str] = None,
    reuse_results: Optional[str] = None,
    reuse_retrieve_k: Optional[int] = None,
    # retrieval / memory build params (forwarded to _build_memory_config)
    **memory_build_kwargs,
) -> Dict[str, Any]:
    """Evaluate LongMemEval with the hierarchical memory system + official prompts.

    Each LongMemEval entry is independent (own memory + cache), so concurrent
    execution via a thread pool is safe: LLM calls are IO-bound. ``num_workers>1``
    processes entries in parallel; results are re-collected in original order.
    ``per_type_summary`` optionally writes an extra JSON with per-question-type
    averages (accuracy + text metrics).
    """
    random.seed(seed)
    np.random.seed(seed)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    log_filename = f"longmemeval_{model}_{backend}_ratio{ratio}_{timestamp}.log"
    log_path = os.path.join(os.path.dirname(__file__), "logs", log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    eval_logger = setup_logger(log_path)

    eval_logger.info("Loading LongMemEval dataset from %s", dataset_path)
    entries = load_longmemeval_dataset(dataset_path)
    eval_logger.info("Loaded %d LongMemEval entries", len(entries))

    ablation_mode = _normalize_ablation_mode(ablation_mode)
    ablation_flags = _resolve_ablation_flags(ablation_mode)
    eval_logger.info("Ablation mode: %s | flags: %s", ablation_mode, ablation_flags)

    memory_config_obj = _build_memory_config(**memory_build_kwargs)
    memory_config_obj.analyze_turn_mode = "batch"  # Enable batched turn analysis in production
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

    memories_dir = os.path.join(
        os.path.dirname(__file__), f"cached_memories_longmemeval_{cache_identity}"
    )
    os.makedirs(memories_dir, exist_ok=True)
    run_top_meta = {
        "model": model,
        "backend": backend,
        "cache_identity": cache_identity,
        "memory_config_signature": memory_config_signature,
        "ablation_mode": ablation_mode,
        "retrieve_k": retrieve_k,
        "answer_prompt_mode": "answer_check",
        "judge_runs": judge_runs,
        "judge_enabled": True,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "judge_prompt_source": "answer_check",
        "judge_base_url": judge_base_url,
    }
    source_index: Dict[str, Dict[str, Any]] = {}
    source_meta: Optional[Dict[str, Any]] = None
    reuse_source_label: Optional[str] = None
    if reuse_results:
        reuse_source_label = reuse_results
        if not os.path.exists(reuse_results):
            raise ValueError(f"reuse_results file not found: {reuse_results}")
        with open(reuse_results, "r", encoding="utf-8") as _reuse_fh:
            source_payload = json.load(_reuse_fh)
        if source_payload.get("benchmark") != "longmemeval":
            raise ValueError(
                "reuse_results is not a LongMemEval result file "
                f"(benchmark={source_payload.get('benchmark')!r})"
            )
        for _rec in source_payload.get("individual_results", []) or []:
            _qid = _rec.get("question_id")
            if _qid is None:
                continue
            if _qid in source_index:
                eval_logger.warning("Duplicate question_id %s in reuse source; using last", _qid)
            source_index[_qid] = _rec
        source_meta = {
            "model": source_payload.get("model"),
            "backend": source_payload.get("backend"),
            "cache_identity": source_payload.get("cache_identity"),
            "memory_config_signature": source_payload.get("memory_config_signature"),
            "ablation_mode": source_payload.get("ablation_mode"),
            "retrieve_k": source_payload.get("retrieve_k"),
            "judge_enabled": source_payload.get("judge_enabled"),
            "judge_model": source_payload.get("judge_model"),
            "judge_backend": source_payload.get("judge_backend"),
            "judge_prompt_source": source_payload.get("judge_prompt_source"),
            "answer_prompt_mode": source_payload.get("answer_prompt_mode"),
            "judge_runs": source_payload.get("judge_runs"),
            "judge_base_url": source_payload.get("judge_base_url"),
        }
        if source_meta.get("retrieve_k") is None and reuse_retrieve_k is not None:
            source_meta["retrieve_k"] = int(reuse_retrieve_k)
            eval_logger.info(
                "reuse source missing retrieve_k; assuming %d via --reuse-retrieve-k",
                int(reuse_retrieve_k),
            )
        eval_logger.info("Loaded reuse source: %d entries from %s", len(source_index), reuse_results)

    precomputed_query_results: List[Dict[str, Any]] = []
    if query_plan_input:
        with open(query_plan_input, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        precomputed_query_results = list(payload.get("individual_results", []) or [])
        if not precomputed_query_results:
            raise ValueError("Query Plan input contains no individual_results")
        eval_logger.info("Loaded %d precomputed Query Plans", len(precomputed_query_results))
    qp_by_qid: Dict[str, Dict[str, Any]] = {
        r["question_id"]: r for r in precomputed_query_results if r.get("question_id")
    }

    if ratio < 1.0:
        num = max(1, int(len(entries) * ratio))
        entries = entries[:num]
        eval_logger.info("Using %d entries (%.1f%%)", num, ratio * 100)

    results: List[Dict[str, Any]] = []
    all_metrics: List[Dict[str, float]] = []
    judge_labels: List[bool] = []
    total_questions = 0
    type_counts = defaultdict(int)
    token_usage_totals = {
        "memory_build": _empty_token_stats(),
        "qa": _empty_token_stats(),
        "answer_generation": _empty_token_stats(),
    }
    time_cost = {"memory_build_seconds": 0.0, "retrieval_seconds": 0.0, "answer_seconds": 0.0}

    # A re-entrant factory keeps every worker's dependency injection local.
    def _build_reused_result(entry_idx, entry, source_record, judge_ok, guard_reason):
        """Reuse a prior result row. answer_question() is never called.

        When ``judge_ok`` is False but judging is enabled, only the judge is
        re-invoked on the reused prediction (L0b); otherwise the judge block is
        reused verbatim (L0a) or left unset when judging is disabled.
        """
        prediction = source_record.get("prediction", "")
        query_plan = copy.deepcopy(source_record.get("query_plan") or {})
        selected_evidence = copy.deepcopy(source_record.get("selected_evidence") or [])
        raw_retrieval_context = source_record.get("raw_retrieval_context", "") or ""
        answer_context = source_record.get("answer_context") or raw_retrieval_context
        src_timing = source_record.get("timing") or {}
        src_tokens = copy.deepcopy(source_record.get("token_usage") or {})
        mb_time = float(src_timing.get("memory_build_seconds", 0.0) or 0.0)
        retrieval_time = float(src_timing.get("retrieval_seconds", 0.0) or 0.0)
        answer_time = float(src_timing.get("answer_seconds", 0.0) or 0.0)
        metrics = {k: v for k, v in (source_record.get("metrics") or {}).items()
                   if k != "llm_score"}
        reuse_info = {"reused": True, "judge_reused": None,
                      "source": reuse_source_label, "reason": guard_reason}
        judge_used = False
        if judge_ok:
            judge_result = copy.deepcopy(source_record.get("judge") or {})
            _label = judge_result.get("label")
            _label_b = bool(_label) if _label is not None else None
            metrics["llm_score"] = (1 if _label_b else 0) if _label_b is not None else None
            reuse_info["judge_reused"] = True
        else:
            judge_result = evaluate_llm_judge(
                question=entry.question,
                gold_answer=str(entry.answer),
                generated_answer=prediction,
                model=judge_model,
                backend=judge_backend,
                base_url=judge_base_url,
                host=judge_sglang_host,
                port=judge_sglang_port,
                source_name="answer_check",
                judge_runs=judge_runs,
            )
            _label_b = bool(judge_result["label"])
            metrics["llm_score"] = 1 if _label_b else 0
            reuse_info["judge_reused"] = False
            reuse_info["reason"] = "rejudge:" + guard_reason
            judge_used = True
        judge_label_log = judge_result.get("label")
        result = {
            "question_id": entry.question_id,
            "question_type": entry.question_type,
            "abstention": entry.is_abstention,
            "question": entry.question,
            "question_date": entry.question_date,
            "prediction": prediction,
            "reference": entry.answer,
            "query_plan": query_plan,
            "selected_evidence": selected_evidence,
            "raw_retrieval_context": raw_retrieval_context,
            "answer_context": answer_context,
            "answer_prompt_mode": "answer_check",
            "metrics": metrics,
            "llm_score": metrics.get("llm_score"),
            "judge": {
                "model": judge_result.get("model") or judge_model,
                "backend": judge_result.get("backend") or judge_backend,
                "prompt_source": judge_result.get("prompt_source") or "answer_check",
                "effective_config": judge_result.get("effective_config", {}),
                "prompt": str(judge_result.get("prompt", "")),
                "system_prompt": str(judge_result.get("system_prompt", "")),
                "raw_response": str(judge_result.get("raw_response", "")),
                "all_raw_responses": judge_result.get("all_raw_responses"),
                "judgments": judge_result.get("judgments"),
                "judge_runs": judge_result.get("judge_runs"),
                "label_scheme": judge_result.get("label_scheme"),
                "label": judge_result.get("label"),
            },
            "memory_loaded_from_cache": source_record.get("memory_loaded_from_cache", False),
            "result_reuse": reuse_info,
            "_timing": {
                "memory_build_seconds": mb_time,
                "retrieval_seconds": retrieval_time,
                "answer_seconds": answer_time,
            },
            "_token_usage": src_tokens,
        }
        _tag = "REUSED+JUDGE" if judge_used else "REUSED"
        eval_logger.info("Q%d %s [%s] -> pred=%r ref=%r judge=%s",
                         entry_idx + 1, entry.question_id, _tag, prediction, entry.answer, judge_label_log)
        return result

    def _process_entry(entry_idx, entry):
        """Build memory + answer + judge for one entry. Thread-safe (no shared state)."""
        eval_logger.info("Entry %d/%d  qid=%s type=%s abstention=%s",
                         entry_idx + 1, len(entries), entry.question_id, entry.question_type,
                         entry.is_abstention)

        # ---- result reuse across ratio (2026-08-11) ----
        reuse_info = {"reused": False, "judge_reused": None,
                      "source": reuse_source_label, "reason": "new_entry"}
        run_meta_entry = _build_run_meta_entry(entry, run_top_meta)
        source_record = source_index.get(entry.question_id) if source_index else None
        if source_record is not None:
            answer_ok, judge_ok, guard_reason = _check_reuse_eligibility(
                source_record, source_meta, run_meta_entry)
            if answer_ok:
                return _build_reused_result(
                    entry_idx, entry, source_record, judge_ok, guard_reason)
            reuse_info["reason"] = guard_reason

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
        )
        agent.memory_system.llm_controller.set_usage_phase("memory_build")
        agent.retriever_llm.set_usage_phase("memory_build")

        cache_file = os.path.join(memories_dir, f"memory_cache_entry_{entry.question_id}.pkl")
        rebuild = True
        loaded_from_cache = False
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    cached_state = pickle.load(f)
                agent.memory_system.load_state(cached_state)
                rebuild = False
                loaded_from_cache = True
                eval_logger.info("Loaded cache for %s", entry.question_id)
            except Exception as exc:
                eval_logger.warning("Cache load failed for %s: %s — rebuilding", entry.question_id, exc)

        mb_start = time.perf_counter()
        if rebuild:
            eval_logger.info("Building memory for %s (%d sessions)", entry.question_id, len(entry.sessions))
            _build_entry_memory(
                agent, entry,
                enable_hierarchy=ablation_flags["enable_hierarchy_memory"],
            )
            _atomic_pickle_dump(agent.memory_system.export_state(), cache_file)
        mb_time = time.perf_counter() - mb_start

        agent.memory_system.llm_controller.set_usage_phase("qa")
        agent.retriever_llm.set_usage_phase("qa")

        precomputed_plan = None
        if precomputed_query_results:
            src = qp_by_qid.get(entry.question_id)
            if src is None and entry_idx < len(precomputed_query_results):
                src = precomputed_query_results[entry_idx]
            if src is not None and str(src.get("question", "")) != str(entry.question):
                raise ValueError(
                    "Query Plan alignment mismatch for question_id {}".format(entry.question_id))
            if src is not None:
                precomputed_plan = src.get("query_plan")

        prediction, user_prompt, ranked_context, query_plan, retrieval_time, answer_time = (
            agent.answer_question(
                entry.question,
                category=0,
                query_plan=precomputed_plan,
                question_date=entry.question_date,
                question_type=entry.question_type,
                answer_prompt_mode="answer_check",
            )
        )
        from eval.longmemeval_prompts import extract_cot_final_answer
        prediction = extract_cot_final_answer(prediction)

        metrics = calculate_metrics(prediction, entry.answer) if entry.answer else {}
        metrics.setdefault("llm_score", None)

        judge_result = evaluate_llm_judge(
            question=entry.question,
            gold_answer=str(entry.answer),
            generated_answer=prediction,
            model=judge_model,
            backend=judge_backend,
            base_url=judge_base_url,
            host=judge_sglang_host,
            port=judge_sglang_port,
            source_name="answer_check",
            judge_runs=judge_runs,
        )
        judge_label = bool(judge_result["label"])
        metrics["llm_score"] = 1 if judge_label else 0

        result = {
            "question_id": entry.question_id,
            "question_type": entry.question_type,
            "abstention": entry.is_abstention,
            "question": entry.question,
            "question_date": entry.question_date,
            "prediction": prediction,
            "reference": entry.answer,
            "query_plan": query_plan,
            "selected_evidence": query_plan.get("selected_evidence", []),
            "raw_retrieval_context": query_plan.get("raw_retrieval_context", ranked_context),
            "answer_context": ranked_context,
            "metrics": metrics,
            "llm_score": (1 if judge_label else 0) if judge_label is not None else None,
            "judge": {
                "model": judge_model,
                "backend": judge_backend,
                "prompt_source": "answer_check",
                "effective_config": judge_result.get("effective_config", {}),
                "prompt": str(judge_result.get("prompt", "")),
                "system_prompt": str(judge_result.get("system_prompt", "")),
                "raw_response": str(judge_result.get("raw_response", "")),
                "all_raw_responses": judge_result.get("all_raw_responses"),
                "judgments": judge_result.get("judgments"),
                "judge_runs": judge_result.get("judge_runs"),
                "label_scheme": judge_result.get("label_scheme"),
                "label": judge_label,
            },
            "memory_loaded_from_cache": loaded_from_cache,
            "answer_prompt_mode": "answer_check",
            "result_reuse": reuse_info,
            "_timing": {
                "memory_build_seconds": mb_time,
                "retrieval_seconds": retrieval_time,
                "answer_seconds": answer_time,
            },
            "_token_usage": {
                "llm": agent.memory_system.llm_controller.get_token_usage_summary(),
                "retriever": agent.retriever_llm.get_token_usage_summary(),
            },
        }
        eval_logger.info("Q%d %s -> pred=%r ref=%r judge=%s",
                         entry_idx + 1, entry.question_id, prediction, entry.answer, judge_label)
        return result

    # ---- concurrent execution ----
    # Results keyed by entry index. Failed entries are recorded (not retried
    # by default) and surfaced via runtime_status/failed_entries/failed_summary.
    idx_to_result: Dict[int, Any] = {}
    failed_entries: List[Dict[str, Any]] = []
    state_lock = threading.Lock()

    def _record_failure(idx: int, exc: BaseException, phase: str = "unknown") -> None:
        entry = entries[idx]
        err_msg = f"{type(exc).__name__}: {exc}"
        _err_short = err_msg[:300]
        with state_lock:
            failed_entries.append({
                "entry_index": idx,
                "question_id": entry.question_id,
                "question_type": entry.question_type,
                "abstention": entry.is_abstention,
                "phase": phase,
                "error": _err_short,
                "failed_at": datetime.now().isoformat(timespec="seconds"),
            })

    def _emit_incremental(finalized: bool = False) -> None:
        """Write an atomic snapshot of current progress to output_path."""
        if not output_path:
            return
        with state_lock:
            snap_results = [idx_to_result[i] for i in range(len(entries)) if idx_to_result.get(i) is not None]
            snap_failed = list(failed_entries)
        completed = len(snap_results)
        failed = len(snap_failed)
        total = len(entries)
        runtime_status = {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": max(0, total - completed - failed),
            "finalized": finalized,
            "last_updated": datetime.now().isoformat(timespec="seconds"),
        }
        payload = {
            "benchmark": "longmemeval",
            "dataset": dataset_path,
            "model": model,
            "backend": backend,
            "ablation_mode": ablation_mode,
            "runtime_status": runtime_status,
            "failed_entries": snap_failed,
            "individual_results": snap_results,
        }
        if finalized:
            payload["failed_summary"] = _summarize_failures(snap_failed)
        try:
            _atomic_write_json(output_path, payload)
        except Exception as exc:
            eval_logger.warning("Incremental write failed: %s", exc)

    if num_workers and num_workers > 1:
        eval_logger.info("Running with %d parallel workers", num_workers)
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            future_to_idx = {ex.submit(_process_entry, i, e): i for i, e in enumerate(entries)}
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    idx_to_result[idx] = fut.result()
                except Exception as exc:
                    eval_logger.error("Entry %d (%s) failed: %s", idx, entries[idx].question_id, exc)
                    idx_to_result[idx] = None
                    _record_failure(idx, exc)
                _emit_incremental(finalized=False)
    else:
        for entry_idx, entry in enumerate(entries):
            try:
                idx_to_result[entry_idx] = _process_entry(entry_idx, entry)
            except Exception as exc:
                eval_logger.error("Entry %d (%s) failed: %s", entry_idx, entries[entry_idx].question_id, exc)
                idx_to_result[entry_idx] = None
                _record_failure(entry_idx, exc)
            _emit_incremental(finalized=False)

    # ---- retry failed entries serially (default 0 rounds; opt-in via env) ----
    max_retry_rounds = int(os.getenv("LONGMEMEVAL_MAX_RETRY_ROUNDS", "0"))
    for round_no in range(1, max_retry_rounds + 1):
        failed_idx = [i for i in range(len(entries)) if idx_to_result.get(i) is None]
        if not failed_idx:
            break
        eval_logger.warning(
            "Retry round %d/%d: %d failed entries (serial, rate-limit-friendly)",
            round_no, max_retry_rounds, len(failed_idx),
        )
        for i in failed_idx:
            entry = entries[i]
            try:
                eval_logger.info("Retrying entry %d (%s)", i + 1, entry.question_id)
                idx_to_result[i] = _process_entry(i, entry)
                if idx_to_result[i] is not None:
                    # retry succeeded: drop prior failure record for this index
                    with state_lock:
                        failed_entries = [f for f in failed_entries if f["entry_index"] != i]
            except Exception as exc:
                eval_logger.error("Retry entry %d (%s) failed: %s", i + 1, entry.question_id, exc)
                idx_to_result[i] = None
                _record_failure(i, exc, phase="retry")
        _emit_incremental(finalized=False)

    # still-failed entries are logged and surfaced (do not raise the whole run)
    final_failed = [entries[i].question_id for i in range(len(entries)) if idx_to_result.get(i) is None]
    if final_failed:
        eval_logger.error("Unrecoverable entries (%d): %s", len(final_failed), final_failed)

    results: List[Dict[str, Any]] = [idx_to_result[i] for i in range(len(entries)) if idx_to_result.get(i) is not None]

    # ---- aggregate after collection ----
    total_questions = len(results)
    _reuse_counts = {"reused": 0, "answer_only": 0, "full_rerun": 0}
    _guard_failures: Dict[str, int] = defaultdict(int)
    for r in results:
        type_counts[r["question_type"]] += 1
        all_metrics.append(r["metrics"])
        judge_labels.append(bool(r["judge"]["label"]))
        tc = r.pop("_timing", {})
        time_cost["memory_build_seconds"] += tc.get("memory_build_seconds", 0.0)
        time_cost["retrieval_seconds"] += tc.get("retrieval_seconds", 0.0)
        time_cost["answer_seconds"] += tc.get("answer_seconds", 0.0)
        tu = r.pop("_token_usage", {})
        _accumulate_phase_usage(token_usage_totals, tu.get("llm", {}))
        _accumulate_phase_usage(token_usage_totals, tu.get("retriever", {}))
        r["timing"] = dict(tc)
        r["token_usage"] = copy.deepcopy(tu)
        _rr = r.get("result_reuse") or {}
        if _rr.get("reused"):
            _reuse_counts["reused"] += 1
            if _rr.get("judge_reused") is False:
                _reuse_counts["answer_only"] += 1
        else:
            _reuse_counts["full_rerun"] += 1
            _reason = _rr.get("reason", "new_entry")
            if _reason != "new_entry":
                _guard_failures[_reason] += 1

    # align entries with results (drop entries whose worker failed)
    aligned_entries = [next(e for e in entries if e.question_id == r["question_id"]) for r in results]

    lme_metrics = _aggregate_longmemeval_metrics(aligned_entries, judge_labels if judge_labels else [False] * len(results))

    # Judge-aggregated view (only when judge enabled, else None placeholders)
    aggregate_judge_metrics = {
        "enabled": True,
        "overall_accuracy": lme_metrics["overall_accuracy"],
        "task_averaged_accuracy": lme_metrics["task_averaged_accuracy"],
        "abstention_accuracy": lme_metrics["abstention_accuracy"],
        "per_question_type": lme_metrics["per_question_type"],
        "count": len(judge_labels),
    }

    question_answering_usage = _combined_token_stats(token_usage_totals["qa"], token_usage_totals["answer_generation"])
    final_results = {
        "benchmark": "longmemeval",
        "dataset": dataset_path,
        "dataset_type": "longmemeval",
        "model": model,
        "backend": backend,
        "ablation_mode": ablation_mode,
        "ablation_flags": ablation_flags,
        "retrieve_k": retrieve_k,
        "memory_layer": "robust_hierarchical",
        "memory_config": memory_config,
        "memory_config_signature": memory_config_signature,
        "memory_cache_signature": memory_cache_signature,
        "cache_family": cache_family,
        "dataset_fingerprint": dataset_fingerprint,
        "cache_identity": cache_identity,
        "cache_version": CACHE_VERSION,
        "judge_model": judge_model,
        "judge_backend": judge_backend,
        "judge_base_url": judge_base_url,
        "judge_prompt_source": "answer_check",
        "judge_runs": judge_runs,
        "answer_prompt_mode": "answer_check",
        "answer_max_tokens": answer_max_tokens,
        "judge_enabled": True,
        "total_questions": total_questions,
        "question_type_distribution": {k: v for k, v in sorted(type_counts.items())},
        "longmemeval_metrics": lme_metrics,
        "aggregate_judge_metrics": aggregate_judge_metrics,
        "routing_summary": _summarize_routing_results(results),
        "answer_length_summary": _summarize_answer_lengths(results),
        "memory_build_diagnostics": {
            "topic_build_mode": memory_config_obj.topic_build_mode,
            "enable_anchors": memory_config_obj.enable_anchors,
        },
        "token_usage": {
            "memory_build": token_usage_totals["memory_build"],
            "qa": token_usage_totals["qa"],
            "answer_generation": token_usage_totals["answer_generation"],
            "question_answering": question_answering_usage,
        },
        "time_cost": {
            **time_cost,
            "memory_build_avg_seconds_per_entry": time_cost["memory_build_seconds"] / len(entries) if entries else 0.0,
            "retrieval_avg_seconds_per_question": time_cost["retrieval_seconds"] / total_questions if total_questions else 0.0,
            "answer_avg_seconds_per_question": time_cost["answer_seconds"] / total_questions if total_questions else 0.0,
        },
        "reuse_meta": {
            "enabled": reuse_results is not None,
            "source": reuse_results,
            "source_model": (source_meta or {}).get("model"),
            "source_cache_identity": (source_meta or {}).get("cache_identity"),
            "reused_count": _reuse_counts["reused"],
            "answer_only_rerun_count": _reuse_counts["answer_only"],
            "full_rerun_count": _reuse_counts["full_rerun"],
            "guard_fail_breakdown": dict(_guard_failures),
        },
        "individual_results": results,
    }

    # Attach runtime status & failure accounting, write atomically.
    final_results["runtime_status"] = {
        "total": len(entries),
        "completed": len(results),
        "failed": len(failed_entries),
        "pending": 0,
        "finalized": True,
        "last_updated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
    }
    final_results["failed_entries"] = failed_entries
    final_results["failed_summary"] = _summarize_failures(failed_entries)
    if output_path:
        _atomic_write_json(output_path, final_results)
        eval_logger.info("Results saved to %s", output_path)

    # Optional extra file: per-question-type averages (accuracy + text metrics)
    if per_type_summary:
        try:
            meta = {
                "model": model,
                "backend": backend,
                "dataset": dataset_path,
                "ablation_mode": ablation_mode,
                "total_questions": total_questions,
                "judge_model": judge_model,
                "judge_prompt_source": "answer_check",
            }
            _write_per_type_summary(
                per_type_summary, aligned_entries, all_metrics, judge_labels,
                judge_enabled=True, meta=meta,
            )
            eval_logger.info("Per-type summary saved to %s", per_type_summary)
        except Exception as exc:
            eval_logger.warning("Failed to write per-type summary: %s", exc)
    return final_results
