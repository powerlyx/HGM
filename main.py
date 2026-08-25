#!/usr/bin/env python3
"""Entry point for HGM evaluation on LoCoMo and LongMemEval.

Usage:
    python main.py --llm-backend openai --llm-model gpt-4.1-mini-2025-04-14 --dataset data/locomo10.json
    python main.py --llm-backend ollama --llm-model qwen2.5:3b --dataset data/locomo10.json
    python main.py --llm-backend openai --llm-model gpt-4.1-mini-2025-04-14 \
        --judge-backend openai --judge-model gpt-4.1-mini-2025-04-14 \
        --dataset data/locomo10.json --output outputs/locomo_results.json
"""

import argparse
import os
import sys

# Ensure the project root is on sys.path so module imports resolve
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from config import Config


def main():
    parser = argparse.ArgumentParser(
        description="HGM — LoCoMo / LongMemEval evaluation with hierarchical graph-structured memory"
    )
    # Dataset & output
    parser.add_argument("--dataset", default="data/locomo10.json")
    parser.add_argument("--output", default=None, help="Path to save evaluation results")
    parser.add_argument("--ratio", type=float, default=1.0,
                        help="Fraction of dataset to evaluate (0-1)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed used for reproducible evaluation")
    parser.add_argument("--query-plan-input", default=None,
                        help="Prior runner JSON whose Query Plans must be reused")
    parser.add_argument("--dataset-type", default="auto",
                        choices=["auto", "locomo", "longmemeval"],
                        help="Force dataset type; auto inspects the file structure")
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrency. LongMemEval: parallel entries; LoCoMo: parallel QAs within each sample (answer+judge). 1=serial (default).")
    parser.add_argument("--per-type-summary", default=None,
                        help="LongMemEval: extra JSON with per-question-type averages")
    parser.add_argument("--reuse-results", default=None,
                        help="LongMemEval: prior result JSON to reuse per-question_id across ratio")
    parser.add_argument("--reuse-retrieve-k", type=int, default=None,
                        help="LongMemEval: assume the reuse-results source used this retrieve_k when its field is missing")
    parser.add_argument("--answer-max-tokens", type=int, default=32768,
                        help="Max output tokens for the answer LLM call (the LongMemEval CoT answer prompt needs room; default 32768)")
    parser.add_argument("--judge-runs", type=int, default=3,
                        help="Number of independent judge runs with majority vote (default 3)")

    # Models & backends
    parser.add_argument("--llm-backend", default="openai",
                        help="LLM backend (openai, ollama, sglang, vllm)")
    parser.add_argument("--llm-model", default="gpt-4.1-mini-2025-04-14",
                        help="LLM model name")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument("--judge-backend", default="openai")

    # API
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--judge-base-url", default=None)

    # Retrieval
    parser.add_argument("--retrieve-k", type=int, default=10,
                        help="Number of memories to retrieve")
    parser.add_argument("--topic-result-limit", type=int, default=None,
                        help="Final topic-layer budget; 0 disables the topic layer")
    parser.add_argument("--episode-result-limit", type=int, default=None,
                        help="Final episode-layer budget; 0 disables the episode layer")
    parser.add_argument("--turn-result-limit", type=int, default=None,
                        help="Final turn-layer budget; 0 disables the turn layer")
    parser.add_argument("--retrieval-semantic-weight", type=float, default=0.68)
    parser.add_argument("--retrieval-lexical-weight", type=float, default=0.24)
    parser.add_argument("--retrieval-recency-weight", type=float, default=0.08)
    parser.add_argument("--retrieval-weight-mode", choices=["static", "adaptive"], default="adaptive")
    parser.add_argument("--retrieval-adaptive-strength", type=float, default=0.35)
    parser.add_argument("--retrieval-temporal-bonus", type=float, default=0.10)
    parser.add_argument("--retrieval-budget-low-multiplier", type=float, default=0.80)
    parser.add_argument("--retrieval-budget-mid-multiplier", type=float, default=1.00)
    parser.add_argument("--retrieval-budget-high-multiplier", type=float, default=1.55)
    parser.add_argument("--retrieval-provenance-boost", type=float, default=0.55)
    parser.add_argument("--retrieval-min-turn-support", type=int, default=2)
    parser.add_argument("--retrieval-uncertainty-temperature", type=float, default=0.08)
    parser.add_argument("--retrieval-uncertainty-margin-scale", type=float, default=0.08)
    parser.add_argument("--retrieval-uncertainty-low-threshold", type=float, default=0.40)
    parser.add_argument("--retrieval-uncertainty-high-threshold", type=float, default=0.70)
    parser.add_argument(
        "--retrieval-final-max-items",
        type=int,
        default=None,
        help="Optional global final-evidence cap; omitted means unbounded",
    )


    # Memory hierarchy
    parser.add_argument("--topic-build-mode", choices=["after_sample", "incremental"],
                        default="after_sample",
                        help="Build Topics after the sample or incrementally during ingestion")
    parser.add_argument("--enable-anchors", action="store_true", default=False,
                        help="Enable structured anchors during memory construction")
    parser.add_argument("--ablation-mode", default="main",
                        choices=["main", "no_hierarchy", "no_uncertainty", "no_evidence_loop",
                                 "no_evidence_loop,no_hierarchy", "no_evidence_loop,no_uncertainty",
                                 "no_hierarchy,no_uncertainty",
                                 "no_evidence_loop,no_hierarchy,no_uncertainty",
                                 "main,answer_driven", "no_evidence_loop,answer_driven"],
                        help="Ablation setting (add 'answer_driven' to enable the answer-driven verification link)")
    parser.add_argument("--answer-verification-gate", default="low_evidence",
                        choices=["low_evidence", "all"],
                        help="Answer-driven verification probe scope: low_evidence (only probe questions with sufficiency_score<0.66, cost-bounded, default) | all (probe every question; ~+73%% tokens). Active only with ablation token 'answer_driven'.")

    # Server hosts
    parser.add_argument("--sglang-host", default="http://localhost")
    parser.add_argument("--sglang-port", type=int, default=30000)
    parser.add_argument("--judge-sglang-host", default="http://localhost")
    parser.add_argument("--judge-sglang-port", type=int, default=30000)

    args = parser.parse_args()

    # Delay the heavy evaluation import until after argument parsing so
    # `main.py --help` works even in lightweight environments.
    from eval.runner import evaluate_dataset
    from config import detect_dataset_type as _detect_dataset_type

    dataset_path = os.path.join(_project_root, args.dataset)
    output_path = os.path.join(_project_root, args.output) if args.output else None

    cfg = Config(
        embedding_model=args.embedding_model,
        llm_backend=args.llm_backend,
        llm_model=args.llm_model,
        llm_base_url=args.llm_base_url,
        judge_model=args.judge_model,
        judge_backend=args.judge_backend,
        judge_base_url=args.judge_base_url,
        retrieve_k=args.retrieve_k,
        topic_result_limit=args.topic_result_limit,
        episode_result_limit=args.episode_result_limit,
        turn_result_limit=args.turn_result_limit,
        retrieval_semantic_weight=args.retrieval_semantic_weight,
        retrieval_lexical_weight=args.retrieval_lexical_weight,
        retrieval_recency_weight=args.retrieval_recency_weight,
        retrieval_weight_mode=args.retrieval_weight_mode,
        retrieval_adaptive_strength=args.retrieval_adaptive_strength,
        retrieval_temporal_bonus=args.retrieval_temporal_bonus,
        retrieval_budget_low_multiplier=args.retrieval_budget_low_multiplier,
        retrieval_budget_mid_multiplier=args.retrieval_budget_mid_multiplier,
        retrieval_budget_high_multiplier=args.retrieval_budget_high_multiplier,
        retrieval_provenance_boost=args.retrieval_provenance_boost,
        retrieval_min_turn_support=args.retrieval_min_turn_support,
        retrieval_uncertainty_temperature=args.retrieval_uncertainty_temperature,
        retrieval_uncertainty_margin_scale=args.retrieval_uncertainty_margin_scale,
        retrieval_uncertainty_low_threshold=args.retrieval_uncertainty_low_threshold,
        retrieval_uncertainty_high_threshold=args.retrieval_uncertainty_high_threshold,
        retrieval_final_max_items=args.retrieval_final_max_items,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        judge_sglang_host=args.judge_sglang_host,
        judge_sglang_port=args.judge_sglang_port,
        ablation_mode=args.ablation_mode,
        topic_build_mode=args.topic_build_mode,
        enable_anchors=args.enable_anchors,
        ratio=args.ratio,
        seed=args.seed,
        query_plan_input=args.query_plan_input,
        dataset_path=args.dataset,
    )

    # ---- dataset routing -------------------------------------------------
    actual_dataset_type = _detect_dataset_type(dataset_path, args.dataset_type)

    if actual_dataset_type == "longmemeval":
        from eval.longmemeval_runner import evaluate_longmemeval_dataset
        # Forward the memory-construction + retrieval parameters that the LongMemEval
        # runner understands (it accepts **memory_build_kwargs -> _build_memory_config).
        evaluate_longmemeval_dataset(
            dataset_path=dataset_path,
            model=cfg.llm_model,
            output_path=output_path,
            ratio=cfg.ratio,
            backend=cfg.llm_backend,
            retrieve_k=cfg.retrieve_k,
            sglang_host=cfg.sglang_host,
            sglang_port=cfg.sglang_port,
            judge_model=cfg.judge_model,
            judge_backend=cfg.judge_backend,
            judge_base_url=cfg.judge_base_url,
            answer_max_tokens=args.answer_max_tokens,
            judge_runs=args.judge_runs,
            judge_sglang_host=cfg.judge_sglang_host,
            judge_sglang_port=cfg.judge_sglang_port,
            num_workers=args.num_workers,
            per_type_summary=(
                os.path.join(_project_root, args.per_type_summary)
                if args.per_type_summary and not os.path.isabs(args.per_type_summary)
                else args.per_type_summary
            ),
            reuse_results=(
                os.path.join(_project_root, args.reuse_results)
                if args.reuse_results and not os.path.isabs(args.reuse_results)
                else args.reuse_results
            ),
            reuse_retrieve_k=args.reuse_retrieve_k,
            ablation_mode=cfg.ablation_mode,


            seed=cfg.seed,
            query_plan_input=(
                os.path.join(_project_root, cfg.query_plan_input)
                if cfg.query_plan_input and not os.path.isabs(cfg.query_plan_input)
                else cfg.query_plan_input
            ),
            topic_result_limit=cfg.topic_result_limit,
            episode_result_limit=cfg.episode_result_limit,
            turn_result_limit=cfg.turn_result_limit,
            retrieval_semantic_weight=cfg.retrieval_semantic_weight,
            retrieval_lexical_weight=cfg.retrieval_lexical_weight,
            retrieval_recency_weight=cfg.retrieval_recency_weight,
            retrieval_weight_mode=cfg.retrieval_weight_mode,
            retrieval_adaptive_strength=cfg.retrieval_adaptive_strength,
            retrieval_temporal_bonus=cfg.retrieval_temporal_bonus,
            retrieval_budget_low_multiplier=cfg.retrieval_budget_low_multiplier,
            retrieval_budget_mid_multiplier=cfg.retrieval_budget_mid_multiplier,
            retrieval_budget_high_multiplier=cfg.retrieval_budget_high_multiplier,
            retrieval_provenance_boost=cfg.retrieval_provenance_boost,
            retrieval_min_turn_support=cfg.retrieval_min_turn_support,
            retrieval_uncertainty_temperature=cfg.retrieval_uncertainty_temperature,
            retrieval_uncertainty_margin_scale=cfg.retrieval_uncertainty_margin_scale,
            retrieval_uncertainty_low_threshold=cfg.retrieval_uncertainty_low_threshold,
            retrieval_uncertainty_high_threshold=cfg.retrieval_uncertainty_high_threshold,
            retrieval_final_max_items=cfg.retrieval_final_max_items,
            topic_build_mode=cfg.topic_build_mode,
            enable_anchors=cfg.enable_anchors,
        )
    else:
        if args.reuse_results:
            import sys as _sys
            print("warning: --reuse-results is only supported for LongMemEval; ignored for LoCoMo",
                  file=_sys.stderr)
        evaluate_dataset(
            dataset_path=dataset_path,
            model=cfg.llm_model,
            output_path=output_path,
            ratio=cfg.ratio,
            seed=cfg.seed,
            query_plan_input=(
                os.path.join(_project_root, cfg.query_plan_input)
                if cfg.query_plan_input and not os.path.isabs(cfg.query_plan_input)
                else cfg.query_plan_input
            ),
            backend=cfg.llm_backend,
            retrieve_k=cfg.retrieve_k,
            topic_result_limit=cfg.topic_result_limit,
            episode_result_limit=cfg.episode_result_limit,
            turn_result_limit=cfg.turn_result_limit,
            retrieval_semantic_weight=cfg.retrieval_semantic_weight,
            retrieval_lexical_weight=cfg.retrieval_lexical_weight,
            retrieval_recency_weight=cfg.retrieval_recency_weight,
            retrieval_weight_mode=cfg.retrieval_weight_mode,
            retrieval_adaptive_strength=cfg.retrieval_adaptive_strength,
            retrieval_temporal_bonus=cfg.retrieval_temporal_bonus,
            retrieval_budget_low_multiplier=cfg.retrieval_budget_low_multiplier,
            retrieval_budget_mid_multiplier=cfg.retrieval_budget_mid_multiplier,
            retrieval_budget_high_multiplier=cfg.retrieval_budget_high_multiplier,
            retrieval_provenance_boost=cfg.retrieval_provenance_boost,
            retrieval_min_turn_support=cfg.retrieval_min_turn_support,
            retrieval_uncertainty_temperature=cfg.retrieval_uncertainty_temperature,
            retrieval_uncertainty_margin_scale=cfg.retrieval_uncertainty_margin_scale,
            retrieval_uncertainty_low_threshold=cfg.retrieval_uncertainty_low_threshold,
            retrieval_uncertainty_high_threshold=cfg.retrieval_uncertainty_high_threshold,
            retrieval_final_max_items=cfg.retrieval_final_max_items,
            sglang_host=cfg.sglang_host,
            sglang_port=cfg.sglang_port,
            judge_model=cfg.judge_model,
            judge_backend=cfg.judge_backend,
            judge_base_url=cfg.judge_base_url,
            answer_max_tokens=args.answer_max_tokens,
            judge_runs=args.judge_runs,
            judge_sglang_host=cfg.judge_sglang_host,
            judge_sglang_port=cfg.judge_sglang_port,
            ablation_mode=cfg.ablation_mode,
            topic_build_mode=cfg.topic_build_mode,
            enable_anchors=cfg.enable_anchors,
            num_workers=args.num_workers,
            answer_verification_gate=args.answer_verification_gate,
        )


if __name__ == "__main__":
    main()
