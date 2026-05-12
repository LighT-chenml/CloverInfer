import argparse
import json
import os
import statistics
import sys
import time
from contextlib import suppress
from typing import Callable, Dict, List, Tuple

import ray
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

RAY_PYTHONPATH_OVERRIDE = os.environ.get("CLOVER_BENCHMARK_PYTHONPATH", os.environ.get("CLOVER_RAY_PYTHONPATH"))
RAY_WORKING_DIR = os.environ.get("CLOVER_BENCHMARK_WORKING_DIR", os.environ.get("CLOVER_RAY_WORKING_DIR", "")).strip()
RAY_PY_MODULES = os.environ.get("CLOVER_BENCHMARK_PY_MODULES", os.environ.get("CLOVER_RAY_PY_MODULES", "")).strip()
RAY_EXCLUDES = os.environ.get("CLOVER_BENCHMARK_EXCLUDES", os.environ.get("CLOVER_RAY_EXCLUDES", "")).strip()

from benchmark_utils import DATASET_FORMAT_CHOICES, load_benchmark_samples, load_tokenizer_for_benchmark
from src.core.config import ClusterConfig, ModelConfig
from src.core.model_adapter import CausalModelAdapter
from src.core.nodes import DecodeDenseNode, PrefillNode
from src.core.scheduler import GlobalScheduler


def _resolve_runtime_path(value: str) -> str:
    if os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(REPO_ROOT, value))


def build_runtime_env() -> Dict[str, object]:
    runtime_env: Dict[str, object] = {}

    if RAY_WORKING_DIR:
        runtime_env["working_dir"] = _resolve_runtime_path(RAY_WORKING_DIR)

    if RAY_PY_MODULES:
        runtime_env["py_modules"] = [
            _resolve_runtime_path(item.strip())
            for item in RAY_PY_MODULES.split(os.pathsep)
            if item.strip()
        ]

    if RAY_EXCLUDES:
        runtime_env["excludes"] = [item.strip() for item in RAY_EXCLUDES.split(os.pathsep) if item.strip()]

    env_vars: Dict[str, str] = {}
    pythonpath = RAY_PYTHONPATH_OVERRIDE
    if pythonpath is None and not runtime_env:
        pythonpath = REPO_ROOT
    if pythonpath:
        env_vars["PYTHONPATH"] = pythonpath

    for env_name in (
        "CLOVER_KVSLOT_MAX_CAPACITY",
        "CLOVER_KVSLOT_MAX_HEADS",
        "CLOVER_KVSLOT_AUTOBUILD",
        "CLOVER_KVSLOT_AUTOBUILD_TIMEOUT_S",
    ):
        env_value = os.environ.get(env_name)
        if env_value is not None:
            env_vars[env_name] = env_value
    if env_vars:
        runtime_env["env_vars"] = env_vars

    return runtime_env


def _normalize_baseline_alias(value: str) -> str:
    return "".join(ch.lower() for ch in value if ch.isalnum())


BASELINE_ALIAS_GROUPS = [
    {
        "canonical_key": "monolithic_gpu",
        "canonical_name": "Monolithic-GPU",
        "internal_baseline": "monolithic_gpu",
        "alias_note": "Paper naming keeps this as the monolithic GPU reference.",
        "aliases": [
            "monolithic_gpu",
            "monolithic-gpu",
            "monolithic",
            "mono",
            "gpu",
        ],
    },
    {
        "canonical_key": "pd",
        "canonical_name": "PD",
        "internal_baseline": "split_gpu_full_decode",
        "alias_note": "PD resolves to the prefill/decoding split baseline implemented by split_gpu_full_decode.",
        "aliases": [
            "pd",
            "pd-baseline",
            "prefill-decoding",
            "prefill_decoding",
            "prefill-decoding-separation",
            "split_gpu_full_decode",
        ],
    },
    {
        "canonical_key": "afd",
        "canonical_name": "AFD",
        "internal_baseline": "disagg_afd",
        "alias_note": (
            "AFD resolves to a disaggregated GPU-attention path where attention and dense decode "
            "share the decode GPU while preserving the split workflow."
        ),
        "aliases": [
            "afd",
            "afd-baseline",
            "af-split",
            "af_split",
            "gpu-attention",
            "gpu_attention",
            "disagg_afd",
        ],
    },
    {
        "canonical_key": "cpu_attention",
        "canonical_name": "CPU-Attention",
        "internal_baseline": "disagg_cpu",
        "alias_note": "CPU-Attention resolves to the disagg_cpu implementation.",
        "aliases": [
            "cpu-attention",
            "cpu_attention",
            "cpu",
            "disagg_cpu",
        ],
    },
    {
        "canonical_key": "naive_pim",
        "canonical_name": "Naive PIM",
        "internal_baseline": "disagg_pim_naive",
        "alias_note": "Naive PIM resolves to the current disagg_pim_naive implementation.",
        "aliases": [
            "naive-pim",
            "naive_pim",
            "pim-naive",
            "pim_naive",
            "disagg_pim_naive",
        ],
    },
    {
        "canonical_key": "cloverinfer",
        "canonical_name": "CloverInfer",
        "internal_baseline": "disagg_cloverinfer",
        "alias_note": "CloverInfer resolves to the current disagg_cloverinfer implementation.",
        "aliases": [
            "cloverinfer",
            "clover-infer",
            "disagg_cloverinfer",
        ],
    },
]


BASELINE_ALIAS_INDEX = {}
for group in BASELINE_ALIAS_GROUPS:
    for alias in group["aliases"]:
        BASELINE_ALIAS_INDEX[_normalize_baseline_alias(alias)] = group


BASELINE_HELP_TEXT = (
    "Comma-separated baseline list. Accepts legacy names "
    "(monolithic_gpu, split_gpu_full_decode, disagg_afd, disagg_cpu, disagg_pim_naive, disagg_cloverinfer) "
    "and paper aliases (PD, AFD, CPU-Attention, Naive PIM, CloverInfer). "
    "AFD uses a standalone GPU-attention path; CPU-Attention remains the CPU attention baseline."
)


def resolve_requested_baselines(value: str) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    ordered: List[Dict[str, object]] = []
    raw_items = [item.strip() for item in value.split(",") if item.strip()]
    if not raw_items:
        raise ValueError("at least one baseline must be specified")

    for raw_item in raw_items:
        normalized = _normalize_baseline_alias(raw_item)
        group = BASELINE_ALIAS_INDEX.get(normalized)
        if group is None:
            supported = sorted({alias for entry in BASELINE_ALIAS_GROUPS for alias in entry["aliases"]})
            raise ValueError(
                f"unsupported baseline alias: {raw_item}. "
                f"Supported aliases include: {', '.join(supported)}"
            )

        key = str(group["canonical_key"])
        if key not in grouped:
            spec = {
                "canonical_key": key,
                "canonical_name": str(group["canonical_name"]),
                "internal_baseline": str(group["internal_baseline"]),
                "alias_note": str(group["alias_note"]),
                "requested_aliases": [raw_item],
                "resolved_aliases": list(group["aliases"]),
            }
            grouped[key] = spec
            ordered.append(spec)
        else:
            if raw_item not in grouped[key]["requested_aliases"]:
                grouped[key]["requested_aliases"].append(raw_item)
    return ordered


def summarize_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    if not metric_list:
        return {
            "avg_latency": 0.0,
            "avg_ttft": 0.0,
            "avg_tpot": 0.0,
            "avg_throughput": 0.0,
            "avg_total_tokens": 0.0,
        }
    return {
        "avg_latency": float(statistics.mean(item["latency"] for item in metric_list)),
        "avg_ttft": float(statistics.mean(item["ttft"] for item in metric_list)),
        "avg_tpot": float(statistics.mean(item["tpot"] for item in metric_list)),
        "avg_throughput": float(statistics.mean(item["throughput"] for item in metric_list)),
        "avg_total_tokens": float(statistics.mean(item["total_tokens"] for item in metric_list)),
    }


def summarize_run(
    records: List[Dict[str, object]],
    wall_time_s: float,
    requested_concurrency: int,
    effective_concurrency: int,
    execution_mode: str,
) -> Dict[str, object]:
    summary = summarize_metrics([record["metrics"] for record in records])
    total_generated_tokens = sum(int(record["metrics"]["total_tokens"]) for record in records)
    summary.update(
        {
            "num_requests": int(len(records)),
            "wall_time_s": float(wall_time_s),
            "request_throughput_rps": float(len(records) / wall_time_s) if wall_time_s > 0 else 0.0,
            "output_token_throughput_tps": float(total_generated_tokens / wall_time_s) if wall_time_s > 0 else 0.0,
            "requested_concurrency": int(requested_concurrency),
            "effective_concurrency": int(effective_concurrency),
            "execution_mode": execution_mode,
            "out_of_order_completions": int(
                sum(
                    1
                    for record in records
                    if int(record["request_index"]) != int(record["completion_index"])
                )
            ),
        }
    )
    return summary


def resolve_pim_dpu_placement_policy(attention_backend: str, requested_policy: str) -> str:
    policy = str(requested_policy)
    if policy != "auto":
        return policy
    if attention_backend == "pim_naive":
        return "load_aware"
    return "rotated"


def resolve_pim_head_grouping_policy(attention_backend: str, requested_policy: str) -> str:
    policy = str(requested_policy)
    if policy != "auto":
        return policy
    if attention_backend == "pim_naive":
        return "coarse"
    return "balanced"


def build_record(
    request_index: int,
    task_id: str,
    completion: str,
    metrics: Dict[str, object],
    submit_started_at: float,
    submit_finished_at: float,
    completion_index: int,
    inflight_at_submit: int,
) -> Dict[str, object]:
    return {
        "request_index": int(request_index),
        "completion_index": int(completion_index),
        "task_id": str(task_id),
        "completion": completion,
        "metrics": metrics,
        "submit_started_at": float(submit_started_at),
        "submit_finished_at": float(submit_finished_at),
        "finished_at": float(time.time()),
        "inflight_at_submit": int(inflight_at_submit),
    }


def run_serial_requests(
    problems: List[Dict[str, object]],
    generate_fn: Callable[[Dict[str, object]], Tuple[str, Dict[str, object]]],
) -> Tuple[List[Dict[str, object]], float]:
    records: List[Dict[str, object]] = []
    run_started_at = time.time()
    for request_index, problem in enumerate(problems, start=1):
        submit_started_at = time.time()
        completion, metrics = generate_fn(problem)
        submit_finished_at = submit_started_at
        records.append(
            build_record(
                request_index=request_index,
                task_id=str(problem["task_id"]),
                completion=completion,
                metrics=metrics,
                submit_started_at=submit_started_at,
                submit_finished_at=submit_finished_at,
                completion_index=request_index,
                inflight_at_submit=1,
            )
        )
    wall_time_s = max(time.time() - run_started_at, 1e-12)
    return records, wall_time_s


def run_scheduler_requests(
    scheduler,
    problems: List[Dict[str, object]],
    max_new_tokens: int,
    concurrency: int,
) -> Tuple[List[Dict[str, object]], float]:
    records: List[Dict[str, object]] = []
    pending: Dict[ray.ObjectRef, Dict[str, object]] = {}
    next_idx = 0
    completion_index = 0
    run_started_at = time.time()

    while next_idx < len(problems) or pending:
        while next_idx < len(problems) and len(pending) < concurrency:
            problem = problems[next_idx]
            request_index = next_idx + 1
            submit_started_at = time.time()
            future = scheduler.submit_request.remote(
                problem["prompt"],
                return_metrics=True,
                max_new_tokens=max_new_tokens,
            )
            submit_finished_at = time.time()
            pending[future] = {
                "request_index": request_index,
                "problem": problem,
                "submit_started_at": submit_started_at,
                "submit_finished_at": submit_finished_at,
                "inflight_at_submit": len(pending) + 1,
            }
            next_idx += 1

        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        future = ready[0]
        meta = pending.pop(future)
        completion, metrics = ray.get(future)
        completion_index += 1
        records.append(
            build_record(
                request_index=int(meta["request_index"]),
                task_id=str(meta["problem"]["task_id"]),
                completion=completion,
                metrics=metrics,
                submit_started_at=float(meta["submit_started_at"]),
                submit_finished_at=float(meta["submit_finished_at"]),
                completion_index=completion_index,
                inflight_at_submit=int(meta["inflight_at_submit"]),
            )
        )

    wall_time_s = max(time.time() - run_started_at, 1e-12)
    return records, wall_time_s


def make_cluster_config(args, attention_backend: str) -> ClusterConfig:
    resident_store_backend = str(args.pim_resident_store_backend)
    if resident_store_backend == "auto":
        resident_store_backend = "upmem_kvslot" if attention_backend in {"pim_naive", "cloverinfer"} else "host"

    qk_full_enabled = bool(args.pim_qk_full_enabled)
    softmax_av_fused_enabled = bool(args.pim_softmax_av_fused_enabled)
    qk_full_shadow_check = bool(args.pim_qk_full_shadow_check)
    softmax_av_shadow_check = bool(args.pim_softmax_av_shadow_check)
    if attention_backend in {"pim_naive", "cloverinfer"}:
        qk_full_enabled = True
        softmax_av_fused_enabled = True
    if attention_backend == "pim_naive":
        qk_full_shadow_check = False
        softmax_av_shadow_check = False
    use_gpu_for_attention = attention_backend == "gpu"
    decode_dense_gpu_fraction = float(args.decode_dense_gpu_fraction)
    attention_gpu_fraction = float(args.attention_gpu_fraction) if use_gpu_for_attention else 0.0
    pim_dpu_placement_policy = resolve_pim_dpu_placement_policy(
        attention_backend,
        args.pim_dpu_placement_policy,
    )
    pim_head_grouping_policy = resolve_pim_head_grouping_policy(
        attention_backend,
        args.pim_head_grouping_policy,
    )
    attention_resource = str(args.attention_resource)
    if attention_backend == "gpu":
        attention_resource = str(args.decode_dense_resource)
        if abs(decode_dense_gpu_fraction - 1.0) < 1e-9 and attention_gpu_fraction <= 0.0:
            decode_dense_gpu_fraction = 0.5
            attention_gpu_fraction = 0.5
        elif decode_dense_gpu_fraction <= 0.0:
            decode_dense_gpu_fraction = 0.5
        if attention_gpu_fraction <= 0.0:
            attention_gpu_fraction = 0.5
        if decode_dense_gpu_fraction + attention_gpu_fraction > 1.0 + 1e-6:
            raise ValueError(
                "AFD shared decode/attention GPU fractions must sum to at most 1.0. "
                f"Got decode_dense_gpu_fraction={decode_dense_gpu_fraction}, "
                f"attention_gpu_fraction={attention_gpu_fraction}."
            )

    return ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource=args.prefill_resource,
        decode_dense_resource=args.decode_dense_resource,
        attention_resource=attention_resource,
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        use_gpu_for_attention=use_gpu_for_attention,
        decode_dense_gpu_fraction=decode_dense_gpu_fraction,
        attention_gpu_fraction=attention_gpu_fraction,
        attention_backend=attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_resident_store_backend=resident_store_backend,
        pim_qk_full_enabled=qk_full_enabled,
        pim_qk_full_shadow_check=qk_full_shadow_check,
        pim_softmax_av_fused_enabled=softmax_av_fused_enabled,
        pim_softmax_av_shadow_check=softmax_av_shadow_check,
        pim_length=args.pim_length,
        pim_block_tokens=args.pim_block_tokens,
        pim_max_resident_groups_per_layer=args.pim_max_resident_groups_per_layer,
        pim_head_grouping_policy=pim_head_grouping_policy,
        pim_dpu_placement_policy=pim_dpu_placement_policy,
        pim_resident_kv_dtype=args.pim_resident_kv_dtype,
        pim_qk_mixed_enabled=args.pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        clover_cpu_shadow_enabled=args.clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=args.clover_shadow_checks_enabled,
        clover_op_profiling_enabled=args.clover_op_profiling_enabled,
        clover_cpu_fast_path_max_context_tokens=(
            args.clover_cpu_fast_path_max_context_tokens if attention_backend == "cloverinfer" else 0
        ),
        clover_shadow_check_token_interval=args.clover_shadow_check_token_interval,
        clover_shadow_check_layer_interval=args.clover_shadow_check_layer_interval,
        clover_host_qk_mixed_enabled=args.clover_host_qk_mixed_enabled,
        clover_pim_attention_enabled=(attention_backend == "cloverinfer"),
        clover_pim_context_fused_experimental_enabled=args.clover_pim_context_fused_experimental_enabled,
        clover_pim_rank_spread_alloc_experimental_enabled=(
            args.clover_pim_rank_spread_alloc_experimental_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_pim_cross_rank_stripe_experimental_enabled=(
            args.clover_pim_cross_rank_stripe_experimental_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_pim_rank_spread_multi_rank_batch_experimental_enabled=(
            args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled
            if attention_backend == "cloverinfer"
            else False
        ),
        clover_pim_layer_rank_rotation_experimental_enabled=(
            args.clover_pim_layer_rank_rotation_experimental_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_pim_slot_spill_alloc_experimental_enabled=(
            args.clover_pim_slot_spill_alloc_experimental_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_pim_slot_pressure_aware_alloc_experimental_enabled=(
            args.clover_pim_slot_pressure_aware_alloc_experimental_enabled
            if attention_backend == "cloverinfer"
            else False
        ),
        clover_pim_emergency_slot_spill_experimental_enabled=(
            args.clover_pim_emergency_slot_spill_experimental_enabled
            if attention_backend == "cloverinfer"
            else False
        ),
        clover_pim_reserve_segment_tail_capacity_experimental_enabled=(
            args.clover_pim_reserve_segment_tail_capacity_experimental_enabled
            if attention_backend == "cloverinfer"
            else False
        ),
        clover_pim_reserve_segment_tail_capacity_tokens=(
            args.clover_pim_reserve_segment_tail_capacity_tokens
            if attention_backend == "cloverinfer"
            else 0
        ),
        clover_pim_perf_guard_enabled=(
            args.clover_pim_perf_guard_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_pim_perf_guard_force_cpu_for_compressed_kv=(
            args.clover_pim_perf_guard_force_cpu_for_compressed_kv
            if attention_backend == "cloverinfer"
            else True
        ),
        clover_pim_perf_guard_min_decode_items=args.clover_pim_perf_guard_min_decode_items,
        clover_pim_perf_guard_slowdown_threshold=args.clover_pim_perf_guard_slowdown_threshold,
        clover_compact_short_segments_enabled=(
            args.clover_compact_short_segments_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_compact_short_segment_min_tokens=args.clover_compact_short_segment_min_tokens,
        clover_fine_head_grouping_experimental_enabled=(
            args.clover_fine_head_grouping_experimental_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_target_heads_per_group_experimental=args.clover_target_heads_per_group_experimental,
        clover_predictive_scheduling_enabled=(
            args.clover_predictive_scheduling_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_predictive_scheduling_alpha=args.clover_predictive_scheduling_alpha,
        clover_predictive_scheduling_min_samples=args.clover_predictive_scheduling_min_samples,
        clover_predictive_scheduling_context_bucket_tokens=args.clover_predictive_scheduling_context_bucket_tokens,
        clover_capacity_aware_batching_enabled=(
            args.clover_capacity_aware_batching_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_capacity_aware_time_gap_threshold=args.clover_capacity_aware_time_gap_threshold,
        clover_capacity_aware_lookahead_window=args.clover_capacity_aware_lookahead_window,
        clover_capacity_aware_pim_a=args.clover_capacity_aware_pim_a,
        clover_capacity_aware_pim_b=args.clover_capacity_aware_pim_b,
        clover_capacity_aware_host_c=args.clover_capacity_aware_host_c,
        clover_capacity_aware_max_tokens_per_dpu=args.clover_capacity_aware_max_tokens_per_dpu,
        clover_capacity_aware_require_slot_headroom=(
            args.clover_capacity_aware_require_slot_headroom if attention_backend == "cloverinfer" else False
        ),
        clover_rankset_overlap_enabled=(
            args.clover_rankset_overlap_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_rankset_overlap_max_ranksets_per_batch=args.clover_rankset_overlap_max_ranksets_per_batch,
        clover_rankset_overlap_transfer_granularity=args.clover_rankset_overlap_transfer_granularity,
        clover_rankset_overlap_async_dispatch_enabled=(
            args.clover_rankset_overlap_async_dispatch_enabled if attention_backend == "cloverinfer" else False
        ),
        clover_rankset_overlap_transfer_latency_s=args.clover_rankset_overlap_transfer_latency_s,
        decode_continuous_batch_window_s=args.decode_continuous_batch_window_s,
        decode_continuous_batch_max_size=args.decode_continuous_batch_max_size,
        attention_rpc_cross_key_batch_enabled=(attention_backend == "cloverinfer"),
        attention_actor_side_batching_enabled=False,
    )


def run_monolithic_gpu(args, problems: List[Dict[str, object]]) -> Dict[str, object]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    adapter = CausalModelAdapter(args.model, "cuda", dtype)

    def _generate(problem: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
        result = adapter.greedy_generate(str(problem["prompt"]), args.max_new_tokens)
        return result["text"], result["metrics"]

    records, wall_time_s = run_serial_requests(problems, _generate)
    return {
        "baseline": "monolithic_gpu",
        "placement": {
            "mode": "single_process",
            "ip": ray.util.get_node_ip_address() if ray.is_initialized() else "local",
            "device": "cuda",
        },
        "records": records,
        "summary": summarize_run(
            records,
            wall_time_s=wall_time_s,
            requested_concurrency=max(1, int(args.concurrency)),
            effective_concurrency=1,
            execution_mode="serial_local_adapter",
        ),
    }


def run_disaggregated(args, problems: List[Dict[str, object]], attention_backend: str) -> Dict[str, object]:
    cluster_conf = make_cluster_config(args, attention_backend)
    model_conf = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    try:
        placement = ray.get(scheduler.initialize_cluster.remote())
        records, wall_time_s = run_scheduler_requests(
            scheduler,
            problems,
            max_new_tokens=args.max_new_tokens,
            concurrency=max(1, int(args.concurrency)),
        )

        return {
            "baseline": f"disagg_{attention_backend}",
            "placement": placement,
            "records": records,
            "summary": summarize_run(
                records,
                wall_time_s=wall_time_s,
                requested_concurrency=max(1, int(args.concurrency)),
                effective_concurrency=max(1, int(args.concurrency)),
                execution_mode="async_scheduler_queue",
            ),
        }
    finally:
        with suppress(Exception):
            ray.kill(scheduler, no_restart=True)


def run_split_gpu(args, problems: List[Dict[str, object]]) -> Dict[str, object]:
    model_conf = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )

    prefill = PrefillNode.options(
        num_gpus=1,
        resources={args.prefill_resource: 0.01},
    ).remote(0, model_conf, True)
    decode = DecodeDenseNode.options(
        num_gpus=1,
        resources={args.decode_dense_resource: 0.01},
    ).remote(0, model_conf, True)

    try:
        placement = {
            "prefill": ray.get(prefill.get_info.remote()),
            "decode_full": ray.get(decode.get_info.remote()),
        }

        def _generate(problem: Dict[str, object]) -> Tuple[str, Dict[str, object]]:
            wall_started = time.time()
            prefill_out = ray.get(prefill.process_prompt.remote(str(problem["prompt"])))
            first_token_ready = time.time()
            decode_out = ray.get(
                decode.continue_full_decode.remote(
                    prefill_out["initial_kv"],
                    prefill_out["prompt_len"],
                    prefill_out["first_token_id"],
                    args.max_new_tokens,
                )
            )
            request_finished = time.time()

            total_tokens = int(len(decode_out["generated_ids"]))
            ttft = float(first_token_ready - wall_started)
            latency = float(request_finished - wall_started)
            tpot = float((latency - ttft) / max(total_tokens - 1, 1))
            throughput = float(total_tokens / latency) if latency > 0 else 0.0

            metrics = {
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "throughput": throughput,
                "total_tokens": total_tokens,
                "stage_timing": {
                    "scheduler": {
                        "prefill_rpc_s": ttft,
                        "decode_full_rpc_s": max(0.0, latency - ttft),
                        "total_rpc_s": latency,
                    },
                    "actors": {
                        "prefill_compute_s": float(prefill_out.get("profile", {}).get("compute_s", 0.0)),
                        "decode_full_compute_s": float(decode_out.get("profile", {}).get("compute_s", 0.0)),
                        "total_compute_s": float(prefill_out.get("profile", {}).get("compute_s", 0.0))
                        + float(decode_out.get("profile", {}).get("compute_s", 0.0)),
                    },
                    "counts": {
                        "decode_steps": max(0, total_tokens - 1),
                    },
                },
            }
            return decode_out["text"], metrics

        records, wall_time_s = run_serial_requests(problems, _generate)
        return {
            "baseline": "split_gpu_full_decode",
            "placement": placement,
            "records": records,
            "summary": summarize_run(
                records,
                wall_time_s=wall_time_s,
                requested_concurrency=max(1, int(args.concurrency)),
                effective_concurrency=1,
                execution_mode="serial_split_pipeline",
            ),
        }
    finally:
        with suppress(Exception):
            ray.kill(prefill, no_restart=True)
        with suppress(Exception):
            ray.kill(decode, no_restart=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/humaneval.jsonl")
    parser.add_argument(
        "--dataset-format",
        default="auto",
        choices=DATASET_FORMAT_CHOICES,
    )
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/Qwen-1_8B")
    parser.add_argument("--model-name", default="qwen-1_8b")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--prompt-token-length", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--baselines",
        default="monolithic_gpu,split_gpu_full_decode,disagg_cpu,disagg_pim_naive",
        help=BASELINE_HELP_TEXT,
    )
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--prefill-resource", default="prefill_gpu")
    parser.add_argument("--decode-dense-resource", default="decode_dense_gpu")
    parser.add_argument("--attention-resource", default="attention_pim")
    parser.add_argument("--decode-dense-gpu-fraction", type=float, default=1.0)
    parser.add_argument("--attention-gpu-fraction", type=float, default=0.0)
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument(
        "--pim-resident-store-backend",
        default="auto",
        choices=["auto", "host", "upmem_kvslot"],
    )
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument("--pim-max-resident-groups-per-layer", type=int, default=0)
    parser.add_argument(
        "--pim-head-grouping-policy",
        default="auto",
        choices=["auto", "legacy", "balanced", "coarse", "segment_aware"],
    )
    parser.add_argument(
        "--pim-dpu-placement-policy",
        default="auto",
        choices=["auto", "identity", "rotated", "rank_spread", "load_aware"],
    )
    parser.add_argument("--pim-resident-kv-dtype", default="fp32", choices=["fp32", "fp16", "int8"])
    parser.add_argument("--pim-qk-full-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-full-enabled", action="store_true")
    parser.add_argument("--pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--no-pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--no-pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--no-pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--no-clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--no-clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--no-clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--clover-cpu-fast-path-max-context-tokens", type=int, default=0)
    parser.add_argument("--clover-shadow-check-token-interval", type=int, default=4)
    parser.add_argument("--clover-shadow-check-layer-interval", type=int, default=4)
    parser.add_argument("--clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-rank-spread-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-rank-spread-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-cross-rank-stripe-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-cross-rank-stripe-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-rank-spread-multi-rank-batch-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-rank-spread-multi-rank-batch-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-layer-rank-rotation-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-layer-rank-rotation-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-slot-spill-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-slot-spill-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-slot-pressure-aware-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-slot-pressure-aware-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-emergency-slot-spill-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-emergency-slot-spill-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-reserve-segment-tail-capacity-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-reserve-segment-tail-capacity-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-reserve-segment-tail-capacity-tokens", type=int, default=0)
    parser.add_argument("--clover-pim-perf-guard-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-perf-guard-enabled", action="store_true")
    parser.add_argument("--clover-pim-perf-guard-force-cpu-for-compressed-kv", action="store_true")
    parser.add_argument("--no-clover-pim-perf-guard-force-cpu-for-compressed-kv", action="store_true")
    parser.add_argument("--clover-pim-perf-guard-min-decode-items", type=int, default=1)
    parser.add_argument("--clover-pim-perf-guard-slowdown-threshold", type=float, default=1.2)
    parser.add_argument("--clover-compact-short-segments-enabled", action="store_true")
    parser.add_argument("--no-clover-compact-short-segments-enabled", action="store_true")
    parser.add_argument("--clover-compact-short-segment-min-tokens", type=int, default=8)
    parser.add_argument("--clover-fine-head-grouping-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-fine-head-grouping-experimental-enabled", action="store_true")
    parser.add_argument("--clover-target-heads-per-group-experimental", type=int, default=0)
    parser.add_argument("--clover-predictive-scheduling-enabled", action="store_true")
    parser.add_argument("--no-clover-predictive-scheduling-enabled", action="store_true")
    parser.add_argument("--clover-predictive-scheduling-alpha", type=float, default=0.2)
    parser.add_argument("--clover-predictive-scheduling-min-samples", type=int, default=4)
    parser.add_argument("--clover-predictive-scheduling-context-bucket-tokens", type=int, default=256)
    parser.add_argument("--clover-capacity-aware-batching-enabled", action="store_true")
    parser.add_argument("--no-clover-capacity-aware-batching-enabled", action="store_true")
    parser.add_argument("--clover-capacity-aware-time-gap-threshold", type=float, default=0.0)
    parser.add_argument("--clover-capacity-aware-lookahead-window", type=int, default=1)
    parser.add_argument("--clover-capacity-aware-pim-a", type=float, default=1.0)
    parser.add_argument("--clover-capacity-aware-pim-b", type=float, default=0.0)
    parser.add_argument("--clover-capacity-aware-host-c", type=float, default=1.0)
    parser.add_argument("--clover-capacity-aware-max-tokens-per-dpu", type=int, default=0)
    parser.add_argument("--clover-capacity-aware-require-slot-headroom", action="store_true")
    parser.add_argument("--no-clover-capacity-aware-require-slot-headroom", action="store_true")
    parser.add_argument("--clover-rankset-overlap-enabled", action="store_true")
    parser.add_argument("--no-clover-rankset-overlap-enabled", action="store_true")
    parser.add_argument("--clover-rankset-overlap-max-ranksets-per-batch", type=int, default=0)
    parser.add_argument(
        "--clover-rankset-overlap-transfer-granularity",
        default="stripe",
        choices=["stripe", "rankset"],
    )
    parser.add_argument("--clover-rankset-overlap-async-dispatch-enabled", action="store_true")
    parser.add_argument("--no-clover-rankset-overlap-async-dispatch-enabled", action="store_true")
    parser.add_argument("--clover-rankset-overlap-transfer-latency-s", type=float, default=0.0)
    parser.add_argument("--decode-continuous-batch-window-s", type=float, default=0.0)
    parser.add_argument("--decode-continuous-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--decode-continuous-batch-max-size", type=int, default=8)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "baseline_comparison.jsonl"),
    )
    args = parser.parse_args()

    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")
    if args.pim_qk_full_enabled and args.no_pim_qk_full_enabled:
        raise ValueError("cannot set both --pim-qk-full-enabled and --no-pim-qk-full-enabled")
    if args.pim_qk_full_shadow_check and args.no_pim_qk_full_shadow_check:
        raise ValueError("cannot set both --pim-qk-full-shadow-check and --no-pim-qk-full-shadow-check")
    if args.pim_softmax_av_fused_enabled and args.no_pim_softmax_av_fused_enabled:
        raise ValueError(
            "cannot set both --pim-softmax-av-fused-enabled and --no-pim-softmax-av-fused-enabled"
        )
    if args.pim_softmax_av_shadow_check and args.no_pim_softmax_av_shadow_check:
        raise ValueError(
            "cannot set both --pim-softmax-av-shadow-check and --no-pim-softmax-av-shadow-check"
        )
    if args.clover_cpu_shadow_enabled and args.no_clover_cpu_shadow_enabled:
        raise ValueError("cannot set both --clover-cpu-shadow-enabled and --no-clover-cpu-shadow-enabled")
    if args.clover_shadow_checks_enabled and args.no_clover_shadow_checks_enabled:
        raise ValueError(
            "cannot set both --clover-shadow-checks-enabled and --no-clover-shadow-checks-enabled"
        )
    if args.clover_op_profiling_enabled and args.no_clover_op_profiling_enabled:
        raise ValueError(
            "cannot set both --clover-op-profiling-enabled and --no-clover-op-profiling-enabled"
        )
    if args.clover_cpu_fast_path_max_context_tokens < 0:
        raise ValueError("--clover-cpu-fast-path-max-context-tokens must be non-negative")
    if args.clover_host_qk_mixed_enabled and args.no_clover_host_qk_mixed_enabled:
        raise ValueError("cannot set both --clover-host-qk-mixed-enabled and --no-clover-host-qk-mixed-enabled")
    if (
        args.clover_pim_context_fused_experimental_enabled
        and args.no_clover_pim_context_fused_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-context-fused-experimental-enabled and "
            "--no-clover-pim-context-fused-experimental-enabled"
        )
    if (
        args.clover_pim_rank_spread_alloc_experimental_enabled
        and args.no_clover_pim_rank_spread_alloc_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-rank-spread-alloc-experimental-enabled and "
            "--no-clover-pim-rank-spread-alloc-experimental-enabled"
        )
    if (
        args.clover_pim_cross_rank_stripe_experimental_enabled
        and args.no_clover_pim_cross_rank_stripe_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-cross-rank-stripe-experimental-enabled and "
            "--no-clover-pim-cross-rank-stripe-experimental-enabled"
        )
    if (
        args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled
        and args.no_clover_pim_rank_spread_multi_rank_batch_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-rank-spread-multi-rank-batch-experimental-enabled and "
            "--no-clover-pim-rank-spread-multi-rank-batch-experimental-enabled"
        )
    if (
        args.clover_pim_layer_rank_rotation_experimental_enabled
        and args.no_clover_pim_layer_rank_rotation_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-layer-rank-rotation-experimental-enabled and "
            "--no-clover-pim-layer-rank-rotation-experimental-enabled"
        )
    if (
        args.clover_pim_slot_spill_alloc_experimental_enabled
        and args.no_clover_pim_slot_spill_alloc_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-slot-spill-alloc-experimental-enabled and "
            "--no-clover-pim-slot-spill-alloc-experimental-enabled"
        )
    if (
        args.clover_pim_slot_pressure_aware_alloc_experimental_enabled
        and args.no_clover_pim_slot_pressure_aware_alloc_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-slot-pressure-aware-alloc-experimental-enabled and "
            "--no-clover-pim-slot-pressure-aware-alloc-experimental-enabled"
        )
    if (
        args.clover_pim_emergency_slot_spill_experimental_enabled
        and args.no_clover_pim_emergency_slot_spill_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-emergency-slot-spill-experimental-enabled and "
            "--no-clover-pim-emergency-slot-spill-experimental-enabled"
        )
    if (
        args.clover_pim_reserve_segment_tail_capacity_experimental_enabled
        and args.no_clover_pim_reserve_segment_tail_capacity_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-reserve-segment-tail-capacity-experimental-enabled and "
            "--no-clover-pim-reserve-segment-tail-capacity-experimental-enabled"
        )
    if args.clover_pim_perf_guard_enabled and args.no_clover_pim_perf_guard_enabled:
        raise ValueError(
            "cannot set both --clover-pim-perf-guard-enabled and "
            "--no-clover-pim-perf-guard-enabled"
        )
    if (
        args.clover_pim_perf_guard_force_cpu_for_compressed_kv
        and args.no_clover_pim_perf_guard_force_cpu_for_compressed_kv
    ):
        raise ValueError(
            "cannot set both --clover-pim-perf-guard-force-cpu-for-compressed-kv and "
            "--no-clover-pim-perf-guard-force-cpu-for-compressed-kv"
        )
    if args.clover_compact_short_segments_enabled and args.no_clover_compact_short_segments_enabled:
        raise ValueError(
            "cannot set both --clover-compact-short-segments-enabled and "
            "--no-clover-compact-short-segments-enabled"
        )
    if (
        args.clover_fine_head_grouping_experimental_enabled
        and args.no_clover_fine_head_grouping_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-fine-head-grouping-experimental-enabled and "
            "--no-clover-fine-head-grouping-experimental-enabled"
        )
    if args.clover_predictive_scheduling_enabled and args.no_clover_predictive_scheduling_enabled:
        raise ValueError(
            "cannot set both --clover-predictive-scheduling-enabled and "
            "--no-clover-predictive-scheduling-enabled"
        )
    if args.clover_capacity_aware_batching_enabled and args.no_clover_capacity_aware_batching_enabled:
        raise ValueError(
            "cannot set both --clover-capacity-aware-batching-enabled and "
            "--no-clover-capacity-aware-batching-enabled"
        )
    if args.clover_rankset_overlap_enabled and args.no_clover_rankset_overlap_enabled:
        raise ValueError(
            "cannot set both --clover-rankset-overlap-enabled and "
            "--no-clover-rankset-overlap-enabled"
        )
    if (
        args.clover_rankset_overlap_async_dispatch_enabled
        and args.no_clover_rankset_overlap_async_dispatch_enabled
    ):
        raise ValueError(
            "cannot set both --clover-rankset-overlap-async-dispatch-enabled and "
            "--no-clover-rankset-overlap-async-dispatch-enabled"
        )

    if not args.pim_qk_mixed_enabled and not args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = False
    args.pim_qk_full_enabled = bool(args.pim_qk_full_enabled)
    if args.no_pim_qk_full_enabled:
        args.pim_qk_full_enabled = False
    args.pim_qk_full_shadow_check = True
    if args.no_pim_qk_full_shadow_check:
        args.pim_qk_full_shadow_check = False
    elif args.pim_qk_full_shadow_check:
        args.pim_qk_full_shadow_check = True
    args.pim_softmax_av_fused_enabled = bool(args.pim_softmax_av_fused_enabled)
    if args.no_pim_softmax_av_fused_enabled:
        args.pim_softmax_av_fused_enabled = False
    args.pim_softmax_av_shadow_check = True
    if args.no_pim_softmax_av_shadow_check:
        args.pim_softmax_av_shadow_check = False
    elif args.pim_softmax_av_shadow_check:
        args.pim_softmax_av_shadow_check = True
    args.clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        args.clover_cpu_shadow_enabled = False
    elif args.clover_cpu_shadow_enabled:
        args.clover_cpu_shadow_enabled = True
    args.clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        args.clover_shadow_checks_enabled = False
    elif args.clover_shadow_checks_enabled:
        args.clover_shadow_checks_enabled = True
    args.clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        args.clover_op_profiling_enabled = False
    elif args.clover_op_profiling_enabled:
        args.clover_op_profiling_enabled = True
    args.clover_host_qk_mixed_enabled = bool(args.clover_host_qk_mixed_enabled)
    if args.no_clover_host_qk_mixed_enabled:
        args.clover_host_qk_mixed_enabled = False
    args.clover_pim_context_fused_experimental_enabled = bool(
        args.clover_pim_context_fused_experimental_enabled
    )
    if args.no_clover_pim_context_fused_experimental_enabled:
        args.clover_pim_context_fused_experimental_enabled = False
    args.clover_pim_rank_spread_alloc_experimental_enabled = bool(
        args.clover_pim_rank_spread_alloc_experimental_enabled
    )
    if args.no_clover_pim_rank_spread_alloc_experimental_enabled:
        args.clover_pim_rank_spread_alloc_experimental_enabled = False
    args.clover_pim_cross_rank_stripe_experimental_enabled = bool(
        args.clover_pim_cross_rank_stripe_experimental_enabled
    )
    if args.no_clover_pim_cross_rank_stripe_experimental_enabled:
        args.clover_pim_cross_rank_stripe_experimental_enabled = False
    args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled = bool(
        args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled
    )
    if args.no_clover_pim_rank_spread_multi_rank_batch_experimental_enabled:
        args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled = False
    args.clover_pim_layer_rank_rotation_experimental_enabled = bool(
        args.clover_pim_layer_rank_rotation_experimental_enabled
    )
    if args.no_clover_pim_layer_rank_rotation_experimental_enabled:
        args.clover_pim_layer_rank_rotation_experimental_enabled = False
    args.clover_pim_slot_spill_alloc_experimental_enabled = bool(
        args.clover_pim_slot_spill_alloc_experimental_enabled
    )
    if args.no_clover_pim_slot_spill_alloc_experimental_enabled:
        args.clover_pim_slot_spill_alloc_experimental_enabled = False
    args.clover_pim_slot_pressure_aware_alloc_experimental_enabled = bool(
        args.clover_pim_slot_pressure_aware_alloc_experimental_enabled
    )
    if args.no_clover_pim_slot_pressure_aware_alloc_experimental_enabled:
        args.clover_pim_slot_pressure_aware_alloc_experimental_enabled = False
    args.clover_pim_emergency_slot_spill_experimental_enabled = bool(
        args.clover_pim_emergency_slot_spill_experimental_enabled
    )
    if args.no_clover_pim_emergency_slot_spill_experimental_enabled:
        args.clover_pim_emergency_slot_spill_experimental_enabled = False
    args.clover_pim_reserve_segment_tail_capacity_experimental_enabled = bool(
        args.clover_pim_reserve_segment_tail_capacity_experimental_enabled
    )
    if args.no_clover_pim_reserve_segment_tail_capacity_experimental_enabled:
        args.clover_pim_reserve_segment_tail_capacity_experimental_enabled = False
    if args.clover_pim_reserve_segment_tail_capacity_tokens < 0:
        raise ValueError("--clover-pim-reserve-segment-tail-capacity-tokens must be non-negative")
    args.clover_pim_perf_guard_enabled = bool(args.clover_pim_perf_guard_enabled)
    if args.no_clover_pim_perf_guard_enabled:
        args.clover_pim_perf_guard_enabled = False
    args.clover_pim_perf_guard_force_cpu_for_compressed_kv = True
    if args.no_clover_pim_perf_guard_force_cpu_for_compressed_kv:
        args.clover_pim_perf_guard_force_cpu_for_compressed_kv = False
    elif args.clover_pim_perf_guard_force_cpu_for_compressed_kv:
        args.clover_pim_perf_guard_force_cpu_for_compressed_kv = True
    if args.clover_pim_perf_guard_min_decode_items <= 0:
        raise ValueError("--clover-pim-perf-guard-min-decode-items must be positive")
    if args.clover_pim_perf_guard_slowdown_threshold < 1.0:
        raise ValueError("--clover-pim-perf-guard-slowdown-threshold must be at least 1.0")
    args.clover_compact_short_segments_enabled = bool(args.clover_compact_short_segments_enabled)
    if args.no_clover_compact_short_segments_enabled:
        args.clover_compact_short_segments_enabled = False
    if args.clover_compact_short_segment_min_tokens <= 0:
        raise ValueError("--clover-compact-short-segment-min-tokens must be positive")
    args.clover_fine_head_grouping_experimental_enabled = bool(
        args.clover_fine_head_grouping_experimental_enabled
    )
    if args.no_clover_fine_head_grouping_experimental_enabled:
        args.clover_fine_head_grouping_experimental_enabled = False
    if args.clover_target_heads_per_group_experimental < 0:
        raise ValueError("--clover-target-heads-per-group-experimental must be non-negative")
    args.clover_predictive_scheduling_enabled = bool(args.clover_predictive_scheduling_enabled)
    if args.no_clover_predictive_scheduling_enabled:
        args.clover_predictive_scheduling_enabled = False
    args.clover_capacity_aware_batching_enabled = bool(args.clover_capacity_aware_batching_enabled)
    if args.no_clover_capacity_aware_batching_enabled:
        args.clover_capacity_aware_batching_enabled = False
    args.clover_capacity_aware_require_slot_headroom = bool(
        args.clover_capacity_aware_require_slot_headroom
    )
    if args.no_clover_capacity_aware_require_slot_headroom:
        args.clover_capacity_aware_require_slot_headroom = False
    args.clover_rankset_overlap_enabled = bool(args.clover_rankset_overlap_enabled)
    if args.no_clover_rankset_overlap_enabled:
        args.clover_rankset_overlap_enabled = False
    args.clover_rankset_overlap_async_dispatch_enabled = bool(
        args.clover_rankset_overlap_async_dispatch_enabled
    )
    if args.no_clover_rankset_overlap_async_dispatch_enabled:
        args.clover_rankset_overlap_async_dispatch_enabled = False
    if args.clover_rankset_overlap_max_ranksets_per_batch < 0:
        raise ValueError("--clover-rankset-overlap-max-ranksets-per-batch must be non-negative")
    if args.clover_rankset_overlap_transfer_latency_s < 0.0:
        raise ValueError("--clover-rankset-overlap-transfer-latency-s must be non-negative")
    if args.decode_continuous_batch_window_s < 0.0:
        raise ValueError("--decode-continuous-batch-window-s must be non-negative")
    if args.decode_continuous_batch_window_ms < 0.0:
        raise ValueError("--decode-continuous-batch-window-ms must be non-negative")
    if args.decode_continuous_batch_max_size <= 0:
        raise ValueError("--decode-continuous-batch-max-size must be positive")
    if args.decode_continuous_batch_window_ms > 0.0:
        args.decode_continuous_batch_window_s += args.decode_continuous_batch_window_ms / 1000.0

    tokenizer = None
    if int(args.prompt_token_length) > 0:
        tokenizer = load_tokenizer_for_benchmark(args.model)

    problems = load_benchmark_samples(
        args.data,
        dataset_format=args.dataset_format,
        limit=args.limit,
        tokenizer=tokenizer,
        prompt_token_length=int(args.prompt_token_length),
    )
    baseline_specs = resolve_requested_baselines(args.baselines)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env=build_runtime_env(),
    )

    results = []
    data_config = {
        "path": args.data,
        "dataset_format": args.dataset_format,
        "loaded_samples": len(problems),
        "prompt_token_length_override": int(args.prompt_token_length),
        "concurrency": max(1, int(args.concurrency)),
        "max_new_tokens": int(args.max_new_tokens),
        "clover_capacity_aware_batching_enabled": bool(args.clover_capacity_aware_batching_enabled),
        "clover_capacity_aware_time_gap_threshold": float(args.clover_capacity_aware_time_gap_threshold),
        "clover_capacity_aware_lookahead_window": int(args.clover_capacity_aware_lookahead_window),
        "clover_capacity_aware_pim_a": float(args.clover_capacity_aware_pim_a),
        "clover_capacity_aware_pim_b": float(args.clover_capacity_aware_pim_b),
        "clover_capacity_aware_host_c": float(args.clover_capacity_aware_host_c),
        "clover_capacity_aware_max_tokens_per_dpu": int(args.clover_capacity_aware_max_tokens_per_dpu),
        "clover_cpu_fast_path_max_context_tokens": int(args.clover_cpu_fast_path_max_context_tokens),
        "clover_capacity_aware_require_slot_headroom": bool(
            args.clover_capacity_aware_require_slot_headroom
        ),
        "clover_pim_rank_spread_alloc_experimental_enabled": bool(
            args.clover_pim_rank_spread_alloc_experimental_enabled
        ),
        "clover_pim_cross_rank_stripe_experimental_enabled": bool(
            args.clover_pim_cross_rank_stripe_experimental_enabled
        ),
        "clover_pim_rank_spread_multi_rank_batch_experimental_enabled": bool(
            args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled
        ),
        "clover_pim_layer_rank_rotation_experimental_enabled": bool(
            args.clover_pim_layer_rank_rotation_experimental_enabled
        ),
        "clover_pim_slot_spill_alloc_experimental_enabled": bool(
            args.clover_pim_slot_spill_alloc_experimental_enabled
        ),
        "clover_pim_slot_pressure_aware_alloc_experimental_enabled": bool(
            args.clover_pim_slot_pressure_aware_alloc_experimental_enabled
        ),
        "clover_pim_emergency_slot_spill_experimental_enabled": bool(
            args.clover_pim_emergency_slot_spill_experimental_enabled
        ),
        "clover_pim_reserve_segment_tail_capacity_experimental_enabled": bool(
            args.clover_pim_reserve_segment_tail_capacity_experimental_enabled
        ),
        "clover_pim_reserve_segment_tail_capacity_tokens": int(
            args.clover_pim_reserve_segment_tail_capacity_tokens
        ),
        "clover_pim_perf_guard_enabled": bool(args.clover_pim_perf_guard_enabled),
        "clover_pim_perf_guard_force_cpu_for_compressed_kv": bool(
            args.clover_pim_perf_guard_force_cpu_for_compressed_kv
        ),
        "clover_pim_perf_guard_min_decode_items": int(args.clover_pim_perf_guard_min_decode_items),
        "clover_pim_perf_guard_slowdown_threshold": float(
            args.clover_pim_perf_guard_slowdown_threshold
        ),
        "clover_compact_short_segments_enabled": bool(args.clover_compact_short_segments_enabled),
        "clover_compact_short_segment_min_tokens": int(args.clover_compact_short_segment_min_tokens),
        "clover_fine_head_grouping_experimental_enabled": bool(
            args.clover_fine_head_grouping_experimental_enabled
        ),
        "clover_target_heads_per_group_experimental": int(
            args.clover_target_heads_per_group_experimental
        ),
        "clover_rankset_overlap_enabled": bool(args.clover_rankset_overlap_enabled),
        "clover_rankset_overlap_max_ranksets_per_batch": int(args.clover_rankset_overlap_max_ranksets_per_batch),
        "clover_rankset_overlap_transfer_granularity": str(args.clover_rankset_overlap_transfer_granularity),
        "clover_rankset_overlap_async_dispatch_enabled": bool(
            args.clover_rankset_overlap_async_dispatch_enabled
        ),
        "clover_rankset_overlap_transfer_latency_s": float(args.clover_rankset_overlap_transfer_latency_s),
        "decode_continuous_batch_window_s": float(args.decode_continuous_batch_window_s),
        "decode_continuous_batch_max_size": int(args.decode_continuous_batch_max_size),
        "resource_layout": {
            "prefill_resource": str(args.prefill_resource),
            "decode_dense_resource": str(args.decode_dense_resource),
            "attention_resource": str(args.attention_resource),
        },
    }

    for baseline_spec in baseline_specs:
        internal_baseline = str(baseline_spec["internal_baseline"])
        if internal_baseline == "monolithic_gpu":
            result = run_monolithic_gpu(args, problems)
        elif internal_baseline == "split_gpu_full_decode":
            result = run_split_gpu(args, problems)
        elif internal_baseline == "disagg_afd":
            result = run_disaggregated(args, problems, "gpu")
        elif internal_baseline == "disagg_cpu":
            result = run_disaggregated(args, problems, "cpu")
        elif internal_baseline == "disagg_pim_naive":
            result = run_disaggregated(args, problems, "pim_naive")
        elif internal_baseline == "disagg_cloverinfer":
            result = run_disaggregated(args, problems, "cloverinfer")
        else:
            raise ValueError(f"unsupported internal baseline: {internal_baseline}")

        result["legacy_baseline"] = result["baseline"]
        result["baseline"] = str(baseline_spec["canonical_name"])
        result["baseline_key"] = str(baseline_spec["canonical_key"])
        result["internal_baseline"] = internal_baseline
        result["requested_aliases"] = list(baseline_spec["requested_aliases"])
        result["resolved_aliases"] = list(baseline_spec["resolved_aliases"])
        result["alias_note"] = str(baseline_spec["alias_note"])
        result["data_config"] = data_config
        result["summary"]["baseline"] = result["baseline"]
        result["summary"]["baseline_key"] = result["baseline_key"]
        result["summary"]["internal_baseline"] = internal_baseline
        result["summary"]["requested_aliases"] = list(baseline_spec["requested_aliases"])
        results.append(result)
        print(json.dumps(result["summary"], ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved baseline comparison results to {args.output}")
    with suppress(Exception):
        ray.shutdown()


if __name__ == "__main__":
    main()
