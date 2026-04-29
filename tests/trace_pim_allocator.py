import argparse
import json
import os
import sys
import time
from typing import Dict, List

import ray
from ray.exceptions import RayTaskError

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def load_prompts(path: str, limit: int | None) -> List[Dict[str, str]]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def summarize_allocator_stats(stats: List[Dict[str, object]]) -> Dict[str, float]:
    if not stats:
        return {
            "max_usage_ratio": 0.0,
            "avg_usage_ratio": 0.0,
            "max_free_range_count": 0.0,
            "min_largest_free_range": 0.0,
            "max_live_slot_count": 0.0,
        }
    usage_ratios = [float(item["usage_ratio"]) for item in stats]
    free_range_counts = [int(item["free_range_count"]) for item in stats]
    largest_free_ranges = [int(item["largest_free_range"]) for item in stats]
    live_slot_counts = [int(item["live_slot_count"]) for item in stats]
    return {
        "max_usage_ratio": max(usage_ratios),
        "avg_usage_ratio": sum(usage_ratios) / len(usage_ratios),
        "max_free_range_count": float(max(free_range_counts)),
        "min_largest_free_range": float(min(largest_free_ranges)),
        "max_live_slot_count": float(max(live_slot_counts)),
    }


def build_trace_row(
    idx: int,
    sample: Dict[str, str],
    completion: str,
    metrics: Dict[str, object],
    submit_started_at: float,
    submit_finished_at: float,
    completion_index: int,
    inflight_at_submit: int,
) -> Dict[str, object]:
    attention_before_free = metrics.get("attention_backend_before_free", {})
    backend_debug = attention_before_free.get("backend_debug", {})
    decode_batching = attention_before_free.get("decode_batching", {})
    resident_store_debug = backend_debug.get("resident_store_debug", {})
    allocator_stats = resident_store_debug.get("allocator_stats", [])
    return {
        "request_index": idx,
        "completion_index": completion_index,
        "task_id": sample.get("task_id", f"sample_{idx}"),
        "prompt_chars": len(sample.get("prompt", "")),
        "submit_started_at": float(submit_started_at),
        "submit_finished_at": float(submit_finished_at),
        "inflight_at_submit": int(inflight_at_submit),
        "completion": completion,
        "latency": float(metrics["latency"]),
        "ttft": float(metrics["ttft"]),
        "tpot": float(metrics["tpot"]),
        "throughput": float(metrics["throughput"]),
        "total_tokens": int(metrics["total_tokens"]),
        "resident_append_ops": int(backend_debug.get("resident_append_ops", 0)),
        "resident_materialize_ops": int(backend_debug.get("resident_materialize_ops", 0)),
        "resident_shadow_max_abs_diff": float(backend_debug.get("resident_shadow_max_abs_diff", 0.0)),
        "resident_av_enabled": bool(backend_debug.get("resident_av_enabled", False)),
        "resident_av_ops": int(backend_debug.get("resident_av_ops", 0)),
        "resident_av_batch_calls": int(backend_debug.get("resident_av_batch_calls", 0)),
        "resident_av_shadow_max_abs_diff": float(backend_debug.get("resident_av_shadow_max_abs_diff", 0.0)),
        "qk_full_enabled": bool(backend_debug.get("qk_full_enabled", False)),
        "qk_full_shadow_check": bool(backend_debug.get("qk_full_shadow_check", False)),
        "qk_full_count": int(backend_debug.get("qk_full_count", 0)),
        "qk_full_batch_calls": int(backend_debug.get("qk_full_batch_calls", 0)),
        "qk_full_shadow_checks": int(backend_debug.get("qk_full_shadow_checks", 0)),
        "qk_full_shadow_max_abs_diff": float(backend_debug.get("qk_full_shadow_max_abs_diff", 0.0)),
        "softmax_av_fused_enabled": bool(backend_debug.get("softmax_av_fused_enabled", False)),
        "softmax_av_shadow_check": bool(backend_debug.get("softmax_av_shadow_check", False)),
        "softmax_av_fused_ops": int(backend_debug.get("softmax_av_fused_ops", 0)),
        "softmax_av_fused_batch_calls": int(backend_debug.get("softmax_av_fused_batch_calls", 0)),
        "softmax_av_fused_shadow_max_abs_diff": float(backend_debug.get("softmax_av_fused_shadow_max_abs_diff", 0.0)),
        "qk_batch_calls": int(backend_debug.get("qk_batch_calls", 0)),
        "decode_batch_calls": int(backend_debug.get("decode_batch_calls", 0)),
        "decode_batch_items": int(backend_debug.get("decode_batch_items", 0)),
        "backend_variant": backend_debug.get("backend_variant", "pim_naive"),
        "clover_cpu_shadow_enabled": bool(backend_debug.get("clover_cpu_shadow_enabled", False)),
        "clover_shadow_checks_enabled": bool(backend_debug.get("clover_shadow_checks_enabled", False)),
        "clover_op_profiling_enabled": bool(backend_debug.get("clover_op_profiling_enabled", False)),
        "clover_shadow_check_token_interval": int(backend_debug.get("clover_shadow_check_token_interval", 0)),
        "clover_shadow_check_layer_interval": int(backend_debug.get("clover_shadow_check_layer_interval", 0)),
        "clover_host_qk_mixed_enabled": bool(backend_debug.get("clover_host_qk_mixed_enabled", False)),
        "clover_shadow_check_invocations": int(backend_debug.get("clover_shadow_check_invocations", 0)),
        "clover_shadow_check_skips": int(backend_debug.get("clover_shadow_check_skips", 0)),
        "clover_op_timing_totals_s": backend_debug.get("clover_op_timing_totals_s", {}),
        "clover_op_timing_counts": backend_debug.get("clover_op_timing_counts", {}),
        "attention_decode_batching": decode_batching,
        "scheduler_decode_step_sync": metrics.get("scheduler_decode_step_sync", {}),
        "scheduler_attention_layer_barrier": metrics.get("scheduler_attention_layer_barrier", {}),
        "scheduler_attention_batching": metrics.get("scheduler_attention_batching", {}),
        "resident_total_live_elems": int(backend_debug.get("resident_total_live_elems", 0)),
        "resident_total_capacity_elems": int(backend_debug.get("resident_total_capacity_elems", 0)),
        "resident_request_footprints": backend_debug.get("resident_request_footprints", []),
        "resident_request_preview": backend_debug.get("resident_request_preview", {}),
        "store_backend": resident_store_debug.get("backend", ""),
        "dpu_allocations": int(resident_store_debug.get("dpu_allocations", 0)),
        "fallback_allocations": int(resident_store_debug.get("fallback_allocations", 0)),
        "dpu_allocate_failures": int(resident_store_debug.get("dpu_allocate_failures", 0)),
        "dpu_capacity_fallbacks": int(resident_store_debug.get("dpu_capacity_fallbacks", 0)),
        "allocator_summary": summarize_allocator_stats(allocator_stats),
        "allocator_stats": allocator_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--data", default="dataset/humaneval.jsonl")
    parser.add_argument("--output", default=os.path.join(REPO_ROOT, "artifacts", "pim_allocator_trace.jsonl"))
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--model-name", default="opt-125m")
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=8)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument("--pim-max-resident-groups-per-layer", type=int, default=0)
    parser.add_argument(
        "--pim-head-grouping-policy",
        default="balanced",
        choices=["legacy", "balanced", "coarse", "segment_aware"],
    )
    parser.add_argument("--pim-dpu-placement-policy", default="rotated", choices=["identity", "rotated"])
    parser.add_argument("--pim-resident-kv-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--pim-resident-store-backend", default="upmem_kvslot", choices=["host", "upmem_kvslot"])
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
    parser.add_argument("--decode-step-sync-window-s", type=float, default=0.0)
    parser.add_argument("--decode-step-sync-max-size", type=int, default=8)
    parser.add_argument("--attention-decode-wave-persist-enabled", action="store_true")
    parser.add_argument("--attention-layer-barrier-window-s", type=float, default=0.0)
    parser.add_argument("--attention-layer-barrier-max-size", type=int, default=8)
    parser.add_argument("--attention-rpc-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-rpc-batch-max-size", type=int, default=8)
    parser.add_argument("--attention-actor-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-actor-batch-max-size", type=int, default=8)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--attention-backend", default="pim_naive", choices=["pim_naive", "cloverinfer"])
    parser.add_argument("--clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--no-clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--no-clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--no-clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--clover-shadow-check-token-interval", type=int, default=4)
    parser.add_argument("--clover-shadow-check-layer-interval", type=int, default=4)
    parser.add_argument("--clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--clover-pim-attention-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-attention-enabled", action="store_true")
    parser.add_argument("--clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-context-fused-experimental-enabled", action="store_true")
    args = parser.parse_args()

    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")
    if args.pim_qk_full_enabled and args.no_pim_qk_full_enabled:
        raise ValueError("cannot set both --pim-qk-full-enabled and --no-pim-qk-full-enabled")
    if args.pim_qk_full_shadow_check and args.no_pim_qk_full_shadow_check:
        raise ValueError("cannot set both --pim-qk-full-shadow-check and --no-pim-qk-full-shadow-check")
    if args.pim_softmax_av_fused_enabled and args.no_pim_softmax_av_fused_enabled:
        raise ValueError("cannot set both --pim-softmax-av-fused-enabled and --no-pim-softmax-av-fused-enabled")
    if args.pim_softmax_av_shadow_check and args.no_pim_softmax_av_shadow_check:
        raise ValueError("cannot set both --pim-softmax-av-shadow-check and --no-pim-softmax-av-shadow-check")
    if args.clover_cpu_shadow_enabled and args.no_clover_cpu_shadow_enabled:
        raise ValueError("cannot set both --clover-cpu-shadow-enabled and --no-clover-cpu-shadow-enabled")
    if args.clover_shadow_checks_enabled and args.no_clover_shadow_checks_enabled:
        raise ValueError("cannot set both --clover-shadow-checks-enabled and --no-clover-shadow-checks-enabled")
    if args.clover_op_profiling_enabled and args.no_clover_op_profiling_enabled:
        raise ValueError("cannot set both --clover-op-profiling-enabled and --no-clover-op-profiling-enabled")
    if args.clover_host_qk_mixed_enabled and args.no_clover_host_qk_mixed_enabled:
        raise ValueError("cannot set both --clover-host-qk-mixed-enabled and --no-clover-host-qk-mixed-enabled")
    if args.clover_pim_attention_enabled and args.no_clover_pim_attention_enabled:
        raise ValueError("cannot set both --clover-pim-attention-enabled and --no-clover-pim-attention-enabled")
    if (
        args.clover_pim_context_fused_experimental_enabled
        and args.no_clover_pim_context_fused_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-context-fused-experimental-enabled and "
            "--no-clover-pim-context-fused-experimental-enabled"
        )

    pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = False
    elif args.pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = True
    pim_qk_full_enabled = False
    if args.pim_qk_full_enabled:
        pim_qk_full_enabled = True
    if args.no_pim_qk_full_enabled:
        pim_qk_full_enabled = False
    pim_qk_full_shadow_check = True
    if args.no_pim_qk_full_shadow_check:
        pim_qk_full_shadow_check = False
    pim_softmax_av_fused_enabled = False
    if args.pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = True
    if args.no_pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = False
    pim_softmax_av_shadow_check = True
    if args.no_pim_softmax_av_shadow_check:
        pim_softmax_av_shadow_check = False
    clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = False
    clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = False
    clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        clover_op_profiling_enabled = False
    clover_host_qk_mixed_enabled = False
    if args.clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = True
    if args.no_clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = False
    clover_pim_attention_enabled = args.attention_backend == "cloverinfer"
    if args.no_clover_pim_attention_enabled:
        clover_pim_attention_enabled = False
    elif args.clover_pim_attention_enabled:
        clover_pim_attention_enabled = True
    clover_pim_context_fused_experimental_enabled = bool(args.clover_pim_context_fused_experimental_enabled)
    if args.no_clover_pim_context_fused_experimental_enabled:
        clover_pim_context_fused_experimental_enabled = False

    prompts = load_prompts(args.data, args.limit)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}},
    )

    cluster = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource="prefill_gpu",
        decode_dense_resource="decode_dense_gpu",
        attention_resource="attention_pim",
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend=args.attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_resident_store_backend=args.pim_resident_store_backend,
        pim_max_resident_groups_per_layer=args.pim_max_resident_groups_per_layer,
        pim_head_grouping_policy=args.pim_head_grouping_policy,
        pim_dpu_placement_policy=args.pim_dpu_placement_policy,
        pim_resident_kv_dtype=args.pim_resident_kv_dtype,
        pim_qk_full_enabled=pim_qk_full_enabled,
        pim_qk_full_shadow_check=pim_qk_full_shadow_check,
        pim_softmax_av_fused_enabled=pim_softmax_av_fused_enabled,
        pim_softmax_av_shadow_check=pim_softmax_av_shadow_check,
        pim_qk_mixed_enabled=pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
        pim_block_tokens=args.pim_block_tokens,
        decode_step_sync_window_s=args.decode_step_sync_window_s,
        decode_step_sync_max_size=args.decode_step_sync_max_size,
        attention_decode_wave_persist_enabled=args.attention_decode_wave_persist_enabled,
        attention_layer_barrier_window_s=args.attention_layer_barrier_window_s,
        attention_layer_barrier_max_size=args.attention_layer_barrier_max_size,
        attention_rpc_batch_window_s=args.attention_rpc_batch_window_s,
        attention_rpc_batch_max_size=args.attention_rpc_batch_max_size,
        attention_rpc_cross_key_batch_enabled=(args.attention_backend == "cloverinfer"),
        attention_actor_side_batching_enabled=False,
        attention_actor_batch_window_s=args.attention_actor_batch_window_s,
        attention_actor_batch_max_size=args.attention_actor_batch_max_size,
        clover_cpu_shadow_enabled=clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=clover_shadow_checks_enabled,
        clover_op_profiling_enabled=clover_op_profiling_enabled,
        clover_shadow_check_token_interval=args.clover_shadow_check_token_interval,
        clover_shadow_check_layer_interval=args.clover_shadow_check_layer_interval,
        clover_host_qk_mixed_enabled=clover_host_qk_mixed_enabled,
        clover_pim_attention_enabled=clover_pim_attention_enabled,
        clover_pim_context_fused_experimental_enabled=clover_pim_context_fused_experimental_enabled,
    )
    model = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )

    scheduler = GlobalScheduler.remote(cluster, model)
    placement = ray.get(scheduler.initialize_cluster.remote())
    print(json.dumps({"placement": placement}, ensure_ascii=False))

    trace_rows = []
    pending = {}
    next_idx = 0
    completion_index = 0
    concurrency = max(1, int(args.concurrency))

    failure = None
    while next_idx < len(prompts) or pending:
        while next_idx < len(prompts) and len(pending) < concurrency:
            sample = prompts[next_idx]
            request_index = next_idx + 1
            submit_started_at = time.time()
            future = scheduler.submit_request.remote(
                sample["prompt"],
                return_metrics=True,
                max_new_tokens=args.max_new_tokens,
            )
            submit_finished_at = time.time()
            pending[future] = {
                "request_index": request_index,
                "sample": sample,
                "submit_started_at": submit_started_at,
                "submit_finished_at": submit_finished_at,
                "inflight_at_submit": len(pending) + 1,
            }
            next_idx += 1

        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        future = ready[0]
        meta = pending.pop(future)
        try:
            completion, metrics = ray.get(future)
        except Exception as exc:
            failure = {
                "request_index": meta["request_index"],
                "task_id": meta["sample"].get("task_id", f"sample_{meta['request_index']}"),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "inflight_at_submit": meta["inflight_at_submit"],
            }
            break
        completion_index += 1
        row = build_trace_row(
            idx=meta["request_index"],
            sample=meta["sample"],
            completion=completion,
            metrics=metrics,
            submit_started_at=meta["submit_started_at"],
            submit_finished_at=meta["submit_finished_at"],
            completion_index=completion_index,
            inflight_at_submit=meta["inflight_at_submit"],
        )
        trace_rows.append(row)
        print(
            json.dumps(
                {
                    "request_index": meta["request_index"],
                    "completion_index": completion_index,
                    "inflight_at_submit": meta["inflight_at_submit"],
                    **row["allocator_summary"],
                },
                ensure_ascii=False,
            )
        )

    if failure is not None:
        print(json.dumps({"failure": failure}, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        for row in trace_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if failure is not None:
            f.write(json.dumps({"failure": failure}, ensure_ascii=False) + "\n")

    summary = {
        "num_requests": len(trace_rows),
        "concurrency": concurrency,
        "max_latency": max(float(row["latency"]) for row in trace_rows) if trace_rows else 0.0,
        "avg_latency": (sum(float(row["latency"]) for row in trace_rows) / len(trace_rows)) if trace_rows else 0.0,
        "max_usage_ratio": max(float(row["allocator_summary"]["max_usage_ratio"]) for row in trace_rows) if trace_rows else 0.0,
        "max_free_range_count": max(float(row["allocator_summary"]["max_free_range_count"]) for row in trace_rows) if trace_rows else 0.0,
        "max_live_slot_count": max(float(row["allocator_summary"]["max_live_slot_count"]) for row in trace_rows) if trace_rows else 0.0,
        "max_dpu_allocate_failures": max(int(row["dpu_allocate_failures"]) for row in trace_rows) if trace_rows else 0,
        "max_dpu_capacity_fallbacks": max(int(row["dpu_capacity_fallbacks"]) for row in trace_rows) if trace_rows else 0,
        "max_fallback_allocations": max(int(row["fallback_allocations"]) for row in trace_rows) if trace_rows else 0,
        "out_of_order_completions": sum(1 for row in trace_rows if int(row["request_index"]) != int(row["completion_index"])),
        "requests_with_overlap_snapshot": sum(1 for row in trace_rows if float(row["allocator_summary"]["max_live_slot_count"]) > 12.0),
        "requests_with_free_ranges": sum(1 for row in trace_rows if float(row["allocator_summary"]["max_free_range_count"]) > 0.0),
        "failed": failure is not None,
    }
    if failure is not None:
        summary["failure_request_index"] = int(failure["request_index"])
        summary["failure_exception_type"] = str(failure["exception_type"])
    print(json.dumps({"summary": summary, "output": args.output}, ensure_ascii=False))


if __name__ == "__main__":
    main()
