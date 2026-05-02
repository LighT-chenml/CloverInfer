import argparse
import json
import os
import sys
import time
from typing import Dict, List

import ray
from ray.exceptions import RayTaskError
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def build_ray_runtime_env(*, force_rebuild_kvslot_helper: bool = False) -> Dict[str, object]:
    excludes = [
        ".git/",
        "artifacts/",
        "model/",
        "**/__pycache__/",
        "*.pyc",
    ]
    env_vars: Dict[str, str] = {
        "PYTHONPATH": REPO_ROOT,
        "CLOVER_SHARED_REPO_ROOT": REPO_ROOT,
    }
    if force_rebuild_kvslot_helper:
        env_vars["CLOVER_FORCE_REBUILD_KVSLOT"] = "1"
    return {
        "working_dir": REPO_ROOT,
        "excludes": excludes,
        "env_vars": env_vars,
    }


def load_prompts(path: str, limit: int | None) -> List[Dict[str, str]]:
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            prompts.append(json.loads(line))
    if limit is not None:
        prompts = prompts[:limit]
    return prompts


def build_prompt_for_target_tokens(tokenizer, target_tokens: int, prefix: str) -> tuple[str, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    filler_ids = tokenizer.encode(" clover", add_special_tokens=False)
    token_ids = list(prefix_ids)
    while len(token_ids) < target_tokens:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:target_tokens]
    prompt = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(actual_ids)


def build_synthetic_prompts(
    tokenizer,
    *,
    prompt_token_length: int,
    count: int,
    prefix: str,
) -> List[Dict[str, str]]:
    prompts: List[Dict[str, str]] = []
    for idx in range(max(count, 0)):
        prompt, actual_tokens = build_prompt_for_target_tokens(
            tokenizer,
            target_tokens=prompt_token_length,
            prefix=f"{prefix} sample_{idx}:",
        )
        prompts.append(
            {
                "task_id": f"synthetic_{idx}",
                "prompt": prompt,
                "prompt_source": "synthetic",
                "prompt_target_token_length": int(prompt_token_length),
                "prompt_token_length": int(actual_tokens),
            }
        )
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


def summarize_live_dpu_balance(live_elems: List[object]) -> Dict[str, float]:
    values = [int(item) for item in live_elems]
    if not values:
        return {
            "num_dpus": 0.0,
            "active_dpus": 0.0,
            "active_ratio": 0.0,
            "total_live_elems": 0.0,
            "max_live_elems": 0.0,
            "min_live_elems": 0.0,
            "avg_live_elems": 0.0,
            "imbalance_ratio": 0.0,
        }
    active_values = [value for value in values if value > 0]
    total_live = sum(values)
    avg_live = float(total_live) / float(len(values)) if values else 0.0
    max_live = max(values)
    min_live = min(values)
    return {
        "num_dpus": float(len(values)),
        "active_dpus": float(len(active_values)),
        "active_ratio": float(len(active_values)) / float(len(values)) if values else 0.0,
        "total_live_elems": float(total_live),
        "max_live_elems": float(max_live),
        "min_live_elems": float(min_live),
        "avg_live_elems": avg_live,
        "imbalance_ratio": (float(max_live) / avg_live) if avg_live > 0.0 else 0.0,
    }


def summarize_rank_balance(
    request_footprints: List[Dict[str, object]],
    helper_topology: Dict[str, object],
) -> Dict[str, object]:
    def safe_rank_index(value: object, default: int = 0) -> int:
        if value is None:
            return int(default)
        return int(value)

    topology_items = helper_topology.get("items", []) if isinstance(helper_topology, dict) else []
    physical_to_rank: Dict[int, int] = {}
    rank_ids = set()
    for item in topology_items:
        logical_dpu_id = int(item.get("logical_dpu_id", 0))
        rank_index = int(item.get("rank_index", 0))
        physical_to_rank[logical_dpu_id] = rank_index
        rank_ids.add(rank_index)

    per_rank_live: Dict[int, int] = {}
    per_rank_blocks: Dict[int, int] = {}
    unique_block_dpustriples = set()
    unique_request_ranks = set()
    total_blocks = 0

    for request in request_footprints:
        for layer in request.get("layers", []):
            for group in layer.get("groups", []):
                slot_debug = group.get("resident_slot", {})
                if slot_debug.get("storage") != "dpu_blocked":
                    physical_dpu = int(slot_debug.get("physical_dpu", 0))
                    rank_index = physical_to_rank.get(
                        physical_dpu,
                        safe_rank_index(slot_debug.get("rank_index", 0)),
                    )
                    live_elems = int(group.get("live_elems", 0))
                    per_rank_live[rank_index] = per_rank_live.get(rank_index, 0) + live_elems
                    per_rank_blocks[rank_index] = per_rank_blocks.get(rank_index, 0) + 1
                    unique_request_ranks.add(rank_index)
                    total_blocks += 1
                    continue
                for block in slot_debug.get("blocks", []):
                    physical_dpu = int(block.get("physical_dpu", 0))
                    rank_index = physical_to_rank.get(
                        physical_dpu,
                        safe_rank_index(block.get("rank_index", 0)),
                    )
                    seq_len = int(block.get("seq_len", 0))
                    capacity = int(block.get("capacity", 0))
                    group_heads = int(slot_debug.get("group_heads", 0))
                    head_dim = int(slot_debug.get("head_dim", 0))
                    live_elems = seq_len * group_heads * head_dim
                    per_rank_live[rank_index] = per_rank_live.get(rank_index, 0) + live_elems
                    per_rank_blocks[rank_index] = per_rank_blocks.get(rank_index, 0) + 1
                    unique_request_ranks.add(rank_index)
                    unique_block_dpustriples.add((physical_dpu, rank_index, capacity))
                    total_blocks += 1

    rank_live_values = list(per_rank_live.values())
    active_rank_count = len(per_rank_live)
    total_rank_count = max(len(rank_ids), active_rank_count)
    avg_rank_live = (
        float(sum(rank_live_values)) / float(active_rank_count)
        if active_rank_count > 0
        else 0.0
    )
    max_rank_live = max(rank_live_values) if rank_live_values else 0
    return {
        "total_ranks": int(total_rank_count),
        "active_ranks": int(active_rank_count),
        "active_rank_ratio": (
            float(active_rank_count) / float(total_rank_count)
            if total_rank_count > 0
            else 0.0
        ),
        "per_rank_live_elems": {str(rank): int(value) for rank, value in sorted(per_rank_live.items())},
        "per_rank_block_count": {str(rank): int(value) for rank, value in sorted(per_rank_blocks.items())},
        "avg_rank_live_elems": float(avg_rank_live),
        "max_rank_live_elems": float(max_rank_live),
        "rank_live_imbalance_ratio": (
            float(max_rank_live) / float(avg_rank_live)
            if avg_rank_live > 0.0
            else 0.0
        ),
        "total_blocks": int(total_blocks),
        "unique_block_dpu_count": int(len({item[0] for item in unique_block_dpustriples})),
        "unique_request_rank_count": int(len(unique_request_ranks)),
    }


def summarize_request_blocks(request_footprints: List[Dict[str, object]]) -> Dict[str, object]:
    total_groups = 0
    blocked_groups = 0
    total_blocks = 0
    max_blocks_per_group = 0
    block_seq_lens: List[int] = []
    block_capacities: List[int] = []

    for request in request_footprints:
        for layer in request.get("layers", []):
            for group in layer.get("groups", []):
                total_groups += 1
                slot_debug = group.get("resident_slot", {})
                blocks = slot_debug.get("blocks", [])
                if str(slot_debug.get("storage", "")) == "dpu_blocked":
                    blocked_groups += 1
                block_count = len(blocks)
                total_blocks += block_count
                if block_count > 0:
                    max_blocks_per_group = max(max_blocks_per_group, block_count)
                for block in blocks:
                    block_seq_lens.append(int(block.get("seq_len", 0)))
                    block_capacities.append(int(block.get("capacity", 0)))

    avg_blocks_per_group = float(total_blocks) / float(total_groups) if total_groups > 0 else 0.0
    return {
        "total_groups": int(total_groups),
        "blocked_groups": int(blocked_groups),
        "blocked_group_ratio": (
            float(blocked_groups) / float(total_groups) if total_groups > 0 else 0.0
        ),
        "total_blocks": int(total_blocks),
        "avg_blocks_per_group": float(avg_blocks_per_group),
        "max_blocks_per_group": int(max_blocks_per_group),
        "max_block_seq_len": int(max(block_seq_lens)) if block_seq_lens else 0,
        "min_block_seq_len": int(min(block_seq_lens)) if block_seq_lens else 0,
        "distinct_block_capacities": sorted({int(value) for value in block_capacities}),
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
    def _topk_timings(totals: object, k: int = 5) -> List[Dict[str, float]]:
        if not isinstance(totals, dict):
            return []
        items = []
        for key, value in totals.items():
            try:
                sec = float(value)
            except Exception:
                continue
            items.append((str(key), sec))
        items.sort(key=lambda kv: kv[1], reverse=True)
        return [{"op": op, "s": float(sec)} for op, sec in items[: max(0, int(k))]]

    attention_before_free = metrics.get("attention_backend_before_free", {})
    backend_debug = attention_before_free.get("backend_debug", {})
    decode_batching = attention_before_free.get("decode_batching", {})
    resident_store_debug = backend_debug.get("resident_store_debug", {})
    allocator_stats = resident_store_debug.get("allocator_stats", [])
    dpu_live_elems_by_dpu = resident_store_debug.get("dpu_live_elems_by_dpu", [])
    helper_topology = resident_store_debug.get("helper_topology", {})
    request_footprints = backend_debug.get("resident_request_footprints", [])
    request_preview = backend_debug.get("resident_request_preview", {})
    analysis_footprints = request_footprints
    if isinstance(request_preview, dict) and request_preview.get("layers"):
        analysis_footprints = [request_preview]
    prompt_token_length = sample.get("prompt_token_length")
    if prompt_token_length is None:
        prompt_token_length = 0
    clover_op_timing_totals_s = backend_debug.get("clover_op_timing_totals_s", {})
    return {
        "request_index": idx,
        "completion_index": completion_index,
        "task_id": sample.get("task_id", f"sample_{idx}"),
        "prompt_source": sample.get("prompt_source", "dataset"),
        "prompt_target_token_length": int(sample.get("prompt_target_token_length", 0)),
        "prompt_token_length": int(prompt_token_length),
        "prompt_chars": len(sample.get("prompt", "")),
        "submit_started_at": float(submit_started_at),
        "submit_finished_at": float(submit_finished_at),
        "inflight_at_submit": int(inflight_at_submit),
        "completion": completion,
        "stage_timing": metrics.get("stage_timing", {}),
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
        "continuous_engine": metrics.get("continuous_engine", {}),
        "clover_cpu_shadow_enabled": bool(backend_debug.get("clover_cpu_shadow_enabled", False)),
        "clover_shadow_checks_enabled": bool(backend_debug.get("clover_shadow_checks_enabled", False)),
        "clover_op_profiling_enabled": bool(backend_debug.get("clover_op_profiling_enabled", False)),
        "clover_shadow_check_token_interval": int(backend_debug.get("clover_shadow_check_token_interval", 0)),
        "clover_shadow_check_layer_interval": int(backend_debug.get("clover_shadow_check_layer_interval", 0)),
        "clover_host_qk_mixed_enabled": bool(backend_debug.get("clover_host_qk_mixed_enabled", False)),
        "clover_shadow_check_invocations": int(backend_debug.get("clover_shadow_check_invocations", 0)),
        "clover_shadow_check_skips": int(backend_debug.get("clover_shadow_check_skips", 0)),
        "clover_op_timing_totals_s": clover_op_timing_totals_s,
        "clover_op_top5": _topk_timings(clover_op_timing_totals_s, k=5),
        "clover_op_timing_counts": backend_debug.get("clover_op_timing_counts", {}),
        "attention_decode_batching": decode_batching,
        "scheduler_decode_step_sync": metrics.get("scheduler_decode_step_sync", {}),
        "scheduler_attention_layer_barrier": metrics.get("scheduler_attention_layer_barrier", {}),
        "scheduler_attention_batching": metrics.get("scheduler_attention_batching", {}),
        "resident_total_live_elems": int(backend_debug.get("resident_total_live_elems", 0)),
        "resident_total_capacity_elems": int(backend_debug.get("resident_total_capacity_elems", 0)),
        "resident_request_footprints": request_footprints,
        "resident_request_preview": request_preview,
        "store_backend": resident_store_debug.get("backend", ""),
        "store_dpu_placement_policy": resident_store_debug.get("dpu_placement_policy", ""),
        "helper_host_profile_totals_s": resident_store_debug.get("helper_host_profile_totals_s", {}),
        "helper_host_profile_counts": resident_store_debug.get("helper_host_profile_counts", {}),
        "helper_dpu_profile": resident_store_debug.get("helper_profile", {}),
        "dpu_allocations": int(resident_store_debug.get("dpu_allocations", 0)),
        "fallback_allocations": int(resident_store_debug.get("fallback_allocations", 0)),
        "dpu_allocate_failures": int(resident_store_debug.get("dpu_allocate_failures", 0)),
        "dpu_capacity_fallbacks": int(resident_store_debug.get("dpu_capacity_fallbacks", 0)),
        "last_allocate_error_stage": resident_store_debug.get("last_allocate_error_stage", ""),
        "last_allocate_error": resident_store_debug.get("last_allocate_error", ""),
        "dpu_live_elems_by_dpu": dpu_live_elems_by_dpu,
        "dpu_balance_summary": summarize_live_dpu_balance(dpu_live_elems_by_dpu),
        "rank_balance_summary": summarize_rank_balance(analysis_footprints, helper_topology),
        "block_summary": summarize_request_blocks(analysis_footprints),
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
    parser.add_argument("--prompt-token-length", type=int, default=0)
    parser.add_argument("--synthetic-prefix", default="Context:")
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
    parser.add_argument(
        "--pim-dpu-placement-policy",
        default="rotated",
        choices=["identity", "rotated", "rank_spread"],
    )
    parser.add_argument("--pim-resident-kv-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--pim-resident-store-backend", default="upmem_kvslot", choices=["host", "upmem_kvslot"])
    parser.add_argument(
        "--pim-force-rebuild-kvslot-helper",
        action="store_true",
        help="Force rebuilding the kvslot helper on Ray workers (sets CLOVER_FORCE_REBUILD_KVSLOT=1).",
    )
    parser.add_argument(
        "--pim-tail-capacity-buckets",
        default="16,32,64,128,256",
        help="Comma-separated capacity buckets (<= pim-block-tokens) for blocked resident KV tail blocks.",
    )
    parser.add_argument("--pim-kvslot-best-round-seed-enabled", action="store_true")
    parser.add_argument("--no-pim-kvslot-best-round-seed-enabled", action="store_true")
    parser.add_argument("--pim-kvslot-shape-rounds-enabled", action="store_true")
    parser.add_argument("--no-pim-kvslot-shape-rounds-enabled", action="store_true")
    parser.add_argument("--pim-kvslot-context-fused-enabled", action="store_true")
    parser.add_argument("--no-pim-kvslot-context-fused-enabled", action="store_true")
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
    parser.add_argument(
        "--attention-wavefront-cohort-policy",
        default="batch",
        choices=["batch", "step"],
        help="How to key attention wavefront batching when decode wave persist is enabled. "
        "'batch' keeps cohort_id in the key; 'step' batches all requests at the same step/layer together.",
    )
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
    parser.add_argument("--decode-continuous-batching-enabled", action="store_true")
    parser.add_argument("--no-decode-continuous-batching-enabled", action="store_true")
    parser.add_argument("--decode-continuous-max-batch-size", type=int, default=8)
    parser.add_argument("--decode-continuous-batch-window-s", type=float, default=0.001)
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
    if args.pim_kvslot_best_round_seed_enabled and args.no_pim_kvslot_best_round_seed_enabled:
        raise ValueError(
            "cannot set both --pim-kvslot-best-round-seed-enabled and --no-pim-kvslot-best-round-seed-enabled"
        )
    if args.pim_kvslot_shape_rounds_enabled and args.no_pim_kvslot_shape_rounds_enabled:
        raise ValueError(
            "cannot set both --pim-kvslot-shape-rounds-enabled and --no-pim-kvslot-shape-rounds-enabled"
        )
    if args.pim_kvslot_context_fused_enabled and args.no_pim_kvslot_context_fused_enabled:
        raise ValueError(
            "cannot set both --pim-kvslot-context-fused-enabled and --no-pim-kvslot-context-fused-enabled"
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
    pim_kvslot_best_round_seed_enabled = bool(args.pim_kvslot_best_round_seed_enabled)
    if args.no_pim_kvslot_best_round_seed_enabled:
        pim_kvslot_best_round_seed_enabled = False
    pim_kvslot_shape_rounds_enabled = bool(args.pim_kvslot_shape_rounds_enabled)
    if args.no_pim_kvslot_shape_rounds_enabled:
        pim_kvslot_shape_rounds_enabled = False
    pim_kvslot_context_fused_enabled = bool(args.pim_kvslot_context_fused_enabled)
    if args.no_pim_kvslot_context_fused_enabled:
        pim_kvslot_context_fused_enabled = False

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.prompt_token_length > 0:
        prompts = build_synthetic_prompts(
            tokenizer,
            prompt_token_length=int(args.prompt_token_length),
            count=int(args.limit),
            prefix=str(args.synthetic_prefix),
        )
    else:
        prompts = load_prompts(args.data, args.limit)
        for sample in prompts:
            prompt = sample.get("prompt", "")
            prompt_token_length = len(tokenizer.encode(prompt, add_special_tokens=False))
            sample["prompt_source"] = sample.get("prompt_source", "dataset")
            sample["prompt_target_token_length"] = int(sample.get("prompt_target_token_length", 0))
            sample["prompt_token_length"] = int(prompt_token_length)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env=build_ray_runtime_env(force_rebuild_kvslot_helper=bool(args.pim_force_rebuild_kvslot_helper)),
    )

    tail_buckets = []
    if str(args.pim_tail_capacity_buckets).strip():
        for item in str(args.pim_tail_capacity_buckets).split(","):
            item = item.strip()
            if not item:
                continue
            tail_buckets.append(int(item))

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
        pim_tail_capacity_buckets=tail_buckets,
        pim_kvslot_best_round_seed_enabled=pim_kvslot_best_round_seed_enabled,
        pim_kvslot_shape_rounds_experimental_enabled=pim_kvslot_shape_rounds_enabled,
        pim_kvslot_context_fused_experimental_enabled=pim_kvslot_context_fused_enabled,
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
        attention_wavefront_cohort_policy=args.attention_wavefront_cohort_policy,
        attention_rpc_cross_key_batch_enabled=(args.attention_backend == "cloverinfer"),
        attention_actor_side_batching_enabled=False,
        attention_actor_batch_window_s=args.attention_actor_batch_window_s,
        attention_actor_batch_max_size=args.attention_actor_batch_max_size,
        decode_continuous_batching_enabled=bool(args.decode_continuous_batching_enabled)
        and not bool(args.no_decode_continuous_batching_enabled),
        decode_continuous_max_batch_size=int(args.decode_continuous_max_batch_size),
        decode_continuous_batch_window_s=float(args.decode_continuous_batch_window_s),
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
                    "prompt_token_length": int(row["prompt_token_length"]),
                    "total_blocks": int(row["block_summary"]["total_blocks"]),
                    "max_blocks_per_group": int(row["block_summary"]["max_blocks_per_group"]),
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
        "max_prompt_token_length": max(int(row["prompt_token_length"]) for row in trace_rows) if trace_rows else 0,
        "max_total_blocks": max(int(row["block_summary"]["total_blocks"]) for row in trace_rows) if trace_rows else 0,
        "max_blocks_per_group": max(int(row["block_summary"]["max_blocks_per_group"]) for row in trace_rows) if trace_rows else 0,
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
