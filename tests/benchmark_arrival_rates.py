from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import statistics
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import ray

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_baselines import build_runtime_env, make_cluster_config, resolve_requested_baselines
from benchmark_utils import (
    DATASET_FORMAT_CHOICES,
    decode_prompt,
    encode_prompt,
    load_benchmark_samples,
    load_tokenizer_for_benchmark,
)
from src.core.config import ModelConfig
from src.core.resident_kv_store import SUPPORTED_RESIDENT_KV_DTYPES, normalize_resident_kv_dtype
from src.core.scheduler import GlobalScheduler


DISAGG_BASELINES = {
    "disagg_afd": "gpu",
    "disagg_cpu": "cpu",
    "disagg_pim_naive": "pim_naive",
}


def parse_csv_list(value: str) -> List[str]:
    items = [part.strip() for part in value.split(",") if part.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def parse_float_csv_list(value: str) -> List[float]:
    rates = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not rates:
        raise argparse.ArgumentTypeError("expected a comma-separated list of rates")
    if any(rate <= 0.0 for rate in rates):
        raise argparse.ArgumentTypeError("arrival rates must be positive")
    return rates


def slugify(value: str) -> str:
    chars: List[str] = []
    previous_dash = False
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
            previous_dash = False
        elif not previous_dash:
            chars.append("-")
            previous_dash = True
    return "".join(chars).strip("-") or "item"


def infer_label_from_path(path: str) -> str:
    stem = os.path.splitext(os.path.basename(os.path.normpath(path)))[0]
    return stem or os.path.basename(path)


def parse_model_specs(value: str) -> List[Dict[str, str]]:
    specs = []
    for item in parse_csv_list(value):
        if "=" in item:
            label, model_path = item.split("=", 1)
        else:
            label, model_path = infer_label_from_path(item), item
        label = label.strip()
        model_path = model_path.strip()
        if not label or not model_path:
            raise argparse.ArgumentTypeError(f"invalid model spec: {item}")
        specs.append({"label": label, "slug": slugify(label), "path": model_path})
    return specs


def parse_dataset_specs(value: str) -> List[Dict[str, str]]:
    specs = []
    valid_formats = set(DATASET_FORMAT_CHOICES)
    for item in parse_csv_list(value):
        if "=" in item:
            label, remainder = item.split("=", 1)
        else:
            label, remainder = infer_label_from_path(item), item
        label = label.strip()
        remainder = remainder.strip()
        dataset_format = "auto"
        if ":" in remainder:
            maybe_path, maybe_format = remainder.rsplit(":", 1)
            if maybe_format in valid_formats:
                remainder = maybe_path
                dataset_format = maybe_format
        if not label or not remainder:
            raise argparse.ArgumentTypeError(f"invalid dataset spec: {item}")
        specs.append(
            {
                "label": label,
                "slug": slugify(label),
                "path": remainder,
                "format": dataset_format,
            }
        )
    return specs


def stable_seed(base_seed: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return (int(base_seed) + int(digest[:8], 16)) % (2**31 - 1)


def percentile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def extract_output_text(sample: Dict[str, object]) -> str:
    raw = dict(sample.get("raw_record", {}) or {})
    for key in ("output", "response", "completion", "canonical_solution", "answer"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value

    answers = raw.get("answers")
    if isinstance(answers, list):
        for value in answers:
            if isinstance(value, str) and value.strip():
                return value
        if answers:
            return str(answers[0])
    elif isinstance(answers, str) and answers.strip():
        return answers

    conversations = raw.get("conversations")
    if isinstance(conversations, list):
        seen_user = False
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", turn.get("role", ""))).lower()
            text = str(turn.get("value", turn.get("content", ""))).strip()
            if not text:
                continue
            if role in {"human", "user"}:
                seen_user = True
                continue
            if seen_user and role in {"gpt", "assistant"}:
                return text
    return ""


def build_work_items(
    samples: List[Dict[str, object]],
    tokenizer,
    *,
    max_input_tokens: int,
    max_output_tokens: int,
    min_output_tokens: int,
    fallback_output_tokens: int,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for idx, sample in enumerate(samples):
        prompt = str(sample["prompt"])
        prompt_ids = encode_prompt(tokenizer, prompt)
        if not prompt_ids:
            continue
        original_input_tokens = len(prompt_ids)
        input_truncated = False
        if max_input_tokens > 0 and len(prompt_ids) > max_input_tokens:
            prompt_ids = prompt_ids[:max_input_tokens]
            prompt = decode_prompt(tokenizer, prompt_ids)
            input_truncated = True

        output_text = extract_output_text(sample)
        output_ids = encode_prompt(tokenizer, output_text) if output_text else []
        output_length_source = "dataset"
        if not output_ids:
            output_tokens = int(fallback_output_tokens)
            output_length_source = "fallback"
        else:
            output_tokens = len(output_ids)

        output_tokens = max(int(min_output_tokens), int(output_tokens))
        output_truncated = False
        if max_output_tokens > 0 and output_tokens > max_output_tokens:
            output_tokens = int(max_output_tokens)
            output_truncated = True

        items.append(
            {
                "task_id": str(sample.get("task_id", f"sample/{idx}")),
                "prompt": prompt,
                "input_tokens": int(len(prompt_ids)),
                "original_input_tokens": int(original_input_tokens),
                "input_truncated": bool(input_truncated),
                "output_tokens": int(output_tokens),
                "output_length_source": output_length_source,
                "output_truncated": bool(output_truncated),
                "source_dataset": str(sample.get("source_dataset", "")),
            }
        )
    if not items:
        raise ValueError("no usable work items after tokenization/capping")
    return items


def sample_workload(items: List[Dict[str, object]], num_requests: int, rng: random.Random) -> List[Dict[str, object]]:
    return [dict(rng.choice(items)) for _ in range(int(num_requests))]


def poisson_offsets(num_requests: int, arrival_rate: float, rng: random.Random) -> List[float]:
    offsets: List[float] = []
    current = 0.0
    for idx in range(int(num_requests)):
        if idx == 0:
            offsets.append(0.0)
            continue
        current += rng.expovariate(float(arrival_rate))
        offsets.append(float(current))
    return offsets


def unit_poisson_offsets(num_requests: int, rng: random.Random) -> List[float]:
    offsets: List[float] = []
    current = 0.0
    for idx in range(int(num_requests)):
        if idx == 0:
            offsets.append(0.0)
            continue
        current += rng.expovariate(1.0)
        offsets.append(float(current))
    return offsets


def scale_poisson_offsets(unit_offsets: List[float], arrival_rate: float) -> List[float]:
    return [float(offset) / float(arrival_rate) for offset in unit_offsets]


def build_case_workload_and_arrivals(
    args,
    *,
    work_items: List[Dict[str, object]],
    model_slug: str,
    dataset_slug: str,
    arrival_rate: float,
) -> tuple[int, List[Dict[str, object]], List[float]]:
    if args.shared_workload_across_rates:
        workload_seed = stable_seed(args.seed, model_slug, dataset_slug, "workload")
    else:
        workload_seed = stable_seed(args.seed, model_slug, dataset_slug, arrival_rate, "workload")
    workload_rng = random.Random(workload_seed)
    workload = sample_workload(
        work_items,
        int(args.num_requests) + int(args.warmup_requests),
        workload_rng,
    )

    if args.shared_arrival_trace_across_rates:
        arrival_seed = stable_seed(args.seed, model_slug, dataset_slug, "arrival_trace")
        arrival_rng = random.Random(arrival_seed)
        arrivals = scale_poisson_offsets(
            unit_poisson_offsets(int(args.num_requests), arrival_rng),
            float(arrival_rate),
        )
    else:
        arrival_seed = stable_seed(args.seed, model_slug, dataset_slug, arrival_rate, "arrival_trace")
        arrival_rng = random.Random(arrival_seed)
        arrivals = poisson_offsets(int(args.num_requests), float(arrival_rate), arrival_rng)

    case_seed = stable_seed(
        args.seed,
        model_slug,
        dataset_slug,
        arrival_rate,
        "shared" if args.shared_workload_across_rates else "independent",
    )
    return case_seed, workload, arrivals


def apply_cluster_defaults(args) -> SimpleNamespace:
    defaults = {
        "prefill_resource": "prefill_gpu",
        "decode_dense_resource": "decode_dense_gpu",
        "attention_resource": "attention_pim",
        "attention_sparse_window": 0,
        "decode_dense_gpu_fraction": 1.0,
        "attention_gpu_fraction": 0.0,
        "pim_num_dpus": 128,
        "pim_resident_store_backend": "upmem_kvslot",
        "pim_length": 2048,
        "pim_block_tokens": 256,
        "pim_max_resident_groups_per_layer": 0,
        "pim_head_grouping_policy": "auto",
        "pim_dpu_placement_policy": "auto",
        "pim_resident_kv_dtype": "fp16",
        "pim_qk_full_enabled": False,
        "pim_qk_full_shadow_check": True,
        "pim_softmax_av_fused_enabled": False,
        "pim_softmax_av_shadow_check": True,
        "pim_qk_mixed_enabled": True,
        "pim_qk_mixed_heads": 2,
        "pim_qk_mixed_window": 128,
        "clover_cpu_shadow_enabled": False,
        "clover_shadow_checks_enabled": False,
        "clover_op_profiling_enabled": False,
        "clover_cpu_fast_path_max_context_tokens": 0,
        "clover_adaptive_routing_enabled": False,
        "clover_adaptive_route_compressed_kv_to_cpu": True,
        "clover_adaptive_route_sparse_window_max": 0,
        "clover_adaptive_route_context_len_max": 0,
        "clover_adaptive_probe_enabled": False,
        "clover_shadow_check_token_interval": 4,
        "clover_shadow_check_layer_interval": 4,
        "clover_host_qk_mixed_enabled": False,
        "clover_pim_context_fused_experimental_enabled": False,
        "clover_pim_qk_only_host_av_experimental_enabled": False,
        "clover_pim_rank_spread_alloc_experimental_enabled": False,
        "clover_pim_cross_rank_stripe_experimental_enabled": False,
        "clover_pim_rank_spread_multi_rank_batch_experimental_enabled": False,
        "clover_pim_layer_rank_rotation_experimental_enabled": False,
        "clover_pim_disjoint_decode_stripe_packing_enabled": False,
        "clover_pim_slot_spill_alloc_experimental_enabled": False,
        "clover_pim_slot_pressure_aware_alloc_experimental_enabled": False,
        "clover_pim_emergency_slot_spill_experimental_enabled": False,
        "clover_pim_reserve_segment_tail_capacity_experimental_enabled": False,
        "clover_pim_reserve_segment_tail_capacity_tokens": 0,
        "clover_pim_perf_guard_enabled": False,
        "clover_pim_perf_guard_force_cpu_for_compressed_kv": True,
        "clover_pim_perf_guard_min_decode_items": 1,
        "clover_pim_perf_guard_slowdown_threshold": 1.2,
        "clover_compact_short_segments_enabled": False,
        "clover_compact_short_segment_min_tokens": 8,
        "clover_fine_head_grouping_experimental_enabled": False,
        "clover_target_heads_per_group_experimental": 0,
        "clover_predictive_scheduling_enabled": False,
        "clover_predictive_scheduling_alpha": 0.2,
        "clover_predictive_scheduling_min_samples": 4,
        "clover_predictive_scheduling_context_bucket_tokens": 256,
        "clover_capacity_aware_batching_enabled": False,
        "clover_capacity_aware_time_gap_threshold": 0.0,
        "clover_capacity_aware_lookahead_window": 1,
        "clover_capacity_aware_pim_a": 1.0,
        "clover_capacity_aware_pim_b": 0.0,
        "clover_capacity_aware_host_c": 1.0,
        "clover_capacity_aware_max_tokens_per_dpu": 0,
        "clover_capacity_aware_require_slot_headroom": False,
        "clover_rankset_overlap_enabled": False,
        "clover_rankset_overlap_max_ranksets_per_batch": 0,
        "clover_rankset_overlap_transfer_granularity": "stripe",
        "clover_rankset_overlap_async_dispatch_enabled": False,
        "clover_rankset_overlap_transfer_latency_s": 0.0,
        "decode_continuous_batch_window_s": 0.0,
        "decode_continuous_batch_max_size": 8,
        "decode_continuous_batch_inflight_target_enabled": False,
        "decode_continuous_batch_startup_grace_s": 0.0,
    }
    merged = vars(args).copy()
    for key, value in defaults.items():
        merged.setdefault(key, value)
    merged["pim_resident_kv_dtype"] = normalize_resident_kv_dtype(merged["pim_resident_kv_dtype"])
    return SimpleNamespace(**merged)


def collect_ready(
    pending: Dict[ray.ObjectRef, Dict[str, object]],
    records: List[Dict[str, object]],
    completion_index: int,
    *,
    timeout_s: float | None,
) -> int:
    if not pending:
        return completion_index
    ready, _ = ray.wait(
        list(pending.keys()),
        num_returns=len(pending),
        timeout=timeout_s,
    )
    for future in ready:
        meta = pending.pop(future)
        finished_at = time.time()
        completion, metrics = ray.get(future)
        completion_index += 1
        output_tokens = int(metrics.get("total_tokens", meta["output_tokens"]))
        internal_latency = float(metrics.get("latency", 0.0))
        latency_from_arrival = max(0.0, finished_at - float(meta["arrival_at"]))
        records.append(
            {
                "request_index": int(meta["request_index"]),
                "completion_index": int(completion_index),
                "task_id": str(meta["task_id"]),
                "completion": completion,
                "metrics": metrics,
                "input_tokens": int(meta["input_tokens"]),
                "original_input_tokens": int(meta["original_input_tokens"]),
                "input_truncated": bool(meta["input_truncated"]),
                "requested_output_tokens": int(meta["output_tokens"]),
                "output_tokens": int(output_tokens),
                "output_length_source": str(meta["output_length_source"]),
                "output_truncated": bool(meta["output_truncated"]),
                "scheduled_arrival_offset_s": float(meta["scheduled_arrival_offset_s"]),
                "arrival_at": float(meta["arrival_at"]),
                "submit_finished_at": float(meta["submit_finished_at"]),
                "finished_at": float(finished_at),
                "latency_from_arrival_s": float(latency_from_arrival),
                "normalized_latency_s_per_output_token": float(latency_from_arrival / max(output_tokens, 1)),
                "normalized_internal_latency_s_per_output_token": float(
                    internal_latency / max(output_tokens, 1)
                ),
            }
        )
    return completion_index


def summarize_records(records: List[Dict[str, object]], wall_time_s: float, arrival_rate: float) -> Dict[str, object]:
    normalized = [float(record["normalized_latency_s_per_output_token"]) for record in records]
    normalized_internal = [
        float(record.get("normalized_internal_latency_s_per_output_token", 0.0))
        for record in records
    ]
    external_latency = [float(record["latency_from_arrival_s"]) for record in records]
    internal_latency = [float(record["metrics"].get("latency", 0.0)) for record in records]
    output_tokens = [int(record["output_tokens"]) for record in records]
    input_tokens = [int(record["input_tokens"]) for record in records]
    return {
        "arrival_rate_rps": float(arrival_rate),
        "num_requests": int(len(records)),
        "wall_time_s": float(wall_time_s),
        "achieved_request_throughput_rps": float(len(records) / wall_time_s) if wall_time_s > 0 else 0.0,
        "achieved_output_throughput_tps": (
            float(sum(output_tokens) / wall_time_s) if wall_time_s > 0 else 0.0
        ),
        "normalized_latency_mean": float(statistics.mean(normalized)) if normalized else 0.0,
        "normalized_latency_p50": percentile(normalized, 0.50),
        "normalized_latency_p90": percentile(normalized, 0.90),
        "normalized_latency_p95": percentile(normalized, 0.95),
        "normalized_internal_latency_mean": (
            float(statistics.mean(normalized_internal)) if normalized_internal else 0.0
        ),
        "normalized_internal_latency_p50": percentile(normalized_internal, 0.50),
        "normalized_internal_latency_p90": percentile(normalized_internal, 0.90),
        "normalized_internal_latency_p95": percentile(normalized_internal, 0.95),
        "latency_from_arrival_mean_s": float(statistics.mean(external_latency)) if external_latency else 0.0,
        "latency_internal_mean_s": float(statistics.mean(internal_latency)) if internal_latency else 0.0,
        "input_tokens_mean": float(statistics.mean(input_tokens)) if input_tokens else 0.0,
        "output_tokens_mean": float(statistics.mean(output_tokens)) if output_tokens else 0.0,
        "output_length_fallback_count": int(
            sum(1 for record in records if record["output_length_source"] == "fallback")
        ),
        "input_truncated_count": int(sum(1 for record in records if record["input_truncated"])),
        "output_truncated_count": int(sum(1 for record in records if record["output_truncated"])),
        "out_of_order_completions": int(
            sum(1 for record in records if int(record["request_index"]) != int(record["completion_index"]))
        ),
    }


def run_warmup(scheduler, workload: List[Dict[str, object]], warmup_requests: int) -> None:
    if warmup_requests <= 0:
        return
    for item in workload[:warmup_requests]:
        ray.get(
            scheduler.submit_request.remote(
                str(item["prompt"]),
                return_metrics=True,
                max_new_tokens=int(item["output_tokens"]),
            )
        )


def run_arrival_case(
    args,
    *,
    model_spec: Dict[str, str],
    dataset_spec: Dict[str, str],
    baseline_spec: Dict[str, object],
    warmup_workload: List[Dict[str, object]],
    workload: List[Dict[str, object]],
    arrivals: List[float],
    arrival_rate: float,
    case_seed: int,
) -> Dict[str, object]:
    internal_baseline = str(baseline_spec["internal_baseline"])
    attention_backend = DISAGG_BASELINES[internal_baseline]
    cluster_conf = make_cluster_config(args, attention_backend)
    model_conf = ModelConfig(
        model_name=str(model_spec["label"]),
        model_path=str(model_spec["path"]),
        max_seq_len=int(args.max_seq_len),
        max_new_tokens=max(int(item["output_tokens"]) for item in workload),
        dtype=str(args.dtype),
    )

    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    placement = {}
    records: List[Dict[str, object]] = []
    pending: Dict[ray.ObjectRef, Dict[str, object]] = {}
    completion_index = 0
    run_started_at = time.time()
    try:
        placement = ray.get(scheduler.initialize_cluster.remote())
        return run_arrival_case_on_scheduler(
            args,
            scheduler=scheduler,
            placement=placement,
            model_spec=model_spec,
            dataset_spec=dataset_spec,
            baseline_spec=baseline_spec,
            warmup_workload=warmup_workload,
            workload=workload,
            arrivals=arrivals,
            arrival_rate=arrival_rate,
            case_seed=case_seed,
        )
    finally:
        with suppress(Exception):
            ray.kill(scheduler, no_restart=True)


def run_arrival_case_on_scheduler(
    args,
    *,
    scheduler,
    placement: Dict[str, object],
    model_spec: Dict[str, str],
    dataset_spec: Dict[str, str],
    baseline_spec: Dict[str, object],
    warmup_workload: List[Dict[str, object]],
    workload: List[Dict[str, object]],
    arrivals: List[float],
    arrival_rate: float,
    case_seed: int,
) -> Dict[str, object]:
    internal_baseline = str(baseline_spec["internal_baseline"])
    records: List[Dict[str, object]] = []
    pending: Dict[ray.ObjectRef, Dict[str, object]] = {}
    completion_index = 0
    run_started_at = time.time()
    run_warmup(scheduler, warmup_workload, int(args.warmup_requests))
    run_started_at = time.time()

    for idx, (item, scheduled_offset) in enumerate(zip(workload, arrivals), start=1):
        target_time = run_started_at + float(scheduled_offset)
        sleep_s = target_time - time.time()
        if sleep_s > 0:
            time.sleep(sleep_s)
        completion_index = collect_ready(pending, records, completion_index, timeout_s=0.0)
        arrival_at = time.time()
        future = scheduler.submit_request.remote(
            str(item["prompt"]),
            return_metrics=True,
            max_new_tokens=int(item["output_tokens"]),
        )
        pending[future] = {
            **item,
            "request_index": int(idx),
            "scheduled_arrival_offset_s": float(scheduled_offset),
            "arrival_at": float(arrival_at),
            "submit_finished_at": float(time.time()),
        }

    while pending:
        completion_index = collect_ready(pending, records, completion_index, timeout_s=None)

    wall_time_s = max(time.time() - run_started_at, 1e-12)
    records.sort(key=lambda record: int(record["request_index"]))
    summary = summarize_records(records, wall_time_s, arrival_rate)
    summary.update(
        {
            "model": str(model_spec["label"]),
            "model_slug": str(model_spec["slug"]),
            "dataset": str(dataset_spec["label"]),
            "dataset_slug": str(dataset_spec["slug"]),
            "baseline": str(baseline_spec["canonical_name"]),
            "baseline_key": str(baseline_spec["canonical_key"]),
            "internal_baseline": internal_baseline,
            "case_seed": int(case_seed),
        }
    )
    return {
        "summary": summary,
        "records": records,
        "placement": placement,
        "data_config": {
            "model": model_spec,
            "dataset": dataset_spec,
            "arrival_rate_rps": float(arrival_rate),
            "num_requests": int(args.num_requests),
            "warmup_requests": int(args.warmup_requests),
            "seed": int(case_seed),
            "max_input_tokens": int(args.max_input_tokens),
            "max_output_tokens": int(args.max_output_tokens),
            "min_output_tokens": int(args.min_output_tokens),
            "fallback_output_tokens": int(args.fallback_output_tokens),
            "max_seq_len": int(args.max_seq_len),
            "dtype": str(args.dtype),
            "pim_num_dpus": int(args.pim_num_dpus),
            "pim_length": int(args.pim_length),
            "pim_resident_kv_dtype": str(args.pim_resident_kv_dtype),
            "decode_continuous_batch_window_s": float(args.decode_continuous_batch_window_s),
            "decode_continuous_batch_max_size": int(args.decode_continuous_batch_max_size),
        },
    }


def write_jsonl(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def iter_case_inputs(args) -> Iterable[tuple[Dict[str, str], Dict[str, str], float]]:
    for model_spec in args.models:
        for dataset_spec in args.datasets:
            for arrival_rate in args.arrival_rates:
                yield model_spec, dataset_spec, float(arrival_rate)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poisson arrival-rate benchmark for AFD baselines.")
    parser.add_argument(
        "--models",
        type=parse_model_specs,
        required=True,
        help="Comma-separated model specs, e.g. qwen-1.8b=model/Qwen-1_8B,opt-1.3b=model/opt-1.3b",
    )
    parser.add_argument(
        "--datasets",
        type=parse_dataset_specs,
        required=True,
        help="Comma-separated dataset specs, optionally label=path:format.",
    )
    parser.add_argument(
        "--baselines",
        default="AFD,CPU-Attention,Naive PIM",
        help="Arrival-rate runner supports disaggregated AFD, CPU-Attention, and Naive PIM.",
    )
    parser.add_argument("--arrival-rates", type=parse_float_csv_list, default=parse_float_csv_list("0.02,0.05,0.1,0.2"))
    parser.add_argument("--num-requests", type=int, default=12)
    parser.add_argument("--warmup-requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=1024)
    parser.add_argument("--max-output-tokens", type=int, default=32)
    parser.add_argument("--min-output-tokens", type=int, default=1)
    parser.add_argument("--fallback-output-tokens", type=int, default=16)
    parser.add_argument("--dataset-load-limit", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "artifacts", "arrival_rate_experiment"))
    parser.add_argument("--prefill-resource", default="prefill_gpu")
    parser.add_argument("--decode-dense-resource", default="decode_dense_gpu")
    parser.add_argument("--attention-resource", default="attention_pim")
    parser.add_argument("--decode-dense-gpu-fraction", type=float, default=1.0)
    parser.add_argument("--attention-gpu-fraction", type=float, default=0.0)
    parser.add_argument("--attention-sparse-window", type=int, default=0)
    parser.add_argument("--pim-num-dpus", type=int, default=128)
    parser.add_argument("--pim-length", type=int, default=2048)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument("--pim-max-resident-groups-per-layer", type=int, default=0)
    parser.add_argument("--pim-head-grouping-policy", default="auto")
    parser.add_argument("--pim-dpu-placement-policy", default="auto")
    parser.add_argument(
        "--pim-resident-store-backend",
        default="upmem_kvslot",
        choices=["auto", "host", "upmem_kvslot"],
    )
    parser.add_argument(
        "--pim-resident-kv-dtype",
        default="fp16",
        choices=sorted(
            SUPPORTED_RESIDENT_KV_DTYPES
            | {"int8_fp16", "int8-fp16", "k_int8_v_fp16", "int8_int16", "int8-int16", "k_int8_v_int16"}
        ),
    )
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--decode-continuous-batch-window-s", type=float, default=0.0)
    parser.add_argument("--decode-continuous-batch-max-size", type=int, default=8)
    parser.add_argument("--attention-actor-batch-window-s", type=float, default=0.001)
    parser.add_argument("--attention-actor-batch-max-size", type=int, default=8)
    parser.add_argument(
        "--fresh-scheduler-per-case",
        action="store_true",
        help="Instantiate a fresh scheduler for every model/dataset/rate case.",
    )
    parser.add_argument(
        "--shared-workload-across-rates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the same sampled requests for all arrival rates within a model/dataset.",
    )
    parser.add_argument(
        "--shared-arrival-trace-across-rates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse one unit-rate Poisson trace and scale it by request rate.",
    )
    args = parser.parse_args()

    if args.num_requests <= 0:
        raise ValueError("--num-requests must be positive")
    if args.warmup_requests < 0:
        raise ValueError("--warmup-requests must be non-negative")
    if args.min_output_tokens <= 0:
        raise ValueError("--min-output-tokens must be positive")
    if args.max_input_tokens < 0 or args.max_output_tokens < 0:
        raise ValueError("token caps must be non-negative")
    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")

    args = apply_cluster_defaults(args)
    if args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = False
    elif args.pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = True
    else:
        args.pim_qk_mixed_enabled = True

    baseline_specs = resolve_requested_baselines(args.baselines)
    unsupported = [
        str(spec["internal_baseline"])
        for spec in baseline_specs
        if str(spec["internal_baseline"]) not in DISAGG_BASELINES
    ]
    if unsupported:
        raise ValueError(f"unsupported baselines for arrival-rate runner: {unsupported}")

    output_dir = Path(args.output_dir)
    result_jsonl = output_dir / "arrival_rate_results.jsonl"
    summary_csv = output_dir / "arrival_rate_summary.csv"
    if result_jsonl.exists():
        result_jsonl.unlink()

    ray.init(address=args.address, ignore_reinit_error=True, runtime_env=build_runtime_env())
    summary_rows: List[Dict[str, object]] = []

    def record_success(result: Dict[str, object]) -> None:
        result["status"] = "ok"
        print(json.dumps(result["summary"], ensure_ascii=False), flush=True)
        write_jsonl(result_jsonl, result)
        summary_rows.append(dict(result["summary"], status="ok", error=""))
        write_summary_csv(summary_csv, summary_rows)

    def record_failure(
        exc: Exception,
        *,
        model_spec: Dict[str, str],
        dataset_spec: Dict[str, str],
        baseline_spec: Dict[str, object],
        arrival_rate: float,
    ) -> None:
        failure = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "summary": {
                "model": model_spec["label"],
                "model_slug": model_spec["slug"],
                "dataset": dataset_spec["label"],
                "dataset_slug": dataset_spec["slug"],
                "arrival_rate_rps": float(arrival_rate),
                "baseline": str(baseline_spec["canonical_name"]),
                "baseline_key": str(baseline_spec["canonical_key"]),
                "internal_baseline": str(baseline_spec["internal_baseline"]),
            },
        }
        print(json.dumps(failure, ensure_ascii=False), flush=True)
        write_jsonl(result_jsonl, failure)
        summary_rows.append(dict(failure["summary"], status="failed", error=failure["error"]))
        write_summary_csv(summary_csv, summary_rows)

    try:
        for model_spec in args.models:
            tokenizer = load_tokenizer_for_benchmark(model_spec["path"])
            work_items_by_dataset: Dict[str, List[Dict[str, object]]] = {}
            max_case_output_tokens = 1
            for dataset_spec in args.datasets:
                samples = load_benchmark_samples(
                    dataset_spec["path"],
                    dataset_format=dataset_spec["format"],
                    limit=args.dataset_load_limit,
                    tokenizer=None,
                    prompt_token_length=0,
                )
                work_items = build_work_items(
                    samples,
                    tokenizer,
                    max_input_tokens=int(args.max_input_tokens),
                    max_output_tokens=int(args.max_output_tokens),
                    min_output_tokens=int(args.min_output_tokens),
                    fallback_output_tokens=int(args.fallback_output_tokens),
                )
                work_items_by_dataset[dataset_spec["slug"]] = work_items
                max_case_output_tokens = max(
                    max_case_output_tokens,
                    max(int(item["output_tokens"]) for item in work_items),
                )

            for baseline_spec in baseline_specs:
                if args.fresh_scheduler_per_case:
                    for dataset_spec in args.datasets:
                        work_items = work_items_by_dataset[dataset_spec["slug"]]
                        for arrival_rate in args.arrival_rates:
                            case_seed, workload, arrivals = build_case_workload_and_arrivals(
                                args,
                                work_items=work_items,
                                model_slug=str(model_spec["slug"]),
                                dataset_slug=str(dataset_spec["slug"]),
                                arrival_rate=float(arrival_rate),
                            )
                            measured_workload = workload[int(args.warmup_requests) :]
                            case_label = (
                                f"{model_spec['slug']} {dataset_spec['slug']} "
                                f"rate={arrival_rate:g} baseline={baseline_spec['canonical_name']}"
                            )
                            print(f"RUN {case_label}", flush=True)
                            try:
                                result = run_arrival_case(
                                    args,
                                    model_spec=model_spec,
                                    dataset_spec=dataset_spec,
                                    baseline_spec=baseline_spec,
                                    warmup_workload=workload[: int(args.warmup_requests)],
                                    workload=measured_workload,
                                    arrivals=arrivals,
                                    arrival_rate=float(arrival_rate),
                                    case_seed=case_seed,
                                )
                                record_success(result)
                            except Exception as exc:
                                record_failure(
                                    exc,
                                    model_spec=model_spec,
                                    dataset_spec=dataset_spec,
                                    baseline_spec=baseline_spec,
                                    arrival_rate=float(arrival_rate),
                                )
                    continue

                internal_baseline = str(baseline_spec["internal_baseline"])
                attention_backend = DISAGG_BASELINES[internal_baseline]
                cluster_conf = make_cluster_config(args, attention_backend)
                model_conf = ModelConfig(
                    model_name=str(model_spec["label"]),
                    model_path=str(model_spec["path"]),
                    max_seq_len=int(args.max_seq_len),
                    max_new_tokens=int(max_case_output_tokens),
                    dtype=str(args.dtype),
                )
                scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
                placement = {}
                group_failed = False
                try:
                    placement = ray.get(scheduler.initialize_cluster.remote())
                    for dataset_spec in args.datasets:
                        if group_failed:
                            break
                        work_items = work_items_by_dataset[dataset_spec["slug"]]
                        for arrival_rate in args.arrival_rates:
                            case_seed, workload, arrivals = build_case_workload_and_arrivals(
                                args,
                                work_items=work_items,
                                model_slug=str(model_spec["slug"]),
                                dataset_slug=str(dataset_spec["slug"]),
                                arrival_rate=float(arrival_rate),
                            )
                            measured_workload = workload[int(args.warmup_requests) :]
                            case_label = (
                                f"{model_spec['slug']} {dataset_spec['slug']} "
                                f"rate={arrival_rate:g} baseline={baseline_spec['canonical_name']}"
                            )
                            print(f"RUN {case_label}", flush=True)
                            try:
                                result = run_arrival_case_on_scheduler(
                                    args,
                                    scheduler=scheduler,
                                    placement=placement,
                                    model_spec=model_spec,
                                    dataset_spec=dataset_spec,
                                    baseline_spec=baseline_spec,
                                    warmup_workload=workload[: int(args.warmup_requests)],
                                    workload=measured_workload,
                                    arrivals=arrivals,
                                    arrival_rate=float(arrival_rate),
                                    case_seed=case_seed,
                                )
                                record_success(result)
                            except Exception as exc:
                                record_failure(
                                    exc,
                                    model_spec=model_spec,
                                    dataset_spec=dataset_spec,
                                    baseline_spec=baseline_spec,
                                    arrival_rate=float(arrival_rate),
                                )
                                group_failed = True
                                break
                except Exception as exc:
                    for dataset_spec in args.datasets:
                        for arrival_rate in args.arrival_rates:
                            record_failure(
                                exc,
                                model_spec=model_spec,
                                dataset_spec=dataset_spec,
                                baseline_spec=baseline_spec,
                                arrival_rate=float(arrival_rate),
                            )
                finally:
                    with suppress(Exception):
                        ray.kill(scheduler, no_restart=True)
    finally:
        with suppress(Exception):
            ray.shutdown()

    print(f"Saved JSONL results to {result_jsonl}")
    print(f"Saved CSV summary to {summary_csv}")


if __name__ == "__main__":
    main()
