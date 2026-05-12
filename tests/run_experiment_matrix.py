from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Dict, Iterable, List

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TESTS_ROOT = os.path.dirname(os.path.abspath(__file__))
for path in (REPO_ROOT, TESTS_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from benchmark_baselines import BASELINE_HELP_TEXT, resolve_requested_baselines
from benchmark_utils import DATASET_FORMAT_CHOICES


def parse_csv_list(value: str) -> List[str]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(part)
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return items


def parse_int_csv_list(value: str) -> List[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return items


def _infer_label_from_path(path: str) -> str:
    normalized = os.path.normpath(path)
    name = os.path.basename(normalized.rstrip(os.sep))
    stem, _ = os.path.splitext(name)
    return stem or name


def slugify(value: str) -> str:
    chars = []
    previous_dash = False
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
            previous_dash = False
        else:
            if not previous_dash:
                chars.append("-")
                previous_dash = True
    slug = "".join(chars).strip("-")
    return slug or "item"


def parse_model_specs(value: str) -> List[Dict[str, str]]:
    specs = []
    for raw_item in parse_csv_list(value):
        if "=" in raw_item:
            label, path = raw_item.split("=", 1)
            label = label.strip()
            path = path.strip()
        else:
            path = raw_item.strip()
            label = _infer_label_from_path(path)
        if not label or not path:
            raise argparse.ArgumentTypeError(f"invalid model spec: {raw_item}")
        specs.append(
            {
                "label": label,
                "slug": slugify(label),
                "path": path,
            }
        )
    return specs


def parse_dataset_specs(value: str) -> List[Dict[str, str]]:
    specs = []
    valid_formats = set(DATASET_FORMAT_CHOICES)
    for raw_item in parse_csv_list(value):
        if "=" in raw_item:
            label, remainder = raw_item.split("=", 1)
            label = label.strip()
            remainder = remainder.strip()
        else:
            remainder = raw_item.strip()
            label = _infer_label_from_path(remainder)

        dataset_format = "auto"
        if ":" in remainder:
            maybe_path, maybe_format = remainder.rsplit(":", 1)
            if maybe_format in valid_formats:
                remainder = maybe_path
                dataset_format = maybe_format

        if not label or not remainder:
            raise argparse.ArgumentTypeError(f"invalid dataset spec: {raw_item}")
        specs.append(
            {
                "label": label,
                "slug": slugify(label),
                "path": remainder,
                "format": dataset_format,
            }
        )
    return specs


def append_optional_flag(cmd: List[str], flag: str, enabled: bool) -> None:
    if enabled:
        cmd.append(flag)


def append_key_value(cmd: List[str], key: str, value) -> None:
    cmd.extend([key, str(value)])


def iter_case_configs(args) -> Iterable[Dict[str, object]]:
    for model_spec, dataset_spec, prompt_tokens, output_tokens, concurrency in itertools.product(
        args.model_specs,
        args.dataset_specs,
        args.prompt_token_lengths,
        args.output_token_lengths,
        args.concurrency_values,
    ):
        case_slug = "__".join(
            [
                model_spec["slug"],
                dataset_spec["slug"],
                f"p{int(prompt_tokens)}" if int(prompt_tokens) > 0 else "pnative",
                f"o{int(output_tokens)}",
                f"c{int(concurrency)}",
            ]
        )
        yield {
            "case_slug": case_slug,
            "model_label": model_spec["label"],
            "model_slug": model_spec["slug"],
            "model_path": model_spec["path"],
            "dataset_label": dataset_spec["label"],
            "dataset_slug": dataset_spec["slug"],
            "dataset_path": dataset_spec["path"],
            "dataset_format": dataset_spec["format"],
            "prompt_token_length": int(prompt_tokens),
            "max_new_tokens": int(output_tokens),
            "concurrency": int(concurrency),
        }


def build_case_command(args, case: Dict[str, object], output_path: str) -> List[str]:
    benchmark_script = os.path.join(TESTS_ROOT, "benchmark_baselines.py")
    cmd = [sys.executable, benchmark_script]
    append_key_value(cmd, "--data", case["dataset_path"])
    append_key_value(cmd, "--dataset-format", case["dataset_format"])
    append_key_value(cmd, "--limit", args.limit)
    append_key_value(cmd, "--model", case["model_path"])
    append_key_value(cmd, "--model-name", case["model_label"])
    append_key_value(cmd, "--max-new-tokens", case["max_new_tokens"])
    append_key_value(cmd, "--prompt-token-length", case["prompt_token_length"])
    append_key_value(cmd, "--concurrency", case["concurrency"])
    append_key_value(cmd, "--dtype", args.dtype)
    append_key_value(cmd, "--baselines", args.baselines)
    append_key_value(cmd, "--address", args.address)
    append_key_value(cmd, "--prefill-resource", args.prefill_resource)
    append_key_value(cmd, "--decode-dense-resource", args.decode_dense_resource)
    append_key_value(cmd, "--attention-resource", args.attention_resource)
    append_key_value(cmd, "--pim-num-dpus", args.pim_num_dpus)
    append_key_value(cmd, "--pim-resident-store-backend", args.pim_resident_store_backend)
    append_key_value(cmd, "--pim-length", args.pim_length)
    append_key_value(cmd, "--pim-block-tokens", args.pim_block_tokens)
    append_key_value(cmd, "--pim-max-resident-groups-per-layer", args.pim_max_resident_groups_per_layer)
    append_key_value(cmd, "--pim-head-grouping-policy", args.pim_head_grouping_policy)
    append_key_value(cmd, "--pim-dpu-placement-policy", args.pim_dpu_placement_policy)
    append_key_value(cmd, "--pim-resident-kv-dtype", args.pim_resident_kv_dtype)
    append_key_value(cmd, "--pim-qk-mixed-heads", args.pim_qk_mixed_heads)
    append_key_value(cmd, "--pim-qk-mixed-window", args.pim_qk_mixed_window)
    append_key_value(cmd, "--clover-cpu-fast-path-max-context-tokens", args.clover_cpu_fast_path_max_context_tokens)
    append_key_value(cmd, "--clover-shadow-check-token-interval", args.clover_shadow_check_token_interval)
    append_key_value(cmd, "--clover-shadow-check-layer-interval", args.clover_shadow_check_layer_interval)
    append_key_value(cmd, "--clover-pim-perf-guard-min-decode-items", args.clover_pim_perf_guard_min_decode_items)
    append_key_value(
        cmd,
        "--clover-pim-perf-guard-slowdown-threshold",
        args.clover_pim_perf_guard_slowdown_threshold,
    )
    append_key_value(cmd, "--output", output_path)

    append_optional_flag(cmd, "--pim-qk-full-enabled", args.pim_qk_full_enabled)
    append_optional_flag(cmd, "--no-pim-qk-full-enabled", args.no_pim_qk_full_enabled)
    append_optional_flag(cmd, "--pim-qk-full-shadow-check", args.pim_qk_full_shadow_check)
    append_optional_flag(cmd, "--no-pim-qk-full-shadow-check", args.no_pim_qk_full_shadow_check)
    append_optional_flag(cmd, "--pim-softmax-av-fused-enabled", args.pim_softmax_av_fused_enabled)
    append_optional_flag(cmd, "--no-pim-softmax-av-fused-enabled", args.no_pim_softmax_av_fused_enabled)
    append_optional_flag(cmd, "--pim-softmax-av-shadow-check", args.pim_softmax_av_shadow_check)
    append_optional_flag(cmd, "--no-pim-softmax-av-shadow-check", args.no_pim_softmax_av_shadow_check)
    append_optional_flag(cmd, "--pim-qk-mixed-enabled", args.pim_qk_mixed_enabled)
    append_optional_flag(cmd, "--no-pim-qk-mixed-enabled", args.no_pim_qk_mixed_enabled)
    append_optional_flag(cmd, "--clover-cpu-shadow-enabled", args.clover_cpu_shadow_enabled)
    append_optional_flag(cmd, "--no-clover-cpu-shadow-enabled", args.no_clover_cpu_shadow_enabled)
    append_optional_flag(cmd, "--clover-shadow-checks-enabled", args.clover_shadow_checks_enabled)
    append_optional_flag(cmd, "--no-clover-shadow-checks-enabled", args.no_clover_shadow_checks_enabled)
    append_optional_flag(cmd, "--clover-op-profiling-enabled", args.clover_op_profiling_enabled)
    append_optional_flag(cmd, "--no-clover-op-profiling-enabled", args.no_clover_op_profiling_enabled)
    append_optional_flag(cmd, "--clover-host-qk-mixed-enabled", args.clover_host_qk_mixed_enabled)
    append_optional_flag(cmd, "--no-clover-host-qk-mixed-enabled", args.no_clover_host_qk_mixed_enabled)
    append_optional_flag(
        cmd,
        "--clover-pim-rank-spread-alloc-experimental-enabled",
        args.clover_pim_rank_spread_alloc_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-rank-spread-alloc-experimental-enabled",
        args.no_clover_pim_rank_spread_alloc_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--clover-pim-cross-rank-stripe-experimental-enabled",
        args.clover_pim_cross_rank_stripe_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-cross-rank-stripe-experimental-enabled",
        args.no_clover_pim_cross_rank_stripe_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--clover-pim-rank-spread-multi-rank-batch-experimental-enabled",
        args.clover_pim_rank_spread_multi_rank_batch_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-rank-spread-multi-rank-batch-experimental-enabled",
        args.no_clover_pim_rank_spread_multi_rank_batch_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--clover-pim-layer-rank-rotation-experimental-enabled",
        args.clover_pim_layer_rank_rotation_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-layer-rank-rotation-experimental-enabled",
        args.no_clover_pim_layer_rank_rotation_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--clover-pim-context-fused-experimental-enabled",
        args.clover_pim_context_fused_experimental_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-context-fused-experimental-enabled",
        args.no_clover_pim_context_fused_experimental_enabled,
    )
    append_optional_flag(cmd, "--clover-pim-perf-guard-enabled", args.clover_pim_perf_guard_enabled)
    append_optional_flag(cmd, "--no-clover-pim-perf-guard-enabled", args.no_clover_pim_perf_guard_enabled)
    append_optional_flag(
        cmd,
        "--clover-pim-perf-guard-force-cpu-for-compressed-kv",
        args.clover_pim_perf_guard_force_cpu_for_compressed_kv,
    )
    append_optional_flag(
        cmd,
        "--no-clover-pim-perf-guard-force-cpu-for-compressed-kv",
        args.no_clover_pim_perf_guard_force_cpu_for_compressed_kv,
    )
    append_optional_flag(cmd, "--clover-predictive-scheduling-enabled", args.clover_predictive_scheduling_enabled)
    append_optional_flag(
        cmd,
        "--no-clover-predictive-scheduling-enabled",
        args.no_clover_predictive_scheduling_enabled,
    )
    append_key_value(cmd, "--clover-predictive-scheduling-alpha", args.clover_predictive_scheduling_alpha)
    append_key_value(
        cmd,
        "--clover-predictive-scheduling-min-samples",
        args.clover_predictive_scheduling_min_samples,
    )
    append_key_value(
        cmd,
        "--clover-predictive-scheduling-context-bucket-tokens",
        args.clover_predictive_scheduling_context_bucket_tokens,
    )
    append_optional_flag(cmd, "--clover-rankset-overlap-enabled", args.clover_rankset_overlap_enabled)
    append_optional_flag(
        cmd,
        "--no-clover-rankset-overlap-enabled",
        args.no_clover_rankset_overlap_enabled,
    )
    append_key_value(
        cmd,
        "--clover-rankset-overlap-max-ranksets-per-batch",
        args.clover_rankset_overlap_max_ranksets_per_batch,
    )
    append_key_value(
        cmd,
        "--clover-rankset-overlap-transfer-granularity",
        args.clover_rankset_overlap_transfer_granularity,
    )
    append_optional_flag(
        cmd,
        "--clover-rankset-overlap-async-dispatch-enabled",
        args.clover_rankset_overlap_async_dispatch_enabled,
    )
    append_optional_flag(
        cmd,
        "--no-clover-rankset-overlap-async-dispatch-enabled",
        args.no_clover_rankset_overlap_async_dispatch_enabled,
    )
    append_key_value(
        cmd,
        "--clover-rankset-overlap-transfer-latency-s",
        args.clover_rankset_overlap_transfer_latency_s,
    )
    append_key_value(cmd, "--decode-continuous-batch-window-s", args.decode_continuous_batch_window_s)
    append_key_value(cmd, "--decode-continuous-batch-window-ms", args.decode_continuous_batch_window_ms)
    append_key_value(cmd, "--decode-continuous-batch-max-size", args.decode_continuous_batch_max_size)
    return cmd


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_case_summary_rows(output_path: str, case: Dict[str, object]) -> List[Dict[str, object]]:
    rows = read_jsonl(output_path)
    summary_rows = []
    for row in rows:
        summary = dict(row.get("summary", {}))
        if not summary:
            continue
        summary_rows.append(
            {
                "case_slug": case["case_slug"],
                "model_label": case["model_label"],
                "model_path": case["model_path"],
                "dataset_label": case["dataset_label"],
                "dataset_path": case["dataset_path"],
                "dataset_format": case["dataset_format"],
                "prompt_token_length": int(case["prompt_token_length"]),
                "max_new_tokens": int(case["max_new_tokens"]),
                "concurrency": int(case["concurrency"]),
                "result_path": output_path,
                **summary,
            }
        )
    return summary_rows


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        default="opt-125m=/home/cml/CloverInfer/model/opt-125m,qwen-1_8b=/home/cml/CloverInfer/model/Qwen-1_8B",
        help="Comma-separated model specs. Each item is label=path or path.",
    )
    parser.add_argument(
        "--datasets",
        default="humaneval=dataset/humaneval.jsonl:auto",
        help=(
            "Comma-separated dataset specs. Each item is label=path[:format] or path[:format]. "
            f"Supported formats: {', '.join(DATASET_FORMAT_CHOICES)}"
        ),
    )
    parser.add_argument("--prompt-token-lengths", type=parse_int_csv_list, default=[0])
    parser.add_argument("--output-token-lengths", type=parse_int_csv_list, default=[8])
    parser.add_argument("--concurrency-values", type=parse_int_csv_list, default=[1])
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--baselines", default="PD,AFD,CPU-Attention,Naive PIM,CloverInfer", help=BASELINE_HELP_TEXT)
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--prefill-resource", default="prefill_gpu")
    parser.add_argument("--decode-dense-resource", default="decode_dense_gpu")
    parser.add_argument("--attention-resource", default="attention_pim")
    parser.add_argument("--prefill-ip", default="192.168.123.4")
    parser.add_argument("--decode-dense-ip", default="192.168.123.3")
    parser.add_argument("--attention-ip", default="192.168.123.7")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-resident-store-backend", default="auto", choices=["auto", "host", "upmem_kvslot"])
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
    parser.add_argument("--clover-pim-rank-spread-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-rank-spread-alloc-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-cross-rank-stripe-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-cross-rank-stripe-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-rank-spread-multi-rank-batch-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-rank-spread-multi-rank-batch-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-layer-rank-rotation-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-layer-rank-rotation-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--clover-pim-perf-guard-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-perf-guard-enabled", action="store_true")
    parser.add_argument("--clover-pim-perf-guard-force-cpu-for-compressed-kv", action="store_true")
    parser.add_argument("--no-clover-pim-perf-guard-force-cpu-for-compressed-kv", action="store_true")
    parser.add_argument("--clover-pim-perf-guard-min-decode-items", type=int, default=1)
    parser.add_argument("--clover-pim-perf-guard-slowdown-threshold", type=float, default=1.2)
    parser.add_argument("--clover-predictive-scheduling-enabled", action="store_true")
    parser.add_argument("--no-clover-predictive-scheduling-enabled", action="store_true")
    parser.add_argument("--clover-predictive-scheduling-alpha", type=float, default=0.2)
    parser.add_argument("--clover-predictive-scheduling-min-samples", type=int, default=4)
    parser.add_argument("--clover-predictive-scheduling-context-bucket-tokens", type=int, default=256)
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
    parser.add_argument("--output-dir", default=os.path.join(REPO_ROOT, "artifacts", "experiment_matrix"))
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.model_specs = parse_model_specs(args.models)
    args.dataset_specs = parse_dataset_specs(args.datasets)
    baseline_specs = resolve_requested_baselines(args.baselines)

    os.makedirs(args.output_dir, exist_ok=True)
    cases_dir = os.path.join(args.output_dir, "cases")
    os.makedirs(cases_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "matrix_summary.jsonl")
    manifest_path = os.path.join(args.output_dir, "matrix_manifest.jsonl")
    meta_path = os.path.join(args.output_dir, "matrix_meta.json")

    case_configs = list(iter_case_configs(args))
    meta = {
        "created_at": time.time(),
        "output_dir": args.output_dir,
        "address": args.address,
        "cluster_layout": {
            "prefill_ip": args.prefill_ip,
            "decode_dense_ip": args.decode_dense_ip,
            "attention_ip": args.attention_ip,
            "prefill_resource": args.prefill_resource,
            "decode_dense_resource": args.decode_dense_resource,
            "attention_resource": args.attention_resource,
        },
        "models": args.model_specs,
        "datasets": args.dataset_specs,
        "prompt_token_lengths": [int(item) for item in args.prompt_token_lengths],
        "output_token_lengths": [int(item) for item in args.output_token_lengths],
        "concurrency_values": [int(item) for item in args.concurrency_values],
        "limit": int(args.limit),
        "dtype": args.dtype,
        "baselines": args.baselines,
        "clover_rankset_overlap_enabled": bool(args.clover_rankset_overlap_enabled),
        "clover_rankset_overlap_max_ranksets_per_batch": int(args.clover_rankset_overlap_max_ranksets_per_batch),
        "clover_rankset_overlap_transfer_granularity": str(args.clover_rankset_overlap_transfer_granularity),
        "clover_rankset_overlap_async_dispatch_enabled": bool(
            args.clover_rankset_overlap_async_dispatch_enabled
        ),
        "clover_rankset_overlap_transfer_latency_s": float(args.clover_rankset_overlap_transfer_latency_s),
        "decode_continuous_batch_window_s": float(args.decode_continuous_batch_window_s),
        "decode_continuous_batch_window_ms": float(args.decode_continuous_batch_window_ms),
        "decode_continuous_batch_max_size": int(args.decode_continuous_batch_max_size),
        "resolved_baselines": baseline_specs,
        "num_cases": len(case_configs),
        "dry_run": bool(args.dry_run),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    if args.dry_run:
        for case in case_configs:
            print(json.dumps(case, ensure_ascii=False))
        print(f"Planned {len(case_configs)} cases. Metadata written to {meta_path}")
        return

    open(summary_path, "w", encoding="utf-8").close()
    open(manifest_path, "w", encoding="utf-8").close()

    for case_index, case in enumerate(case_configs, start=1):
        case_dir = os.path.join(cases_dir, case["case_slug"])
        os.makedirs(case_dir, exist_ok=True)
        result_path = os.path.join(case_dir, "results.jsonl")
        stdout_path = os.path.join(case_dir, "stdout.log")
        stderr_path = os.path.join(case_dir, "stderr.log")
        command_path = os.path.join(case_dir, "command.json")
        case_meta_path = os.path.join(case_dir, "case_meta.json")

        cmd = build_case_command(args, case, result_path)
        with open(command_path, "w", encoding="utf-8") as f:
            json.dump(cmd, f, indent=2, ensure_ascii=False)
        with open(case_meta_path, "w", encoding="utf-8") as f:
            json.dump({"case_index": case_index, "case": case}, f, indent=2, ensure_ascii=False)

        if args.skip_existing and os.path.exists(result_path):
            summary_rows = load_case_summary_rows(result_path, case)
            with open(summary_path, "a", encoding="utf-8") as f:
                for row in summary_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "case_index": case_index,
                            "case_slug": case["case_slug"],
                            "status": "skipped_existing",
                            "result_path": result_path,
                            "num_summaries": len(summary_rows),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            print(json.dumps({"case_index": case_index, "case_slug": case["case_slug"], "status": "skipped_existing"}))
            continue

        started_at = time.time()
        completed = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        finished_at = time.time()

        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(completed.stdout)
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(completed.stderr)

        manifest_entry = {
            "case_index": case_index,
            "case_slug": case["case_slug"],
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_s": float(finished_at - started_at),
            "returncode": int(completed.returncode),
            "result_path": result_path,
            "stdout_path": stdout_path,
            "stderr_path": stderr_path,
            "case": case,
        }

        if completed.returncode != 0:
            manifest_entry["status"] = "failed"
            failure_summary = {
                "case_slug": case["case_slug"],
                "model_label": case["model_label"],
                "model_path": case["model_path"],
                "dataset_label": case["dataset_label"],
                "dataset_path": case["dataset_path"],
                "dataset_format": case["dataset_format"],
                "prompt_token_length": int(case["prompt_token_length"]),
                "max_new_tokens": int(case["max_new_tokens"]),
                "concurrency": int(case["concurrency"]),
                "status": "failed",
                "returncode": int(completed.returncode),
                "result_path": result_path,
                "stdout_path": stdout_path,
                "stderr_path": stderr_path,
            }
            with open(summary_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(failure_summary, ensure_ascii=False) + "\n")
            with open(manifest_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
            print(json.dumps({"case_index": case_index, "case_slug": case["case_slug"], "status": "failed"}))
            if not args.continue_on_error:
                raise RuntimeError(
                    f"Case {case['case_slug']} failed with return code {completed.returncode}. "
                    f"See {stderr_path}"
                )
            continue

        summary_rows = load_case_summary_rows(result_path, case)
        with open(summary_path, "a", encoding="utf-8") as f:
            for row in summary_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        manifest_entry["status"] = "completed"
        manifest_entry["num_summaries"] = len(summary_rows)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")

        print(
            json.dumps(
                {
                    "case_index": case_index,
                    "case_slug": case["case_slug"],
                    "status": "completed",
                    "num_summaries": len(summary_rows),
                },
                ensure_ascii=False,
            )
        )

    print(f"Saved matrix metadata to {meta_path}")
    print(f"Saved matrix manifest to {manifest_path}")
    print(f"Saved matrix summary to {summary_path}")


if __name__ == "__main__":
    main()
