from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = REPO_ROOT / "tests"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TESTS_ROOT) not in sys.path:
    sys.path.insert(0, str(TESTS_ROOT))

from benchmark_utils import build_prompt_for_target_tokens, encode_prompt, load_benchmark_samples, load_tokenizer_for_benchmark


WORKLOAD_SPECS: Dict[str, Dict[str, str]] = {
    "humaneval": {
        "path": "dataset/humaneval.jsonl",
        "format": "humaneval_jsonl",
    },
    "sharegpt": {
        "path": "dataset/sharegpt_processed.json",
        "format": "sharegpt_json",
    },
    "qasper": {
        "path": "dataset/longbench/qasper.jsonl",
        "format": "longbench_jsonl",
    },
    "gov_report": {
        "path": "dataset/longbench/gov_report.jsonl",
        "format": "longbench_jsonl",
    },
}
SYNTHETIC_IMBALANCE_WORKLOADS = {"mixed_synthetic", "synthetic_mixed", "mixed"}


def _parse_csv(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> List[int]:
    return [int(item) for item in _parse_csv(value)]


def _parse_prompt_length_csv(value: str) -> List[int]:
    lengths: List[int] = []
    for item in _parse_csv(value):
        lowered = item.lower()
        if lowered in {"native", "dataset", "raw", "none", "0"}:
            lengths.append(0)
        else:
            lengths.append(int(item))
    return lengths


def _format_prompt_length(length: int) -> str:
    return "native" if int(length) <= 0 else str(int(length))


def _slug(value: str) -> str:
    chars: List[str] = []
    last_dash = False
    for ch in value.lower():
        if ch.isalnum():
            chars.append(ch)
            last_dash = False
        elif not last_dash:
            chars.append("-")
            last_dash = True
    return "".join(chars).strip("-") or "case"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_path(obj: Any, path: Sequence[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return default
            cur = cur[key]
        else:
            if not isinstance(cur, dict) or key not in cur:
                return default
            cur = cur[key]
    return cur


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return (sum(vals) / len(vals)) if vals else 0.0


def _std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    if not vals:
        return 0.0
    mean = sum(vals) / len(vals)
    return (sum((value - mean) ** 2 for value in vals) / len(vals)) ** 0.5


def _pct(values: Iterable[float], pct: float) -> float:
    vals = sorted(float(value) for value in values)
    if not vals:
        return 0.0
    if len(vals) == 1:
        return vals[0]
    index = (len(vals) - 1) * pct
    lo = int(index)
    hi = min(lo + 1, len(vals) - 1)
    weight = index - lo
    return vals[lo] * (1.0 - weight) + vals[hi] * weight


def _attention_debug(record: Dict[str, Any]) -> Dict[str, Any]:
    candidates = [
        ["metrics", "attention_backend_before_free", "backend_debug"],
        ["metrics", "attention_backend", "backend_debug"],
    ]
    for path in candidates:
        value = _get_path(record, path, default={})
        if isinstance(value, dict) and value:
            return value
    return {}


def _result_attention_debugs(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    records = row.get("records")
    if isinstance(records, list):
        for record in records:
            if not isinstance(record, dict):
                continue
            debug = _attention_debug(record)
            if debug:
                out.append(debug)
    return out


def _result_attention_debug(row: Dict[str, Any]) -> Dict[str, Any]:
    debugs = _result_attention_debugs(row)
    if debugs:
        return debugs[0]
    return {}


def _debug_store(debug: Dict[str, Any]) -> Dict[str, Any]:
    store = debug.get("resident_store_debug")
    return store if isinstance(store, dict) else {}


def _debug_balance(debug: Dict[str, Any]) -> Dict[str, Any]:
    store = _debug_store(debug)
    balance = store.get("dpu_balance_summary")
    return balance if isinstance(balance, dict) else {}


def _debug_imbalance_score(debug: Dict[str, Any]) -> tuple[float, float, float]:
    balance = _debug_balance(debug)
    store = _debug_store(debug)
    dpu_live_elems = [
        _to_float(item)
        for item in (store.get("dpu_live_elems_by_dpu") if isinstance(store.get("dpu_live_elems_by_dpu"), list) else [])
    ]
    all_mean = _mean(dpu_live_elems)
    all_imbalance = (max(dpu_live_elems) / all_mean) if dpu_live_elems and all_mean > 0 else 0.0
    all_cv = (_std(dpu_live_elems) / all_mean) if dpu_live_elems and all_mean > 0 else 0.0
    return (
        all_imbalance,
        all_cv,
        _to_float(balance.get("imbalance_ratio")),
        _to_float(balance.get("spread_ratio")),
        _to_float(balance.get("max_live_elems")),
    )


def _select_peak_balance_debug(debugs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not debugs:
        return {}
    return max(debugs, key=_debug_imbalance_score)


def _select_peak_timing_debug(debugs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not debugs:
        return {}
    return max(debugs, key=_extract_backend_pim_seconds)


def _record_metrics(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    records = row.get("records")
    if not isinstance(records, list):
        return []
    out = []
    for record in records:
        if isinstance(record, dict) and isinstance(record.get("metrics"), dict):
            out.append(record["metrics"])
    return out


def _sum_actor_time(metrics: Dict[str, Any], keys: Sequence[str]) -> float:
    actors = _get_path(metrics, ["stage_timing", "actors"], default={})
    if not isinstance(actors, dict):
        return 0.0
    return sum(_to_float(actors.get(key)) for key in keys)


def _extract_backend_pim_seconds(debug: Dict[str, Any]) -> float:
    clover_totals = debug.get("clover_op_timing_totals_s")
    if isinstance(clover_totals, dict):
        return (
            _to_float(clover_totals.get("resident_qk_batch_s"))
            + _to_float(clover_totals.get("resident_av_s"))
            + _to_float(clover_totals.get("softmax_av_s"))
        )

    store = debug.get("resident_store_debug")
    if isinstance(store, dict):
        store_totals = store.get("op_timing_totals_s")
        if isinstance(store_totals, dict):
            return (
                _to_float(store_totals.get("qk_softmax_weighted_value_sum_batch_dpu"))
                + _to_float(store_totals.get("qk_slot_scores_batch_dpu"))
                + _to_float(store_totals.get("weighted_value_sum_batch_dpu"))
                + _to_float(store_totals.get("softmax_weighted_value_sum_batch_dpu"))
            )
    return 0.0


def _extract_dense_seconds(metrics_list: Sequence[Dict[str, Any]]) -> float:
    keys = [
        "dense_start_token_compute_s",
        "dense_prepare_attention_compute_s",
        "dense_finish_layer_compute_s",
        "dense_sample_next_token_compute_s",
        "dense_decode_tokens_compute_s",
        "decode_full_compute_s",
    ]
    return sum(_sum_actor_time(metrics, keys) for metrics in metrics_list)


def _extract_dense_batching(metrics_list: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    last: Dict[str, Any] = {}
    for metrics in metrics_list:
        value = metrics.get("scheduler_dense_continuous_batching")
        if isinstance(value, dict) and value:
            last = value
    return last


def _baseline_kind(baseline: str) -> str:
    lowered = str(baseline).strip().lower().replace("_", " ")
    if "clover" in lowered:
        return "clover"
    if "naive" in lowered or "pim" in lowered:
        return "naive"
    return "other"


def _resolve_imbalance_policy(args: argparse.Namespace, baseline: str, policy_type: str) -> str:
    if bool(getattr(args, "imbalance_use_baseline_specific_policies", False)):
        attr = f"imbalance_{_baseline_kind(baseline)}_{policy_type}_policy"
        value = getattr(args, attr, "")
        if value:
            return str(value)
    return str(getattr(args, f"imbalance_{policy_type}_policy"))


def _run_command(cmd: List[str], cwd: Path, stdout_path: Path, stderr_path: Path) -> int:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=stdout, stderr=stderr, text=True)
    duration = time.time() - started
    meta_path = stdout_path.parent / "run_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "cmd": cmd,
                "returncode": int(proc.returncode),
                "duration_s": float(duration),
                "stdout": str(stdout_path),
                "stderr": str(stderr_path),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return int(proc.returncode)


def _common_benchmark_cmd(args: argparse.Namespace, output_path: Path) -> List[str]:
    return [
        sys.executable,
        str(TESTS_ROOT / "benchmark_baselines.py"),
        "--model",
        args.model,
        "--model-name",
        args.model_name,
        "--dtype",
        args.dtype,
        "--address",
        args.address,
        "--prefill-resource",
        args.prefill_resource,
        "--decode-dense-resource",
        args.decode_dense_resource,
        "--attention-resource",
        args.attention_resource,
        "--pim-num-dpus",
        str(args.pim_num_dpus),
        "--pim-resident-store-backend",
        args.pim_resident_store_backend,
        "--pim-length",
        str(args.pim_length),
        "--pim-block-tokens",
        str(args.pim_block_tokens),
        "--pim-resident-kv-dtype",
        args.pim_resident_kv_dtype,
        "--pim-qk-full-enabled",
        "--pim-softmax-av-fused-enabled",
        "--no-pim-qk-full-shadow-check",
        "--no-pim-softmax-av-shadow-check",
        "--no-clover-cpu-shadow-enabled",
        "--no-clover-shadow-checks-enabled",
        "--clover-op-profiling-enabled",
        "--no-clover-adaptive-routing-enabled",
        "--no-clover-pim-perf-guard-enabled",
        "--output",
        str(output_path),
    ]


def _case_cmd(
    args: argparse.Namespace,
    *,
    data_path: str,
    dataset_format: str,
    limit: int,
    max_seq_len: int,
    max_new_tokens: int,
    prompt_token_length: int,
    concurrency: int,
    baselines: str,
    output_path: Path,
    head_grouping_policy: str,
    dpu_placement_policy: str,
    decode_batch_max_size: Optional[int] = None,
    decode_batch_window_ms: Optional[float] = None,
    startup_grace_ms: Optional[float] = None,
) -> List[str]:
    cmd = _common_benchmark_cmd(args, output_path)
    cmd.extend(
        [
            "--data",
            data_path,
            "--dataset-format",
            dataset_format,
            "--limit",
            str(limit),
            "--max-seq-len",
            str(max_seq_len),
            "--max-new-tokens",
            str(max_new_tokens),
            "--prompt-token-length",
            str(prompt_token_length),
            "--concurrency",
            str(concurrency),
            "--baselines",
            baselines,
            "--pim-head-grouping-policy",
            head_grouping_policy,
            "--pim-dpu-placement-policy",
            dpu_placement_policy,
        ]
    )
    if decode_batch_max_size is not None:
        cmd.extend(["--decode-continuous-batch-max-size", str(decode_batch_max_size)])
    if decode_batch_window_ms is not None:
        cmd.extend(["--decode-continuous-batch-window-ms", str(decode_batch_window_ms)])
    if startup_grace_ms is not None:
        cmd.extend(["--decode-continuous-batch-startup-grace-ms", str(startup_grace_ms)])
    return cmd


def _summarize_result_file(
    result_path: Path,
    *,
    experiment: str,
    case: Dict[str, Any],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for result_index, row in enumerate(_read_jsonl(result_path)):
        summary = row.get("summary") if isinstance(row.get("summary"), dict) else {}
        debugs = _result_attention_debugs(row)
        balance_debug = _select_peak_balance_debug(debugs)
        timing_debug = _select_peak_timing_debug(debugs)
        debug = balance_debug or timing_debug or _result_attention_debug(row)
        store = debug.get("resident_store_debug") if isinstance(debug.get("resident_store_debug"), dict) else {}
        balance = store.get("dpu_balance_summary") if isinstance(store.get("dpu_balance_summary"), dict) else {}
        rank_balance = store.get("rank_balance_summary") if isinstance(store.get("rank_balance_summary"), dict) else {}
        block_summary = store.get("block_summary") if isinstance(store.get("block_summary"), dict) else {}
        allocator = store.get("allocator_summary") if isinstance(store.get("allocator_summary"), dict) else {}
        metrics_list = _record_metrics(row)
        total_tokens = sum(_to_int(metrics.get("total_tokens")) for metrics in metrics_list)
        dense_s = _extract_dense_seconds(metrics_list)
        pim_s = sum(_extract_backend_pim_seconds(item) for item in debugs) if debugs else 0.0
        if pim_s <= 0.0:
            pim_s = _extract_backend_pim_seconds(debug)
        dense_tps = (float(total_tokens) / dense_s) if dense_s > 0 else 0.0
        pim_tps = (float(total_tokens) / pim_s) if pim_s > 0 else 0.0
        gap_ratio = (
            abs(dense_tps - pim_tps) / max(dense_tps, pim_tps)
            if max(dense_tps, pim_tps) > 0
            else 0.0
        )
        batching = _extract_dense_batching(metrics_list)
        dpu_live_elems = [
            _to_int(item)
            for item in (store.get("dpu_live_elems_by_dpu") if isinstance(store.get("dpu_live_elems_by_dpu"), list) else [])
        ]
        all_dpu_live_elems = [float(item) for item in dpu_live_elems]
        active_dpu_live_elems = [float(item) for item in dpu_live_elems if int(item) > 0]
        dpu_live_cv = (
            _std(active_dpu_live_elems) / _mean(active_dpu_live_elems)
            if active_dpu_live_elems and _mean(active_dpu_live_elems) > 0
            else 0.0
        )
        dpu_live_cv_all = (
            _std(all_dpu_live_elems) / _mean(all_dpu_live_elems)
            if all_dpu_live_elems and _mean(all_dpu_live_elems) > 0
            else 0.0
        )
        dpu_imbalance_ratio_all = (
            (max(all_dpu_live_elems) / _mean(all_dpu_live_elems))
            if all_dpu_live_elems and _mean(all_dpu_live_elems) > 0
            else 0.0
        )
        dpu_spread_ratio_all = (
            ((max(all_dpu_live_elems) - min(all_dpu_live_elems)) / _mean(all_dpu_live_elems))
            if all_dpu_live_elems and _mean(all_dpu_live_elems) > 0
            else 0.0
        )
        rows.append(
            {
                "experiment": experiment,
                "result_path": str(result_path),
                "result_index": int(result_index),
                **case,
                "baseline": row.get("baseline") or summary.get("baseline") or "",
                "baseline_key": row.get("baseline_key") or summary.get("baseline_key") or "",
                "internal_baseline": row.get("internal_baseline") or summary.get("internal_baseline") or "",
                "output_token_throughput_tps": _to_float(summary.get("output_token_throughput_tps")),
                "avg_latency_s": _to_float(summary.get("avg_latency")),
                "avg_tpot_s": _to_float(summary.get("avg_tpot")),
                "wall_time_s": _to_float(summary.get("wall_time_s")),
                "num_requests": _to_int(summary.get("num_requests")),
                "num_dpus": _to_int(debug.get("num_dpus") or store.get("num_dpus")),
                "resident_kv_dtype": debug.get("resident_kv_dtype") or store.get("kv_dtype") or "",
                "head_grouping_policy": debug.get("head_grouping_policy") or store.get("placement_policy") or "",
                "dpu_placement_policy": debug.get("dpu_placement_policy") or store.get("placement_policy") or "",
                "dpu_active_count": _to_int(balance.get("active_dpus")),
                "dpu_active_ratio": _to_float(balance.get("active_ratio")),
                "dpu_avg_live_elems_active": _to_float(balance.get("avg_live_elems_active")),
                "dpu_max_live_elems": _to_float(balance.get("max_live_elems")),
                "dpu_min_live_elems_active": _to_float(balance.get("min_live_elems_active")),
                "dpu_imbalance_ratio": _to_float(balance.get("imbalance_ratio")),
                "dpu_spread_ratio": _to_float(balance.get("spread_ratio")),
                "dpu_live_elems_cv": float(dpu_live_cv),
                "dpu_imbalance_ratio_all": float(dpu_imbalance_ratio_all),
                "dpu_spread_ratio_all": float(dpu_spread_ratio_all),
                "dpu_live_elems_cv_all": float(dpu_live_cv_all),
                "dpu_live_elems_by_dpu": json.dumps(dpu_live_elems, ensure_ascii=False),
                "rank_imbalance_ratio": _to_float(rank_balance.get("imbalance_ratio")),
                "block_fill_ratio": _to_float(block_summary.get("block_fill_ratio")),
                "avg_blocks_per_slot": _to_float(block_summary.get("avg_blocks_per_slot")),
                "allocator_max_usage_ratio": _to_float(allocator.get("max_usage_ratio")),
                "allocator_avg_usage_ratio": _to_float(allocator.get("avg_usage_ratio")),
                "fallback_allocations": _to_int(store.get("fallback_allocations")),
                "dpu_allocate_failures": _to_int(store.get("dpu_allocate_failures")),
                "total_tokens": int(total_tokens),
                "dense_compute_s": float(dense_s),
                "pim_compute_s": float(pim_s),
                "dense_throughput_tps": float(dense_tps),
                "pim_throughput_tps": float(pim_tps),
                "throughput_gap_ratio": float(gap_ratio),
                "decode_batch_max_observed_size": _to_int(batching.get("max_observed_size")),
                "decode_batch_config_max_size": _to_int(batching.get("max_size")),
                "decode_batch_flushes": _to_int(batching.get("flushes")),
                "decode_batch_wait_s": _to_float(batching.get("wait_s")),
                "decode_batch_reordered_flushes": _to_int(batching.get("reordered_flushes")),
                "decode_batch_size_histogram": json.dumps(
                    batching.get("size_histogram", {}) if isinstance(batching.get("size_histogram"), dict) else {},
                    sort_keys=True,
                    ensure_ascii=False,
                ),
            }
        )
    return rows


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _balance_metrics(loads: Sequence[int | float]) -> Dict[str, float | int | str]:
    values = [float(value) for value in loads]
    active = [value for value in values if value > 0.0]
    active_mean = _mean(active)
    all_mean = _mean(values)
    active_max = max(active) if active else 0.0
    active_min = min(active) if active else 0.0
    all_max = max(values) if values else 0.0
    all_min = min(values) if values else 0.0
    return {
        "dpu_active_count": int(len(active)),
        "dpu_active_ratio": (float(len(active)) / float(len(values))) if values else 0.0,
        "dpu_avg_live_elems_active": float(active_mean),
        "dpu_max_live_elems": float(active_max),
        "dpu_min_live_elems_active": float(active_min),
        "dpu_imbalance_ratio": (active_max / active_mean) if active_mean > 0 else 0.0,
        "dpu_spread_ratio": ((active_max - active_min) / active_mean) if active_mean > 0 else 0.0,
        "dpu_live_elems_cv": (_std(active) / active_mean) if active_mean > 0 else 0.0,
        "dpu_imbalance_ratio_all": (all_max / all_mean) if all_mean > 0 else 0.0,
        "dpu_spread_ratio_all": ((all_max - all_min) / all_mean) if all_mean > 0 else 0.0,
        "dpu_live_elems_cv_all": (_std(values) / all_mean) if all_mean > 0 else 0.0,
        "dpu_live_elems_by_dpu": json.dumps([int(value) for value in values], ensure_ascii=False),
    }


def _simulate_naive_head_only_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
    loads = [0 for _ in range(max(1, int(num_dpus)))]
    heads = max(1, int(num_heads))
    for head_id in range(heads):
        dpu_id = int(head_id) % len(loads)
        for seq_len in seq_lens:
            loads[dpu_id] += int(seq_len)
    return loads


def _head_dpu_groups(num_dpus: int, num_heads: int) -> List[List[int]]:
    dpu_count = max(1, int(num_dpus))
    head_count = max(1, int(num_heads))
    group_count = min(dpu_count, head_count)
    base = dpu_count // group_count
    extra = dpu_count % group_count
    groups: List[List[int]] = []
    cursor = 0
    for group_idx in range(group_count):
        width = base + (1 if group_idx < extra else 0)
        groups.append(list(range(cursor, cursor + width)))
        cursor += width
    return groups


def _simulate_naive_static_request_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
    """Length-oblivious request placement inside each head group.

    This is the motivation baseline we want to expose: all DPUs are available,
    but each request's KV for a head is pinned to one DPU by arrival order rather
    than split or length-balanced. Mixed long/short prompts create stragglers.
    """

    loads = [0 for _ in range(max(1, int(num_dpus)))]
    for group in _head_dpu_groups(num_dpus, num_heads):
        if not group:
            continue
        for request_idx, seq_len in enumerate(seq_lens):
            dpu_id = group[int(request_idx) % len(group)]
            loads[int(dpu_id)] += int(seq_len)
    return loads


def _simulate_balanced_token_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
    loads = [0 for _ in range(max(1, int(num_dpus)))]
    for group in _head_dpu_groups(num_dpus, num_heads):
        if not group:
            continue
        for seq_len in seq_lens:
            remaining = int(seq_len)
            remaining_dpus = len(group)
            for dpu_id in sorted(group, key=lambda idx: (loads[idx], idx)):
                chunk = int((remaining + remaining_dpus - 1) // max(1, remaining_dpus))
                loads[dpu_id] += chunk
                remaining -= chunk
                remaining_dpus -= 1
    return loads


def _load_prompt_lengths_from_dataset(
    *,
    args: argparse.Namespace,
    data_path: str,
    dataset_format: str,
    limit: int,
    prompt_token_length: int,
) -> List[int]:
    tokenizer = load_tokenizer_for_benchmark(args.model)
    samples = load_benchmark_samples(
        data_path,
        dataset_format=dataset_format,
        limit=limit,
        tokenizer=tokenizer,
        prompt_token_length=max(0, int(prompt_token_length)),
    )
    lengths: List[int] = []
    for sample in samples:
        sample_len = sample.get("prompt_token_length")
        if sample_len is None:
            sample_len = len(encode_prompt(tokenizer, str(sample["prompt"])))
        lengths.append(int(sample_len))
    return lengths


def _run_imbalance_model(args: argparse.Namespace, output_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    workloads = _parse_csv(args.imbalance_workloads)
    lengths = _parse_prompt_length_csv(args.imbalance_prompt_lengths)
    baselines = _parse_csv(args.imbalance_model_baselines)
    for workload in workloads:
        if workload in SYNTHETIC_IMBALANCE_WORKLOADS:
            data_path = str(
                _generate_synthetic_dataset(
                    model=args.model,
                    output_dir=output_dir,
                    lengths_csv=args.imbalance_mixed_prompt_lengths,
                    limit=args.imbalance_limit,
                    name="imbalance_mixed",
                    prefix="Imbalance mixed-length",
                )
            )
            dataset_format = "prompt_jsonl"
            workload_label = "mixed_synthetic"
        else:
            if workload not in WORKLOAD_SPECS:
                choices = sorted(list(WORKLOAD_SPECS) + sorted(SYNTHETIC_IMBALANCE_WORKLOADS))
                raise ValueError(f"unknown workload {workload}; choices: {', '.join(choices)}")
            spec = WORKLOAD_SPECS[workload]
            data_path = spec["path"]
            dataset_format = spec["format"]
            workload_label = workload
        for length in lengths:
            prompt_label = _format_prompt_length(length)
            seq_lens = _load_prompt_lengths_from_dataset(
                args=args,
                data_path=data_path,
                dataset_format=dataset_format,
                limit=args.imbalance_limit,
                prompt_token_length=length,
            )
            if not seq_lens:
                failures.append(
                    {
                        "experiment": "imbalance_model",
                        "case_slug": f"imbalance_model__{_slug(workload_label)}__p{_slug(prompt_label)}",
                        "returncode": 1,
                        "stderr_path": "no prompt lengths loaded",
                    }
                )
                continue
            for baseline in baselines:
                baseline_lower = str(baseline).strip().lower()
                kind = _baseline_kind(baseline)
                if kind == "clover" or "balanced" in baseline_lower:
                    loads = _simulate_balanced_token_loads(
                        seq_lens,
                        num_dpus=args.imbalance_model_num_dpus,
                        num_heads=args.imbalance_model_num_heads,
                    )
                    baseline_name = "Clover Planner"
                    internal_baseline = "balanced_token_sharding_model"
                elif "head" in baseline_lower or "under" in baseline_lower:
                    loads = _simulate_naive_head_only_loads(
                        seq_lens,
                        num_dpus=args.imbalance_model_num_dpus,
                        num_heads=args.imbalance_model_num_heads,
                    )
                    baseline_name = "Naive Head-Only PIM"
                    internal_baseline = "head_only_no_token_sharding_model"
                else:
                    loads = _simulate_naive_static_request_loads(
                        seq_lens,
                        num_dpus=args.imbalance_model_num_dpus,
                        num_heads=args.imbalance_model_num_heads,
                    )
                    baseline_name = "Naive Static PIM"
                    internal_baseline = "length_oblivious_static_request_sharding_model"
                metrics = _balance_metrics(loads)
                rows.append(
                    {
                        "experiment": "imbalance_model",
                        "result_path": "",
                        "result_index": 0,
                        "workload": workload_label,
                        "prompt_token_length": int(length),
                        "prompt_length_label": prompt_label,
                        "mixed_prompt_lengths": args.imbalance_mixed_prompt_lengths
                        if workload in SYNTHETIC_IMBALANCE_WORKLOADS
                        else "",
                        "max_new_tokens": int(args.max_new_tokens),
                        "limit": int(args.imbalance_limit),
                        "concurrency": int(args.imbalance_concurrency),
                        "decode_batch_max_size": int(args.imbalance_decode_batch_max_size),
                        "configured_head_grouping_policy": "model",
                        "configured_dpu_placement_policy": "model",
                        "baseline": baseline_name,
                        "baseline_key": _slug(baseline_name),
                        "internal_baseline": internal_baseline,
                        "tail_latency_proxy_tokens": int(max(loads) if loads else 0),
                        "avg_active_dpu_tokens": float(_mean([value for value in loads if value > 0])),
                        "output_token_throughput_tps": 0.0,
                        "avg_latency_s": 0.0,
                        "avg_tpot_s": 0.0,
                        "wall_time_s": 0.0,
                        "num_requests": int(len(seq_lens)),
                        "num_dpus": int(args.imbalance_model_num_dpus),
                        "resident_kv_dtype": "",
                        "head_grouping_policy": "model",
                        "dpu_placement_policy": "model",
                        **metrics,
                        "rank_imbalance_ratio": 0.0,
                        "block_fill_ratio": 0.0,
                        "avg_blocks_per_slot": 0.0,
                        "allocator_max_usage_ratio": 0.0,
                        "allocator_avg_usage_ratio": 0.0,
                        "fallback_allocations": 0,
                        "dpu_allocate_failures": 0,
                        "total_tokens": int(sum(seq_lens)),
                        "dense_compute_s": 0.0,
                        "pim_compute_s": 0.0,
                        "dense_throughput_tps": 0.0,
                        "pim_throughput_tps": 0.0,
                        "throughput_gap_ratio": 0.0,
                        "decode_batch_max_observed_size": 0,
                        "decode_batch_config_max_size": int(args.imbalance_decode_batch_max_size),
                        "decode_batch_flushes": 0,
                        "decode_batch_wait_s": 0.0,
                        "decode_batch_reordered_flushes": 0,
                        "decode_batch_size_histogram": "{}",
                        "seq_lens": json.dumps(seq_lens, ensure_ascii=False),
                        "seq_len_min": int(min(seq_lens)),
                        "seq_len_max": int(max(seq_lens)),
                        "seq_len_mean": float(_mean(seq_lens)),
                    }
                )
    return rows, failures


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return lines


def _fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_markdown(rows: Sequence[Dict[str, Any]], failures: Sequence[Dict[str, Any]], output_dir: Path) -> str:
    lines: List[str] = [
        "# CloverInfer Motivation Experiments",
        "",
        f"Output directory: `{output_dir}`",
        f"Completed result rows: {len(rows)}",
        f"Failed cases: {len(failures)}",
        "",
        "## Metric Definitions",
        "",
        "- Experiment 1 uses `dpu_imbalance_ratio = max_live_elems / avg_live_elems_active` as the main straggler metric; `tail_latency_proxy_tokens = max_live_elems` is the proxy for the slowest DPU. `dpu_imbalance_ratio_all` is only an auxiliary under-utilization metric because it counts idle DPUs in the denominator.",
        "- Experiment 2 uses `throughput_gap_ratio = abs(dense_throughput_tps - pim_throughput_tps) / max(dense_throughput_tps, pim_throughput_tps)`; larger means the dense side and PIM side are less aligned.",
        "",
    ]

    imbalance_rows = [row for row in rows if row.get("experiment") in {"imbalance", "imbalance_model"}]
    if imbalance_rows:
        lines.extend(["## Experiment 1: DPU Load Imbalance", ""])
        grouped: Dict[tuple[Any, Any, Any, Any], List[Dict[str, Any]]] = {}
        for row in imbalance_rows:
            key = (
                row.get("experiment"),
                row.get("workload"),
                row.get("prompt_length_label", row.get("prompt_token_length")),
                row.get("baseline"),
            )
            grouped.setdefault(key, []).append(row)
        table_rows = []
        for (experiment, workload, length, baseline), group in sorted(grouped.items()):
            table_rows.append(
                [
                    experiment,
                    workload,
                    length,
                    baseline,
                    _fmt(_mean(row["dpu_imbalance_ratio"] for row in group)),
                    _fmt(_pct((row["dpu_imbalance_ratio"] for row in group), 0.95)),
                    _fmt(_mean(row.get("tail_latency_proxy_tokens", row["dpu_max_live_elems"]) for row in group)),
                    _fmt(_mean(row["dpu_imbalance_ratio_all"] for row in group)),
                    _fmt(_mean(row["dpu_spread_ratio"] for row in group)),
                    _fmt(_mean(row["dpu_live_elems_cv"] for row in group)),
                    _fmt(_mean(row["dpu_live_elems_cv_all"] for row in group)),
                    _fmt(_mean(row["block_fill_ratio"] for row in group)),
                    _fmt(_mean(row["output_token_throughput_tps"] for row in group)),
                    group[0].get("dpu_live_elems_by_dpu", "[]"),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "experiment",
                    "workload",
                    "length",
                    "baseline",
                    "mean imbalance",
                    "p95 imbalance",
                    "tail proxy",
                    "all-DPU imbalance",
                    "mean spread",
                    "live CV",
                    "all-DPU CV",
                    "block fill",
                    "tok/s",
                    "peak DPU live elems",
                ],
                table_rows,
            )
        )
        lines.append("")

    micro_rows = [row for row in rows if row.get("experiment") == "microbatch"]
    if micro_rows:
        lines.extend(["## Experiment 2: Dense/PIM Throughput Mismatch", ""])
        table_rows = []
        for row in sorted(micro_rows, key=lambda r: (int(r.get("decode_batch_max_size", 0)), str(r.get("baseline")))):
            table_rows.append(
                [
                    row.get("decode_batch_max_size"),
                    row.get("baseline"),
                    row.get("decode_batch_max_observed_size"),
                    _fmt(row.get("dense_throughput_tps", 0.0)),
                    _fmt(row.get("pim_throughput_tps", 0.0)),
                    _fmt(row.get("throughput_gap_ratio", 0.0)),
                    _fmt(row.get("output_token_throughput_tps", 0.0)),
                    row.get("decode_batch_size_histogram", "{}"),
                ]
            )
        lines.extend(
            _markdown_table(
                [
                    "configured max batch",
                    "baseline",
                    "observed max batch",
                    "dense tok/s",
                    "PIM tok/s",
                    "gap ratio",
                    "end-to-end tok/s",
                    "batch histogram",
                ],
                table_rows,
            )
        )
        lines.append("")

    if failures:
        lines.extend(["## Failed Cases", ""])
        table_rows = [
            [item.get("experiment"), item.get("case_slug"), item.get("returncode"), item.get("stderr_path")]
            for item in failures
        ]
        lines.extend(_markdown_table(["experiment", "case", "returncode", "stderr"], table_rows))
        lines.append("")

    return "\n".join(lines)


def _generate_synthetic_dataset(
    *,
    model: str,
    output_dir: Path,
    lengths_csv: str,
    limit: int,
    name: str,
    prefix: str,
) -> Path:
    lengths = _parse_int_csv(lengths_csv)
    tokenizer = load_tokenizer_for_benchmark(model)
    dataset_dir = output_dir / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    path = dataset_dir / f"{name}_lengths_{'_'.join(str(item) for item in lengths)}_n{int(limit)}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for idx in range(max(1, int(limit))):
            target = lengths[idx % len(lengths)]
            prompt, actual_tokens = build_prompt_for_target_tokens(
                tokenizer,
                target,
                base_prompt=f"{prefix} motivation sample {idx}.",
            )
            f.write(
                json.dumps(
                    {
                        "task_id": f"{name}/{idx}",
                        "prompt": prompt,
                        "target_prompt_tokens": int(target),
                        "actual_prompt_tokens": int(actual_tokens),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return path


def _generate_mixed_dataset(args: argparse.Namespace, output_dir: Path) -> Path:
    return _generate_synthetic_dataset(
        model=args.model,
        output_dir=output_dir,
        lengths_csv=args.microbatch_mixed_prompt_lengths,
        limit=args.microbatch_limit,
        name="mixed",
        prefix="Mixed-length",
    )


def _run_imbalance(args: argparse.Namespace, output_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    workloads = _parse_csv(args.imbalance_workloads)
    lengths = _parse_prompt_length_csv(args.imbalance_prompt_lengths)
    baselines = _parse_csv(args.imbalance_baselines)
    for workload in workloads:
        if workload in SYNTHETIC_IMBALANCE_WORKLOADS:
            data_path = str(
                _generate_synthetic_dataset(
                    model=args.model,
                    output_dir=output_dir,
                    lengths_csv=args.imbalance_mixed_prompt_lengths,
                    limit=args.imbalance_limit,
                    name="imbalance_mixed",
                    prefix="Imbalance mixed-length",
                )
            )
            dataset_format = "prompt_jsonl"
            workload_label = "mixed_synthetic"
            synthetic_max_prompt = max(_parse_int_csv(args.imbalance_mixed_prompt_lengths))
        else:
            if workload not in WORKLOAD_SPECS:
                choices = sorted(list(WORKLOAD_SPECS) + sorted(SYNTHETIC_IMBALANCE_WORKLOADS))
                raise ValueError(f"unknown workload {workload}; choices: {', '.join(choices)}")
            spec = WORKLOAD_SPECS[workload]
            data_path = spec["path"]
            dataset_format = spec["format"]
            workload_label = workload
            synthetic_max_prompt = 0
        for length in lengths:
            for baseline in baselines:
                prompt_label = _format_prompt_length(length)
                case_slug = f"imbalance__{_slug(workload_label)}__p{_slug(prompt_label)}__{_slug(baseline)}"
                case_dir = output_dir / "cases" / case_slug
                result_path = case_dir / "results.jsonl"
                head_grouping_policy = _resolve_imbalance_policy(args, baseline, "head_grouping")
                dpu_placement_policy = _resolve_imbalance_policy(args, baseline, "dpu_placement")
                effective_prompt_bound = max(int(length), int(synthetic_max_prompt))
                cmd = _case_cmd(
                    args,
                    data_path=data_path,
                    dataset_format=dataset_format,
                    limit=args.imbalance_limit,
                    max_seq_len=max(args.max_seq_len, effective_prompt_bound + args.max_new_tokens + 8),
                    max_new_tokens=args.max_new_tokens,
                    prompt_token_length=length,
                    concurrency=args.imbalance_concurrency,
                    baselines=baseline,
                    output_path=result_path,
                    head_grouping_policy=head_grouping_policy,
                    dpu_placement_policy=dpu_placement_policy,
                    decode_batch_max_size=args.imbalance_decode_batch_max_size,
                    decode_batch_window_ms=args.imbalance_decode_batch_window_ms,
                )
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                returncode = _run_command(cmd, REPO_ROOT, case_dir / "stdout.log", case_dir / "stderr.log")
                case = {
                    "workload": workload_label,
                    "prompt_token_length": int(length),
                    "prompt_length_label": prompt_label,
                    "mixed_prompt_lengths": args.imbalance_mixed_prompt_lengths if workload in SYNTHETIC_IMBALANCE_WORKLOADS else "",
                    "max_new_tokens": int(args.max_new_tokens),
                    "limit": int(args.imbalance_limit),
                    "concurrency": int(args.imbalance_concurrency),
                    "decode_batch_max_size": int(args.imbalance_decode_batch_max_size),
                    "configured_head_grouping_policy": head_grouping_policy,
                    "configured_dpu_placement_policy": dpu_placement_policy,
                }
                if returncode != 0:
                    failures.append(
                        {
                            "experiment": "imbalance",
                            "case_slug": case_slug,
                            "returncode": int(returncode),
                            "stderr_path": str(case_dir / "stderr.log"),
                        }
                    )
                    if not args.continue_on_error:
                        return rows, failures
                    continue
                rows.extend(_summarize_result_file(result_path, experiment="imbalance", case=case))
    return rows, failures


def _run_microbatch(args: argparse.Namespace, output_dir: Path) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    batch_sizes = _parse_int_csv(args.microbatch_sizes)
    baselines = _parse_csv(args.microbatch_baselines)
    dataset_path = _generate_mixed_dataset(args, output_dir)
    max_prompt = max(_parse_int_csv(args.microbatch_mixed_prompt_lengths))
    for batch_size in batch_sizes:
        for baseline in baselines:
            case_slug = f"microbatch__bs{batch_size}__{_slug(baseline)}"
            case_dir = output_dir / "cases" / case_slug
            result_path = case_dir / "results.jsonl"
            cmd = _case_cmd(
                args,
                data_path=str(dataset_path),
                dataset_format="prompt_jsonl",
                limit=args.microbatch_limit,
                max_seq_len=max(args.max_seq_len, max_prompt + args.max_new_tokens + 8),
                max_new_tokens=args.microbatch_max_new_tokens,
                prompt_token_length=0,
                concurrency=args.microbatch_concurrency,
                baselines=baseline,
                output_path=result_path,
                head_grouping_policy=args.microbatch_head_grouping_policy,
                dpu_placement_policy=args.microbatch_dpu_placement_policy,
                decode_batch_max_size=batch_size,
                decode_batch_window_ms=args.microbatch_decode_batch_window_ms,
                startup_grace_ms=args.microbatch_startup_grace_ms,
            )
            if args.dry_run:
                print(" ".join(cmd))
                continue
            returncode = _run_command(cmd, REPO_ROOT, case_dir / "stdout.log", case_dir / "stderr.log")
            case = {
                "workload": "mixed_synthetic",
                "prompt_token_length": 0,
                "mixed_prompt_lengths": args.microbatch_mixed_prompt_lengths,
                "max_new_tokens": int(args.microbatch_max_new_tokens),
                "limit": int(args.microbatch_limit),
                "concurrency": int(args.microbatch_concurrency),
                "decode_batch_max_size": int(batch_size),
            }
            if returncode != 0:
                failures.append(
                    {
                        "experiment": "microbatch",
                        "case_slug": case_slug,
                        "returncode": int(returncode),
                        "stderr_path": str(case_dir / "stderr.log"),
                    }
                )
                if not args.continue_on_error:
                    return rows, failures
                continue
            rows.extend(_summarize_result_file(result_path, experiment="microbatch", case=case))
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="imbalance,microbatch")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "artifacts" / f"motivation_{time.strftime('%Y%m%d_%H%M%S')}"))
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/Qwen-1_8B")
    parser.add_argument("--model-name", default="qwen-1_8b")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--prefill-resource", default="prefill_gpu")
    parser.add_argument("--decode-dense-resource", default="decode_dense_gpu")
    parser.add_argument("--attention-resource", default="attention_pim")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-resident-store-backend", default="upmem_kvslot", choices=["auto", "host", "upmem_kvslot"])
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument("--pim-resident-kv-dtype", default="fp32")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kvslot-slim-qk-slot-only", default="0")
    parser.add_argument("--kvslot-autobuild", default="1")
    parser.add_argument("--kvslot-autobuild-timeout-s", default="300")
    parser.add_argument("--kvslot-dpu-opt", default="-Os")

    parser.add_argument("--imbalance-workloads", default="humaneval,sharegpt,qasper")
    parser.add_argument("--imbalance-prompt-lengths", default="512,1024,2048")
    parser.add_argument("--imbalance-limit", type=int, default=1)
    parser.add_argument("--imbalance-concurrency", type=int, default=1)
    parser.add_argument("--imbalance-baselines", default="Naive PIM,CloverInfer")
    parser.add_argument("--imbalance-head-grouping-policy", default="auto")
    parser.add_argument("--imbalance-dpu-placement-policy", default="auto")
    parser.add_argument("--imbalance-use-baseline-specific-policies", action="store_true")
    parser.add_argument("--imbalance-naive-head-grouping-policy", default="legacy")
    parser.add_argument("--imbalance-naive-dpu-placement-policy", default="identity")
    parser.add_argument("--imbalance-clover-head-grouping-policy", default="balanced")
    parser.add_argument("--imbalance-clover-dpu-placement-policy", default="rotated")
    parser.add_argument("--imbalance-other-head-grouping-policy", default="")
    parser.add_argument("--imbalance-other-dpu-placement-policy", default="")
    parser.add_argument("--imbalance-mixed-prompt-lengths", default="64,512,1536")
    parser.add_argument("--imbalance-model-baselines", default="Naive PIM,CloverInfer")
    parser.add_argument("--imbalance-model-num-dpus", type=int, default=16)
    parser.add_argument("--imbalance-model-num-heads", type=int, default=4)
    parser.add_argument("--imbalance-decode-batch-max-size", type=int, default=8)
    parser.add_argument("--imbalance-decode-batch-window-ms", type=float, default=0.0)

    parser.add_argument("--microbatch-sizes", default="1,2,4,8")
    parser.add_argument("--microbatch-limit", type=int, default=8)
    parser.add_argument("--microbatch-concurrency", type=int, default=8)
    parser.add_argument("--microbatch-max-new-tokens", type=int, default=4)
    parser.add_argument("--microbatch-baselines", default="CloverInfer")
    parser.add_argument("--microbatch-mixed-prompt-lengths", default="512,2048")
    parser.add_argument("--microbatch-head-grouping-policy", default="balanced")
    parser.add_argument("--microbatch-dpu-placement-policy", default="rotated")
    parser.add_argument("--microbatch-decode-batch-window-ms", type=float, default=20.0)
    parser.add_argument("--microbatch-startup-grace-ms", type=float, default=20.0)
    args = parser.parse_args()

    # Motivation experiments intentionally exercise both legacy grouped QK/AV
    # and fused Clover paths.  The slim QK-only helper build cannot serve the
    # legacy command set, so default to the full kvslot helper unless callers
    # explicitly override the setting.
    os.environ.setdefault("CLOVER_KVSLOT_SLIM_QK_SLOT_ONLY", str(args.kvslot_slim_qk_slot_only))
    os.environ.setdefault("CLOVER_KVSLOT_AUTOBUILD", str(args.kvslot_autobuild))
    os.environ.setdefault("CLOVER_KVSLOT_AUTOBUILD_TIMEOUT_S", str(args.kvslot_autobuild_timeout_s))
    os.environ.setdefault("CLOVER_KVSLOT_DPU_OPT", str(args.kvslot_dpu_opt))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    selected = set(_parse_csv(args.experiments))
    all_rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    if "imbalance_model" in selected:
        rows, failed = _run_imbalance_model(args, output_dir)
        all_rows.extend(rows)
        failures.extend(failed)
    if "imbalance" in selected:
        rows, failed = _run_imbalance(args, output_dir)
        all_rows.extend(rows)
        failures.extend(failed)
    if "microbatch" in selected:
        rows, failed = _run_microbatch(args, output_dir)
        all_rows.extend(rows)
        failures.extend(failed)

    _write_jsonl(output_dir / "motivation_metrics.jsonl", all_rows)
    _write_csv(output_dir / "motivation_metrics.csv", all_rows)
    _write_jsonl(output_dir / "failures.jsonl", failures)
    report = _build_markdown(all_rows, failures, output_dir)
    (output_dir / "summary.md").write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
