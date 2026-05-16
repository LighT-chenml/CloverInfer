from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _get_path(obj: Any, path: Iterable[Any], default: Any = None) -> Any:
    cur = obj
    for key in path:
        if isinstance(key, int):
            if not isinstance(cur, list) or key >= len(cur):
                return default
            cur = cur[key]
            continue
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_present(row: Dict[str, Any], candidate_paths: List[List[Any]], default: Any = None) -> Any:
    for path in candidate_paths:
        value = _get_path(row, path, default=None)
        if value is not None:
            return value
    return default


def _attention_debug(row: Dict[str, Any]) -> Dict[str, Any]:
    # Request-level snapshots preserve counters before actor teardown. The top-level placement
    # snapshot is often taken before any decode work has happened.
    candidates = [
        ["records", 0, "metrics", "attention_backend_before_free", "backend_debug"],
        ["records", 0, "metrics", "attention_backend", "backend_debug"],
        ["placement", "attention", "backend_debug"],
    ]
    for path in candidates:
        value = _get_path(row, path, default={})
        if isinstance(value, dict) and value:
            return value
    return {}


def _ns_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value) / 1_000_000_000.0
    except (TypeError, ValueError):
        return None


def _fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_int(value: Any) -> str:
    if value is None:
        return ""
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def summarize_row(row: Dict[str, Any], path: Path, row_index: int) -> Dict[str, Any]:
    summary = row.get("summary", {}) if isinstance(row.get("summary"), dict) else {}
    debug = _attention_debug(row)
    store = debug.get("resident_store_debug", {}) if isinstance(debug.get("resident_store_debug"), dict) else {}
    batch_totals = store.get("batch_item_totals", {}) if isinstance(store.get("batch_item_totals"), dict) else {}
    helper_profile = store.get("helper_profile", {}) if isinstance(store.get("helper_profile"), dict) else {}

    return {
        "path": str(path),
        "row_index": int(row_index),
        "baseline": row.get("baseline") or summary.get("baseline") or "",
        "baseline_key": row.get("baseline_key") or summary.get("baseline_key") or "",
        "internal_baseline": row.get("internal_baseline") or summary.get("internal_baseline") or "",
        "output_token_throughput_tps": summary.get("output_token_throughput_tps"),
        "avg_latency_s": summary.get("avg_latency"),
        "avg_ttft_s": summary.get("avg_ttft"),
        "avg_tpot_s": summary.get("avg_tpot"),
        "num_requests": summary.get("num_requests"),
        "prompt_tokens": _first_present(row, [["data_config", "prompt_token_length_override"]]),
        "max_new_tokens": _first_present(row, [["data_config", "max_new_tokens"]]),
        "max_seq_len": _first_present(row, [["data_config", "max_seq_len"]]),
        "num_dpus": debug.get("num_dpus"),
        "resident_kv_dtype": debug.get("resident_kv_dtype"),
        "head_grouping_policy": debug.get("head_grouping_policy"),
        "dpu_placement_policy": debug.get("dpu_placement_policy"),
        "context_fused": debug.get("clover_pim_context_fused_experimental_enabled"),
        "perf_guard": debug.get("clover_pim_perf_guard_enabled"),
        "cpu_fast_path_items": debug.get("clover_cpu_fast_path_decode_items"),
        "qk_full_batch_calls": debug.get("qk_full_batch_calls"),
        "softmax_av_fused_batch_calls": debug.get("softmax_av_fused_batch_calls"),
        "qk_softmax_total": batch_totals.get("qk_softmax_weighted_value_sum_batch_total"),
        "qk_softmax_dpu_items": batch_totals.get("qk_softmax_weighted_value_sum_batch_dpu_items"),
        "qk_softmax_host_fallback_items": batch_totals.get(
            "qk_softmax_weighted_value_sum_batch_host_fallback_items"
        ),
        "qk_batched_round_s": _ns_to_seconds(helper_profile.get("qk_batched_round_total_ns")),
        "qk_batched_launch_s": _ns_to_seconds(helper_profile.get("qk_batched_launch_ns")),
    }


def summarize_file(path: Path) -> List[Dict[str, Any]]:
    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"no JSONL records found in {path}")
    return [summarize_row(row, path, row_index) for row_index, row in enumerate(rows)]


def _baseline_sort_key(row: Dict[str, Any]) -> tuple[int, str]:
    key = str(row.get("baseline_key") or row.get("baseline") or "").lower()
    if key == "naive_pim":
        return (0, key)
    if key == "cloverinfer":
        return (1, key)
    return (2, key)


def _case_key(row: Dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        row.get("prompt_tokens"),
        row.get("max_new_tokens"),
        row.get("max_seq_len"),
    )


def _find_naive_row(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in rows:
        key = str(row.get("baseline_key") or row.get("baseline") or "").lower()
        name = str(row.get("baseline") or "").lower()
        if key == "naive_pim" or name == "naive pim":
            return row
    return None


def build_markdown(rows: List[Dict[str, Any]]) -> str:
    rows = sorted(rows, key=lambda row: (_case_key(row), _baseline_sort_key(row)))
    naive_tps_by_case: Dict[tuple[Any, Any, Any], float] = {}
    for row in rows:
        key = str(row.get("baseline_key") or row.get("baseline") or "").lower()
        name = str(row.get("baseline") or "").lower()
        if key != "naive_pim" and name != "naive pim":
            continue
        try:
            naive_tps_by_case[_case_key(row)] = float(row.get("output_token_throughput_tps"))
        except (TypeError, ValueError):
            continue

    headers = [
        "prompt",
        "out",
        "baseline",
        "tok/s",
        "speedup vs Naive",
        "latency s",
        "TPOT s",
        "DPUs",
        "KV",
        "grouping",
        "placement",
        "fused",
        "QK launches s",
        "DPU items",
        "host fallback",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        tps = row.get("output_token_throughput_tps")
        naive_tps = naive_tps_by_case.get(_case_key(row))
        speedup = ""
        if naive_tps and tps is not None:
            try:
                speedup = f"{float(tps) / naive_tps:.2f}x"
            except (TypeError, ValueError, ZeroDivisionError):
                speedup = ""
        fused = row.get("context_fused")
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt_int(row.get("prompt_tokens")),
                    _fmt_int(row.get("max_new_tokens")),
                    str(row.get("baseline") or ""),
                    _fmt_float(tps, 4),
                    speedup,
                    _fmt_float(row.get("avg_latency_s"), 2),
                    _fmt_float(row.get("avg_tpot_s"), 2),
                    _fmt_int(row.get("num_dpus")),
                    str(row.get("resident_kv_dtype") or ""),
                    str(row.get("head_grouping_policy") or ""),
                    str(row.get("dpu_placement_policy") or ""),
                    "" if fused is None else str(bool(fused)),
                    _fmt_float(row.get("qk_batched_launch_s"), 2),
                    _fmt_int(row.get("qk_softmax_dpu_items")),
                    _fmt_int(row.get("qk_softmax_host_fallback_items")),
                ]
            )
            + " |"
        )

    if naive_tps_by_case:
        winners = [
            row
            for row in rows
            if str(row.get("baseline_key") or "").lower() == "cloverinfer"
            and row.get("output_token_throughput_tps") is not None
            and _case_key(row) in naive_tps_by_case
            and float(row["output_token_throughput_tps"]) > naive_tps_by_case[_case_key(row)]
        ]
        status = "PASS" if winners else "FAIL"
        lines.append("")
        lines.append(f"Target status: {status} for at least one matched `CloverInfer > Naive PIM` case.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", nargs="+", help="Benchmark JSONL files to compare.")
    parser.add_argument("--output-md", default="", help="Optional markdown summary output path.")
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    for path in args.jsonl:
        rows.extend(summarize_file(Path(path)))
    report = build_markdown(rows)
    print(report)
    if args.output_md:
        Path(args.output_md).write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
