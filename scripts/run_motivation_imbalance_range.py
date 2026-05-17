from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Sequence


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _head_dpu_groups(num_dpus: int, num_heads: int) -> List[List[int]]:
    group_count = max(1, min(int(num_dpus), int(num_heads)))
    base = int(num_dpus) // group_count
    extra = int(num_dpus) % group_count
    groups: List[List[int]] = []
    cursor = 0
    for group_idx in range(group_count):
        width = base + (1 if group_idx < extra else 0)
        groups.append(list(range(cursor, cursor + width)))
        cursor += width
    return groups


def _adversarial_lengths(group_width: int, *, short_len: int, long_len: int, cycles: int) -> List[int]:
    group_width = max(1, int(group_width))
    lengths: List[int] = []
    hot_slot = max(0, group_width // 2)
    for _ in range(max(1, int(cycles))):
        for slot in range(group_width):
            lengths.append(int(long_len) if slot == hot_slot else int(short_len))
    return lengths


def _naive_static_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
    loads = [0 for _ in range(max(1, int(num_dpus)))]
    for group in _head_dpu_groups(num_dpus, num_heads):
        if not group:
            continue
        for request_idx, seq_len in enumerate(seq_lens):
            loads[group[int(request_idx) % len(group)]] += int(seq_len)
    return loads


def _balanced_token_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
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


def _metrics(loads: Sequence[int]) -> Dict[str, Any]:
    values = [float(value) for value in loads]
    active = [value for value in values if value > 0]
    active_mean = _mean(active)
    all_mean = _mean(values)
    active_max = max(active) if active else 0.0
    active_min = min(active) if active else 0.0
    all_max = max(values) if values else 0.0
    all_min = min(values) if values else 0.0
    return {
        "active_dpus": int(len(active)),
        "active_ratio": float(len(active)) / float(len(values)) if values else 0.0,
        "avg_active_tokens": active_mean,
        "max_tokens": active_max,
        "min_active_tokens": active_min,
        "straggler_ratio": active_max / active_mean if active_mean > 0 else 0.0,
        "spread_ratio": (active_max - active_min) / active_mean if active_mean > 0 else 0.0,
        "cv_active": pstdev(active) / active_mean if active_mean > 0 and len(active) > 1 else 0.0,
        "all_dpu_imbalance": all_max / all_mean if all_mean > 0 else 0.0,
        "all_dpu_spread": (all_max - all_min) / all_mean if all_mean > 0 else 0.0,
        "loads": json.dumps([int(value) for value in values], ensure_ascii=False),
    }


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (float, int)) else str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/motivation_imbalance_range")
    parser.add_argument("--dpu-counts", default="16,32,64")
    parser.add_argument("--head-counts", default="4")
    parser.add_argument("--long-short-ratios", default="8,32,128")
    parser.add_argument("--short-len", type=int, default=128)
    parser.add_argument("--cycles", type=int, default=8)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    pair_index: Dict[tuple[int, int, int], Dict[str, Dict[str, Any]]] = {}
    for num_dpus in _parse_int_csv(args.dpu_counts):
        for num_heads in _parse_int_csv(args.head_counts):
            groups = _head_dpu_groups(num_dpus, num_heads)
            if not groups:
                continue
            group_width = max(len(group) for group in groups)
            for ratio in _parse_int_csv(args.long_short_ratios):
                short_len = int(args.short_len)
                long_len = int(short_len * int(ratio))
                seq_lens = _adversarial_lengths(
                    group_width,
                    short_len=short_len,
                    long_len=long_len,
                    cycles=args.cycles,
                )
                pair_key = (int(num_dpus), int(num_heads), int(ratio))
                pair_index[pair_key] = {}
                for baseline, load_fn in [
                    ("Naive Static PIM", _naive_static_loads),
                    ("Clover Planner", _balanced_token_loads),
                ]:
                    loads = load_fn(seq_lens, num_dpus, num_heads)
                    row = {
                        "num_dpus": int(num_dpus),
                        "num_heads": int(num_heads),
                        "dpu_per_head_group": int(group_width),
                        "short_len": int(short_len),
                        "long_len": int(long_len),
                        "long_short_ratio": int(ratio),
                        "request_count": int(len(seq_lens)),
                        "baseline": baseline,
                        "seq_lens": json.dumps(seq_lens, ensure_ascii=False),
                        "seq_len_mean": float(_mean(seq_lens)),
                        "seq_len_cv": (pstdev(seq_lens) / _mean(seq_lens)) if _mean(seq_lens) > 0 else 0.0,
                        **_metrics(loads),
                    }
                    rows.append(row)
                    pair_index[pair_key][baseline] = row

    for pair in pair_index.values():
        naive = pair.get("Naive Static PIM")
        clover = pair.get("Clover Planner")
        if not naive or not clover:
            continue
        tail_reduction = (
            float(naive["max_tokens"]) / float(clover["max_tokens"])
            if float(clover["max_tokens"]) > 0
            else 0.0
        )
        straggler_reduction = (
            float(naive["straggler_ratio"]) / float(clover["straggler_ratio"])
            if float(clover["straggler_ratio"]) > 0
            else 0.0
        )
        naive["tail_reduction_vs_clover"] = tail_reduction
        naive["straggler_reduction_vs_clover"] = straggler_reduction
        clover["tail_reduction_vs_clover"] = 1.0
        clover["straggler_reduction_vs_clover"] = 1.0

    _write_csv(output_dir / "imbalance_range_metrics.csv", rows)
    with (output_dir / "imbalance_range_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_rows = []
    for row in rows:
        if row["baseline"] != "Naive Static PIM":
            continue
        summary_rows.append(
            [
                row["num_dpus"],
                row["num_heads"],
                row["dpu_per_head_group"],
                row["long_short_ratio"],
                _fmt(row["straggler_ratio"]),
                _fmt(row["max_tokens"], 0),
                _fmt(row.get("tail_reduction_vs_clover", 0.0)),
                _fmt(row.get("straggler_reduction_vs_clover", 0.0)),
            ]
        )
    report = "\n".join(
        [
            "# Motivation 1 Imbalance Range Sweep",
            "",
            "Main metric: `straggler_ratio = max_active_dpu_tokens / avg_active_dpu_tokens`.",
            "Tail proxy: `max_active_dpu_tokens`; smaller is better.",
            "",
            _markdown_table(
                [
                    "D",
                    "H",
                    "D/H",
                    "long/short",
                    "naive straggler",
                    "naive tail",
                    "tail reduction vs Clover",
                    "straggler reduction vs Clover",
                ],
                summary_rows,
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
