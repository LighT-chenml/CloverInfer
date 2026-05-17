from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Sequence


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(value: Any, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}" if isinstance(value, (float, int)) else str(value)


def _svg_text(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(item) for item in row) + " |")
    return "\n".join(lines)


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


def _fixed_hotspot_lengths(
    *,
    request_count: int,
    short_len: int,
    long_len: int,
    long_period: int,
    long_offset: int,
) -> List[int]:
    lengths: List[int] = []
    period = max(1, int(long_period))
    offset = int(long_offset) % period
    for request_idx in range(max(1, int(request_count))):
        lengths.append(int(long_len) if request_idx % period == offset else int(short_len))
    return lengths


def _naive_static_loads(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> List[int]:
    loads = [0 for _ in range(max(1, int(num_dpus)))]
    for group in _head_dpu_groups(num_dpus, num_heads):
        if not group:
            continue
        for request_idx, seq_len in enumerate(seq_lens):
            loads[group[int(request_idx) % len(group)]] += int(seq_len)
    return loads


def _sweep_row(seq_lens: Sequence[int], num_dpus: int, num_heads: int) -> Dict[str, Any]:
    loads = _naive_static_loads(seq_lens, num_dpus, num_heads)
    active_loads = [float(load) for load in loads if load > 0]
    avg_load = _mean(active_loads)
    max_load = max(active_loads) if active_loads else 0.0
    min_load = min(active_loads) if active_loads else 0.0
    return {
        "num_dpus": int(num_dpus),
        "num_heads": int(num_heads),
        "dpu_per_head_group": int(max((len(group) for group in _head_dpu_groups(num_dpus, num_heads)), default=0)),
        "request_count": int(len(seq_lens)),
        "long_request_count": int(sum(1 for seq_len in seq_lens if seq_len == max(seq_lens))),
        "seq_len_min": int(min(seq_lens)) if seq_lens else 0,
        "seq_len_max": int(max(seq_lens)) if seq_lens else 0,
        "seq_len_mean": float(_mean(seq_lens)),
        "seq_len_cv": (pstdev(seq_lens) / _mean(seq_lens)) if _mean(seq_lens) > 0 else 0.0,
        "avg_load_tokens": avg_load,
        "max_load_tokens": max_load,
        "min_load_tokens": min_load,
        "straggler_ratio": max_load / avg_load if avg_load > 0 else 0.0,
        "loads": json.dumps([int(load) for load in loads], ensure_ascii=False),
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


def _plot_load_svg(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    rows = sorted(rows, key=lambda row: int(row["num_dpus"]))
    width = 850
    height = 530
    margin_left = 82
    margin_right = 170
    margin_top = 62
    margin_bottom = 76
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    dpu_counts = [int(row["num_dpus"]) for row in rows]
    x_values = {dpu_count: math.log2(float(dpu_count)) for dpu_count in dpu_counts}
    x_min = min(x_values.values())
    x_max = max(x_values.values())
    if x_max == x_min:
        x_max = x_min + 1.0

    y_max = max(max(float(row["max_load_tokens"]), float(row["avg_load_tokens"])) for row in rows)
    y_max = max(1.0, math.ceil(y_max / 5000.0) * 5000.0)

    def x_pos(num_dpus: int) -> float:
        return margin_left + ((x_values[int(num_dpus)] - x_min) / (x_max - x_min)) * chart_w

    def y_pos(tokens: float) -> float:
        return margin_top + (1.0 - (float(tokens) / y_max)) * chart_h

    max_color = "#D55E00"
    avg_color = "#0072B2"
    svg: List[str] = [
        '<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Georgia, 'Times New Roman', serif; fill: #1f2933; }",
        ".axis { stroke: #1f2933; stroke-width: 1.4; }",
        ".grid { stroke: #d8dee9; stroke-width: 1; }",
        ".tick { stroke: #1f2933; stroke-width: 1; }",
        ".caption { fill: #52616b; font-size: 13px; }",
        ".label { font-size: 15px; font-weight: 600; }",
        ".small { font-size: 12px; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{margin_left}" y="30" font-size="22" font-weight="700">'
        "Naive PIM DPU load imbalance</text>",
        f'<text x="{margin_left}" y="51" class="caption">'
        "Fixed workload/config; only the DPU count changes.</text>",
    ]

    tick_step = 5000.0
    if y_max > 60000:
        tick_step = 10000.0
    y_tick = 0.0
    while y_tick <= y_max + 1e-9:
        y = y_pos(y_tick)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + chart_w}" '
            f'y2="{y:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="small">{int(y_tick):,}</text>'
        )
        y_tick += tick_step

    for dpu_count in dpu_counts:
        x = x_pos(dpu_count)
        svg.append(
            f'<line x1="{x:.2f}" y1="{margin_top}" x2="{x:.2f}" '
            f'y2="{margin_top + chart_h}" class="grid" opacity="0.35"/>'
        )
        svg.append(
            f'<line x1="{x:.2f}" y1="{margin_top + chart_h}" x2="{x:.2f}" '
            f'y2="{margin_top + chart_h + 6}" class="tick"/>'
        )
        svg.append(
            f'<text x="{x:.2f}" y="{margin_top + chart_h + 25}" '
            f'text-anchor="middle" class="small">{dpu_count}</text>'
        )

    svg.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top + chart_h}" '
            f'x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" class="axis"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + chart_h}" class="axis"/>',
            f'<text x="{margin_left + chart_w / 2:.2f}" y="{height - 22}" '
            'text-anchor="middle" class="label">Number of DPUs</text>',
            f'<text x="24" y="{margin_top + chart_h / 2:.2f}" '
            'text-anchor="middle" class="label" '
            f'transform="rotate(-90 24 {margin_top + chart_h / 2:.2f})">DPU load (tokens)</text>',
        ]
    )

    series = [
        ("Max DPU load (straggler)", "max_load_tokens", max_color),
        ("Average DPU load", "avg_load_tokens", avg_color),
    ]
    for label, field, color in series:
        points = [
            f'{x_pos(int(row["num_dpus"])):.2f},{y_pos(float(row[field])):.2f}'
            for row in rows
        ]
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{" ".join(points)}"/>'
        )
        for row in rows:
            x = x_pos(int(row["num_dpus"]))
            y = y_pos(float(row[field]))
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.6" fill="{color}" '
                f'stroke="#ffffff" stroke-width="1.2"/>'
            )

    legend_x = margin_left + chart_w + 34
    legend_y = margin_top + 18
    svg.append(f'<text x="{legend_x}" y="{legend_y - 18}" class="label">Metric</text>')
    for idx, (label, _field, color) in enumerate(series):
        y = legend_y + idx * 30
        svg.append(
            f'<line x1="{legend_x}" y1="{y:.2f}" x2="{legend_x + 24}" '
            f'y2="{y:.2f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>'
        )
        svg.append(f'<text x="{legend_x + 32}" y="{y + 4:.2f}" class="small">{_svg_text(label)}</text>')

    last = rows[-1]
    svg.append(
        f'<text x="{legend_x}" y="{legend_y + 86}" class="caption">'
        f'At D={int(last["num_dpus"])}, max/avg = {float(last["straggler_ratio"]):.2f}x</text>'
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/motivation_imbalance_dpu_sweep")
    parser.add_argument("--dpu-counts", default="16,32,64,128")
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--request-count", type=int, default=128)
    parser.add_argument("--short-len", type=int, default=128)
    parser.add_argument("--long-len", type=int, default=8192)
    parser.add_argument("--long-period", type=int, default=32)
    parser.add_argument("--long-offset", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    seq_lens = _fixed_hotspot_lengths(
        request_count=args.request_count,
        short_len=args.short_len,
        long_len=args.long_len,
        long_period=args.long_period,
        long_offset=args.long_offset,
    )
    rows = [
        _sweep_row(seq_lens, num_dpus=num_dpus, num_heads=args.num_heads)
        for num_dpus in _parse_int_csv(args.dpu_counts)
    ]

    _write_csv(output_dir / "imbalance_dpu_sweep.csv", rows)
    with (output_dir / "imbalance_dpu_sweep.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    _plot_load_svg(output_dir / "imbalance_dpu_load.svg", rows)

    table_rows = [
        [
            row["num_dpus"],
            row["dpu_per_head_group"],
            _fmt(row["avg_load_tokens"], 0),
            _fmt(row["max_load_tokens"], 0),
            _fmt(row["straggler_ratio"]),
        ]
        for row in rows
    ]
    report = "\n".join(
        [
            "# Motivation 1 DPU Load Sweep",
            "",
            "This sweep fixes the request distribution and model head count, then varies only the number of DPUs.",
            "The plotted quantities are the naive static PIM `max DPU load` and `average DPU load`.",
            "",
            f"- Figure: `{(output_dir / 'imbalance_dpu_load.svg').as_posix()}`",
            f"- Plot CSV: `{(output_dir / 'imbalance_dpu_sweep.csv').as_posix()}`",
            "",
            _markdown_table(
                [
                    "D",
                    "D/H",
                    "avg load",
                    "max load",
                    "max/avg",
                ],
                table_rows,
            ),
            "",
        ]
    )
    (output_dir / "summary.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
