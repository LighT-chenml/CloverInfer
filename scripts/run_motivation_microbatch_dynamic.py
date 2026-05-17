from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, Iterable, List, Sequence

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.clover_planner import plan_sharding


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _scenario_lengths(name: str, request_count: int) -> List[int]:
    name = str(name).strip().lower()
    patterns = {
        "short": [128],
        "medium": [512],
        "long": [2048],
        "very_long": [4096],
        "mixed": [128, 2048, 256, 1024],
        "bursty": [128, 128, 128, 4096, 128, 2048, 256, 4096],
        "capacity_pressure": [2048, 4096, 4096, 2048, 1024, 4096],
    }
    if name not in patterns:
        if ":" in name:
            pattern = [int(item) for item in name.split(":", 1)[1].split("-") if item]
        else:
            raise ValueError(f"unknown scenario {name}; choices: {', '.join(sorted(patterns))}")
    else:
        pattern = patterns[name]
    return [int(pattern[idx % len(pattern)]) for idx in range(max(1, int(request_count)))]


def _batches(seq_lens: Sequence[int], batch_size: int) -> List[List[int]]:
    size = max(1, int(batch_size))
    return [list(seq_lens[idx : idx + size]) for idx in range(0, len(seq_lens), size)]


def _batch_model(
    seq_lens: Sequence[int],
    *,
    num_dpus: int,
    num_heads: int,
    capacity_tokens_per_dpu: int,
    pim_fixed_s: float,
    pim_per_token_s: float,
    pim_per_request_s: float,
    host_fixed_s: float,
    host_per_request_s: float,
    host_alpha: float,
) -> Dict[str, Any]:
    requests = [
        {
            "request_id": f"r{idx}",
            "seq_len": int(seq_len),
            "num_new_tokens": 1,
        }
        for idx, seq_len in enumerate(seq_lens)
    ]
    plan = plan_sharding(requests, D=int(num_dpus), H=int(num_heads))
    dpu_loads = [int(value) for value in dict(plan.get("dpu_loads", {}) or {}).values()]
    max_dpu_tokens = max(dpu_loads, default=0)
    batch_size = len(seq_lens)
    pim_time = float(pim_fixed_s + pim_per_token_s * max_dpu_tokens + pim_per_request_s * batch_size)
    host_time = float(host_fixed_s + host_per_request_s * (batch_size ** float(host_alpha)))
    stage_time = max(pim_time, host_time)
    capacity_ok = max_dpu_tokens <= int(capacity_tokens_per_dpu)
    alignment_ratio = min(pim_time, host_time) / stage_time if stage_time > 0 else 0.0
    return {
        "batch_size_observed": int(batch_size),
        "max_dpu_tokens": int(max_dpu_tokens),
        "capacity_ok": bool(capacity_ok),
        "capacity_usage": (float(max_dpu_tokens) / float(capacity_tokens_per_dpu))
        if capacity_tokens_per_dpu > 0
        else 0.0,
        "pim_time_s": pim_time,
        "host_time_s": host_time,
        "stage_time_s": stage_time,
        "alignment_ratio": alignment_ratio,
        "stage_bottleneck": "pim" if pim_time >= host_time else "host",
        "dpu_loads": json.dumps(dpu_loads, ensure_ascii=False),
    }


def _evaluate_fixed_batch(
    seq_lens: Sequence[int],
    batch_size: int,
    *,
    num_dpus: int,
    num_heads: int,
    capacity_tokens_per_dpu: int,
    pim_fixed_s: float,
    pim_per_token_s: float,
    pim_per_request_s: float,
    host_fixed_s: float,
    host_per_request_s: float,
    host_alpha: float,
) -> Dict[str, Any]:
    modeled_batches = [
        _batch_model(
            batch,
            num_dpus=num_dpus,
            num_heads=num_heads,
            capacity_tokens_per_dpu=capacity_tokens_per_dpu,
            pim_fixed_s=pim_fixed_s,
            pim_per_token_s=pim_per_token_s,
            pim_per_request_s=pim_per_request_s,
            host_fixed_s=host_fixed_s,
            host_per_request_s=host_per_request_s,
            host_alpha=host_alpha,
        )
        for batch in _batches(seq_lens, batch_size)
    ]
    feasible_batches = [batch for batch in modeled_batches if bool(batch["capacity_ok"])]
    capacity_ok = len(feasible_batches) == len(modeled_batches)
    total_requests = sum(int(batch["batch_size_observed"]) for batch in modeled_batches)
    total_stage_time = sum(float(batch["stage_time_s"]) for batch in modeled_batches)
    total_pim_time = sum(float(batch["pim_time_s"]) for batch in modeled_batches)
    total_host_time = sum(float(batch["host_time_s"]) for batch in modeled_batches)
    throughput = float(total_requests) / total_stage_time if capacity_ok and total_stage_time > 0 else 0.0
    avg_alignment = _mean(batch["alignment_ratio"] for batch in modeled_batches)
    min_alignment = min((float(batch["alignment_ratio"]) for batch in modeled_batches), default=0.0)
    avg_gap = _mean(1.0 - float(batch["alignment_ratio"]) for batch in modeled_batches)
    max_gap = max((1.0 - float(batch["alignment_ratio"]) for batch in modeled_batches), default=0.0)
    p95_capacity = sorted(float(batch["capacity_usage"]) for batch in modeled_batches)
    p95_capacity_usage = p95_capacity[int(0.95 * (len(p95_capacity) - 1))] if p95_capacity else 0.0
    return {
        "batch_size": int(batch_size),
        "batch_count": int(len(modeled_batches)),
        "capacity_ok": bool(capacity_ok),
        "throughput_rps": throughput,
        "total_stage_time_s": total_stage_time,
        "total_pim_time_s": total_pim_time,
        "total_host_time_s": total_host_time,
        "avg_pim_time_s": _mean(batch["pim_time_s"] for batch in modeled_batches),
        "avg_host_time_s": _mean(batch["host_time_s"] for batch in modeled_batches),
        "avg_stage_time_s": _mean(batch["stage_time_s"] for batch in modeled_batches),
        "avg_alignment_ratio": avg_alignment,
        "min_alignment_ratio": min_alignment,
        "gap_ratio": avg_gap,
        "avg_gap_ratio": avg_gap,
        "max_gap_ratio": max_gap,
        "avg_capacity_usage": _mean(batch["capacity_usage"] for batch in modeled_batches),
        "p95_capacity_usage": p95_capacity_usage,
        "max_dpu_tokens_peak": max((int(batch["max_dpu_tokens"]) for batch in modeled_batches), default=0),
        "pim_bottleneck_batches": sum(1 for batch in modeled_batches if batch["stage_bottleneck"] == "pim"),
        "host_bottleneck_batches": sum(1 for batch in modeled_batches if batch["stage_bottleneck"] == "host"),
        "modeled_batches": json.dumps(modeled_batches, ensure_ascii=False),
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


def _gap_curve_rows(rows: Sequence[Dict[str, Any]], scenario_order: Sequence[str]) -> List[Dict[str, Any]]:
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)

    curve_rows: List[Dict[str, Any]] = []
    for scenario in scenario_order:
        scenario_rows = sorted(by_scenario.get(str(scenario), []), key=lambda row: int(row["batch_size"]))
        feasible = [row for row in scenario_rows if bool(row["capacity_ok"])]
        best = (
            min(
                feasible,
                key=lambda row: (
                    float(row["gap_ratio"]),
                    -float(row["throughput_rps"]),
                ),
            )
            if feasible
            else None
        )
        for row in scenario_rows:
            stage_bottleneck = "mixed"
            if int(row["pim_bottleneck_batches"]) and not int(row["host_bottleneck_batches"]):
                stage_bottleneck = "pim"
            elif int(row["host_bottleneck_batches"]) and not int(row["pim_bottleneck_batches"]):
                stage_bottleneck = "host"
            curve_rows.append(
                {
                    "scenario": scenario,
                    "batch_size": int(row["batch_size"]),
                    "gap_ratio": float(row["gap_ratio"]),
                    "avg_alignment_ratio": float(row["avg_alignment_ratio"]),
                    "avg_pim_time_s": float(row["avg_pim_time_s"]),
                    "avg_host_time_s": float(row["avg_host_time_s"]),
                    "throughput_rps": float(row["throughput_rps"]),
                    "capacity_ok": bool(row["capacity_ok"]),
                    "p95_capacity_usage": float(row["p95_capacity_usage"]),
                    "stage_bottleneck": stage_bottleneck,
                    "is_gap_optimal": bool(best is not None and int(row["batch_size"]) == int(best["batch_size"])),
                }
            )
    return curve_rows


def _plot_gap_ratio_svg(
    path: Path,
    rows: Sequence[Dict[str, Any]],
    *,
    scenario_order: Sequence[str],
    batch_sizes: Sequence[int],
    x_scale: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [row for row in rows if str(row.get("scenario")) in set(scenario_order)]
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    width = 980
    height = 600
    margin_left = 82
    margin_right = 260
    margin_top = 62
    margin_bottom = 84
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom

    batch_sizes = sorted({int(batch_size) for batch_size in batch_sizes if int(batch_size) > 0})
    if not batch_sizes:
        batch_sizes = sorted({int(row["batch_size"]) for row in rows})
    if x_scale == "linear":
        x_values = {batch_size: float(batch_size) for batch_size in batch_sizes}
    else:
        x_values = {batch_size: math.log2(float(batch_size)) for batch_size in batch_sizes}
    x_min = min(x_values.values())
    x_max = max(x_values.values())
    if x_max == x_min:
        x_max = x_min + 1.0

    max_gap = max(float(row["gap_ratio"]) for row in rows)
    y_max = max(0.2, min(1.0, math.ceil(max_gap * 10.0) / 10.0))
    if y_max <= 0.0:
        y_max = 0.2

    def x_pos(batch_size: int) -> float:
        return margin_left + ((x_values[int(batch_size)] - x_min) / (x_max - x_min)) * chart_w

    def y_pos(gap_ratio: float) -> float:
        return margin_top + (1.0 - (float(gap_ratio) / y_max)) * chart_h

    colors = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#E69F00",
        "#56B4E9",
        "#7F7F7F",
        "#332288",
        "#88CCEE",
    ]
    by_scenario: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario"]), []).append(row)
    for scenario_rows in by_scenario.values():
        scenario_rows.sort(key=lambda row: int(row["batch_size"]))

    y_tick_count = int(round(y_max * 10.0))
    y_ticks = [idx / 10.0 for idx in range(0, y_tick_count + 1, 2)]
    if not y_ticks or y_ticks[-1] < y_max:
        y_ticks.append(y_max)

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
        "Micro-batch alignment gap across workloads</text>",
        f'<text x="{margin_left}" y="51" class="caption">'
        "Lower is better; diamond marks the best capacity-feasible batch size.</text>",
    ]

    for tick in y_ticks:
        y = y_pos(tick)
        svg.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{margin_left + chart_w}" '
            f'y2="{y:.2f}" class="grid"/>'
        )
        svg.append(
            f'<text x="{margin_left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'class="small">{tick:.1f}</text>'
        )

    for batch_size in batch_sizes:
        x = x_pos(batch_size)
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
            f'text-anchor="middle" class="small">{batch_size}</text>'
        )

    svg.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top + chart_h}" '
            f'x2="{margin_left + chart_w}" y2="{margin_top + chart_h}" class="axis"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" '
            f'x2="{margin_left}" y2="{margin_top + chart_h}" class="axis"/>',
            f'<text x="{margin_left + chart_w / 2:.2f}" y="{height - 24}" '
            'text-anchor="middle" class="label">Micro-batch size</text>',
            f'<text x="25" y="{margin_top + chart_h / 2:.2f}" '
            'text-anchor="middle" class="label" '
            f'transform="rotate(-90 25 {margin_top + chart_h / 2:.2f})">Gap ratio</text>',
        ]
    )

    legend_x = margin_left + chart_w + 34
    legend_y = margin_top + 8
    svg.append(f'<text x="{legend_x}" y="{legend_y - 18}" class="label">Workload</text>')

    for idx, scenario in enumerate(scenario_order):
        scenario_rows = by_scenario.get(str(scenario), [])
        if not scenario_rows:
            continue
        color = colors[idx % len(colors)]
        points = [
            f'{x_pos(int(row["batch_size"])):.2f},{y_pos(float(row["gap_ratio"])):.2f}'
            for row in scenario_rows
        ]
        svg.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.8" '
            f'stroke-linejoin="round" stroke-linecap="round" points="{" ".join(points)}"/>'
        )

        best = next((row for row in scenario_rows if bool(row["is_gap_optimal"])), None)
        for row in scenario_rows:
            x = x_pos(int(row["batch_size"]))
            y = y_pos(float(row["gap_ratio"]))
            feasible = bool(row["capacity_ok"])
            fill = color if feasible else "#ffffff"
            opacity = "1.0" if feasible else "0.45"
            svg.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.2" fill="{fill}" '
                f'stroke="{color}" stroke-width="1.8" opacity="{opacity}"/>'
            )

        if best is not None:
            bx = x_pos(int(best["batch_size"]))
            by = y_pos(float(best["gap_ratio"]))
            svg.append(
                f'<path d="M {bx:.2f} {by - 7:.2f} L {bx + 7:.2f} {by:.2f} '
                f'L {bx:.2f} {by + 7:.2f} L {bx - 7:.2f} {by:.2f} Z" '
                f'fill="{color}" stroke="#ffffff" stroke-width="1.4"/>'
            )

        ly = legend_y + idx * 30
        svg.append(
            f'<line x1="{legend_x}" y1="{ly:.2f}" x2="{legend_x + 24}" '
            f'y2="{ly:.2f}" stroke="{color}" stroke-width="2.8" stroke-linecap="round"/>'
        )
        if best is not None:
            label = (
                f'{_svg_text(scenario)} '
                f'(best B={int(best["batch_size"])}, gap={float(best["gap_ratio"]):.3f})'
            )
        else:
            label = f'{_svg_text(scenario)} (no feasible B)'
        svg.append(f'<text x="{legend_x + 32}" y="{ly + 4:.2f}" class="small">{label}</text>')

    svg.append(
        f'<text x="{margin_left}" y="{height - 8}" class="caption">'
        "Gap ratio = |T_pim - T_host| / max(T_pim, T_host); hollow points exceed DPU capacity.</text>"
    )
    svg.append("</svg>")
    path.write_text("\n".join(svg) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/motivation_microbatch_dynamic")
    parser.add_argument("--scenarios", default="short,medium,long,very_long,mixed,bursty,capacity_pressure")
    parser.add_argument(
        "--plot-scenarios",
        default="",
        help="Comma-separated subset/order to draw in the SVG. Defaults to --scenarios.",
    )
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--plot-x-scale", choices=("log2", "linear"), default="log2")
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--num-dpus", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--capacity-tokens-per-dpu", type=int, default=4096)
    parser.add_argument("--pim-fixed-s", type=float, default=0.005)
    parser.add_argument("--pim-per-token-s", type=float, default=0.000035)
    parser.add_argument("--pim-per-request-s", type=float, default=0.0004)
    parser.add_argument("--host-fixed-s", type=float, default=0.018)
    parser.add_argument("--host-per-request-s", type=float, default=0.004)
    parser.add_argument("--host-alpha", type=float, default=0.72)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[List[Any]] = []
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    plot_scenarios = [item.strip() for item in args.plot_scenarios.split(",") if item.strip()] or scenarios
    batch_sizes = _parse_int_csv(args.batch_sizes)
    for scenario in scenarios:
        seq_lens = _scenario_lengths(scenario, args.request_count)
        scenario_rows: List[Dict[str, Any]] = []
        for batch_size in batch_sizes:
            row = {
                "scenario": scenario,
                "seq_len_min": int(min(seq_lens)),
                "seq_len_max": int(max(seq_lens)),
                "seq_len_mean": float(_mean(seq_lens)),
                "seq_len_cv": (pstdev(seq_lens) / _mean(seq_lens)) if _mean(seq_lens) > 0 else 0.0,
                "request_count": int(len(seq_lens)),
                "seq_lens": json.dumps(seq_lens, ensure_ascii=False),
                **_evaluate_fixed_batch(
                    seq_lens,
                    batch_size,
                    num_dpus=args.num_dpus,
                    num_heads=args.num_heads,
                    capacity_tokens_per_dpu=args.capacity_tokens_per_dpu,
                    pim_fixed_s=args.pim_fixed_s,
                    pim_per_token_s=args.pim_per_token_s,
                    pim_per_request_s=args.pim_per_request_s,
                    host_fixed_s=args.host_fixed_s,
                    host_per_request_s=args.host_per_request_s,
                    host_alpha=args.host_alpha,
                ),
            }
            scenario_rows.append(row)
            metric_rows.append(row)

        feasible = [row for row in scenario_rows if bool(row["capacity_ok"])]
        best_throughput = max(feasible, key=lambda row: float(row["throughput_rps"])) if feasible else None
        best_gap = (
            min(
                feasible,
                key=lambda row: (
                    float(row["gap_ratio"]),
                    -float(row["throughput_rps"]),
                ),
            )
            if feasible
            else None
        )
        summary_rows.append(
            [
                scenario,
                int(min(seq_lens)),
                int(max(seq_lens)),
                _fmt((pstdev(seq_lens) / _mean(seq_lens)) if _mean(seq_lens) > 0 else 0.0),
                "none" if best_gap is None else best_gap["batch_size"],
                "none" if best_gap is None else _fmt(best_gap["gap_ratio"]),
                "none" if best_throughput is None else best_throughput["batch_size"],
                "none" if best_throughput is None else _fmt(best_throughput["throughput_rps"]),
                "none" if best_gap is None else _fmt(best_gap["p95_capacity_usage"]),
            ]
        )

    _write_csv(output_dir / "microbatch_dynamic_metrics.csv", metric_rows)
    curve_rows = _gap_curve_rows(metric_rows, plot_scenarios)
    _write_csv(output_dir / "microbatch_gap_curve.csv", curve_rows)
    _plot_gap_ratio_svg(
        output_dir / "microbatch_gap_ratio.svg",
        curve_rows,
        scenario_order=plot_scenarios,
        batch_sizes=batch_sizes,
        x_scale=args.plot_x_scale,
    )
    with (output_dir / "microbatch_dynamic_metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in metric_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    report = "\n".join(
        [
            "# Motivation 2 Dynamic Micro-Batch Sweep",
            "",
            "This sweep shows that the best fixed micro-batch size depends on context length distribution and capacity pressure.",
            "The plotted gap ratio is `|T_pim - T_host| / max(T_pim, T_host)`, so lower is better.",
            "The gap optimum minimizes this ratio among capacity-feasible fixed batch sizes.",
            "",
            f"- Figure: `{(output_dir / 'microbatch_gap_ratio.svg').as_posix()}`",
            f"- Plot CSV: `{(output_dir / 'microbatch_gap_curve.csv').as_posix()}`",
            "",
            _markdown_table(
                [
                    "scenario",
                    "min len",
                    "max len",
                    "seq CV",
                    "gap-opt B",
                    "min gap",
                    "throughput-opt B",
                    "throughput",
                    "p95 capacity at gap-opt",
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
