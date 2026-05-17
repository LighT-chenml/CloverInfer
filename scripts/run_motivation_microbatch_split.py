from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import pstdev
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = REPO_ROOT / "scripts"
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from run_motivation_microbatch_dynamic import (  # type: ignore
    _evaluate_fixed_batch,
    _fmt,
    _gap_curve_rows,
    _markdown_table,
    _plot_gap_ratio_svg,
    _write_csv,
    _mean,
    _scenario_lengths,
)


def _parse_int_csv(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _repeat_pattern(pattern: Sequence[int], request_count: int) -> List[int]:
    pattern = [int(item) for item in pattern]
    return [int(pattern[idx % len(pattern)]) for idx in range(max(1, int(request_count)))]


def _pattern_lengths(name: str, request_count: int, base_length: int) -> List[int]:
    base_length = max(1, int(base_length))
    short = max(1, base_length // 16)
    quarter = max(1, base_length // 8)
    half = max(1, base_length // 2)
    patterns = {
        "homogeneous": [base_length],
        "mixed": [short, base_length, quarter, half],
        "bursty": [short, short, short, base_length * 2, short, base_length, quarter, base_length * 2],
        "capacity_pressure": [base_length, base_length * 2, base_length * 2, base_length, half, base_length * 2],
    }
    if name not in patterns:
        raise ValueError(f"unknown pattern scenario {name}; choices: {', '.join(sorted(patterns))}")
    return _repeat_pattern(patterns[name], request_count)


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_suite(
    *,
    output_dir: Path,
    title: str,
    subtitle: str,
    scenarios: Sequence[Tuple[str, List[int]]],
    batch_sizes: Sequence[int],
    num_dpus: int,
    num_heads: int,
    capacity_tokens_per_dpu: int,
    pim_fixed_s: float,
    pim_per_token_s: float,
    pim_per_request_s: float,
    host_fixed_s: float,
    host_per_request_s: float,
    host_alpha: float,
    plot_x_scale: str,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_rows: List[Dict[str, Any]] = []
    summary_rows: List[List[Any]] = []

    for scenario_name, seq_lens in scenarios:
        scenario_rows: List[Dict[str, Any]] = []
        for batch_size in batch_sizes:
            seq_len_mean = _mean(seq_lens)
            seq_len_cv = (pstdev(seq_lens) / seq_len_mean) if seq_len_mean > 0 else 0.0
            row = {
                "scenario": scenario_name,
                "seq_len_min": int(min(seq_lens)),
                "seq_len_max": int(max(seq_lens)),
                "seq_len_mean": float(seq_len_mean),
                "seq_len_cv": float(seq_len_cv),
                "request_count": int(len(seq_lens)),
                "seq_lens": json.dumps(seq_lens, ensure_ascii=False),
                **_evaluate_fixed_batch(
                    seq_lens,
                    batch_size,
                    num_dpus=num_dpus,
                    num_heads=num_heads,
                    capacity_tokens_per_dpu=capacity_tokens_per_dpu,
                    pim_fixed_s=pim_fixed_s,
                    pim_per_token_s=pim_per_token_s,
                    pim_per_request_s=pim_per_request_s,
                    host_fixed_s=host_fixed_s,
                    host_per_request_s=host_per_request_s,
                    host_alpha=host_alpha,
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
                scenario_name,
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
    _write_jsonl(output_dir / "microbatch_dynamic_metrics.jsonl", metric_rows)
    curve_rows = _gap_curve_rows(metric_rows, [name for name, _ in scenarios])
    _write_csv(output_dir / "microbatch_gap_curve.csv", curve_rows)
    _plot_gap_ratio_svg(
        output_dir / "microbatch_gap_ratio.svg",
        curve_rows,
        scenario_order=[name for name, _ in scenarios],
        batch_sizes=batch_sizes,
        x_scale=plot_x_scale,
    )

    report = "\n".join(
        [
            f"# {title}",
            "",
            subtitle,
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
    return {
        "title": title,
        "output_dir": output_dir,
        "summary": report,
        "rows": metric_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/motivation_microbatch_split")
    parser.add_argument("--length-scenarios", default="short,medium,long,very_long")
    parser.add_argument("--pattern-scenarios", default="homogeneous,mixed,bursty,capacity_pressure")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--request-count", type=int, default=64)
    parser.add_argument("--pattern-base-length", type=int, default=2048)
    parser.add_argument("--num-dpus", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--capacity-tokens-per-dpu", type=int, default=4096)
    parser.add_argument("--pim-fixed-s", type=float, default=0.005)
    parser.add_argument("--pim-per-token-s", type=float, default=0.000035)
    parser.add_argument("--pim-per-request-s", type=float, default=0.0004)
    parser.add_argument("--host-fixed-s", type=float, default=0.018)
    parser.add_argument("--host-per-request-s", type=float, default=0.004)
    parser.add_argument("--host-alpha", type=float, default=0.72)
    parser.add_argument("--plot-x-scale", choices=("log2", "linear"), default="log2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    batch_sizes = _parse_int_csv(args.batch_sizes)
    length_names = [item.strip() for item in args.length_scenarios.split(",") if item.strip()]
    pattern_names = [item.strip() for item in args.pattern_scenarios.split(",") if item.strip()]

    length_scenarios = [
        (name, _scenario_lengths(name, args.request_count))
        for name in length_names
    ]
    pattern_scenarios = [
        (name, _pattern_lengths(name, args.request_count, args.pattern_base_length))
        for name in pattern_names
    ]

    length_result = _run_suite(
        output_dir=output_dir / "length_sweep",
        title="Motivation 2 Length Sweep",
        subtitle=(
            "This figure fixes the request pattern to homogeneous batches and only changes the context-length scale. "
            "It shows that the best fixed micro-batch size moves as sequences get longer."
        ),
        scenarios=length_scenarios,
        batch_sizes=batch_sizes,
        num_dpus=args.num_dpus,
        num_heads=args.num_heads,
        capacity_tokens_per_dpu=args.capacity_tokens_per_dpu,
        pim_fixed_s=args.pim_fixed_s,
        pim_per_token_s=args.pim_per_token_s,
        pim_per_request_s=args.pim_per_request_s,
        host_fixed_s=args.host_fixed_s,
        host_per_request_s=args.host_per_request_s,
        host_alpha=args.host_alpha,
        plot_x_scale=args.plot_x_scale,
    )

    pattern_result = _run_suite(
        output_dir=output_dir / "pattern_sweep",
        title="Motivation 2 Pattern Sweep",
        subtitle=(
            "This figure fixes the base context length and varies request-shape patterns such as mixed, bursty, "
            "and capacity-pressure. It isolates workload shape from length scale."
        ),
        scenarios=pattern_scenarios,
        batch_sizes=batch_sizes,
        num_dpus=args.num_dpus,
        num_heads=args.num_heads,
        capacity_tokens_per_dpu=args.capacity_tokens_per_dpu,
        pim_fixed_s=args.pim_fixed_s,
        pim_per_token_s=args.pim_per_token_s,
        pim_per_request_s=args.pim_per_request_s,
        host_fixed_s=args.host_fixed_s,
        host_per_request_s=args.host_per_request_s,
        host_alpha=args.host_alpha,
        plot_x_scale=args.plot_x_scale,
    )

    top_summary = "\n".join(
        [
            "# Motivation 2 Split Sweep",
            "",
            "This split avoids mixing context-length scale and workload-shape effects in one crowded plot.",
            "",
            f"- Length figure: `{(output_dir / 'length_sweep' / 'microbatch_gap_ratio.svg').as_posix()}`",
            f"- Pattern figure: `{(output_dir / 'pattern_sweep' / 'microbatch_gap_ratio.svg').as_posix()}`",
            f"- Length summary: `{(output_dir / 'length_sweep' / 'summary.md').as_posix()}`",
            f"- Pattern summary: `{(output_dir / 'pattern_sweep' / 'summary.md').as_posix()}`",
            "",
            "See the per-figure summaries for full tables.",
            "",
        ]
    )
    # Use the per-suite summaries as the authoritative writeout.
    (output_dir / "summary.md").write_text(top_summary, encoding="utf-8")

    print(length_result["summary"])
    print(pattern_result["summary"])


if __name__ == "__main__":
    main()
