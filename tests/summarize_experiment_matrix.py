from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_case_result_rows(matrix_dir: str) -> List[Dict[str, object]]:
    cases_dir = os.path.join(matrix_dir, "cases")
    result_rows: List[Dict[str, object]] = []
    if not os.path.isdir(cases_dir):
        return result_rows

    for case_slug in sorted(os.listdir(cases_dir)):
        result_path = os.path.join(cases_dir, case_slug, "results.jsonl")
        if not os.path.exists(result_path):
            continue
        for row in read_jsonl(result_path):
            summary = dict(row.get("summary", {}))
            if not summary:
                continue
            last_metrics = {}
            records = row.get("records", [])
            if records:
                last_metrics = dict(records[-1].get("metrics", {}))
            dense_batching = dict(last_metrics.get("scheduler_dense_continuous_batching", {}))
            predictive_last_batch = dict(dense_batching.get("predictive_last_batch", {}))
            result_rows.append(
                {
                    "case_slug": case_slug,
                    "baseline": row.get("baseline"),
                    "baseline_key": row.get("baseline_key"),
                    "internal_baseline": row.get("internal_baseline"),
                    "avg_ttft": summary.get("avg_ttft"),
                    "avg_tpot": summary.get("avg_tpot"),
                    "avg_latency": summary.get("avg_latency"),
                    "avg_throughput": summary.get("avg_throughput"),
                    "avg_total_tokens": summary.get("avg_total_tokens"),
                    "num_requests": summary.get("num_requests"),
                    "wall_time_s": summary.get("wall_time_s"),
                    "request_throughput_rps": summary.get("request_throughput_rps"),
                    "output_token_throughput_tps": summary.get("output_token_throughput_tps"),
                    "requested_concurrency": summary.get("requested_concurrency"),
                    "effective_concurrency": summary.get("effective_concurrency"),
                    "predictive_enabled": dense_batching.get("predictive_enabled"),
                    "predictive_batch_decisions": dense_batching.get("predictive_batch_decisions"),
                    "predictive_batch_fallbacks": dense_batching.get("predictive_batch_fallbacks"),
                    "predictive_model_updates": dense_batching.get("predictive_model_updates"),
                    "predictive_ready_requests": dense_batching.get("predictive_ready_requests"),
                    "predictive_unready_requests": dense_batching.get("predictive_unready_requests"),
                    "dense_flushes": dense_batching.get("flushes"),
                    "dense_max_batch_size": dense_batching.get("max_observed_size"),
                    "dense_reordered_flushes": dense_batching.get("reordered_flushes"),
                    "dense_window_s": dense_batching.get("window_s"),
                    "dense_wait_s": dense_batching.get("wait_s"),
                    "predictive_last_batch_used": predictive_last_batch.get("used"),
                    "predictive_last_batch_size": len(predictive_last_batch.get("request_ids", [])),
                }
            )
    return result_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-dir", required=True)
    parser.add_argument("--output-jsonl", default="")
    parser.add_argument("--print-limit", type=int, default=20)
    args = parser.parse_args()

    rows = load_case_result_rows(args.matrix_dir)
    if args.output_jsonl:
        with open(args.output_jsonl, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({"matrix_dir": args.matrix_dir, "num_rows": len(rows)}, ensure_ascii=False))
    for row in rows[: max(0, int(args.print_limit))]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
