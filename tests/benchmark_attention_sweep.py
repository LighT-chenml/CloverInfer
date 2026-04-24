import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Tuple

import ray
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def parse_int_list(value: str) -> List[int]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return items


def parse_str_list(value: str) -> List[str]:
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(part)
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated list of strings")
    return items


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def summarize_stage_timing(stage_timing: Dict[str, object]) -> Dict[str, float]:
    scheduler = stage_timing["scheduler"]
    actors = stage_timing["actors"]
    counts = stage_timing["counts"]
    return {
        "prefill_rpc_s": float(scheduler["prefill_rpc_s"]),
        "attention_decode_rpc_s": float(scheduler["attention_decode_rpc_s"]),
        "prepare_attention_rpc_s": float(scheduler["prepare_attention_rpc_s"]),
        "finish_layer_rpc_s": float(scheduler["finish_layer_rpc_s"]),
        "sample_next_token_rpc_s": float(scheduler["sample_next_token_rpc_s"]),
        "total_rpc_s": float(scheduler["total_rpc_s"]),
        "prefill_compute_s": float(actors["prefill_compute_s"]),
        "attention_decode_compute_s": float(actors["attention_decode_compute_s"]),
        "dense_prepare_attention_compute_s": float(actors["dense_prepare_attention_compute_s"]),
        "dense_finish_layer_compute_s": float(actors["dense_finish_layer_compute_s"]),
        "dense_sample_next_token_compute_s": float(actors["dense_sample_next_token_compute_s"]),
        "total_compute_s": float(actors["total_compute_s"]),
        "decode_steps": float(counts["decode_steps"]),
        "decode_layers": float(counts["decode_layers"]),
        "scheduler_overhead_s": float(stage_timing["scheduler_overhead_s"]),
    }


def build_prompt_for_target_tokens(tokenizer, target_tokens: int) -> Tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target token length must be positive")

    prefix_ids = tokenizer.encode("Context:", add_special_tokens=False)
    filler_ids = tokenizer.encode(" clover", add_special_tokens=False)
    if not prefix_ids or not filler_ids:
        raise ValueError("failed to build prompt tokens for sweep")

    token_ids = list(prefix_ids)
    while len(token_ids) < target_tokens:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:target_tokens]

    prompt = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Decode/encode is expected to round-trip cleanly for the local tokenizer.
    # Keep the actual length in the record either way, so experiments remain
    # explicit even if tokenizer normalization changes in the future.
    return prompt, len(actual_ids)


def make_scheduler(args, attention_backend: str, mixed_heads: int) -> ray.actor.ActorHandle:
    cluster = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource="prefill_gpu",
        decode_dense_resource="decode_dense_gpu",
        attention_resource="attention_pim",
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend=attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_qk_mixed_enabled=args.pim_qk_mixed_enabled,
        pim_qk_mixed_heads=mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
    )
    model = ModelConfig(model_path=args.model, max_new_tokens=args.max_new_tokens)
    return GlobalScheduler.remote(cluster, model)


def run_case(
    args,
    attention_backend: str,
    mixed_heads: int,
    prompt: str,
    prompt_token_length: int,
) -> Dict[str, object]:
    scheduler = make_scheduler(args, attention_backend, mixed_heads)
    info = ray.get(scheduler.initialize_cluster.remote())

    records = []
    ttfts = []
    tpots = []
    latencies = []
    throughputs = []
    qk_diffs = []
    qk_counts = []
    qk_failures = []
    stage_summaries = []

    for repeat_idx in range(args.repeats):
        started_at = time.time()
        output, metrics = ray.get(
            scheduler.submit_request.remote(
                prompt,
                return_metrics=True,
                max_new_tokens=args.max_new_tokens,
            )
        )
        finished_at = time.time()
        backend_debug = metrics["attention_backend"].get("backend_debug", {})

        ttfts.append(float(metrics["ttft"]))
        tpots.append(float(metrics["tpot"]))
        latencies.append(float(metrics["latency"]))
        throughputs.append(float(metrics["throughput"]))
        qk_diffs.append(float(backend_debug.get("qk_mixed_last_max_abs_diff", 0.0)))
        qk_counts.append(int(backend_debug.get("qk_mixed_count", 0)))
        qk_failures.append(int(backend_debug.get("qk_check_failures", 0)))
        stage_summaries.append(summarize_stage_timing(metrics["stage_timing"]))

        record = {
            "case": {
                "attention_backend": attention_backend,
                "pim_qk_mixed_enabled": bool(args.pim_qk_mixed_enabled),
                "pim_qk_mixed_heads": int(mixed_heads),
                "pim_qk_mixed_window": int(args.pim_qk_mixed_window),
                "pim_num_dpus": int(args.pim_num_dpus),
                "pim_length": int(args.pim_length),
                "max_new_tokens": int(args.max_new_tokens),
                "prompt_token_length": int(prompt_token_length),
                "repeat_idx": int(repeat_idx),
            },
            "placement": info,
            "prompt": prompt,
            "output": output,
            "metrics": metrics,
            "started_at": started_at,
            "finished_at": finished_at,
        }
        records.append(record)

    summary = {
        "attention_backend": attention_backend,
        "pim_qk_mixed_enabled": bool(args.pim_qk_mixed_enabled),
        "pim_qk_mixed_heads": int(mixed_heads),
        "prompt_token_length": int(prompt_token_length),
        "repeats": int(args.repeats),
        "avg_ttft": statistics.mean(ttfts),
        "avg_tpot": statistics.mean(tpots),
        "avg_latency": statistics.mean(latencies),
        "avg_throughput": statistics.mean(throughputs),
        "p95_latency": percentile(latencies, 0.95),
        "max_qk_mixed_last_abs_diff": max(qk_diffs) if qk_diffs else 0.0,
        "final_qk_mixed_count": qk_counts[-1] if qk_counts else 0,
        "final_qk_check_failures": qk_failures[-1] if qk_failures else 0,
        "avg_stage_timing": {
            key: statistics.mean([stage[key] for stage in stage_summaries])
            for key in stage_summaries[0]
        } if stage_summaries else {},
    }
    return {"records": records, "summary": summary}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--prompt", default="Hello CloverInfer")
    parser.add_argument("--prompt-token-lengths", type=parse_int_list, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--attention-backends", type=parse_str_list, default=["cpu", "pim_naive"])
    parser.add_argument("--pim-qk-mixed-heads-list", type=parse_int_list, default=[0, 1, 2, 4])
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "attention_sweep.jsonl"),
    )
    args = parser.parse_args()

    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")
    if not args.pim_qk_mixed_enabled and not args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = False

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompt_cases = [
        (
            args.prompt,
            len(tokenizer.encode(args.prompt, add_special_tokens=False)),
        )
    ]
    if args.prompt_token_lengths:
        prompt_cases = []
        for target_tokens in args.prompt_token_lengths:
            prompt, actual_tokens = build_prompt_for_target_tokens(tokenizer, target_tokens)
            prompt_cases.append((prompt, actual_tokens))

    ray.init(
        address=args.address,
        runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}},
    )

    summaries = []
    with open(args.output, "w", encoding="utf-8") as f:
        for prompt, prompt_token_length in prompt_cases:
            for backend in args.attention_backends:
                mixed_heads_list = args.pim_qk_mixed_heads_list if backend == "pim_naive" else [0]
                for mixed_heads in mixed_heads_list:
                    result = run_case(args, backend, mixed_heads, prompt, prompt_token_length)
                    for record in result["records"]:
                        f.write(json.dumps({"type": "record", **record}) + "\n")
                    f.write(json.dumps({"type": "summary", **result["summary"]}) + "\n")
                    summaries.append(result["summary"])
                    print(json.dumps(result["summary"], ensure_ascii=True))

    print(f"Saved benchmark records to {args.output}")


if __name__ == "__main__":
    main()
