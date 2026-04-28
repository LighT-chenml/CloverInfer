import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

import ray
from transformers import AutoTokenizer

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


def parse_int_list(value: str) -> List[int]:
    values = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    if not values:
        raise argparse.ArgumentTypeError("expected a comma-separated list of integers")
    return values


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


def build_prompt_for_target_tokens(tokenizer, target_tokens: int) -> tuple[str, int]:
    prefix_ids = tokenizer.encode("Context:", add_special_tokens=False)
    filler_ids = tokenizer.encode(" clover", add_special_tokens=False)
    token_ids = list(prefix_ids)
    while len(token_ids) < target_tokens:
        token_ids.extend(filler_ids)
    token_ids = token_ids[:target_tokens]
    prompt = tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
    actual_ids = tokenizer.encode(prompt, add_special_tokens=False)
    return prompt, len(actual_ids)


def make_scheduler(args, token_interval: int, layer_interval: int):
    cluster = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource="prefill_gpu",
        decode_dense_resource="decode_dense_gpu",
        attention_resource="attention_pim",
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend="cloverinfer",
        pim_num_dpus=args.pim_num_dpus,
        pim_resident_store_backend=args.pim_resident_store_backend,
        pim_qk_mixed_enabled=True,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
        clover_cpu_shadow_enabled=True,
        clover_shadow_checks_enabled=True,
        clover_op_profiling_enabled=True,
        clover_shadow_check_token_interval=token_interval,
        clover_shadow_check_layer_interval=layer_interval,
        clover_host_qk_mixed_enabled=False,
    )
    model = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    return GlobalScheduler.remote(cluster, model)


def summarize_case(case_records: List[Dict[str, object]]) -> Dict[str, object]:
    latencies = [float(item["metrics"]["latency"]) for item in case_records]
    ttfts = [float(item["metrics"]["ttft"]) for item in case_records]
    tpots = [float(item["metrics"]["tpot"]) for item in case_records]
    throughputs = [float(item["metrics"]["throughput"]) for item in case_records]
    invocations = [
        int(item["metrics"]["attention_backend"]["backend_debug"].get("clover_shadow_check_invocations", 0))
        for item in case_records
    ]
    skips = [
        int(item["metrics"]["attention_backend"]["backend_debug"].get("clover_shadow_check_skips", 0))
        for item in case_records
    ]
    timing_keys = [
        "prepare_decode_record_s",
        "resident_append_s",
        "cpu_shadow_append_s",
        "resident_materialize_s",
        "resident_shadow_check_s",
        "qk_mixed_batch_s",
        "finalize_decode_records_s",
        "softmax_av_s",
    ]
    avg_timings = {}
    for key in timing_keys:
        values = [
            float(
                item["metrics"]["attention_backend"]["backend_debug"]
                .get("clover_op_timing_totals_s", {})
                .get(key, 0.0)
            )
            for item in case_records
        ]
        avg_timings[key] = statistics.mean(values) if values else 0.0

    return {
        "repeats": len(case_records),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": percentile(latencies, 0.95),
        "avg_ttft": statistics.mean(ttfts) if ttfts else 0.0,
        "avg_tpot": statistics.mean(tpots) if tpots else 0.0,
        "avg_throughput": statistics.mean(throughputs) if throughputs else 0.0,
        "avg_shadow_check_invocations": statistics.mean(invocations) if invocations else 0.0,
        "avg_shadow_check_skips": statistics.mean(skips) if skips else 0.0,
        "avg_clover_timings_s": avg_timings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--model-name", default="opt-125m")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--prompt", default="Hello CloverInfer")
    parser.add_argument("--prompt-token-length", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-resident-store-backend", default="host", choices=["host", "upmem_kvslot"])
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--shadow-token-intervals", type=parse_int_list, default=[4, 1])
    parser.add_argument("--shadow-layer-intervals", type=parse_int_list, default=[4, 1])
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "clover_shadow_sweep.jsonl"),
    )
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if args.prompt_token_length and args.prompt_token_length > 0:
        prompt, prompt_token_length = build_prompt_for_target_tokens(tokenizer, args.prompt_token_length)
    else:
        prompt = args.prompt
        prompt_token_length = len(tokenizer.encode(prompt, add_special_tokens=False))

    ray.init(
        address=args.address,
        runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}},
        ignore_reinit_error=True,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        for token_interval in args.shadow_token_intervals:
            for layer_interval in args.shadow_layer_intervals:
                scheduler = make_scheduler(args, token_interval, layer_interval)
                placement = ray.get(scheduler.initialize_cluster.remote())
                records = []
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
                    record = {
                        "case": {
                            "shadow_token_interval": int(token_interval),
                            "shadow_layer_interval": int(layer_interval),
                            "prompt_token_length": int(prompt_token_length),
                            "max_new_tokens": int(args.max_new_tokens),
                            "repeat_idx": int(repeat_idx),
                        },
                        "placement": placement,
                        "prompt": prompt,
                        "output": output,
                        "metrics": metrics,
                        "started_at": started_at,
                        "finished_at": finished_at,
                    }
                    records.append(record)
                    f.write(json.dumps({"type": "record", **record}) + "\n")

                summary = {
                    "type": "summary",
                    "shadow_token_interval": int(token_interval),
                    "shadow_layer_interval": int(layer_interval),
                    "prompt_token_length": int(prompt_token_length),
                    "max_new_tokens": int(args.max_new_tokens),
                    **summarize_case(records),
                }
                f.write(json.dumps(summary) + "\n")
                print(json.dumps(summary, ensure_ascii=True))

    print(f"Saved CloverInfer shadow sweep to {args.output}")


if __name__ == "__main__":
    main()
