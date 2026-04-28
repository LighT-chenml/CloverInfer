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


def make_scheduler(args):
    mode = str(args.mode)
    clover_pim_context_fused_experimental_enabled = False
    if mode == "host_best":
        pim_resident_store_backend = "host"
        pim_qk_full_enabled = False
        pim_softmax_av_fused_enabled = False
        clover_cpu_shadow_enabled = True
        clover_shadow_checks_enabled = True
        clover_host_qk_mixed_enabled = False
        clover_pim_attention_enabled = False
    elif mode == "pim_full":
        pim_resident_store_backend = "upmem_kvslot"
        pim_qk_full_enabled = True
        pim_softmax_av_fused_enabled = True
        clover_cpu_shadow_enabled = True
        clover_shadow_checks_enabled = True
        clover_host_qk_mixed_enabled = False
        clover_pim_attention_enabled = True
    else:
        raise ValueError(f"Unsupported mode: {mode}")

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
        pim_resident_store_backend=pim_resident_store_backend,
        pim_qk_full_enabled=pim_qk_full_enabled,
        pim_softmax_av_fused_enabled=pim_softmax_av_fused_enabled,
        pim_qk_mixed_enabled=True,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        pim_length=args.pim_length,
        clover_cpu_shadow_enabled=clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=clover_shadow_checks_enabled,
        clover_op_profiling_enabled=True,
        clover_shadow_check_token_interval=4,
        clover_shadow_check_layer_interval=4,
        clover_host_qk_mixed_enabled=clover_host_qk_mixed_enabled,
        clover_pim_attention_enabled=clover_pim_attention_enabled,
        clover_pim_context_fused_experimental_enabled=clover_pim_context_fused_experimental_enabled,
    )
    model = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    return GlobalScheduler.remote(cluster, model)


def build_record(
    repeat_idx: int,
    completion_index: int,
    inflight_at_submit: int,
    submit_started_at: float,
    submit_finished_at: float,
    output: str,
    metrics: Dict[str, object],
) -> Dict[str, object]:
    return {
        "type": "record",
        "repeat_idx": int(repeat_idx),
        "completion_index": int(completion_index),
        "inflight_at_submit": int(inflight_at_submit),
        "submit_started_at": float(submit_started_at),
        "submit_finished_at": float(submit_finished_at),
        "finished_at": time.time(),
        "output": output,
        "metrics": metrics,
    }


def run_requests(
    scheduler,
    prompt: str,
    max_new_tokens: int,
    num_requests: int,
    concurrency: int,
) -> Tuple[List[Dict[str, object]], Dict[str, object] | None]:
    records: List[Dict[str, object]] = []
    pending: Dict[ray.ObjectRef, Dict[str, object]] = {}
    next_idx = 0
    completion_index = 0
    failure = None

    while next_idx < num_requests or pending:
        while next_idx < num_requests and len(pending) < concurrency:
            submit_started_at = time.time()
            future = scheduler.submit_request.remote(
                prompt,
                return_metrics=True,
                max_new_tokens=max_new_tokens,
            )
            submit_finished_at = time.time()
            pending[future] = {
                "repeat_idx": next_idx,
                "submit_started_at": submit_started_at,
                "submit_finished_at": submit_finished_at,
                "inflight_at_submit": len(pending) + 1,
            }
            next_idx += 1

        ready, _ = ray.wait(list(pending.keys()), num_returns=1)
        future = ready[0]
        meta = pending.pop(future)
        try:
            output, metrics = ray.get(future)
        except Exception as exc:
            failure = {
                "repeat_idx": int(meta["repeat_idx"]),
                "completion_index": int(completion_index),
                "inflight_at_submit": int(meta["inflight_at_submit"]),
                "exception_type": type(exc).__name__,
                "message": str(exc),
            }
            break

        records.append(
            build_record(
                repeat_idx=int(meta["repeat_idx"]),
                completion_index=completion_index,
                inflight_at_submit=int(meta["inflight_at_submit"]),
                submit_started_at=float(meta["submit_started_at"]),
                submit_finished_at=float(meta["submit_finished_at"]),
                output=output,
                metrics=metrics,
            )
        )
        completion_index += 1

    return records, failure


def summarize_runs(records: List[Dict[str, object]]) -> Dict[str, object]:
    latencies = [float(item["metrics"]["latency"]) for item in records]
    ttfts = [float(item["metrics"]["ttft"]) for item in records]
    tpots = [float(item["metrics"]["tpot"]) for item in records]
    throughputs = [float(item["metrics"]["throughput"]) for item in records]
    total_tokens = [int(item["metrics"]["total_tokens"]) for item in records]
    client_observed_latencies = [
        float(item["finished_at"]) - float(item["submit_started_at"]) for item in records
    ]
    submit_durations = [
        float(item["submit_finished_at"]) - float(item["submit_started_at"]) for item in records
    ]
    scheduler_totals = [
        float(item["metrics"]["stage_timing"]["scheduler"]["total_rpc_s"]) for item in records
    ]
    actor_totals = [
        float(item["metrics"]["stage_timing"]["actors"]["total_compute_s"]) for item in records
    ]
    scheduler_overheads = [
        float(item["metrics"]["stage_timing"].get("scheduler_overhead_s", 0.0)) for item in records
    ]
    clover_prepare = [
        float(
            item["metrics"]["attention_backend"]["backend_debug"]
            .get("clover_op_timing_totals_s", {})
            .get("prepare_decode_record_s", 0.0)
        )
        for item in records
    ]
    clover_finalize = [
        float(
            item["metrics"]["attention_backend"]["backend_debug"]
            .get("clover_op_timing_totals_s", {})
            .get("finalize_decode_records_s", 0.0)
        )
        for item in records
    ]
    clover_cpu_shadow_append = [
        float(
            item["metrics"]["attention_backend"]["backend_debug"]
            .get("clover_op_timing_totals_s", {})
            .get("cpu_shadow_append_s", 0.0)
        )
        for item in records
    ]
    out_of_order_completions = sum(
        1 for item in records if int(item["repeat_idx"]) != int(item["completion_index"])
    )

    return {
        "repeats": len(records),
        "avg_latency": statistics.mean(latencies) if latencies else 0.0,
        "p95_latency": percentile(latencies, 0.95),
        "avg_client_observed_latency": statistics.mean(client_observed_latencies)
        if client_observed_latencies
        else 0.0,
        "p95_client_observed_latency": percentile(client_observed_latencies, 0.95),
        "avg_ttft": statistics.mean(ttfts) if ttfts else 0.0,
        "avg_tpot": statistics.mean(tpots) if tpots else 0.0,
        "avg_throughput": statistics.mean(throughputs) if throughputs else 0.0,
        "avg_total_tokens": statistics.mean(total_tokens) if total_tokens else 0.0,
        "avg_submit_duration_s": statistics.mean(submit_durations) if submit_durations else 0.0,
        "avg_scheduler_rpc_s": statistics.mean(scheduler_totals) if scheduler_totals else 0.0,
        "avg_actor_compute_s": statistics.mean(actor_totals) if actor_totals else 0.0,
        "avg_scheduler_overhead_s": statistics.mean(scheduler_overheads) if scheduler_overheads else 0.0,
        "avg_clover_prepare_decode_record_s": statistics.mean(clover_prepare) if clover_prepare else 0.0,
        "avg_clover_finalize_decode_records_s": statistics.mean(clover_finalize) if clover_finalize else 0.0,
        "avg_clover_cpu_shadow_append_s": statistics.mean(clover_cpu_shadow_append)
        if clover_cpu_shadow_append
        else 0.0,
        "out_of_order_completions": int(out_of_order_completions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/opt-125m")
    parser.add_argument("--model-name", default="opt-125m")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--mode", default="host_best", choices=["host_best", "pim_full"])
    parser.add_argument("--prompt", default="Hello CloverInfer")
    parser.add_argument("--prompt-token-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "clover_host_best_benchmark.jsonl"),
    )
    args = parser.parse_args()
    concurrency = max(1, int(args.concurrency))

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

    scheduler = make_scheduler(args)
    placement = ray.get(scheduler.initialize_cluster.remote())

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "type": "meta",
                    "config": {
                        "prompt_token_length": int(prompt_token_length),
                        "mode": str(args.mode),
                        "max_new_tokens": int(args.max_new_tokens),
                        "warmup_runs": int(args.warmup_runs),
                        "repeats": int(args.repeats),
                        "concurrency": concurrency,
                        "pim_num_dpus": int(args.pim_num_dpus),
                        "pim_length": int(args.pim_length),
                        "pim_qk_mixed_heads": int(args.pim_qk_mixed_heads),
                        "pim_qk_mixed_window": int(args.pim_qk_mixed_window),
                        "clover_shadow_check_token_interval": 4,
                        "clover_shadow_check_layer_interval": 4,
                        "clover_host_qk_mixed_enabled": False,
                        "clover_pim_attention_enabled": bool(args.mode == "pim_full"),
                        "clover_pim_context_fused_experimental_enabled": False,
                    },
                    "placement": placement,
                }
            )
            + "\n"
        )

        warmup_records, warmup_failure = run_requests(
            scheduler=scheduler,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_requests=int(args.warmup_runs),
            concurrency=concurrency,
        )
        for warmup_record in warmup_records:
            f.write(
                json.dumps(
                    {
                        "type": "warmup",
                        "warmup_idx": int(warmup_record["repeat_idx"]),
                        "completion_index": int(warmup_record["completion_index"]),
                        "inflight_at_submit": int(warmup_record["inflight_at_submit"]),
                        "submit_started_at": float(warmup_record["submit_started_at"]),
                        "submit_finished_at": float(warmup_record["submit_finished_at"]),
                        "finished_at": float(warmup_record["finished_at"]),
                        "output": warmup_record["output"],
                        "metrics": warmup_record["metrics"],
                    }
                )
                + "\n"
            )
        if warmup_failure is not None:
            f.write(json.dumps({"type": "warmup_failure", **warmup_failure}) + "\n")
            raise RuntimeError(f"Warmup failed: {warmup_failure}")

        benchmark_started_at = time.time()
        records, failure = run_requests(
            scheduler=scheduler,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            num_requests=int(args.repeats),
            concurrency=concurrency,
        )
        benchmark_finished_at = time.time()

        for record in records:
            f.write(json.dumps(record) + "\n")

        if failure is not None:
            f.write(json.dumps({"type": "failure", **failure}) + "\n")
            raise RuntimeError(f"Benchmark failed: {failure}")

        wall_time_s = max(benchmark_finished_at - benchmark_started_at, 1e-12)
        total_generated_tokens = sum(int(item["metrics"]["total_tokens"]) for item in records)
        summary = {
            "type": "summary",
            "prompt_token_length": int(prompt_token_length),
            "max_new_tokens": int(args.max_new_tokens),
            "concurrency": concurrency,
            "benchmark_started_at": float(benchmark_started_at),
            "benchmark_finished_at": float(benchmark_finished_at),
            "wall_time_s": float(wall_time_s),
            "request_throughput_rps": float(len(records) / wall_time_s),
            "output_token_throughput_tps": float(total_generated_tokens / wall_time_s),
            **summarize_runs(records),
        }
        f.write(json.dumps(summary) + "\n")
        print(json.dumps(summary, ensure_ascii=True))

    print(f"Saved CloverInfer host best benchmark to {args.output}")


if __name__ == "__main__":
    main()
