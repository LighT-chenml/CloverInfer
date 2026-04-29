import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List

import ray
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.config import ClusterConfig, ModelConfig
from src.core.model_adapter import CausalModelAdapter
from src.core.nodes import DecodeDenseNode, PrefillNode
from src.core.scheduler import GlobalScheduler


def summarize_metrics(metric_list: List[Dict[str, float]]) -> Dict[str, float]:
    return {
        "avg_latency": float(statistics.mean(item["latency"] for item in metric_list)),
        "avg_ttft": float(statistics.mean(item["ttft"] for item in metric_list)),
        "avg_tpot": float(statistics.mean(item["tpot"] for item in metric_list)),
        "avg_throughput": float(statistics.mean(item["throughput"] for item in metric_list)),
        "avg_total_tokens": float(statistics.mean(item["total_tokens"] for item in metric_list)),
    }


def load_prompts(path: str, limit: int | None) -> List[Dict[str, str]]:
    problems = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            problems.append(json.loads(line))
    if limit is not None:
        problems = problems[:limit]
    return problems


def run_monolithic_gpu(args, problems: List[Dict[str, str]]) -> Dict[str, object]:
    dtype = torch.float16 if args.dtype == "float16" else torch.float32
    adapter = CausalModelAdapter(args.model, "cuda", dtype)

    records = []
    for problem in problems:
        result = adapter.greedy_generate(problem["prompt"], args.max_new_tokens)
        records.append(
            {
                "task_id": problem["task_id"],
                "completion": result["text"],
                "metrics": result["metrics"],
            }
        )

    return {
        "baseline": "monolithic_gpu",
        "placement": {
            "mode": "single_process",
            "ip": ray.util.get_node_ip_address() if ray.is_initialized() else "local",
            "device": "cuda",
        },
        "records": records,
        "summary": summarize_metrics([record["metrics"] for record in records]),
    }


def make_cluster_config(args, attention_backend: str) -> ClusterConfig:
    resident_store_backend = str(args.pim_resident_store_backend)
    if resident_store_backend == "auto":
        resident_store_backend = "upmem_kvslot" if attention_backend == "cloverinfer" else "host"

    qk_full_enabled = bool(args.pim_qk_full_enabled)
    softmax_av_fused_enabled = bool(args.pim_softmax_av_fused_enabled)
    if attention_backend == "cloverinfer":
        qk_full_enabled = True
        softmax_av_fused_enabled = True

    return ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource=args.prefill_resource,
        decode_dense_resource=args.decode_dense_resource,
        attention_resource=args.attention_resource,
        use_gpu_for_prefill=True,
        use_gpu_for_decode_dense=True,
        attention_backend=attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_resident_store_backend=resident_store_backend,
        pim_qk_full_enabled=qk_full_enabled,
        pim_qk_full_shadow_check=args.pim_qk_full_shadow_check,
        pim_softmax_av_fused_enabled=softmax_av_fused_enabled,
        pim_softmax_av_shadow_check=args.pim_softmax_av_shadow_check,
        pim_length=args.pim_length,
        pim_max_resident_groups_per_layer=args.pim_max_resident_groups_per_layer,
        pim_head_grouping_policy=args.pim_head_grouping_policy,
        pim_dpu_placement_policy=args.pim_dpu_placement_policy,
        pim_resident_kv_dtype=args.pim_resident_kv_dtype,
        pim_qk_mixed_enabled=args.pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        clover_cpu_shadow_enabled=args.clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=args.clover_shadow_checks_enabled,
        clover_op_profiling_enabled=args.clover_op_profiling_enabled,
        clover_shadow_check_token_interval=args.clover_shadow_check_token_interval,
        clover_shadow_check_layer_interval=args.clover_shadow_check_layer_interval,
        clover_host_qk_mixed_enabled=args.clover_host_qk_mixed_enabled,
        clover_pim_attention_enabled=(attention_backend == "cloverinfer"),
        clover_pim_context_fused_experimental_enabled=args.clover_pim_context_fused_experimental_enabled,
        attention_rpc_cross_key_batch_enabled=(attention_backend == "cloverinfer"),
        attention_actor_side_batching_enabled=False,
    )


def run_disaggregated(args, problems: List[Dict[str, str]], attention_backend: str) -> Dict[str, object]:
    cluster_conf = make_cluster_config(args, attention_backend)
    model_conf = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )
    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    placement = ray.get(scheduler.initialize_cluster.remote())

    records = []
    for problem in problems:
        completion, metrics = ray.get(
            scheduler.submit_request.remote(
                problem["prompt"],
                return_metrics=True,
                max_new_tokens=args.max_new_tokens,
            )
        )
        records.append(
            {
                "task_id": problem["task_id"],
                "completion": completion,
                "metrics": metrics,
            }
        )

    return {
        "baseline": f"disagg_{attention_backend}",
        "placement": placement,
        "records": records,
        "summary": summarize_metrics([record["metrics"] for record in records]),
    }


def run_split_gpu(args, problems: List[Dict[str, str]]) -> Dict[str, object]:
    model_conf = ModelConfig(
        model_name=args.model_name,
        model_path=args.model,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )

    prefill = PrefillNode.options(
        num_gpus=1,
        resources={args.prefill_resource: 0.01},
    ).remote(0, model_conf, True)
    decode = DecodeDenseNode.options(
        num_gpus=1,
        resources={args.decode_dense_resource: 0.01},
    ).remote(0, model_conf, True)

    placement = {
        "prefill": ray.get(prefill.get_info.remote()),
        "decode_full": ray.get(decode.get_info.remote()),
    }

    records = []
    for problem in problems:
        wall_started = time.time()
        prefill_out = ray.get(prefill.process_prompt.remote(problem["prompt"]))
        first_token_ready = time.time()
        decode_out = ray.get(
            decode.continue_full_decode.remote(
                prefill_out["initial_kv"],
                prefill_out["prompt_len"],
                prefill_out["first_token_id"],
                args.max_new_tokens,
            )
        )
        request_finished = time.time()

        total_tokens = int(len(decode_out["generated_ids"]))
        ttft = float(first_token_ready - wall_started)
        latency = float(request_finished - wall_started)
        tpot = float((latency - ttft) / max(total_tokens - 1, 1))
        throughput = float(total_tokens / latency) if latency > 0 else 0.0

        metrics = {
            "ttft": ttft,
            "tpot": tpot,
            "latency": latency,
            "throughput": throughput,
            "total_tokens": total_tokens,
            "stage_timing": {
                "scheduler": {
                    "prefill_rpc_s": ttft,
                    "decode_full_rpc_s": max(0.0, latency - ttft),
                    "total_rpc_s": latency,
                },
                "actors": {
                    "prefill_compute_s": float(prefill_out.get("profile", {}).get("compute_s", 0.0)),
                    "decode_full_compute_s": float(decode_out.get("profile", {}).get("compute_s", 0.0)),
                    "total_compute_s": float(prefill_out.get("profile", {}).get("compute_s", 0.0))
                    + float(decode_out.get("profile", {}).get("compute_s", 0.0)),
                },
                "counts": {
                    "decode_steps": max(0, total_tokens - 1),
                },
            },
        }

        records.append(
            {
                "task_id": problem["task_id"],
                "completion": decode_out["text"],
                "metrics": metrics,
            }
        )

    return {
        "baseline": "split_gpu_full_decode",
        "placement": placement,
        "records": records,
        "summary": summarize_metrics([record["metrics"] for record in records]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataset/humaneval.jsonl")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--model", default="/home/cml/CloverInfer/model/Qwen-1_8B")
    parser.add_argument("--model-name", default="qwen-1_8b")
    parser.add_argument("--max-new-tokens", type=int, default=3)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--baselines",
        default="monolithic_gpu,split_gpu_full_decode,disagg_cpu,disagg_pim_naive",
        help=(
            "Comma-separated subset of "
            "monolithic_gpu,split_gpu_full_decode,disagg_cpu,disagg_pim_naive,disagg_cloverinfer"
        ),
    )
    parser.add_argument("--address", default="192.168.123.4:26379")
    parser.add_argument("--prefill-resource", default="prefill_gpu")
    parser.add_argument("--decode-dense-resource", default="decode_dense_gpu")
    parser.add_argument("--attention-resource", default="attention_pim")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument(
        "--pim-resident-store-backend",
        default="auto",
        choices=["auto", "host", "upmem_kvslot"],
    )
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-max-resident-groups-per-layer", type=int, default=0)
    parser.add_argument(
        "--pim-head-grouping-policy",
        default="balanced",
        choices=["legacy", "balanced", "coarse", "segment_aware"],
    )
    parser.add_argument(
        "--pim-dpu-placement-policy",
        default="rotated",
        choices=["identity", "rotated"],
    )
    parser.add_argument("--pim-resident-kv-dtype", default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--pim-qk-full-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-full-enabled", action="store_true")
    parser.add_argument("--pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--no-pim-qk-full-shadow-check", action="store_true")
    parser.add_argument("--pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--no-pim-softmax-av-fused-enabled", action="store_true")
    parser.add_argument("--pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--no-pim-softmax-av-shadow-check", action="store_true")
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--no-clover-cpu-shadow-enabled", action="store_true")
    parser.add_argument("--clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--no-clover-shadow-checks-enabled", action="store_true")
    parser.add_argument("--clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--no-clover-op-profiling-enabled", action="store_true")
    parser.add_argument("--clover-shadow-check-token-interval", type=int, default=4)
    parser.add_argument("--clover-shadow-check-layer-interval", type=int, default=4)
    parser.add_argument("--clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-clover-host-qk-mixed-enabled", action="store_true")
    parser.add_argument("--clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument("--no-clover-pim-context-fused-experimental-enabled", action="store_true")
    parser.add_argument(
        "--output",
        default=os.path.join(REPO_ROOT, "artifacts", "baseline_comparison.jsonl"),
    )
    args = parser.parse_args()

    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")
    if args.pim_qk_full_enabled and args.no_pim_qk_full_enabled:
        raise ValueError("cannot set both --pim-qk-full-enabled and --no-pim-qk-full-enabled")
    if args.pim_qk_full_shadow_check and args.no_pim_qk_full_shadow_check:
        raise ValueError("cannot set both --pim-qk-full-shadow-check and --no-pim-qk-full-shadow-check")
    if args.pim_softmax_av_fused_enabled and args.no_pim_softmax_av_fused_enabled:
        raise ValueError(
            "cannot set both --pim-softmax-av-fused-enabled and --no-pim-softmax-av-fused-enabled"
        )
    if args.pim_softmax_av_shadow_check and args.no_pim_softmax_av_shadow_check:
        raise ValueError(
            "cannot set both --pim-softmax-av-shadow-check and --no-pim-softmax-av-shadow-check"
        )
    if args.clover_cpu_shadow_enabled and args.no_clover_cpu_shadow_enabled:
        raise ValueError("cannot set both --clover-cpu-shadow-enabled and --no-clover-cpu-shadow-enabled")
    if args.clover_shadow_checks_enabled and args.no_clover_shadow_checks_enabled:
        raise ValueError(
            "cannot set both --clover-shadow-checks-enabled and --no-clover-shadow-checks-enabled"
        )
    if args.clover_op_profiling_enabled and args.no_clover_op_profiling_enabled:
        raise ValueError(
            "cannot set both --clover-op-profiling-enabled and --no-clover-op-profiling-enabled"
        )
    if args.clover_host_qk_mixed_enabled and args.no_clover_host_qk_mixed_enabled:
        raise ValueError("cannot set both --clover-host-qk-mixed-enabled and --no-clover-host-qk-mixed-enabled")
    if (
        args.clover_pim_context_fused_experimental_enabled
        and args.no_clover_pim_context_fused_experimental_enabled
    ):
        raise ValueError(
            "cannot set both --clover-pim-context-fused-experimental-enabled and "
            "--no-clover-pim-context-fused-experimental-enabled"
        )
    if not args.pim_qk_mixed_enabled and not args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        args.pim_qk_mixed_enabled = False
    args.pim_qk_full_enabled = bool(args.pim_qk_full_enabled)
    if args.no_pim_qk_full_enabled:
        args.pim_qk_full_enabled = False
    args.pim_qk_full_shadow_check = True
    if args.no_pim_qk_full_shadow_check:
        args.pim_qk_full_shadow_check = False
    elif args.pim_qk_full_shadow_check:
        args.pim_qk_full_shadow_check = True
    args.pim_softmax_av_fused_enabled = bool(args.pim_softmax_av_fused_enabled)
    if args.no_pim_softmax_av_fused_enabled:
        args.pim_softmax_av_fused_enabled = False
    args.pim_softmax_av_shadow_check = True
    if args.no_pim_softmax_av_shadow_check:
        args.pim_softmax_av_shadow_check = False
    elif args.pim_softmax_av_shadow_check:
        args.pim_softmax_av_shadow_check = True
    args.clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        args.clover_cpu_shadow_enabled = False
    elif args.clover_cpu_shadow_enabled:
        args.clover_cpu_shadow_enabled = True
    args.clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        args.clover_shadow_checks_enabled = False
    elif args.clover_shadow_checks_enabled:
        args.clover_shadow_checks_enabled = True
    args.clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        args.clover_op_profiling_enabled = False
    elif args.clover_op_profiling_enabled:
        args.clover_op_profiling_enabled = True
    args.clover_host_qk_mixed_enabled = False
    if args.clover_host_qk_mixed_enabled:
        args.clover_host_qk_mixed_enabled = True
    elif args.no_clover_host_qk_mixed_enabled:
        args.clover_host_qk_mixed_enabled = False
    args.clover_pim_context_fused_experimental_enabled = bool(
        args.clover_pim_context_fused_experimental_enabled
    )
    if args.no_clover_pim_context_fused_experimental_enabled:
        args.clover_pim_context_fused_experimental_enabled = False

    problems = load_prompts(args.data, args.limit)
    baselines = [item.strip() for item in args.baselines.split(",") if item.strip()]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ray.init(
        address=args.address,
        ignore_reinit_error=True,
        runtime_env={"env_vars": {"PYTHONPATH": REPO_ROOT}},
    )

    results = []
    for baseline in baselines:
        if baseline == "monolithic_gpu":
            result = run_monolithic_gpu(args, problems)
        elif baseline == "split_gpu_full_decode":
            result = run_split_gpu(args, problems)
        elif baseline == "disagg_cpu":
            result = run_disaggregated(args, problems, "cpu")
        elif baseline == "disagg_pim_naive":
            result = run_disaggregated(args, problems, "pim_naive")
        elif baseline == "disagg_cloverinfer":
            result = run_disaggregated(args, problems, "cloverinfer")
        else:
            raise ValueError(f"unsupported baseline: {baseline}")

        results.append(result)
        print(json.dumps({"baseline": result["baseline"], **result["summary"]}, ensure_ascii=False))

    with open(args.output, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved baseline comparison results to {args.output}")


if __name__ == "__main__":
    main()
