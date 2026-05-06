import argparse
import json
import os
import sys
import time

import ray

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/humaneval.jsonl")
    parser.add_argument("--output", type=str, default="humaneval_results.jsonl")
    parser.add_argument(
        "--model",
        type=str,
        default="model/opt-125m",
        help="Path to model directory or HF model name",
    )
    parser.add_argument("--model-name", type=str, default="custom")
    parser.add_argument("--address", type=str, default=None)
    parser.add_argument("--attention-backend", choices=["cpu", "pim_naive", "cloverinfer"], default="cpu")
    parser.add_argument("--prefill-resource", type=str, default=None)
    parser.add_argument("--decode-dense-resource", type=str, default=None)
    parser.add_argument("--attention-resource", type=str, default=None)
    parser.add_argument("--use-gpu-for-prefill", action="store_true")
    parser.add_argument("--no-gpu-for-prefill", action="store_true")
    parser.add_argument("--use-gpu-for-decode-dense", action="store_true")
    parser.add_argument("--no-gpu-for-decode-dense", action="store_true")
    parser.add_argument("--prefill-gpu-fraction", type=float, default=1.0)
    parser.add_argument("--decode-dense-gpu-fraction", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-block-tokens", type=int, default=256)
    parser.add_argument(
        "--pim-max-resident-groups-per-layer",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--pim-head-grouping-policy",
        type=str,
        default="balanced",
        choices=["legacy", "balanced", "coarse", "segment_aware"],
    )
    parser.add_argument(
        "--pim-resident-store-backend",
        type=str,
        default="host",
        choices=["host", "upmem_kvslot"],
    )
    parser.add_argument(
        "--pim-dpu-placement-policy",
        type=str,
        default="rotated",
        choices=["identity", "rotated", "rank_spread", "load_aware"],
    )
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
    parser.add_argument("--decode-continuous-batch-window-ms", type=float, default=0.0)
    parser.add_argument("--decode-continuous-batch-max-size", type=int, default=8)
    parser.add_argument("--attention-rpc-batch-window-ms", type=float, default=1.0)
    parser.add_argument("--attention-rpc-batch-max-size", type=int, default=8)
    parser.add_argument("--attention-rpc-cross-key-batch-enabled", action="store_true")
    parser.add_argument("--no-attention-rpc-cross-key-batch-enabled", action="store_true")
    parser.add_argument("--attention-actor-side-batching-enabled", action="store_true")
    parser.add_argument("--no-attention-actor-side-batching-enabled", action="store_true")
    parser.add_argument("--attention-actor-batch-window-ms", type=float, default=1.0)
    parser.add_argument("--attention-actor-batch-max-size", type=int, default=8)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.use_gpu_for_prefill and args.no_gpu_for_prefill:
        raise ValueError("cannot set both --use-gpu-for-prefill and --no-gpu-for-prefill")
    if args.use_gpu_for_decode_dense and args.no_gpu_for_decode_dense:
        raise ValueError("cannot set both --use-gpu-for-decode-dense and --no-gpu-for-decode-dense")
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
        raise ValueError("cannot set both --clover-shadow-checks-enabled and --no-clover-shadow-checks-enabled")
    if args.clover_op_profiling_enabled and args.no_clover_op_profiling_enabled:
        raise ValueError("cannot set both --clover-op-profiling-enabled and --no-clover-op-profiling-enabled")
    if args.clover_host_qk_mixed_enabled and args.no_clover_host_qk_mixed_enabled:
        raise ValueError("cannot set both --clover-host-qk-mixed-enabled and --no-clover-host-qk-mixed-enabled")
    if args.attention_rpc_cross_key_batch_enabled and args.no_attention_rpc_cross_key_batch_enabled:
        raise ValueError(
            "cannot set both --attention-rpc-cross-key-batch-enabled and --no-attention-rpc-cross-key-batch-enabled"
        )
    if args.attention_actor_side_batching_enabled and args.no_attention_actor_side_batching_enabled:
        raise ValueError(
            "cannot set both --attention-actor-side-batching-enabled and --no-attention-actor-side-batching-enabled"
        )

    use_gpu_for_prefill = True
    if args.no_gpu_for_prefill:
        use_gpu_for_prefill = False
    elif args.use_gpu_for_prefill:
        use_gpu_for_prefill = True

    use_gpu_for_decode_dense = True
    if args.no_gpu_for_decode_dense:
        use_gpu_for_decode_dense = False
    elif args.use_gpu_for_decode_dense:
        use_gpu_for_decode_dense = True

    # This benchmark is used primarily for throughput comparisons, so keep the
    # mixed-head QK shadow path opt-in instead of on by default.
    pim_qk_mixed_enabled = False
    if args.no_pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = False
    elif args.pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = True

    pim_qk_full_enabled = False
    if args.no_pim_qk_full_enabled:
        pim_qk_full_enabled = False
    elif args.pim_qk_full_enabled:
        pim_qk_full_enabled = True

    pim_qk_full_shadow_check = True
    if args.no_pim_qk_full_shadow_check:
        pim_qk_full_shadow_check = False
    elif args.pim_qk_full_shadow_check:
        pim_qk_full_shadow_check = True

    pim_softmax_av_fused_enabled = False
    if args.no_pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = False
    elif args.pim_softmax_av_fused_enabled:
        pim_softmax_av_fused_enabled = True

    pim_softmax_av_shadow_check = True
    if args.no_pim_softmax_av_shadow_check:
        pim_softmax_av_shadow_check = False
    elif args.pim_softmax_av_shadow_check:
        pim_softmax_av_shadow_check = True

    clover_cpu_shadow_enabled = True
    if args.no_clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = False
    elif args.clover_cpu_shadow_enabled:
        clover_cpu_shadow_enabled = True

    clover_shadow_checks_enabled = True
    if args.no_clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = False
    elif args.clover_shadow_checks_enabled:
        clover_shadow_checks_enabled = True

    clover_op_profiling_enabled = True
    if args.no_clover_op_profiling_enabled:
        clover_op_profiling_enabled = False
    elif args.clover_op_profiling_enabled:
        clover_op_profiling_enabled = True

    clover_host_qk_mixed_enabled = False
    if args.clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = True
    elif args.no_clover_host_qk_mixed_enabled:
        clover_host_qk_mixed_enabled = False

    attention_rpc_cross_key_batch_enabled = False
    if args.no_attention_rpc_cross_key_batch_enabled:
        attention_rpc_cross_key_batch_enabled = False
    elif args.attention_rpc_cross_key_batch_enabled:
        attention_rpc_cross_key_batch_enabled = True

    attention_actor_side_batching_enabled = False
    if args.no_attention_actor_side_batching_enabled:
        attention_actor_side_batching_enabled = False
    elif args.attention_actor_side_batching_enabled:
        attention_actor_side_batching_enabled = True

    # Init Ray
    if not ray.is_initialized():
        runtime_env = {
            "env_vars": {
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "PYTHONPATH": REPO_ROOT,
            }
        }
        ray.init(address=args.address, ignore_reinit_error=True, runtime_env=runtime_env)

    print("Initialize Configuration...")

    # Resolve absolute path for model if it exists locally
    model_path = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    print(f"Using model: {model_path}")

    cluster_conf = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        prefill_resource=args.prefill_resource,
        decode_dense_resource=args.decode_dense_resource,
        attention_resource=args.attention_resource,
        use_gpu_for_prefill=use_gpu_for_prefill,
        use_gpu_for_decode_dense=use_gpu_for_decode_dense,
        prefill_gpu_fraction=args.prefill_gpu_fraction,
        decode_dense_gpu_fraction=args.decode_dense_gpu_fraction,
        attention_backend=args.attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_length=args.pim_length,
        pim_block_tokens=args.pim_block_tokens,
        pim_resident_store_backend=args.pim_resident_store_backend,
        pim_max_resident_groups_per_layer=args.pim_max_resident_groups_per_layer,
        pim_head_grouping_policy=args.pim_head_grouping_policy,
        pim_dpu_placement_policy=args.pim_dpu_placement_policy,
        pim_qk_full_enabled=pim_qk_full_enabled,
        pim_qk_full_shadow_check=pim_qk_full_shadow_check,
        pim_softmax_av_fused_enabled=pim_softmax_av_fused_enabled,
        pim_softmax_av_shadow_check=pim_softmax_av_shadow_check,
        pim_qk_mixed_enabled=pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
        clover_cpu_shadow_enabled=clover_cpu_shadow_enabled,
        clover_shadow_checks_enabled=clover_shadow_checks_enabled,
        clover_op_profiling_enabled=clover_op_profiling_enabled,
        clover_shadow_check_token_interval=args.clover_shadow_check_token_interval,
        clover_shadow_check_layer_interval=args.clover_shadow_check_layer_interval,
        clover_host_qk_mixed_enabled=clover_host_qk_mixed_enabled,
        attention_rpc_batch_window_s=args.attention_rpc_batch_window_ms / 1000.0,
        attention_rpc_batch_max_size=args.attention_rpc_batch_max_size,
        attention_rpc_cross_key_batch_enabled=attention_rpc_cross_key_batch_enabled,
        attention_actor_side_batching_enabled=attention_actor_side_batching_enabled,
        attention_actor_batch_window_s=args.attention_actor_batch_window_ms / 1000.0,
        attention_actor_batch_max_size=args.attention_actor_batch_max_size,
        decode_continuous_batch_window_s=args.decode_continuous_batch_window_ms / 1000.0,
        decode_continuous_batch_max_size=args.decode_continuous_batch_max_size,
    )
    model_conf = ModelConfig(
        model_name=args.model_name,
        model_path=model_path,
        max_seq_len=2048,
        max_new_tokens=args.max_new_tokens,
        dtype=args.dtype,
    )

    print("Deploying Scheduler...")
    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    placement = ray.get(scheduler.initialize_cluster.remote())
    print(json.dumps({"placement": placement}, ensure_ascii=False))
    
    # Load Data
    print(f"Loading data from {args.data}...")
    if not os.path.exists(args.data):
        print(f"Error: {args.data} not found.")
        return

    problems = []
    with open(args.data, 'r') as f:
        for line in f:
            problems.append(json.loads(line))

    if args.limit:
        problems = problems[:args.limit]

    start_time = time.time()
    results = []

    print(f"Submitting {len(problems)} tasks...")

    if args.sequential:
        results_list = []
        for problem in problems:
            results_list.append(
                ray.get(
                    scheduler.submit_request.remote(
                        problem["prompt"],
                        return_metrics=True,
                        max_new_tokens=args.max_new_tokens,
                    )
                )
            )
    else:
        futures = []
        for problem in problems:
            futures.append(
                scheduler.submit_request.remote(
                    problem["prompt"],
                    return_metrics=True,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        results_list = ray.get(futures)
    
    # Save Results and compute stats
    total_metrics = {
        "ttft": [],
        "tpot": [],
        "latency": [],
        "throughput": []
    }
    
    with open(args.output, "w") as f:
        for problem, res_tuple in zip(problems, results_list):
            if isinstance(res_tuple, tuple) and len(res_tuple) == 2:
                completion, metrics = res_tuple
            else:
                completion = str(res_tuple)
                metrics = {}
            
            res = {
                "task_id": problem["task_id"],
                "completion": completion,
                "metrics": metrics,
            }
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            results.append(res)
            
            if metrics:
                total_metrics["ttft"].append(metrics["ttft"])
                total_metrics["tpot"].append(metrics["tpot"])
                total_metrics["latency"].append(metrics["latency"])
                total_metrics["throughput"].append(metrics["throughput"])

    end_time = time.time()
    print(f"Finished. Saved {len(results_list)} results to {args.output}")
    
    if total_metrics["latency"]:
        avg_ttft = sum(total_metrics["ttft"]) / len(total_metrics["ttft"])
        avg_tpot = sum(total_metrics["tpot"]) / len(total_metrics["tpot"])
        avg_latency = sum(total_metrics["latency"]) / len(total_metrics["latency"])
        avg_throughput = sum(total_metrics["throughput"]) / len(total_metrics["throughput"])
        attention_before_free = results[-1]["metrics"].get("attention_backend_before_free", {}) if results else {}
        backend_debug = attention_before_free.get("backend_debug", {})
        resident_store_debug = backend_debug.get("resident_store_debug", {})
        
        print("\nBenchmark Results Summary:")
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"Average TTFT:    {avg_ttft:.4f} s")
        print(f"Average TPOT:    {avg_tpot:.4f} s")
        print(f"Avg Throughput:  {avg_throughput:.2f} tokens/s")
        if resident_store_debug:
            print("\nResident Store Summary:")
            print(
                json.dumps(
                    {
                        "placement_policy": resident_store_debug.get("placement_policy", ""),
                        "allocator_summary": resident_store_debug.get("allocator_summary", {}),
                        "dpu_balance_summary": resident_store_debug.get("dpu_balance_summary", {}),
                        "rank_balance_summary": resident_store_debug.get("rank_balance_summary", {}),
                        "block_summary": resident_store_debug.get("block_summary", {}),
                    },
                    ensure_ascii=False,
                )
            )
            helper_profile = resident_store_debug.get("helper_profile", {})
            if helper_profile:
                print("\nHelper Profile Summary:")
                print(
                    json.dumps(
                        {
                            "qk_rounds_total": int(helper_profile.get("qk_rounds_total", 0)),
                            "qk_batched_rounds": int(helper_profile.get("qk_batched_rounds", 0)),
                            "qk_fallback_rounds": int(helper_profile.get("qk_fallback_rounds", 0)),
                            "qk_round_items_total": int(helper_profile.get("qk_round_items_total", 0)),
                            "qk_batched_items_total": int(helper_profile.get("qk_batched_items_total", 0)),
                            "qk_active_ranks_total": int(helper_profile.get("qk_active_ranks_total", 0)),
                            "qk_max_round_size": int(helper_profile.get("qk_max_round_size", 0)),
                            "av_rounds_total": int(helper_profile.get("av_rounds_total", 0)),
                            "av_batched_rounds": int(helper_profile.get("av_batched_rounds", 0)),
                            "av_fallback_rounds": int(helper_profile.get("av_fallback_rounds", 0)),
                            "av_round_items_total": int(helper_profile.get("av_round_items_total", 0)),
                            "av_batched_items_total": int(helper_profile.get("av_batched_items_total", 0)),
                            "av_active_ranks_total": int(helper_profile.get("av_active_ranks_total", 0)),
                            "av_max_round_size": int(helper_profile.get("av_max_round_size", 0)),
                        },
                        ensure_ascii=False,
                    )
                )
    else:
        print("\nNo metrics collected.")

    if results:
        print("\nFirst result preview:")
        print(json.dumps(results[0], ensure_ascii=False))

if __name__ == "__main__":
    main()
