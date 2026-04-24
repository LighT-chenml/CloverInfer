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
    parser.add_argument("--attention-backend", choices=["cpu", "pim_naive"], default="cpu")
    parser.add_argument("--prefill-resource", type=str, default=None)
    parser.add_argument("--decode-dense-resource", type=str, default=None)
    parser.add_argument("--attention-resource", type=str, default=None)
    parser.add_argument("--use-gpu-for-prefill", action="store_true")
    parser.add_argument("--no-gpu-for-prefill", action="store_true")
    parser.add_argument("--use-gpu-for-decode-dense", action="store_true")
    parser.add_argument("--no-gpu-for-decode-dense", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--pim-num-dpus", type=int, default=4)
    parser.add_argument("--pim-length", type=int, default=128)
    parser.add_argument("--pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--no-pim-qk-mixed-enabled", action="store_true")
    parser.add_argument("--pim-qk-mixed-heads", type=int, default=2)
    parser.add_argument("--pim-qk-mixed-window", type=int, default=128)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.use_gpu_for_prefill and args.no_gpu_for_prefill:
        raise ValueError("cannot set both --use-gpu-for-prefill and --no-gpu-for-prefill")
    if args.use_gpu_for_decode_dense and args.no_gpu_for_decode_dense:
        raise ValueError("cannot set both --use-gpu-for-decode-dense and --no-gpu-for-decode-dense")
    if args.pim_qk_mixed_enabled and args.no_pim_qk_mixed_enabled:
        raise ValueError("cannot set both --pim-qk-mixed-enabled and --no-pim-qk-mixed-enabled")

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

    pim_qk_mixed_enabled = True
    if args.no_pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = False
    elif args.pim_qk_mixed_enabled:
        pim_qk_mixed_enabled = True

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
        attention_backend=args.attention_backend,
        pim_num_dpus=args.pim_num_dpus,
        pim_length=args.pim_length,
        pim_qk_mixed_enabled=pim_qk_mixed_enabled,
        pim_qk_mixed_heads=args.pim_qk_mixed_heads,
        pim_qk_mixed_window=args.pim_qk_mixed_window,
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
        
        print("\nBenchmark Results Summary:")
        print(f"Average Latency: {avg_latency:.4f} s")
        print(f"Average TTFT:    {avg_ttft:.4f} s")
        print(f"Average TPOT:    {avg_tpot:.4f} s")
        print(f"Avg Throughput:  {avg_throughput:.2f} tokens/s")
    else:
        print("\nNo metrics collected.")

    if results:
        print("\nFirst result preview:")
        print(json.dumps(results[0], ensure_ascii=False))

if __name__ == "__main__":
    main()
