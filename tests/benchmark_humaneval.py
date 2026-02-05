import ray
import json
import time
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.scheduler import GlobalScheduler
from src.core.config import ClusterConfig, ModelConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset/humaneval.jsonl")
    parser.add_argument("--output", type=str, default="humaneval_results.jsonl")
    parser.add_argument("--model", type=str, default="model/opt-125m", help="Path to model directory or HF model name")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Init Ray
    if not ray.is_initialized():
        # Pass environment variables to workers to ensure they find torch libraries
        import os
        runtime_env = {
            "env_vars": {
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
                "PYTHONPATH": os.environ.get("PYTHONPATH", "")
            }
        }
        ray.init(ignore_reinit_error=True, runtime_env=runtime_env)

    print("Initialize Configuration...")
    
    # Resolve absolute path for model if it exists locally
    model_path = os.path.abspath(args.model) if os.path.exists(args.model) else args.model
    print(f"Using model: {model_path}")

    # Setup for OPT-125M (Code generation might be poor but we are testing pipeline)
    cluster_conf = ClusterConfig(
        num_prefill_workers=1,
        num_decode_nodes=1 
    )
    model_conf = ModelConfig(
        model_name=model_path,
        max_seq_len=2048,
        dtype="float16"
    )

    print("Deploying Scheduler...")
    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    ray.get(scheduler.initialize_cluster.remote())
    
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

    # Run Benchmark
    start_time = time.time()
    results = []
    
    print(f"Submitting {len(problems)} tasks...")
    
    # We will submit sequentially or loosely parallel for this test
    # CloverInfer submit_request is async
    futures = []
    for problem in problems:
        prompt = problem["prompt"]
        task_id = problem["task_id"]
        # Submit request with metrics
        futures.append(scheduler.submit_request.remote(prompt, return_metrics=True))
        
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
                "metrics": metrics
            }
            f.write(json.dumps(res) + "\n")
            
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

if __name__ == "__main__":
    main()
