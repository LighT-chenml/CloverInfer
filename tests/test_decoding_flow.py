import ray
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.getcwd())

from src.core.scheduler import GlobalScheduler
from src.core.config import ClusterConfig, ModelConfig

async def test_decoding_flow():
    if not ray.is_initialized():
        # Ship the local repo to the cluster so remote workers import the same sources.
        ray.init(
            ignore_reinit_error=True,
            runtime_env={
                "working_dir": os.getcwd(),
                "excludes": [".git/", "artifacts/", "model/", "**/__pycache__/", "*.pyc"],
                "env_vars": {"PYTHONPATH": os.getcwd()},
            },
        )

    cluster_conf = ClusterConfig(
        num_prefill_workers=1,
        num_attention_nodes=1,
        num_decode_dense_nodes=1,
        use_gpu_for_prefill=False,
        use_gpu_for_decode_dense=False,
    )
    model_conf = ModelConfig(
        model_name="opt-125m",
        model_path=os.path.abspath("model/opt-125m"),
        max_new_tokens=2,
    )

    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    cluster_info = await scheduler.initialize_cluster.remote()
    output, metrics = await scheduler.submit_request.remote("Test Prompt", return_metrics=True, max_new_tokens=2)

    print(f"Cluster Info: {cluster_info}")
    print(f"Test Output: {output}")
    print(f"Metrics: {metrics}")

    assert isinstance(output, str)
    assert cluster_info["attention"]["backend"] == "cpu"
    assert metrics["total_tokens"] >= 1
    assert metrics["latency"] >= metrics["ttft"] >= 0
    assert metrics["stage_timing"]["counts"]["decode_steps"] >= 1
    assert metrics["stage_timing"]["scheduler"]["total_rpc_s"] >= 0

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_decoding_flow())
