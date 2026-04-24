import ray
import sys
import os
import time

# Ensure src is in path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
sys.path.append(src_path)

from core.config import ClusterConfig, ModelConfig
from core.scheduler import GlobalScheduler

def test_integration():
    print("Testing Ray + CPU-Attention Integration...")

    ray.init(
        ignore_reinit_error=True, 
        runtime_env={"env_vars": {"PYTHONPATH": src_path}}
    )
    try:
        cluster_cfg = ClusterConfig(
            num_prefill_workers=1,
            num_attention_nodes=1,
            num_decode_dense_nodes=1,
            use_gpu_for_prefill=False,
            use_gpu_for_decode_dense=False,
        )
        model_cfg = ModelConfig(
            model_name="opt-125m",
            model_path=os.path.abspath(os.path.join(os.path.dirname(__file__), "../model/opt-125m")),
            max_new_tokens=2,
        )

        scheduler = GlobalScheduler.remote(cluster_cfg, model_cfg)

        print("Calling initialize_cluster...")
        cluster_info = ray.get(scheduler.initialize_cluster.remote())

        print("Cluster Init complete.")

        output, metrics = ray.get(scheduler.submit_request.remote("Test Prompt", return_metrics=True, max_new_tokens=2))
        assert isinstance(output, str)
        assert cluster_info["attention"]["backend"] == "cpu"
        assert metrics["total_tokens"] >= 1
        assert metrics["stage_timing"]["actors"]["total_compute_s"] >= 0
    finally:
        ray.shutdown()

if __name__ == "__main__":
    test_integration()
