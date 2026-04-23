import ray
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.config import ClusterConfig, ModelConfig
from src.core.scheduler import GlobalScheduler

def test_skeleton():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": repo_root}})
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
        cluster_info = ray.get(scheduler.initialize_cluster.remote())
        output = ray.get(scheduler.submit_request.remote("Hello CloverInfer", max_new_tokens=2))

        assert isinstance(output, str)
        assert cluster_info["prefill"]["role"] == "prefill"
        assert cluster_info["decode_dense"]["role"] == "decode_dense"
        print("Skeleton Test Passed: Cluster initialized and request submitted.")
    finally:
        ray.shutdown()

if __name__ == "__main__":
    test_skeleton()
