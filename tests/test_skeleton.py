import ray
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.config import ClusterConfig, ModelConfig
from core.scheduler import GlobalScheduler

def test_skeleton():
    # Initialize Ray (simulating single node cluster)
    ray.init(ignore_reinit_error=True)
    
    cluster_cfg = ClusterConfig(num_prefill_workers=1)
    model_cfg = ModelConfig()
    
    scheduler = GlobalScheduler.remote(cluster_cfg, model_cfg)
    
    # Trigger initialization
    ray.get(scheduler.initialize_cluster.remote())
    
    # Submit explicit dummy request
    ray.get(scheduler.submit_request.remote("Hello CloverInfer"))
    
    print("Skeleton Test Passed: Cluster initialized and request submitted.")
    ray.shutdown()

if __name__ == "__main__":
    test_skeleton()
