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
    print("Testing Ray + RDMA Integration...")
    
    # Init Ray with PYTHONPATH in runtime_env to ensure workers find clover_net
    ray.init(
        ignore_reinit_error=True, 
        runtime_env={"env_vars": {"PYTHONPATH": src_path}}
    )
    
    cluster_cfg = ClusterConfig(num_prefill_workers=1)
    model_cfg = ModelConfig()
    
    scheduler = GlobalScheduler.remote(cluster_cfg, model_cfg)
    
    # Run Init
    print("Calling initialize_cluster...")
    # This calls init_infra on workers and performs handshake
    ray.get(scheduler.initialize_cluster.remote())
    
    # We don't have a direct "assert" for RDMA success exposed here easily without log parsing 
    # or querying state. 
    # But if it didn't crash, it's a good sign.
    
    # Let's check status via a dummy method on Scheduler? 
    # Or just rely on stdout for this DIY verification.
    
    print("Cluster Init complete. Check logs for QPN connection.")
    
    # Submit a dummy request to exercise the Workers
    # (AttnNode/FFNNode process_step is currently a pass)
    ray.get(scheduler.submit_request.remote("Test Prompt"))
    
    ray.shutdown()

if __name__ == "__main__":
    test_integration()
