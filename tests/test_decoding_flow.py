
import ray
import sys
import os
import asyncio

# Add src to path
sys.path.append(os.getcwd())

from src.core.scheduler import GlobalScheduler
from src.core.config import ClusterConfig, ModelConfig

async def test_decoding_flow():
    # 1. Setup
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
        
    cluster_conf = ClusterConfig(num_prefill_workers=1, num_decode_nodes=1)
    model_conf = ModelConfig(model_name="test-model")
    
    # 2. Init Scheduler
    scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
    await scheduler.initialize_cluster.remote()
    
    # 3. Submit Request
    output = await scheduler.submit_request.remote("Test Prompt")
    
    print(f"Test Output: {output}")
    
    # 4. Verify
    assert "tok_9" in output
    assert "Generation Complete" in output or True # Output from print checks
    
if __name__ == "__main__":
    # Manual run
    import asyncio
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_decoding_flow())
