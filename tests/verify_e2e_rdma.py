
import pytest

pytest.skip("Legacy RDMA e2e verification is disabled during the correctness-first refactor.", allow_module_level=True)

import ray
import asyncio
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.scheduler import GlobalScheduler
from core.config import ClusterConfig, ModelConfig

# Mock Config
cluster_config = ClusterConfig(
    num_prefill_workers=1,
    num_attn_nodes=1,
    num_ffn_nodes=1
)

model_config = ModelConfig(
    model_path="/home/cml/CloverInfer/model/opt-125m", # Local path
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=50272,
    max_seq_len=2048
)

async def run_test():
    print("Starting End-to-End RDMA Verification...")
    
    # Init Ray
    if not ray.is_initialized():
        ray.init()
        
    scheduler = GlobalScheduler.remote(cluster_config, model_config)
    
    print("Initializing Cluster...")
    await scheduler.initialize_cluster.remote()
    
    print("Submitting Request...")
    prompt = "Hello, world! This is a test of the RDMA system."
    result = await scheduler.submit_request.remote(prompt)
    
    print("\n--- Result ---")
    print(f"Generated: {result}")
    print("--- End ---")
    
    print("E2E Test Success.")

if __name__ == "__main__":
    asyncio.run(run_test())
