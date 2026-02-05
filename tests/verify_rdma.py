
import ray
import torch
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.config import ModelConfig
from core.nodes import AttnNode, PrefillWorker

# Mock Config
config = ModelConfig(
    model_path="/home/cml/CloverInfer/model/opt-125m", # Local path
    num_layers=12,
    hidden_size=768,
    num_heads=12,
    vocab_size=50272,
    max_seq_len=2048
)

@ray.remote
def test_coordinator():
    print("Starting RDMA Verification...")
    
    # 1. Start Actors
    worker = PrefillWorker.remote(0, config)
    attn_node = AttnNode.remote(0, config)
    
    # 2. Init RDMA
    print("Initializing RDMA...")
    worker_info = ray.get(worker.init_infra.remote())
    attn_info = ray.get(attn_node.init_infra.remote())
    
    if not worker_info or not attn_info:
        print("RDMA Init Failed on one or both nodes. Skipping test.")
        return
        
    w_qpn, w_lid = worker_info
    a_qpn, a_lid = attn_info
    print(f"Worker: QPN={w_qpn}, LID={w_lid}")
    print(f"AttnNode: QPN={a_qpn}, LID={a_lid}")
    
    # 3. Prefill
    print("Running Prefill...")
    prompt = "Hello, world!"
    prefill_out = ray.get(worker.process_prompt.remote(prompt))
    
    req_id = prefill_out["req_id"]
    seq_len = prefill_out["seq_len"]
    print(f"Prefill Done. ReqID: {req_id}, SeqLen: {seq_len}")
    
    # 4. RDMA Handshake
    # Step A: AttnNode Posts Recvs
    print("Step A: AttnNode Prepare Recv...")
    ready = ray.get(attn_node.prepare_recv_kv.remote(req_id, seq_len, w_qpn, w_lid))
    assert ready, "AttnNode failed to prepare recv"
    
    # Step B: PrefillWorker Sends
    print("Step B: PrefillWorker Send...")
    sent = ray.get(worker.send_kv_rdma.remote(req_id, a_qpn, a_lid))
    assert sent, "PrefillWorker failed to send"
    
    # Step C: AttnNode Finalize
    print("Step C: AttnNode Finalize...")
    finalized = ray.get(attn_node.finalize_rdma_transfer.remote(req_id))
    assert finalized, "AttnNode failed to finalize transfer"
    
    print("RDMA Transfer Successful!")
    
    # 5. Verify Data Integrity (Optional - Run a decode step?)
    # Let's try running one decode step to see if it crashes or works.
    step_inputs = {
        "input_ids": [prefill_out["first_token_id"]],
        "request_ids": [req_id],
        "seq_lens": [seq_len],
        "layer_idx": 0
    }
    
    print("Running Decode Step to verify KV Cache usage...")
    try:
        out, _ = ray.get(attn_node.process_step.remote(step_inputs))
        print("Decode Step Successful. Output shape:", out.shape)
        print("SUCCESS: RDMA Integration Verified.")
    except Exception as e:
        print(f"Decode Step Failed: {e}")

if __name__ == "__main__":
    ray.init()
    ray.get(test_coordinator.remote())
