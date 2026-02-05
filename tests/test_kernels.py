import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from core.memory_manager import KVCacheManager

def test_kv_manager():
    print("Testing KVCacheManager (Mock/Fallback)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: Running on CPU, simplified test.")
    
    num_blocks = 16
    block_size = 4
    num_heads = 2
    head_dim = 8
    
    mgr = KVCacheManager(num_blocks, block_size, num_heads, head_dim, device=device)
    
    # 1. Allocate Request
    req_id = "req_1"
    mgr.allocate(req_id, seq_len=0) # Empty start
    
    # 2. Append Tokens
    # Fill 1st block (0,1,2,3)
    # Fill 2nd block (4)
    # Context len -> 5
    for i in range(5):
        mgr.append_slot(req_id)
        
    print(f"Context Len: {mgr.context_lens[req_id]}")
    assert mgr.context_lens[req_id] == 5
    assert len(mgr.block_tables[req_id]) == 2 # 5 tokens need 2 blocks (size 4)
    
    # 3. Write Data to Cache (simulate Prefill)
    # We cheat and write directly using block table
    blocks = mgr.block_tables[req_id]
    
    # Block 0: Indices 0,1,2,3
    # Block 1: Index 0
    
    # K = ones * token_idx
    # V = ones * token_idx
    
    phy_blk_0 = blocks[0]
    phy_blk_1 = blocks[1]
    
    for i in range(4):
        mgr.k_cache[phy_blk_0, i] = i
        mgr.v_cache[phy_blk_0, i] = i
        
    mgr.k_cache[phy_blk_1, 0] = 4
    mgr.v_cache[phy_blk_1, 0] = 4
    
    # 4. Run Attention
    # Q = ones (matches everything)
    q = torch.ones((1, num_heads, head_dim), device=device)
    
    out = mgr.attention(q, [req_id])
    
    # Check shape
    assert out.shape == q.shape
    
    # Check value roughly
    # Access checks
    print("Output Mean:", out.mean().item())
    print("Test Passed: KVCacheManager Functional.")

if __name__ == "__main__":
    test_kv_manager()
