import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from core.memory_manager import KVCacheManager
from core.graph_compiler import AttnPartition

class MockAttention(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.out_proj = nn.Linear(hidden, hidden, bias=False)
        self.num_heads = heads
        self.head_dim = hidden // heads

class MockLayer(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.self_attn = MockAttention(hidden, heads)
        self.self_attn_layer_norm = nn.LayerNorm(hidden)

def test_real_attention_flow():
    print("Testing Real Attention Flow...")
    
    # 1. Setup Infra
    hidden_size = 64
    num_heads = 4
    head_dim = 16
    kv_manager = KVCacheManager(
        num_blocks=10,
        block_size=4,
        num_heads=num_heads,
        head_dim=head_dim,
        device="cpu"
    )
    
    # 2. Setup Partition
    mock_layer = MockLayer(hidden_size, num_heads)
    attn_partition = AttnPartition(mock_layer)
    
    # 3. Simulate Prefill Data
    req_id = "req_1"
    seq_len = 5
    # Random K, V for prefill [Seq, Heads, Dim]
    k_prefill = torch.randn(seq_len, num_heads, head_dim)
    v_prefill = torch.randn(seq_len, num_heads, head_dim)
    
    # Load into manager
    kv_manager.load_initial_kv(req_id, k_prefill, v_prefill)
    
    print(f"Initial Context Len: {kv_manager.context_lens[req_id]}")
    assert kv_manager.context_lens[req_id] == 5
    
    # 4. Run Decode Step
    # Input x: [Batch=1, Seq=1, Hidden]
    x = torch.randn(1, 1, hidden_size)
    
    # Run Forward
    try:
        out = attn_partition(
            x, 
            kv_manager=kv_manager, 
            request_ids=[req_id]
        )
        print("Forward pass successful.")
        print(f"Output shape: {out.shape}")
        
        # Verify Context Len Increased
        new_len = kv_manager.context_lens[req_id]
        print(f"New Context Len: {new_len}")
        assert new_len == 6
        assert out.shape == (1, 1, hidden_size)
        
        print("SUCCESS: Real Attention Test Passed.")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()


def check_kernel_vs_reference():
    print("\n--- Checking Kernel vs Reference Accuracy ---")
    kv_manager = KVCacheManager(
        num_blocks=100,
        block_size=16,
        num_heads=4,
        head_dim=32,
        dtype=torch.float16,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Setup dummy data
    req_id = "req_test_kernel"
    kv_manager.allocate(req_id, seq_len=0)
    
    # Fill some data
    k = torch.randn(4, 32, dtype=torch.float16, device=kv_manager.device)
    v = torch.randn(4, 32, dtype=torch.float16, device=kv_manager.device)
    
    # Write 50 tokens
    for _ in range(50):
        kv_manager.write_new_kv(req_id, k, v)
        
    # Query
    q = torch.randn(1, 4, 32, dtype=torch.float16, device=kv_manager.device)
    
    # 1. Run Kernel (default if installed)
    # Note: KVCacheManager automatically tries to use kernel if installed.
    # We trust it is installed.
    try:
        out_kernel = kv_manager.attention(q, [req_id])
    except Exception as e:
        print(f"Kernel Execution Failed: {e}")
        return

    # 2. Run Reference (Force calling private method)
    out_ref = kv_manager._paged_attention_ref(q, [req_id])
    
    print(f"Kernel Output Mean: {out_kernel.mean().item()}")
    print(f"Reference Output Mean: {out_ref.mean().item()}")
    
    # Compare
    if torch.allclose(out_kernel, out_ref, atol=1e-2, rtol=1e-2):
        print("SUCCESS: Kernel matches Reference Implementation!")
    else:
        diff = (out_kernel - out_ref).abs().mean()
        print(f"FAILURE: Kernel output mismatch! Mean Diff: {diff}")
        print("Kernel:", out_kernel[0,0,:5])
        print("Reference:", out_ref[0,0,:5])

if __name__ == "__main__":
    test_real_attention_flow()
    check_kernel_vs_reference()

