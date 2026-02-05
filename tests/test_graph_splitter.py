import torch
import torch.nn as nn
import sys
import os

# Add src to pythonpath
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from core.graph_compiler import split_transformer_layer

class DummyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.Linear(hidden_dim, hidden_dim) # Mock Attn
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x):
        # Pre-LN implementation
        # x = x + attn(norm1(x))
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x)
        x = x + residual
        
        # x = x + mlp(norm2(x))
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x

def test_split():
    torch.manual_seed(42)
    dim = 32
    layer = DummyTransformerLayer(dim)
    
    # Random input
    x = torch.randn(1, 10, dim)
    
    # 1. Run Original
    with torch.no_grad():
        original_output = layer(x)
    
    # 2. Split
    attn_part, ffn_part = split_transformer_layer(layer)
    
    # 3. Run Split Parts
    with torch.no_grad():
        # Attn Part: expects x, residual=x
        attn_out = attn_part(x, residual=x)
        
        # FFN Part: expects attn_out, residual=attn_out
        ffn_out = ffn_part(attn_out, residual=attn_out)
    
    # 4. Compare
    print("Original Output Mean:", original_output.mean().item())
    print("Split Output Mean:", ffn_out.mean().item())
    
    if torch.allclose(original_output, ffn_out, atol=1e-6):
        print("SUCCESS: Split execution matches original.")
    else:
        print("FAILURE: Mismatch!")
        # Debug
        print("Diff:", (original_output - ffn_out).abs().max().item())

if __name__ == "__main__":
    test_split()
