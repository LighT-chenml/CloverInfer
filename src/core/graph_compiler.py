import torch
import torch.nn as nn
import torch.fx as fx
import copy

class GraphSplitter:
    def __init__(self):
        pass

    def split_decode_model(self, model: nn.Module):
        """
        Splits a Transformer model (or layer) into Attention-only and FFN-only submodules.
        This is a heuristic-based splitter assuming standard naming conventions (e.g. self_attn, mlp).
        
        Returns:
            attn_graph (nn.Module): The Attention partition
            ffn_graph (nn.Module): The FFN partition
        """
        # We trace the model to get the graph
        traced = fx.symbolic_trace(model)
        
        # In a real scenario, we need a sophisticated partitioning algorithm.
        # For this "DIY" framework, we will create two copies of the model
        # and "dead code elimination" the parts we don't need based on a mask or
        # manually rewriting the graph.
        
        # However, a cleaner way for verified structure is to extract submodules.
        # Let's assume the user provides a "TransformerBlock".
        # We want to pull out `self_attn` and `mlp`.
        
        # Approach: Return wrappers that execute valid paths.
        # But designing a generic splitter is complex. 
        # Let's implement a manual split for the "Clover" architecture where 
        # we assume we have control over the model definition or can monkey-patch it.
        
        # Generic approach using FX:
        # 1. Inspect graph nodes.
        # 2. Identify Attn nodes and FFN nodes. 
        # 3. Cut edges.
        
        # Simplified Implementation for v1: Focus on Layer-wise split.
        # We assume the input model is a `TransformerLayer`.
        
        attn_part = AttnPartition(model)
        ffn_part = FFNPartition(model)
        
        return attn_part, ffn_part

class AttnPartition(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        # Deepcopy to avoid sharing state meant to be split, 
        # though weights should theoretically be separate? 
        # For V100 single node, sharing weights in memory is fine.
        # For distributed, we'd need to serialize only relevant parts.
        self.layer = copy.deepcopy(original_layer)
        
        # Pruning: Remove MLP parts to save memory (Simulated)
        if hasattr(self.layer, 'mlp'):
            self.layer.mlp = nn.Identity() 
    
    def forward(self, x, **kwargs):
        # Only execute Attention part
        # Assuming standard Llama/OPT structure pre-norm: 
        # y = x + self.attn(self.norm1(x))
        # But we want to separate the residual usually? 
        # For simplicity in V1, we compute the residual block internal to this node if it fits.
        # Flow: Norm -> Attn -> Residual Add
        
        residual = x
        
        kv_manager = kwargs.get("kv_manager")
        request_ids = kwargs.get("request_ids")
        
        # 1. Norm
        norm1 = getattr(self.layer, "input_layernorm", None) or getattr(self.layer, "norm1", None) or getattr(self.layer, "self_attn_layer_norm", None)
        if norm1:
            x_norm = norm1(x)
        else:
            x_norm = x
            
        if kv_manager:
            # REAL ATTENTION IMPLEMENTATION
            
            # 2. QKV Projection
            # OPT / Llama Support
            attn_module = self.layer.self_attn
            
            if hasattr(attn_module, "q_proj"):
                q = attn_module.q_proj(x_norm)
                k = attn_module.k_proj(x_norm)
                v = attn_module.v_proj(x_norm)
            else:
                # Fallback or unknown
                raise ValueError(f"Unknown attention module structure: {attn_module}")

            # Reshape Q, K, V to [Batch, Heads, Dim]
            # Assuming x is [Batch, 1, Hidden]
            # Proj output is [Batch, 1, Hidden] -> [Batch, Heads, HeadDim]
            # Note: num_heads and head_dim logic?
            # Usually Linear layer output is [Batch, 1, Hidden].
            # We need to reshape.
            
            b_size, seq_len, hidden_dim = q.shape
            num_heads = kv_manager.num_heads
            head_dim = kv_manager.head_dim
            
            q = q.view(b_size, seq_len, num_heads, head_dim)
            k = k.view(b_size, seq_len, num_heads, head_dim)
            v = v.view(b_size, seq_len, num_heads, head_dim)
            
            # 3. Write KV Cache
            # Iterate batch and write
            # Assumes b_size == len(request_ids)
            for i, rid in enumerate(request_ids):
                # k[i] is [1, Heads, Dim] -> squeeze -> [Heads, Dim]
                kv_manager.write_new_kv(rid, k[i], v[i])
                
            # 4. Paged Attention
            # q needs to be [Batch, Heads, Dim] (squeeze seq_len=1)
            q_squeezed = q.squeeze(1) 
            attn_output = kv_manager.attention(q_squeezed, request_ids) # [Batch, Heads, Dim]
            
            # Reshape back to [Batch, 1, Hidden]
            attn_output = attn_output.view(b_size, seq_len, hidden_dim)
            
            # 5. Output Projection
            if hasattr(attn_module, "out_proj"):
                final = attn_module.out_proj(attn_output)
            elif hasattr(attn_module, "o_proj"):
                final = attn_module.o_proj(attn_output)
            else:
                final = attn_output # Should not happen usually
                
            x = residual + final
            return x

        else:
            # ORIGINAL / FALLBACK (No KV Manager)
            # 2. Attention
            # We need to handle KVCache here ideally, but for now let's assume standard forward
            # calls into the underlying self_attn which might be patched or we pass kwargs.
            
            # Note: We must pass x_norm if we computed it, but original code did x = norm(x); x = attn(x).
            # We reused logic above.
            
            out = self.layer.self_attn(x_norm, **kwargs)[0] # [0] for attn_output
            
            # 3. Residual
            x = residual + out
            return x

class FFNPartition(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.layer = copy.deepcopy(original_layer)
        # Prune Attention
        if hasattr(self.layer, 'self_attn'):
             self.layer.self_attn = nn.Identity()
             
    def forward(self, x):
        # Flow: Norm -> MLP -> Residual Add
        residual = x
        
        # Norm
        norm2 = getattr(self.layer, "post_attention_layernorm", None) or getattr(self.layer, "norm2", None) or getattr(self.layer, "final_layer_norm", None)
        if norm2:
            x = norm2(x)
            
        # MLP
        # Logic for Llama (gate_proj etc) vs OPT (fc1/fc2)
        if hasattr(self.layer, "mlp"):
             x = self.layer.mlp(x)
        elif hasattr(self.layer, "fc1"): # OPT
             x = self.layer.activation_fn(self.layer.fc1(x))
             x = self.layer.fc2(x)
             
        # Residual
        x = residual + x
        return x

# Better Approach: FX Graph Rewriting
def split_transformer_layer(layer: nn.Module):
    """
    Splits a standard transformer layer into Attn and FFN graphs.
    Supports:
    - Standard: self_attn + mlp
    - OPT: self_attn + fc1/fc2
    """
    # OPT Support
    if hasattr(layer, "fc1") and hasattr(layer, "fc2"):
        # Create a mock MLP container for OPT
        mlp = nn.Sequential(
            layer.fc1,
            layer.activation_fn if hasattr(layer, "activation_fn") else nn.ReLU(),
            layer.fc2
        )
        # OPT norms: self_attn_layer_norm, final_layer_norm
        norm1 = getattr(layer, "self_attn_layer_norm", None)
        norm2 = getattr(layer, "final_layer_norm", None)
        
        return LayerSplit(layer.self_attn, norm1), LayerSplit(mlp, norm2)

    # Llama / Standard Support
    # Assumes self_attn and mlp
    mlp = getattr(layer, "mlp", None)
    if mlp is None:
        # Fallback or specific check
        raise ValueError(f"Unsupported layer structure: {layer}")

    norm1 = getattr(layer, "input_layernorm", None) or getattr(layer, "norm1", None)
    norm2 = getattr(layer, "post_attention_layernorm", None) or getattr(layer, "norm2", None)

    return LayerSplit(layer.self_attn, norm1), LayerSplit(mlp, norm2)

class LayerSplit(nn.Module):
    def __init__(self, main_module, norm_module=None):
        super().__init__()
        self.main = main_module
        self.norm = norm_module
    
    def forward(self, x, residual=None):
        # Standard Pre-Norm: x + Main(Norm(x))
        # But some are Post-Norm. 
        # For this simplified frame, we assume Pre-Norm as in Llama/OPT (mostly).
        # OPT is Pre-Norm? actually OPT is Pre-Norm.
        
        # Save residual for connection
        res_connection = x if residual is None else residual
        
        out = x
        if self.norm:
            out = self.norm(out)
        
        out = self.main(out)
        
        if residual is not None:
             out = out + res_connection
             
        return out
