from pydantic import BaseModel, ConfigDict
from typing import Optional, List

class NodeConfig(BaseModel):
    gpu_id: int
    worker_id: int
    node_type: str  # "prefill", "attn", "ffn"

class ClusterConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    num_prefill_workers: int = 1
    num_decode_layers: int = 2  # Kept for compatibility with older scripts.
    num_attention_nodes: int = 1
    num_decode_dense_nodes: int = 1
    head_node_ip: str = "localhost"
    redis_port: int = 6379
    prefill_resource: Optional[str] = None
    decode_dense_resource: Optional[str] = None
    attention_resource: Optional[str] = None
    use_gpu_for_prefill: bool = True
    use_gpu_for_decode_dense: bool = True
    attention_backend: str = "cpu"

class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model_name: str = "custom"
    model_path: str = "/home/cml/CloverInfer/model/opt-125m" 
    max_seq_len: int = 2048
    dtype: str = "float16" # or "int8"
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
    max_new_tokens: int = 20
