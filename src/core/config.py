from pydantic import BaseModel
from typing import Optional, List

class NodeConfig(BaseModel):
    gpu_id: int
    worker_id: int
    node_type: str  # "prefill", "attn", "ffn"

class ClusterConfig(BaseModel):
    num_prefill_workers: int = 1
    num_decode_layers: int = 2  # Simulating 2 layers for now
    head_node_ip: str = "localhost"
    redis_port: int = 6379

class ModelConfig(BaseModel):
    model_name: str = "custom"
    model_path: str = "/home/cml/CloverInfer/model/opt-125m" 
    max_seq_len: int = 2048
    dtype: str = "float16" # or "int8"
    hidden_size: int = 768
    num_heads: int = 12
    num_layers: int = 12
