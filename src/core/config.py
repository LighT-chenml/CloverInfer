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
    redis_port: int = 26379
    prefill_resource: Optional[str] = None
    decode_dense_resource: Optional[str] = None
    attention_resource: Optional[str] = None
    use_gpu_for_prefill: bool = True
    use_gpu_for_decode_dense: bool = True
    attention_backend: str = "cpu"
    pim_num_dpus: int = 4
    pim_resident_store_backend: str = "host"
    pim_qk_full_enabled: bool = False
    pim_qk_full_shadow_check: bool = True
    pim_softmax_av_fused_enabled: bool = False
    pim_softmax_av_shadow_check: bool = True
    pim_qk_mixed_enabled: bool = True
    pim_qk_mixed_heads: int = 2
    pim_qk_mixed_window: int = 128
    pim_length: int = 128
    pim_block_tokens: int = 256
    pim_max_resident_groups_per_layer: int = 0
    pim_head_grouping_policy: str = "balanced"
    pim_dpu_placement_policy: str = "rotated"
    pim_resident_kv_dtype: str = "fp32"
    clover_cpu_shadow_enabled: bool = True
    clover_shadow_checks_enabled: bool = True
    clover_op_profiling_enabled: bool = True
    clover_shadow_check_token_interval: int = 4
    clover_shadow_check_layer_interval: int = 4
    clover_host_qk_mixed_enabled: bool = False
    clover_pim_attention_enabled: bool = False
    clover_pim_context_fused_experimental_enabled: bool = False
    clover_pim_rank_spread_alloc_experimental_enabled: bool = False
    clover_fine_head_grouping_experimental_enabled: bool = False
    clover_target_heads_per_group_experimental: int = 0
    decode_step_sync_window_s: float = 0.0
    decode_step_sync_max_size: int = 8
    attention_decode_wave_persist_enabled: bool = False
    attention_layer_barrier_window_s: float = 0.0
    attention_layer_barrier_max_size: int = 8
    attention_rpc_batch_window_s: float = 0.001
    attention_rpc_batch_max_size: int = 8
    attention_rpc_cross_key_batch_enabled: bool = False
    attention_actor_side_batching_enabled: bool = False
    attention_actor_batch_window_s: float = 0.001
    attention_actor_batch_max_size: int = 8

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
