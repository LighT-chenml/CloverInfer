from __future__ import annotations

import os
import socket
import time
from typing import Dict, List

import ray
import torch
from transformers import AutoTokenizer, OPTForCausalLM
from transformers.utils import logging

from .attention_backend import CpuAttentionBackend
from .config import ModelConfig

logging.disable_progress_bar()
logging.set_verbosity_error()


def _dtype_from_config(config: ModelConfig, device: str):
    if device == "cpu":
        return torch.float32
    return torch.float16 if config.dtype == "float16" else torch.float32


def _select_device(prefer_gpu: bool) -> str:
    return "cuda" if prefer_gpu and torch.cuda.is_available() else "cpu"


def _actor_info(node_id: int, role: str, device: str) -> Dict[str, str]:
    return {
        "node_id": str(node_id),
        "role": role,
        "hostname": socket.gethostname(),
        "ip": socket.gethostbyname(socket.gethostname()),
        "device": device,
    }


@ray.remote
class PrefillNode:
    def __init__(self, node_id: int, config: ModelConfig, prefer_gpu: bool = True):
        self.node_id = node_id
        self.config = config
        self.device = _select_device(prefer_gpu)
        self.dtype = _dtype_from_config(config, self.device)

        print(f"PrefillNode {node_id} loading model from {config.model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
        self.model = OPTForCausalLM.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.model_config = self.model.config

    def get_info(self):
        return _actor_info(self.node_id, "prefill", self.device)

    def get_model_spec(self):
        return {
            "num_layers": int(self.model.config.num_hidden_layers),
            "hidden_size": int(self.model.config.hidden_size),
            "num_heads": int(self.model.config.num_attention_heads),
            "vocab_size": int(self.model.config.vocab_size),
        }

    def process_prompt(self, prompt: str):
        request_id = f"req_{time.time_ns()}"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, use_cache=True)
            logits = outputs.logits[:, -1, :]
            first_token_id = int(torch.argmax(logits, dim=-1).item())

        initial_kv: List[Dict[str, torch.Tensor]] = []
        for layer_kv in outputs.past_key_values:
            key = layer_kv.key_cache if hasattr(layer_kv, "key_cache") else layer_kv[0]
            value = layer_kv.value_cache if hasattr(layer_kv, "value_cache") else layer_kv[1]

            # HF OPT cache is [batch, heads, seq, head_dim]. Store [seq, heads, head_dim].
            initial_kv.append(
                {
                    "key": key[0].permute(1, 0, 2).contiguous().cpu(),
                    "value": value[0].permute(1, 0, 2).contiguous().cpu(),
                }
            )

        return {
            "request_id": request_id,
            "prompt_len": int(input_ids.shape[1]),
            "first_token_id": first_token_id,
            "initial_kv": initial_kv,
        }


@ray.remote
class AttentionNode:
    def __init__(self, node_id: int, config: ModelConfig, backend: str = "cpu"):
        if backend != "cpu":
            raise ValueError(f"Unsupported attention backend for now: {backend}")
        self.node_id = node_id
        self.config = config
        self.backend_name = backend
        self.backend = CpuAttentionBackend()
        self.device = "cpu"
        print(f"AttentionNode {node_id} initialized with {backend} backend")

    def get_info(self):
        info = _actor_info(self.node_id, "attention", self.device)
        info["backend"] = self.backend_name
        return info

    def init_request(self, request_id: str, initial_kv):
        return self.backend.init_request(request_id, initial_kv)

    def decode_layer(self, payload):
        return self.backend.decode_layer(
            payload["request_id"],
            int(payload["layer_idx"]),
            payload["query"],
            payload["key"],
            payload["value"],
        )

    def get_context_len(self, request_id: str):
        return self.backend.get_context_len(request_id)

    def free_request(self, request_id: str):
        self.backend.free_request(request_id)
        return True


@ray.remote
class DecodeDenseNode:
    def __init__(self, node_id: int, config: ModelConfig, prefer_gpu: bool = True):
        self.node_id = node_id
        self.config = config
        self.device = _select_device(prefer_gpu)
        self.dtype = _dtype_from_config(config, self.device)

        print(f"DecodeDenseNode {node_id} loading model from {config.model_path} on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path, local_files_only=True)
        self.model = OPTForCausalLM.from_pretrained(
            config.model_path,
            local_files_only=True,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()
        self.decoder = self.model.model.decoder
        self.layers = self.decoder.layers
        self.num_layers = len(self.layers)
        self.hidden_size = self.model.config.hidden_size
        self.num_heads = self.model.config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

    def get_info(self):
        return _actor_info(self.node_id, "decode_dense", self.device)

    def get_model_spec(self):
        return {
            "num_layers": int(self.num_layers),
            "hidden_size": int(self.hidden_size),
            "num_heads": int(self.num_heads),
            "vocab_size": int(self.model.config.vocab_size),
        }

    def start_token(self, token_id: int, position: int):
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([[position]], dtype=torch.long, device=self.device)
        attention_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
        hidden = self.decoder.embed_tokens(input_ids)
        hidden = hidden + self.decoder.embed_positions(attention_mask, position_ids=position_ids)
        return hidden.detach().cpu()

    def prepare_attention(self, hidden_state, layer_idx: int, request_id: str):
        layer = self.layers[layer_idx]
        hidden = hidden_state.to(self.device)
        residual = hidden

        if layer.do_layer_norm_before:
            hidden = layer.self_attn_layer_norm(hidden)

        attn = layer.self_attn
        batch, seq_len, _ = hidden.shape
        if batch != 1 or seq_len != 1:
            raise ValueError("The first refactor supports batch=1, seq_len=1 decode only")

        query = attn.q_proj(hidden) * attn.scaling
        key = attn.k_proj(hidden)
        value = attn.v_proj(hidden)

        query = query.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)
        key = key.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)
        value = value.view(batch, seq_len, self.num_heads, self.head_dim).squeeze(1)

        return {
            "request_id": request_id,
            "layer_idx": int(layer_idx),
            "residual": residual.detach().cpu(),
            "query": query.detach().cpu(),
            "key": key.detach().cpu(),
            "value": value.detach().cpu(),
        }

    def finish_layer(self, residual, attention_context, layer_idx: int):
        layer = self.layers[layer_idx]
        residual = residual.to(self.device)
        context = attention_context.to(self.device)

        if context.dim() == 3:
            context = context.unsqueeze(1)
        attn_output = context.reshape(1, 1, self.hidden_size).contiguous()
        hidden = layer.self_attn.out_proj(attn_output)
        hidden = residual + hidden

        if not layer.do_layer_norm_before:
            hidden = layer.self_attn_layer_norm(hidden)

        hidden_shape = hidden.shape
        hidden = hidden.reshape(-1, hidden.size(-1))
        residual_ffn = hidden

        if layer.do_layer_norm_before:
            hidden = layer.final_layer_norm(hidden)

        hidden = layer.fc1(hidden)
        hidden = layer.activation_fn(hidden)
        hidden = layer.fc2(hidden)
        hidden = residual_ffn + hidden
        hidden = hidden.view(hidden_shape)

        if not layer.do_layer_norm_before:
            hidden = layer.final_layer_norm(hidden)

        return hidden.detach().cpu()

    def sample_next_token(self, hidden_state):
        hidden = hidden_state.to(self.device)
        final_norm = getattr(self.decoder, "final_layer_norm", None)
        if final_norm is not None:
            hidden = final_norm(hidden)
        logits = self.model.lm_head(hidden)
        return int(torch.argmax(logits[:, -1, :], dim=-1).item())

    def decode_tokens(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)


# Compatibility names for older scripts. RDMA-specific methods are intentionally
# not preserved in the correctness-first refactor.
PrefillWorker = PrefillNode
FFNNode = DecodeDenseNode
AttnNode = AttentionNode
