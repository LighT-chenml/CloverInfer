from __future__ import annotations

import os
import socket
import time
from typing import Dict, List

import ray
import torch
from transformers.utils import logging

from .attention_backend import CpuAttentionBackend, PimNaiveAttentionBackend
from .config import ModelConfig
from .model_adapter import CausalModelAdapter

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
        "ip": ray.util.get_node_ip_address(),
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
        self.adapter = CausalModelAdapter(config.model_path, self.device, self.dtype)

    def get_info(self):
        info = _actor_info(self.node_id, "prefill", self.device)
        info["model_type"] = self.adapter.model_type
        return info

    def get_model_spec(self):
        return self.adapter.get_model_spec()

    def process_prompt(self, prompt: str):
        started_at = time.perf_counter()
        prefill_out = self.adapter.prefill(prompt)
        finished_at = time.perf_counter()
        prefill_out["profile"] = {
            "compute_s": float(finished_at - started_at),
        }
        return prefill_out


@ray.remote
class AttentionNode:
    def __init__(
        self,
        node_id: int,
        config: ModelConfig,
        backend: str = "cpu",
        backend_kwargs: Dict[str, object] | None = None,
    ):
        self.node_id = node_id
        self.config = config
        self.backend_name = backend
        self.device = "cpu"
        backend_kwargs = backend_kwargs or {}
        if backend == "cpu":
            self.backend = CpuAttentionBackend()
        elif backend == "pim_naive":
            self.backend = PimNaiveAttentionBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unsupported attention backend for now: {backend}")
        print(f"AttentionNode {node_id} initialized with {backend} backend")

    def get_info(self):
        info = _actor_info(self.node_id, "attention", self.device)
        info["backend"] = self.backend_name
        if hasattr(self.backend, "get_debug_info"):
            info["backend_debug"] = self.backend.get_debug_info()
        return info

    def init_request(self, request_id: str, initial_kv):
        started_at = time.perf_counter()
        context_len = self.backend.init_request(request_id, initial_kv)
        finished_at = time.perf_counter()
        return {
            "context_len": int(context_len),
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def decode_layer(self, payload):
        started_at = time.perf_counter()
        context = self.backend.decode_layer(
            payload["request_id"],
            int(payload["layer_idx"]),
            payload["query"],
            payload["key"],
            payload["value"],
            float(payload.get("score_scale", 1.0)),
        )
        finished_at = time.perf_counter()
        return {
            "context": context,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

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
        self.adapter = CausalModelAdapter(config.model_path, self.device, self.dtype)

    def get_info(self):
        info = _actor_info(self.node_id, "decode_dense", self.device)
        info["model_type"] = self.adapter.model_type
        return info

    def get_model_spec(self):
        return self.adapter.get_model_spec()

    def start_token(self, token_id: int, position: int):
        started_at = time.perf_counter()
        hidden = self.adapter.start_token(token_id, position)
        finished_at = time.perf_counter()
        return {
            "hidden": hidden,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def prepare_attention(self, hidden_state, layer_idx: int, request_id: str, context_len: int):
        started_at = time.perf_counter()
        prepared = self.adapter.prepare_attention(hidden_state, layer_idx, request_id, context_len)
        finished_at = time.perf_counter()
        return {
            "request_id": request_id,
            "layer_idx": int(layer_idx),
            "residual": prepared["residual"],
            "query": prepared["query"],
            "key": prepared["key"],
            "value": prepared["value"],
            "score_scale": float(prepared.get("score_scale", 1.0)),
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def finish_layer(self, residual, attention_context, layer_idx: int):
        started_at = time.perf_counter()
        hidden = self.adapter.finish_layer(residual, attention_context, layer_idx)
        finished_at = time.perf_counter()
        return {
            "hidden": hidden,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def sample_next_token(self, hidden_state):
        started_at = time.perf_counter()
        token_id = self.adapter.sample_next_token(hidden_state)
        finished_at = time.perf_counter()
        return {
            "token_id": token_id,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def decode_tokens(self, token_ids: List[int]) -> str:
        started_at = time.perf_counter()
        text = self.adapter.decode_tokens(token_ids)
        finished_at = time.perf_counter()
        return {
            "text": text,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }


# Compatibility names for older scripts. RDMA-specific methods are intentionally
# not preserved in the correctness-first refactor.
PrefillWorker = PrefillNode
FFNNode = DecodeDenseNode
AttnNode = AttentionNode
