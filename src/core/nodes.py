from __future__ import annotations

import asyncio
import importlib
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
        decode_batch_window_s: float = 0.001,
        decode_batch_max_size: int = 8,
    ):
        self.node_id = node_id
        self.config = config
        self.backend_name = backend
        self.device = "cpu"
        backend_kwargs = backend_kwargs or {}
        decode_batch_window_s = float(backend_kwargs.pop("decode_batch_window_s", decode_batch_window_s))
        decode_batch_max_size = int(backend_kwargs.pop("decode_batch_max_size", decode_batch_max_size))
        if backend == "cpu":
            self.backend = CpuAttentionBackend()
        elif backend == "pim_naive":
            self.backend = PimNaiveAttentionBackend(**backend_kwargs)
        elif backend == "cloverinfer":
            importlib.invalidate_caches()
            from .clover_attention_backend import CloverInferAttentionBackend

            self.backend = CloverInferAttentionBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unsupported attention backend for now: {backend}")
        self.decode_batch_window_s = max(0.0, float(decode_batch_window_s))
        self.decode_batch_max_size = max(1, int(decode_batch_max_size))
        self.decode_batch_enabled = hasattr(self.backend, "decode_layer_batch")
        self.decode_batch_flushes = 0
        self.decode_batch_total_items = 0
        self.decode_batch_max_observed = 0
        self._decode_batch_queue: list[tuple[dict, asyncio.Future]] = []
        self._decode_batch_task: asyncio.Task | None = None
        print(f"AttentionNode {node_id} initialized with {backend} backend")

    def get_info(self):
        info = _actor_info(self.node_id, "attention", self.device)
        info["backend"] = self.backend_name
        info["decode_batching"] = {
            "enabled": bool(self.decode_batch_enabled),
            "window_s": float(self.decode_batch_window_s),
            "max_size": int(self.decode_batch_max_size),
            "flushes": int(self.decode_batch_flushes),
            "total_items": int(self.decode_batch_total_items),
            "max_observed_size": int(self.decode_batch_max_observed),
            "pending": len(self._decode_batch_queue),
        }
        if hasattr(self.backend, "get_debug_info"):
            info["backend_debug"] = self.backend.get_debug_info()
        return info

    def init_request(self, request_id: str, initial_kv, decode_reserve_tokens: int = 0):
        started_at = time.perf_counter()
        context_len = self.backend.init_request(
            request_id,
            initial_kv,
            decode_reserve_tokens=int(decode_reserve_tokens),
        )
        finished_at = time.perf_counter()
        return {
            "context_len": int(context_len),
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    async def _flush_decode_layer_batch(self):
        try:
            if self.decode_batch_window_s > 0:
                await asyncio.sleep(self.decode_batch_window_s)
            while self._decode_batch_queue:
                batch = self._decode_batch_queue[: self.decode_batch_max_size]
                del self._decode_batch_queue[: self.decode_batch_max_size]
                payloads = [item[0] for item in batch]
                futures = [item[1] for item in batch]
                self.decode_batch_flushes += 1
                self.decode_batch_total_items += len(payloads)
                self.decode_batch_max_observed = max(self.decode_batch_max_observed, len(payloads))
                started_at = time.perf_counter()
                try:
                    contexts = self.backend.decode_layer_batch(payloads)
                    per_item_compute_s = float(time.perf_counter() - started_at) / max(len(payloads), 1)
                    for future, context in zip(futures, contexts):
                        if not future.done():
                            future.set_result(
                                {
                                    "context": context,
                                    "profile": {
                                        "compute_s": per_item_compute_s,
                                        "batch_size": len(payloads),
                                    },
                                }
                            )
                except Exception as exc:
                    for future in futures:
                        if not future.done():
                            future.set_exception(exc)
        finally:
            self._decode_batch_task = None
            if self._decode_batch_queue:
                self._decode_batch_task = asyncio.create_task(self._flush_decode_layer_batch())

    async def decode_layer(self, payload):
        if not self.decode_batch_enabled:
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
                    "batch_size": 1,
                },
            }

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._decode_batch_queue.append((payload, future))
        if self._decode_batch_task is None:
            self._decode_batch_task = asyncio.create_task(self._flush_decode_layer_batch())
        return await future

    def decode_layer_batch(self, payloads):
        started_at = time.perf_counter()
        contexts = self.backend.decode_layer_batch(payloads)
        per_item_compute_s = float(time.perf_counter() - started_at) / max(len(contexts), 1)
        return [
            {
                "context": context,
                "profile": {
                    "compute_s": per_item_compute_s,
                    "batch_size": len(contexts),
                },
            }
            for context in contexts
        ]

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

    def continue_full_decode(self, initial_kv, prompt_len: int, first_token_id: int, max_new_tokens: int):
        started_at = time.perf_counter()
        result = self.adapter.continue_greedy_generate(
            initial_kv=initial_kv,
            prompt_len=int(prompt_len),
            first_token_id=int(first_token_id),
            max_new_tokens=int(max_new_tokens),
        )
        finished_at = time.perf_counter()
        result["profile"] = {
            "compute_s": float(finished_at - started_at),
        }
        return result


# Compatibility names for older scripts. RDMA-specific methods are intentionally
# not preserved in the correctness-first refactor.
PrefillWorker = PrefillNode
FFNNode = DecodeDenseNode
AttnNode = AttentionNode
