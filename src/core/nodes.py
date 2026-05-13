from __future__ import annotations

import asyncio
import importlib
import socket
import time
from typing import Dict, List

import ray
import torch
from transformers.utils import logging

from .attention_backend import CpuAttentionBackend, GpuAttentionBackend, PimNaiveAttentionBackend
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
        prefer_gpu: bool = False,
        decode_batch_window_s: float = 0.001,
        decode_batch_max_size: int = 8,
    ):
        self.node_id = node_id
        self.config = config
        self.backend_name = backend
        self.device = _select_device(prefer_gpu)
        backend_kwargs = backend_kwargs or {}
        attention_sparse_window = max(0, int(backend_kwargs.pop("attention_sparse_window", 0)))
        self.rankset_overlap_async_dispatch_enabled = bool(
            backend_kwargs.pop("rankset_overlap_async_dispatch_enabled", False)
        )
        self.rankset_overlap_transfer_latency_s = max(
            0.0,
            float(backend_kwargs.pop("rankset_overlap_transfer_latency_s", 0.0)),
        )
        decode_batch_window_s = float(backend_kwargs.pop("decode_batch_window_s", decode_batch_window_s))
        decode_batch_max_size = int(backend_kwargs.pop("decode_batch_max_size", decode_batch_max_size))
        if backend == "cpu":
            self.device = "cpu"
            self.backend = CpuAttentionBackend(attention_sparse_window=attention_sparse_window)
        elif backend == "gpu":
            self.device = _select_device(True)
            self.backend = GpuAttentionBackend(attention_sparse_window=attention_sparse_window)
        elif backend == "pim_naive":
            self.device = "cpu"
            self.backend = PimNaiveAttentionBackend(attention_sparse_window=attention_sparse_window, **backend_kwargs)
        elif backend == "cloverinfer":
            importlib.invalidate_caches()
            from .clover_attention_backend import CloverInferAttentionBackend

            self.device = "cpu"
            self.backend = CloverInferAttentionBackend(attention_sparse_window=attention_sparse_window, **backend_kwargs)
        else:
            raise ValueError(f"Unsupported attention backend for now: {backend}")
        self.decode_batch_window_s = max(0.0, float(decode_batch_window_s))
        self.decode_batch_max_size = max(1, int(decode_batch_max_size))
        self.decode_batch_enabled = hasattr(self.backend, "decode_layer_batch")
        self.decode_batch_flushes = 0
        self.decode_batch_total_items = 0
        self.decode_batch_max_observed = 0
        self.rankset_task_graph_exec_batches = 0
        self.rankset_task_graph_exec_work_items = 0
        self.rankset_task_graph_exec_max_work_items = 0
        self.rankset_task_graph_exec_fallback_batches = 0
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
        info["rankset_task_graph_execution"] = {
            "batches": int(self.rankset_task_graph_exec_batches),
            "work_items": int(self.rankset_task_graph_exec_work_items),
            "max_work_items": int(self.rankset_task_graph_exec_max_work_items),
            "fallback_batches": int(self.rankset_task_graph_exec_fallback_batches),
            "async_dispatch_enabled": bool(self.rankset_overlap_async_dispatch_enabled),
            "transfer_latency_s": float(self.rankset_overlap_transfer_latency_s),
        }
        if hasattr(self.backend, "get_debug_info"):
            info["backend_debug"] = self.backend.get_debug_info()
        return info

    def init_request(
        self,
        request_id: str,
        initial_kv,
        decode_reserve_tokens: int = 0,
        expected_decode_batch_size: int = 1,
    ):
        started_at = time.perf_counter()
        if hasattr(self.backend, "expected_decode_batch_max_size"):
            self.backend.expected_decode_batch_max_size = max(1, int(expected_decode_batch_size))
        context_len = self.backend.init_request(
            request_id,
            initial_kv,
            decode_reserve_tokens=int(decode_reserve_tokens),
        )
        packing_hint = {}
        if hasattr(self.backend, "get_request_packing_hint"):
            packing_hint = dict(self.backend.get_request_packing_hint(request_id) or {})
        finished_at = time.perf_counter()
        return {
            "context_len": int(context_len),
            "packing_hint": packing_hint,
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def _fallback_rankset_execution(
        self,
        payloads,
        execution_mode: str,
        fallback_reason: str,
    ):
        started_at = time.perf_counter()
        contexts = self.backend.decode_layer_batch(payloads)
        per_item_compute_s = float(time.perf_counter() - started_at) / max(len(contexts), 1)
        self.rankset_task_graph_exec_fallback_batches += 1
        return [
            {
                "context": context,
                "profile": {
                    "compute_s": per_item_compute_s,
                    "batch_size": len(contexts),
                },
                "rankset_execution": {
                    "enabled": bool(dict(payload.get("rankset_task_graph", {}) or {}).get("enabled", False)),
                    "layer_idx": int(payload.get("layer_idx", 0)),
                    "request_id": str(payload.get("request_id", "")),
                    "execution_mode": execution_mode,
                    "work_item_count": int(dict(payload.get("rankset_task_graph", {}) or {}).get("work_item_count", 0)),
                    "executed_work_item_ids": [],
                    "executed_work_item_count": 0,
                    "rankset_ids": [
                        str(item.get("rankset_id"))
                        for item in list(payload.get("request_rankset_plan", []) or [])
                        if item.get("rankset_id") is not None
                    ],
                    "request_rankset_count": int(len(list(payload.get("request_rankset_plan", []) or []))),
                    "fallback_to_full_batch": True,
                    "fallback_reason": str(fallback_reason),
                    "work_item_events": [],
                },
            }
            for payload, context in zip(payloads, contexts)
        ]

    async def _execute_rankset_task_graph_batch(self, payloads):
        if not payloads:
            return []
        task_graph = dict(payloads[0].get("rankset_task_graph", {}) or {})
        if not bool(task_graph.get("enabled", False)):
            return self._fallback_rankset_execution(
                payloads,
                execution_mode="scaffold_serial_attention",
                fallback_reason="task_graph_disabled",
            )

        work_items = list(task_graph.get("work_items", []) or [])
        if not work_items:
            return self._fallback_rankset_execution(
                payloads,
                execution_mode="scaffold_serial_attention",
                fallback_reason="task_graph_empty",
            )

        payload_by_request_id = {}
        for payload in payloads:
            request_id = str(payload.get("request_id", ""))
            if request_id in payload_by_request_id:
                return self._fallback_rankset_execution(
                    payloads,
                    execution_mode="scaffold_serial_attention",
                    fallback_reason="duplicate_request_id_in_payload_batch",
                )
            payload_by_request_id[request_id] = payload

        request_to_work_items = {}
        covered_request_ids = []
        for work_item in work_items:
            for request_id in work_item.get("request_ids", []) or []:
                request_id = str(request_id)
                request_to_work_items.setdefault(request_id, []).append(str(work_item.get("work_item_id", "")))
                covered_request_ids.append(request_id)

        payload_request_ids = set(payload_by_request_id.keys())
        covered_request_id_set = set(covered_request_ids)
        if payload_request_ids != covered_request_id_set:
            return self._fallback_rankset_execution(
                payloads,
                execution_mode="scaffold_serial_attention",
                fallback_reason="task_graph_request_coverage_mismatch",
            )
        multi_work_item_per_request = any(len(work_item_ids) != 1 for work_item_ids in request_to_work_items.values())
        if multi_work_item_per_request and not (
            hasattr(self.backend, "prepare_decode_records")
            and hasattr(self.backend, "compute_rankset_partial_contexts")
            and hasattr(self.backend, "finalize_rankset_decode_records")
        ):
            return self._fallback_rankset_execution(
                payloads,
                execution_mode="scaffold_serial_attention",
                fallback_reason="multi_work_item_per_request_not_supported_yet",
            )

        results_by_request_id = {}
        executed_work_item_ids = []
        work_item_events = []
        self.rankset_task_graph_exec_batches += 1
        self.rankset_task_graph_exec_work_items += len(work_items)
        self.rankset_task_graph_exec_max_work_items = max(
            self.rankset_task_graph_exec_max_work_items,
            len(work_items),
        )

        prepared_records = None
        if multi_work_item_per_request:
            prepared_records = self.backend.prepare_decode_records(payloads)
        if self.rankset_overlap_async_dispatch_enabled and len(work_items) > 1:
            work_item_results = await self._execute_rankset_task_graph_batch_async(
                payload_by_request_id, work_items, prepared_records=prepared_records
            )
        else:
            work_item_results = await self._execute_rankset_task_graph_batch_serial(
                payload_by_request_id, work_items, prepared_records=prepared_records
            )

        partials_by_request = {}
        for item_result in work_item_results:
            work_item_id = str(item_result["work_item_id"])
            subpayloads = list(item_result["subpayloads"])
            contexts = list(item_result.get("contexts", []))
            per_item_compute_s = float(item_result["per_item_compute_s"])
            executed_work_item_ids.append(work_item_id)
            work_item_events.append(dict(item_result["event"]))
            partial_contexts = dict(item_result.get("partial_contexts_by_request", {}) or {})
            for request_id, partials in partial_contexts.items():
                partials_by_request.setdefault(str(request_id), []).extend(list(partials or []))
            for payload, context in zip(subpayloads, contexts):
                request_id = str(payload.get("request_id", ""))
                results_by_request_id[request_id] = {
                    "context": context,
                    "profile": {
                        "compute_s": per_item_compute_s,
                        "batch_size": len(subpayloads),
                    },
                    "rankset_execution": {
                        "enabled": True,
                        "layer_idx": int(payload.get("layer_idx", 0)),
                        "request_id": request_id,
                        "execution_mode": str(item_result["execution_mode"]),
                        "work_item_count": int(task_graph.get("work_item_count", 0)),
                        "executed_work_item_ids": list(executed_work_item_ids),
                        "executed_work_item_count": int(len(executed_work_item_ids)),
                        "rankset_ids": [
                            str(item.get("rankset_id"))
                            for item in list(payload.get("request_rankset_plan", []) or [])
                            if item.get("rankset_id") is not None
                        ],
                        "request_rankset_count": int(len(list(payload.get("request_rankset_plan", []) or []))),
                        "fallback_to_full_batch": False,
                        "fallback_reason": "",
                        "work_item_events": list(work_item_events),
                    },
                }

        if multi_work_item_per_request and prepared_records is not None:
            final_contexts = self.backend.finalize_rankset_decode_records(prepared_records, partials_by_request)
            for payload, context in zip(payloads, final_contexts):
                request_id = str(payload.get("request_id", ""))
                results_by_request_id[request_id] = {
                    "context": context,
                    "profile": {
                        "compute_s": 0.0,
                        "batch_size": len(payloads),
                    },
                    "rankset_execution": {
                        "enabled": True,
                        "layer_idx": int(payload.get("layer_idx", 0)),
                        "request_id": request_id,
                        "execution_mode": (
                            "rankset_task_graph_async_dispatch_serial_compute"
                            if self.rankset_overlap_async_dispatch_enabled and len(work_items) > 1
                            else "rankset_task_graph_serial"
                        ),
                        "work_item_count": int(task_graph.get("work_item_count", 0)),
                        "executed_work_item_ids": list(executed_work_item_ids),
                        "executed_work_item_count": int(len(executed_work_item_ids)),
                        "rankset_ids": [
                            str(item.get("rankset_id"))
                            for item in list(payload.get("request_rankset_plan", []) or [])
                            if item.get("rankset_id") is not None
                        ],
                        "request_rankset_count": int(len(list(payload.get("request_rankset_plan", []) or []))),
                        "fallback_to_full_batch": False,
                        "fallback_reason": "",
                        "work_item_events": list(work_item_events),
                    },
                }

        if len(results_by_request_id) != len(payloads):
            return self._fallback_rankset_execution(
                payloads,
                execution_mode="scaffold_serial_attention",
                fallback_reason="task_graph_execution_incomplete",
            )
        return [results_by_request_id[str(payload.get("request_id", ""))] for payload in payloads]

    def _build_rankset_subpayloads(self, payload_by_request_id, work_item):
        return [
            payload_by_request_id[str(request_id)]
            for request_id in work_item.get("request_ids", []) or []
            if str(request_id) in payload_by_request_id
        ]

    async def _execute_single_rankset_work_item(self, work_item, subpayloads, transferred_at: float, prepared_records=None):
        compute_started_at = time.perf_counter()
        partial_contexts_by_request = {}
        if prepared_records is not None:
            partial_contexts_by_request = self.backend.compute_rankset_partial_contexts(prepared_records, work_item)
            contexts = []
        else:
            contexts = self.backend.decode_layer_batch(subpayloads)
        compute_finished_at = time.perf_counter()
        compute_duration_s = float(compute_finished_at - compute_started_at)
        transfer_started_at = float(work_item.get("transfer_started_at", transferred_at))
        return {
            "work_item_id": str(work_item.get("work_item_id", "")),
            "subpayloads": subpayloads,
            "contexts": contexts,
            "partial_contexts_by_request": partial_contexts_by_request,
            "per_item_compute_s": compute_duration_s / max(len(contexts), 1),
            "execution_mode": "rankset_task_graph_async_dispatch_serial_compute",
            "event": {
                "work_item_id": str(work_item.get("work_item_id", "")),
                "layer_idx": int(work_item.get("layer_idx", 0)),
                "rankset_id": str(work_item.get("rankset_id", "")),
                "request_ids": [str(request_id) for request_id in work_item.get("request_ids", []) or []],
                "batch_size": int(len(subpayloads)),
                "transfer_started_at": float(transfer_started_at),
                "transfer_finished_at": float(transferred_at),
                "transfer_duration_s": max(0.0, float(transferred_at - transfer_started_at)),
                "compute_started_at": float(compute_started_at),
                "compute_finished_at": float(compute_finished_at),
                "compute_duration_s": float(compute_duration_s),
                "started_at": float(transfer_started_at),
                "finished_at": float(compute_finished_at),
                "duration_s": max(0.0, float(compute_finished_at - transfer_started_at)),
                "overlap_ready_at": float(transferred_at),
            },
        }

    async def _execute_rankset_task_graph_batch_serial(self, payload_by_request_id, work_items, prepared_records=None):
        results = []
        for work_item in work_items:
            subpayloads = self._build_rankset_subpayloads(payload_by_request_id, work_item)
            if not subpayloads:
                continue
            transfer_started_at = time.perf_counter()
            transferred_at = transfer_started_at
            if self.rankset_overlap_transfer_latency_s > 0.0:
                await asyncio.sleep(self.rankset_overlap_transfer_latency_s)
                transferred_at = time.perf_counter()
            work_item["transfer_started_at"] = transfer_started_at
            results.append(
                await self._execute_single_rankset_work_item(
                    work_item,
                    subpayloads,
                    transferred_at,
                    prepared_records=prepared_records,
                )
            )
        return results

    async def _simulate_rankset_transfer(self, work_item, subpayloads):
        transfer_started_at = time.perf_counter()
        if self.rankset_overlap_transfer_latency_s > 0.0:
            await asyncio.sleep(self.rankset_overlap_transfer_latency_s)
        transferred_at = time.perf_counter()
        return {
            "work_item": work_item,
            "subpayloads": subpayloads,
            "transfer_started_at": float(transfer_started_at),
            "transferred_at": float(transferred_at),
        }

    async def _execute_rankset_task_graph_batch_async(self, payload_by_request_id, work_items, prepared_records=None):
        transfer_tasks = []
        for work_item in work_items:
            subpayloads = self._build_rankset_subpayloads(payload_by_request_id, work_item)
            if not subpayloads:
                continue
            transfer_tasks.append(asyncio.create_task(self._simulate_rankset_transfer(work_item, subpayloads)))
        if not transfer_tasks:
            return []

        ready_items = []
        for transfer_task in asyncio.as_completed(transfer_tasks):
            ready_items.append(await transfer_task)
        ready_items.sort(key=lambda item: (float(item["transferred_at"]), str(item["work_item"].get("work_item_id", ""))))

        results = []
        for ready_item in ready_items:
            work_item = dict(ready_item["work_item"])
            work_item["transfer_started_at"] = float(ready_item["transfer_started_at"])
            results.append(
                await self._execute_single_rankset_work_item(
                    work_item,
                    list(ready_item["subpayloads"]),
                    float(ready_item["transferred_at"]),
                    prepared_records=prepared_records,
                )
            )
        return results

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
                try:
                    results = await self._execute_rankset_task_graph_batch(payloads)
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)
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

    async def decode_layer_batch(self, payloads):
        return await self._execute_rankset_task_graph_batch(payloads)

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

    def start_token_batch(self, token_ids: List[int], positions: List[int]):
        started_at = time.perf_counter()
        hidden = self.adapter.start_token_batch(token_ids, positions)
        finished_at = time.perf_counter()
        return {
            "hidden": hidden,
            "profile": {
                "compute_s": float(finished_at - started_at),
                "batch_size": len(token_ids),
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
            "num_query_heads": int(prepared.get("num_query_heads", prepared["query"].shape[-2])),
            "num_key_value_heads": int(prepared.get("num_key_value_heads", prepared["key"].shape[-2])),
            "profile": {
                "compute_s": float(finished_at - started_at),
            },
        }

    def prepare_attention_batch(
        self,
        hidden_states,
        layer_idx: int,
        request_ids: List[str],
        context_lens: List[int],
    ):
        started_at = time.perf_counter()
        prepared_items = self.adapter.prepare_attention_batch(
            hidden_states,
            layer_idx,
            request_ids,
            context_lens,
        )
        finished_at = time.perf_counter()
        per_item_compute_s = float(finished_at - started_at) / max(len(prepared_items), 1)
        return [
            {
                "request_id": request_id,
                "layer_idx": int(layer_idx),
                "residual": prepared["residual"],
                "query": prepared["query"],
                "key": prepared["key"],
                "value": prepared["value"],
                "score_scale": float(prepared.get("score_scale", 1.0)),
                "num_query_heads": int(prepared.get("num_query_heads", prepared["query"].shape[-2])),
                "num_key_value_heads": int(prepared.get("num_key_value_heads", prepared["key"].shape[-2])),
                "profile": {
                    "compute_s": per_item_compute_s,
                    "batch_size": len(prepared_items),
                },
            }
            for request_id, prepared in zip(request_ids, prepared_items)
        ]

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

    def finish_layer_batch(self, residuals, attention_contexts, layer_idx: int):
        started_at = time.perf_counter()
        hidden_states = self.adapter.finish_layer_batch(residuals, attention_contexts, layer_idx)
        finished_at = time.perf_counter()
        per_item_compute_s = float(finished_at - started_at) / max(len(hidden_states), 1)
        return [
            {
                "hidden": hidden,
                "profile": {
                    "compute_s": per_item_compute_s,
                    "batch_size": len(hidden_states),
                },
            }
            for hidden in hidden_states
        ]

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

    def sample_next_token_batch(self, hidden_states):
        started_at = time.perf_counter()
        token_ids = self.adapter.sample_next_token_batch(hidden_states)
        finished_at = time.perf_counter()
        per_item_compute_s = float(finished_at - started_at) / max(len(token_ids), 1)
        return [
            {
                "token_id": int(token_id),
                "profile": {
                    "compute_s": per_item_compute_s,
                    "batch_size": len(token_ids),
                },
            }
            for token_id in token_ids
        ]

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
