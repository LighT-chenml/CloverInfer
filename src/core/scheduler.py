from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import ray

from .config import ClusterConfig, ModelConfig
from .nodes import AttentionNode, DecodeDenseNode, PrefillNode


def _actor_options(resource_name: str | None, num_gpus: float = 0):
    options = {"num_gpus": num_gpus}
    if resource_name:
        options["resources"] = {resource_name: 0.01}
    return options


def _empty_stage_timing() -> Dict[str, object]:
    return {
        "scheduler": {
            "prefill_rpc_s": 0.0,
            "attention_init_rpc_s": 0.0,
            "start_token_rpc_s": 0.0,
            "prepare_attention_rpc_s": 0.0,
            "attention_decode_rpc_s": 0.0,
            "finish_layer_rpc_s": 0.0,
            "sample_next_token_rpc_s": 0.0,
            "decode_tokens_rpc_s": 0.0,
            "free_request_rpc_s": 0.0,
        },
        "actors": {
            "prefill_compute_s": 0.0,
            "attention_init_compute_s": 0.0,
            "dense_start_token_compute_s": 0.0,
            "dense_prepare_attention_compute_s": 0.0,
            "attention_decode_compute_s": 0.0,
            "dense_finish_layer_compute_s": 0.0,
            "dense_sample_next_token_compute_s": 0.0,
            "dense_decode_tokens_compute_s": 0.0,
        },
        "counts": {
            "decode_steps": 0,
            "decode_layers": 0,
        },
    }


@ray.remote
class GlobalScheduler:
    def __init__(self, cluster_config: ClusterConfig, model_config: ModelConfig):
        self.cluster_config = cluster_config
        self.model_config = model_config
        self.prefill_nodes = []
        self.attention_nodes = []
        self.decode_dense_nodes = []
        self.runtime_model_spec = {
            "num_layers": int(model_config.num_layers),
            "hidden_size": int(model_config.hidden_size),
            "num_heads": int(model_config.num_heads),
            "vocab_size": 0,
        }
        self.decode_step_sync_window_s = max(0.0, float(cluster_config.decode_step_sync_window_s))
        self.decode_step_sync_max_size = max(1, int(cluster_config.decode_step_sync_max_size))
        self.attention_decode_wave_persist_enabled = bool(
            getattr(cluster_config, "attention_decode_wave_persist_enabled", False)
        )
        self.decode_step_sync_flushes = 0
        self.decode_step_sync_total_items = 0
        self.decode_step_sync_max_observed = 0
        self._decode_step_sync_next_cohort_id = 1
        self._decode_step_sync_batches: dict[int, list[tuple[asyncio.Future, str]]] = {}
        self._decode_step_sync_tasks: dict[int, asyncio.Task] = {}
        self._inflight_request_count = 0
        self.attention_layer_barrier_window_s = max(
            0.0, float(cluster_config.attention_layer_barrier_window_s)
        )
        self.attention_layer_barrier_max_size = max(
            1, int(cluster_config.attention_layer_barrier_max_size)
        )
        self.attention_layer_barrier_flushes = 0
        self.attention_layer_barrier_total_items = 0
        self.attention_layer_barrier_max_observed = 0
        self._attention_layer_barrier_batches: dict[tuple[int, int], list[asyncio.Future]] = {}
        self._attention_layer_barrier_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self.attention_batch_window_s = max(0.0, float(cluster_config.attention_rpc_batch_window_s))
        self.attention_batch_max_size = max(1, int(cluster_config.attention_rpc_batch_max_size))
        self.attention_batch_flushes = 0
        self.attention_batch_total_items = 0
        self.attention_batch_max_observed = 0
        self._attention_wavefront_batches: dict[tuple[int, int], list[tuple[dict, asyncio.Future, object]]] = {}
        self._attention_wavefront_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._attention_wavefront_expected_sizes: dict[tuple[int, int], int] = {}
        self._active_decode_requests = 0

    def _attention_batch_target_size(self) -> int:
        return max(1, min(self._active_decode_requests, self.attention_batch_max_size))

    def _decode_step_sync_target_size(self) -> int:
        return max(1, min(self._inflight_request_count, self.decode_step_sync_max_size))

    def _attention_layer_barrier_target_size(self) -> int:
        return max(1, min(self._active_decode_requests, self.attention_layer_barrier_max_size))

    async def _flush_decode_step_sync(self, step: int):
        try:
            if self.decode_step_sync_window_s > 0:
                await asyncio.sleep(self.decode_step_sync_window_s)
            await self._execute_decode_step_sync(step)
        except asyncio.CancelledError:
            return

    async def _execute_decode_step_sync(self, step: int):
        batch = self._decode_step_sync_batches.pop(step, [])
        self._decode_step_sync_tasks.pop(step, None)
        if not batch:
            return
        group_size = len(batch)
        cohort = None
        if self.attention_decode_wave_persist_enabled:
            cohort_id = f"step{int(step)}_cohort{self._decode_step_sync_next_cohort_id}"
            self._decode_step_sync_next_cohort_id += 1
            cohort = {
                "cohort_id": cohort_id,
                "group_size": int(group_size),
            }
        self.decode_step_sync_flushes += 1
        self.decode_step_sync_total_items += group_size
        self.decode_step_sync_max_observed = max(self.decode_step_sync_max_observed, group_size)
        for future, request_id in batch:
            if not future.done():
                if cohort is None:
                    future.set_result(None)
                else:
                    result = dict(cohort)
                    result["request_id"] = request_id
                    future.set_result(result)

    async def _maybe_flush_decode_step_syncs(self):
        target_size = self._decode_step_sync_target_size()
        ready_steps = [
            step
            for step, batch in self._decode_step_sync_batches.items()
            if len(batch) >= target_size
        ]
        for step in ready_steps:
            task = self._decode_step_sync_tasks.pop(step, None)
            if task is not None:
                task.cancel()
            await self._execute_decode_step_sync(step)

    async def _synchronize_decode_step(self, step: int, request_id: str):
        if self.decode_step_sync_window_s <= 0:
            if not self.attention_decode_wave_persist_enabled:
                return None
            return {
                "cohort_id": f"step{int(step)}_solo",
                "group_size": 1,
                "request_id": request_id,
            }
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        batch = self._decode_step_sync_batches.setdefault(int(step), [])
        batch.append((future, request_id))
        target_size = self._decode_step_sync_target_size()
        if len(batch) >= target_size:
            task = self._decode_step_sync_tasks.pop(int(step), None)
            if task is not None:
                task.cancel()
            await self._execute_decode_step_sync(int(step))
        elif int(step) not in self._decode_step_sync_tasks:
            self._decode_step_sync_tasks[int(step)] = asyncio.create_task(
                self._flush_decode_step_sync(int(step))
            )
        return await future

    async def _flush_attention_layer_barrier_key(self, key: tuple[int, int]):
        try:
            if self.attention_layer_barrier_window_s > 0:
                await asyncio.sleep(self.attention_layer_barrier_window_s)
            await self._execute_attention_layer_barrier_key(key)
        except asyncio.CancelledError:
            return

    async def _execute_attention_layer_barrier_key(self, key: tuple[int, int]):
        batch = self._attention_layer_barrier_batches.pop(key, [])
        self._attention_layer_barrier_tasks.pop(key, None)
        if not batch:
            return
        group_size = len(batch)
        self.attention_layer_barrier_flushes += 1
        self.attention_layer_barrier_total_items += group_size
        self.attention_layer_barrier_max_observed = max(
            self.attention_layer_barrier_max_observed, group_size
        )
        for future in batch:
            if not future.done():
                future.set_result(group_size)

    async def _maybe_flush_attention_layer_barriers(self):
        target_size = self._attention_layer_barrier_target_size()
        ready_keys = [
            key
            for key, batch in self._attention_layer_barrier_batches.items()
            if len(batch) >= target_size
        ]
        for key in ready_keys:
            task = self._attention_layer_barrier_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_layer_barrier_key(key)

    async def _synchronize_attention_layer(self, key: tuple[int, int]) -> int | None:
        if self.attention_layer_barrier_window_s <= 0:
            return None
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        batch = self._attention_layer_barrier_batches.setdefault(key, [])
        batch.append(future)
        target_size = self._attention_layer_barrier_target_size()
        if len(batch) >= target_size:
            task = self._attention_layer_barrier_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_layer_barrier_key(key)
        elif key not in self._attention_layer_barrier_tasks:
            self._attention_layer_barrier_tasks[key] = asyncio.create_task(
                self._flush_attention_layer_barrier_key(key)
            )
        return await future

    async def _flush_attention_wavefront_key(self, key: tuple[int, int]):
        try:
            if self.attention_batch_window_s > 0:
                await asyncio.sleep(self.attention_batch_window_s)
            await self._execute_attention_wavefront_key(key)
        except asyncio.CancelledError:
            return

    async def _execute_attention_wavefront_key(self, key: tuple[int, int]):
        batch = self._attention_wavefront_batches.pop(key, [])
        self._attention_wavefront_tasks.pop(key, None)
        self._attention_wavefront_expected_sizes.pop(key, None)
        if not batch:
            return
        payloads = [item[0] for item in batch]
        futures = [item[1] for item in batch]
        attention = batch[0][2]
        self.attention_batch_flushes += 1
        self.attention_batch_total_items += len(payloads)
        self.attention_batch_max_observed = max(self.attention_batch_max_observed, len(payloads))
        try:
            results = await attention.decode_layer_batch.remote(payloads)
            for future, result in zip(futures, results):
                if not future.done():
                    future.set_result(result)
        except Exception as exc:
            for future in futures:
                if not future.done():
                    future.set_exception(exc)

    async def _maybe_flush_attention_wavefronts(self):
        ready_keys = []
        default_target_size = self._attention_batch_target_size()
        for key, batch in self._attention_wavefront_batches.items():
            target_size = self._attention_wavefront_expected_sizes.get(key, default_target_size)
            if len(batch) >= target_size:
                ready_keys.append(key)
        for key in ready_keys:
            task = self._attention_wavefront_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_wavefront_key(key)

    async def _batched_attention_decode(
        self,
        attention,
        prepared,
        decode_step: int,
        decode_wave: dict[str, object] | None = None,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        key: tuple[int, int] | tuple[int, int, str]
        cohort_size = None
        if self.attention_decode_wave_persist_enabled and decode_wave is not None:
            cohort_id = str(decode_wave.get("cohort_id", "default"))
            if "group_size" in decode_wave:
                cohort_size = max(1, int(decode_wave["group_size"]))
            key = (int(decode_step), int(prepared["layer_idx"]), cohort_id)
        else:
            key = (int(decode_step), int(prepared["layer_idx"]))
        barrier_group_size = await self._synchronize_attention_layer(key)
        batch = self._attention_wavefront_batches.setdefault(key, [])
        batch.append((prepared, future, attention))
        if cohort_size is not None:
            current_expected = self._attention_wavefront_expected_sizes.get(key, 1)
            self._attention_wavefront_expected_sizes[key] = max(current_expected, cohort_size)
        if barrier_group_size is not None:
            current_expected = self._attention_wavefront_expected_sizes.get(key, 1)
            self._attention_wavefront_expected_sizes[key] = max(current_expected, int(barrier_group_size))
        target_size = self._attention_wavefront_expected_sizes.get(
            key, self._attention_batch_target_size()
        )
        if len(batch) >= target_size:
            task = self._attention_wavefront_tasks.pop(key, None)
            if task is not None:
                task.cancel()
            await self._execute_attention_wavefront_key(key)
        elif key not in self._attention_wavefront_tasks:
            self._attention_wavefront_tasks[key] = asyncio.create_task(
                self._flush_attention_wavefront_key(key)
            )
        return await future

    async def initialize_cluster(self):
        gpu_prefill = 1 if self.cluster_config.use_gpu_for_prefill else 0
        gpu_dense = 1 if self.cluster_config.use_gpu_for_decode_dense else 0
        attention_backend_kwargs = {}
        if self.cluster_config.attention_backend in {"pim_naive", "cloverinfer"}:
            attention_backend_kwargs = {
                "num_dpus": int(self.cluster_config.pim_num_dpus),
                "length": int(self.cluster_config.pim_length),
                "resident_store_backend": str(self.cluster_config.pim_resident_store_backend),
                "max_resident_groups_per_layer": int(self.cluster_config.pim_max_resident_groups_per_layer),
                "head_grouping_policy": str(self.cluster_config.pim_head_grouping_policy),
                "dpu_placement_policy": str(self.cluster_config.pim_dpu_placement_policy),
                "resident_kv_dtype": str(self.cluster_config.pim_resident_kv_dtype),
                "qk_full_enabled": bool(self.cluster_config.pim_qk_full_enabled),
                "qk_full_shadow_check": bool(self.cluster_config.pim_qk_full_shadow_check),
                "softmax_av_fused_enabled": bool(self.cluster_config.pim_softmax_av_fused_enabled),
                "softmax_av_shadow_check": bool(self.cluster_config.pim_softmax_av_shadow_check),
                "qk_mixed_enabled": bool(self.cluster_config.pim_qk_mixed_enabled),
                "qk_mixed_heads": int(self.cluster_config.pim_qk_mixed_heads),
                "qk_mixed_window": int(self.cluster_config.pim_qk_mixed_window),
                "decode_batch_window_s": float(self.cluster_config.attention_actor_batch_window_s),
                "decode_batch_max_size": int(self.cluster_config.attention_actor_batch_max_size),
            }
            if self.cluster_config.attention_backend == "cloverinfer":
                attention_backend_kwargs.update(
                    {
                        "cpu_shadow_enabled": bool(self.cluster_config.clover_cpu_shadow_enabled),
                        "shadow_checks_enabled": bool(self.cluster_config.clover_shadow_checks_enabled),
                        "op_profiling_enabled": bool(self.cluster_config.clover_op_profiling_enabled),
                    }
                )

        self.prefill_nodes = [
            PrefillNode.options(
                **_actor_options(self.cluster_config.prefill_resource, gpu_prefill)
            ).remote(i, self.model_config, self.cluster_config.use_gpu_for_prefill)
            for i in range(self.cluster_config.num_prefill_workers)
        ]

        self.attention_nodes = [
            AttentionNode.options(
                **_actor_options(self.cluster_config.attention_resource, 0)
            ).remote(
                i,
                self.model_config,
                self.cluster_config.attention_backend,
                attention_backend_kwargs,
            )
            for i in range(self.cluster_config.num_attention_nodes)
        ]

        self.decode_dense_nodes = [
            DecodeDenseNode.options(
                **_actor_options(self.cluster_config.decode_dense_resource, gpu_dense)
            ).remote(i, self.model_config, self.cluster_config.use_gpu_for_decode_dense)
            for i in range(self.cluster_config.num_decode_dense_nodes)
        ]

        infos = {
            "prefill": await self.prefill_nodes[0].get_info.remote(),
            "attention": await self.attention_nodes[0].get_info.remote(),
            "decode_dense": await self.decode_dense_nodes[0].get_info.remote(),
        }
        self.runtime_model_spec = await self.decode_dense_nodes[0].get_model_spec.remote()
        print(f"Cluster initialized: {infos}")
        return infos

    async def submit_request(self, prompt: str, return_metrics: bool = False, max_new_tokens: int | None = None):
        self._inflight_request_count += 1
        request_start = time.time()
        stage_timing = _empty_stage_timing()
        prefill = self.prefill_nodes[0]
        attention = self.attention_nodes[0]
        dense = self.decode_dense_nodes[0]

        rpc_started = time.perf_counter()
        prefill_out = await prefill.process_prompt.remote(prompt)
        stage_timing["scheduler"]["prefill_rpc_s"] += time.perf_counter() - rpc_started
        stage_timing["actors"]["prefill_compute_s"] += float(prefill_out.get("profile", {}).get("compute_s", 0.0))
        request_id = prefill_out["request_id"]
        prompt_len = int(prefill_out["prompt_len"])
        first_token = int(prefill_out["first_token_id"])
        first_token_time = time.time()

        generated_ids: List[int] = [first_token]
        current_token = first_token
        max_tokens = int(max_new_tokens or self.model_config.max_new_tokens)

        rpc_started = time.perf_counter()
        init_result = await attention.init_request.remote(
            request_id,
            prefill_out["initial_kv"],
            max_tokens,
        )
        self._active_decode_requests += 1
        stage_timing["scheduler"]["attention_init_rpc_s"] += time.perf_counter() - rpc_started
        stage_timing["actors"]["attention_init_compute_s"] += float(
            init_result.get("profile", {}).get("compute_s", 0.0)
        )

        try:
            for step in range(1, max_tokens):
                stage_timing["counts"]["decode_steps"] += 1
                position = prompt_len + step - 1
                rpc_started = time.perf_counter()
                decode_wave = await self._synchronize_decode_step(step, request_id)
                stage_timing["scheduler"].setdefault("decode_step_sync_wait_s", 0.0)
                stage_timing["scheduler"]["decode_step_sync_wait_s"] += time.perf_counter() - rpc_started
                rpc_started = time.perf_counter()
                start_token_result = await dense.start_token.remote(current_token, position)
                stage_timing["scheduler"]["start_token_rpc_s"] += time.perf_counter() - rpc_started
                stage_timing["actors"]["dense_start_token_compute_s"] += float(
                    start_token_result.get("profile", {}).get("compute_s", 0.0)
                )
                hidden = start_token_result["hidden"]

                for layer_idx in range(self.runtime_model_spec["num_layers"]):
                    stage_timing["counts"]["decode_layers"] += 1

                    rpc_started = time.perf_counter()
                    prepared = await dense.prepare_attention.remote(hidden, layer_idx, request_id, prompt_len + step)
                    stage_timing["scheduler"]["prepare_attention_rpc_s"] += time.perf_counter() - rpc_started
                    stage_timing["actors"]["dense_prepare_attention_compute_s"] += float(
                        prepared.get("profile", {}).get("compute_s", 0.0)
                    )

                    rpc_started = time.perf_counter()
                    attention_result = await self._batched_attention_decode(
                        attention,
                        prepared,
                        step,
                        decode_wave=decode_wave,
                    )
                    stage_timing["scheduler"]["attention_decode_rpc_s"] += time.perf_counter() - rpc_started
                    stage_timing["actors"]["attention_decode_compute_s"] += float(
                        attention_result.get("profile", {}).get("compute_s", 0.0)
                    )
                    context = attention_result["context"]

                    rpc_started = time.perf_counter()
                    finish_result = await dense.finish_layer.remote(prepared["residual"], context, layer_idx)
                    stage_timing["scheduler"]["finish_layer_rpc_s"] += time.perf_counter() - rpc_started
                    stage_timing["actors"]["dense_finish_layer_compute_s"] += float(
                        finish_result.get("profile", {}).get("compute_s", 0.0)
                    )
                    hidden = finish_result["hidden"]

                rpc_started = time.perf_counter()
                sample_result = await dense.sample_next_token.remote(hidden)
                stage_timing["scheduler"]["sample_next_token_rpc_s"] += time.perf_counter() - rpc_started
                stage_timing["actors"]["dense_sample_next_token_compute_s"] += float(
                    sample_result.get("profile", {}).get("compute_s", 0.0)
                )
                next_token = int(sample_result["token_id"])
                generated_ids.append(next_token)
                current_token = next_token

                if next_token == 2:
                    break
        finally:
            attention_debug_before_free = None
            if return_metrics:
                attention_debug_before_free = await attention.get_info.remote()
            self._inflight_request_count = max(0, self._inflight_request_count - 1)
            self._active_decode_requests = max(0, self._active_decode_requests - 1)
            await self._maybe_flush_decode_step_syncs()
            await self._maybe_flush_attention_layer_barriers()
            await self._maybe_flush_attention_wavefronts()
            rpc_started = time.perf_counter()
            await attention.free_request.remote(request_id)
            stage_timing["scheduler"]["free_request_rpc_s"] += time.perf_counter() - rpc_started

        rpc_started = time.perf_counter()
        decode_result = await dense.decode_tokens.remote(generated_ids)
        stage_timing["scheduler"]["decode_tokens_rpc_s"] += time.perf_counter() - rpc_started
        stage_timing["actors"]["dense_decode_tokens_compute_s"] += float(
            decode_result.get("profile", {}).get("compute_s", 0.0)
        )
        generated_text = decode_result["text"]

        if return_metrics:
            request_end = time.time()
            latency = request_end - request_start
            total_tokens = len(generated_ids)
            ttft = first_token_time - request_start
            tpot = (latency - ttft) / max(total_tokens - 1, 1)
            throughput = total_tokens / latency if latency > 0 else 0.0
            metrics = {
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "throughput": throughput,
                "total_tokens": total_tokens,
            }
            scheduler_rpc_total = sum(stage_timing["scheduler"].values())
            actor_compute_total = sum(stage_timing["actors"].values())
            metrics["stage_timing"] = stage_timing
            metrics["stage_timing"]["scheduler"]["total_rpc_s"] = scheduler_rpc_total
            metrics["stage_timing"]["actors"]["total_compute_s"] = actor_compute_total
            metrics["stage_timing"]["scheduler_overhead_s"] = max(
                0.0,
                float(latency - scheduler_rpc_total),
            )
            metrics["scheduler_attention_batching"] = {
                "window_s": float(self.attention_batch_window_s),
                "max_size": int(self.attention_batch_max_size),
                "flushes": int(self.attention_batch_flushes),
                "total_items": int(self.attention_batch_total_items),
                "max_observed_size": int(self.attention_batch_max_observed),
                "pending": sum(len(batch) for batch in self._attention_wavefront_batches.values()),
            }
            metrics["scheduler_decode_step_sync"] = {
                "window_s": float(self.decode_step_sync_window_s),
                "max_size": int(self.decode_step_sync_max_size),
                "flushes": int(self.decode_step_sync_flushes),
                "total_items": int(self.decode_step_sync_total_items),
                "max_observed_size": int(self.decode_step_sync_max_observed),
                "pending": sum(len(batch) for batch in self._decode_step_sync_batches.values()),
            }
            metrics["scheduler_attention_layer_barrier"] = {
                "window_s": float(self.attention_layer_barrier_window_s),
                "max_size": int(self.attention_layer_barrier_max_size),
                "flushes": int(self.attention_layer_barrier_flushes),
                "total_items": int(self.attention_layer_barrier_total_items),
                "max_observed_size": int(self.attention_layer_barrier_max_observed),
                "pending": sum(len(batch) for batch in self._attention_layer_barrier_batches.values()),
            }
            metrics["attention_backend_before_free"] = attention_debug_before_free
            metrics["attention_backend"] = await attention.get_info.remote()
            return generated_text, metrics

        return generated_text
