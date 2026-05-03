from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import ray
import torch

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
            "model_type": "unknown",
            "num_layers": int(model_config.num_layers),
            "hidden_size": int(model_config.hidden_size),
            "num_heads": int(model_config.num_heads),
            "num_kv_heads": int(model_config.num_heads),
            "head_dim": int(model_config.hidden_size) // max(int(model_config.num_heads), 1),
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
        self.attention_rpc_cross_key_batch_enabled = bool(
            getattr(cluster_config, "attention_rpc_cross_key_batch_enabled", False)
        )
        self.attention_wavefront_cohort_policy = str(
            getattr(cluster_config, "attention_wavefront_cohort_policy", "batch")
        ).strip().lower()
        if self.attention_wavefront_cohort_policy not in {"batch", "step"}:
            self.attention_wavefront_cohort_policy = "batch"
        self.attention_actor_side_batching_enabled = bool(
            getattr(cluster_config, "attention_actor_side_batching_enabled", False)
        )
        self.attention_batch_flushes = 0
        self.attention_batch_total_items = 0
        self.attention_batch_max_observed = 0
        self.attention_batch_multi_key_flushes = 0
        self.attention_batch_total_keys = 0
        self.attention_batch_max_keys_observed = 0
        self._attention_wavefront_batches: dict[tuple[int, int], list[tuple[dict, asyncio.Future, object]]] = {}
        self._attention_wavefront_tasks: dict[tuple[int, int], asyncio.Task] = {}
        self._attention_wavefront_expected_sizes: dict[tuple[int, int], int] = {}
        self._active_decode_requests = 0
        # Continuous decode engine state (optional).
        self._continuous_engine = None

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

    def _pop_attention_wavefront_bundle(
        self, seed_key: tuple[int, int] | tuple[int, int, str]
    ) -> list[tuple[tuple[int, int] | tuple[int, int, str], list[tuple[dict, asyncio.Future, object]]]]:
        seed_batch = self._attention_wavefront_batches.pop(seed_key, [])
        self._attention_wavefront_tasks.pop(seed_key, None)
        self._attention_wavefront_expected_sizes.pop(seed_key, None)
        if not seed_batch:
            return []

        bundle = [(seed_key, seed_batch)]
        if not self.attention_rpc_cross_key_batch_enabled:
            return bundle

        attention = seed_batch[0][2]
        merge_keys = []
        for key, batch in self._attention_wavefront_batches.items():
            if batch and batch[0][2] is attention:
                merge_keys.append(key)

        for key in merge_keys:
            batch = self._attention_wavefront_batches.pop(key, [])
            self._attention_wavefront_tasks.pop(key, None)
            self._attention_wavefront_expected_sizes.pop(key, None)
            if batch:
                bundle.append((key, batch))
        return bundle

    async def _execute_attention_wavefront_key(self, key: tuple[int, int]):
        bundle = self._pop_attention_wavefront_bundle(key)
        if not bundle:
            return
        payloads = []
        futures = []
        attention = bundle[0][1][0][2]
        for _, batch in bundle:
            payloads.extend(item[0] for item in batch)
            futures.extend(item[1] for item in batch)
        self.attention_batch_flushes += 1
        self.attention_batch_total_items += len(payloads)
        self.attention_batch_max_observed = max(self.attention_batch_max_observed, len(payloads))
        self.attention_batch_total_keys += len(bundle)
        self.attention_batch_max_keys_observed = max(
            self.attention_batch_max_keys_observed,
            len(bundle),
        )
        if len(bundle) > 1:
            self.attention_batch_multi_key_flushes += 1
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
        if self.attention_actor_side_batching_enabled:
            key: tuple[int, int] | tuple[int, int, str]
            if self.attention_decode_wave_persist_enabled and decode_wave is not None:
                cohort_id = str(decode_wave.get("cohort_id", "default"))
                key = (int(decode_step), int(prepared["layer_idx"]), cohort_id)
            else:
                key = (int(decode_step), int(prepared["layer_idx"]))
            await self._synchronize_attention_layer(key)
            return await attention.decode_layer.remote(prepared)

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        key: tuple[int, int] | tuple[int, int, str]
        cohort_size = None
        if self.attention_decode_wave_persist_enabled and decode_wave is not None:
            cohort_id = str(decode_wave.get("cohort_id", "default"))
            if "group_size" in decode_wave:
                cohort_size = max(1, int(decode_wave["group_size"]))
            if self.attention_wavefront_cohort_policy == "step":
                # Force different requests at the same decode step to share the same key,
                # maximizing batch formation even if submission cohort ids differ.
                key = (int(decode_step), int(prepared["layer_idx"]))
            else:
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
                "block_tokens": int(self.cluster_config.pim_block_tokens),
                "resident_store_backend": str(self.cluster_config.pim_resident_store_backend),
                "max_resident_groups_per_layer": int(self.cluster_config.pim_max_resident_groups_per_layer),
                "head_grouping_policy": str(self.cluster_config.pim_head_grouping_policy),
                "dpu_placement_policy": str(self.cluster_config.pim_dpu_placement_policy),
                "resident_kv_dtype": str(self.cluster_config.pim_resident_kv_dtype),
                "tail_capacity_buckets": list(self.cluster_config.pim_tail_capacity_buckets),
                "qk_full_enabled": bool(self.cluster_config.pim_qk_full_enabled),
                "qk_full_shadow_check": bool(self.cluster_config.pim_qk_full_shadow_check),
                "softmax_av_fused_enabled": bool(self.cluster_config.pim_softmax_av_fused_enabled),
                "softmax_av_shadow_check": bool(self.cluster_config.pim_softmax_av_shadow_check),
                "qk_mixed_enabled": bool(self.cluster_config.pim_qk_mixed_enabled),
                "qk_mixed_heads": int(self.cluster_config.pim_qk_mixed_heads),
                "qk_mixed_window": int(self.cluster_config.pim_qk_mixed_window),
                "decode_batch_window_s": float(self.cluster_config.attention_actor_batch_window_s),
                "decode_batch_max_size": int(self.cluster_config.attention_actor_batch_max_size),
                "best_round_seed_enabled": bool(self.cluster_config.pim_kvslot_best_round_seed_enabled),
                "shape_rounds_enabled": bool(self.cluster_config.pim_kvslot_shape_rounds_experimental_enabled),
                "context_fused_enabled": bool(self.cluster_config.pim_kvslot_context_fused_experimental_enabled),
            }
            if self.cluster_config.attention_backend == "cloverinfer":
                attention_backend_kwargs.update(
                    {
                        "cpu_shadow_enabled": bool(self.cluster_config.clover_cpu_shadow_enabled),
                        "shadow_checks_enabled": bool(self.cluster_config.clover_shadow_checks_enabled),
                        "op_profiling_enabled": bool(self.cluster_config.clover_op_profiling_enabled),
                        "shadow_check_token_interval": int(self.cluster_config.clover_shadow_check_token_interval),
                        "shadow_check_layer_interval": int(self.cluster_config.clover_shadow_check_layer_interval),
                        "host_qk_mixed_enabled": bool(self.cluster_config.clover_host_qk_mixed_enabled),
                        "pim_attention_enabled": bool(self.cluster_config.clover_pim_attention_enabled),
                        "pim_context_fused_experimental_enabled": bool(
                            self.cluster_config.clover_pim_context_fused_experimental_enabled
                        ),
                        "pim_rank_spread_alloc_experimental_enabled": bool(
                            self.cluster_config.clover_pim_rank_spread_alloc_experimental_enabled
                        ),
                        "fine_head_grouping_experimental_enabled": bool(
                            self.cluster_config.clover_fine_head_grouping_experimental_enabled
                        ),
                        "target_heads_per_group_experimental": int(
                            self.cluster_config.clover_target_heads_per_group_experimental
                        ),
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

    async def shutdown_cluster(self) -> bool:
        """Best-effort cleanup to release external resources (e.g. kvslot helper)."""
        # Stop continuous engine loop so it doesn't keep issuing decode work while we free state.
        engine = getattr(self, "_continuous_engine", None)
        if engine is not None:
            try:
                await engine.shutdown()
            except Exception:
                pass
            self._continuous_engine = None

        # Kill actors to ensure subprocesses / DPU handles are released.
        actors = []
        actors.extend(getattr(self, "prefill_nodes", []) or [])
        actors.extend(getattr(self, "attention_nodes", []) or [])
        actors.extend(getattr(self, "decode_dense_nodes", []) or [])
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self.prefill_nodes = []
        self.attention_nodes = []
        self.decode_dense_nodes = []
        return True

    async def submit_request(self, prompt: str, return_metrics: bool = False, max_new_tokens: int | None = None):
        if bool(getattr(self.cluster_config, "decode_continuous_batching_enabled", False)):
            return await self._submit_request_via_continuous_engine(
                prompt,
                return_metrics=return_metrics,
                max_new_tokens=max_new_tokens,
            )
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
                "cross_key_batch_enabled": bool(self.attention_rpc_cross_key_batch_enabled),
                "actor_side_batching_enabled": bool(self.attention_actor_side_batching_enabled),
                "flushes": int(self.attention_batch_flushes),
                "total_items": int(self.attention_batch_total_items),
                "max_observed_size": int(self.attention_batch_max_observed),
                "multi_key_flushes": int(self.attention_batch_multi_key_flushes),
                "total_keys": int(self.attention_batch_total_keys),
                "max_keys_observed": int(self.attention_batch_max_keys_observed),
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

    async def _submit_request_via_continuous_engine(
        self,
        prompt: str,
        *,
        return_metrics: bool = False,
        max_new_tokens: int | None = None,
    ):
        # Phase-1 engine: step-batched decode.
        if self._continuous_engine is None:
            self._continuous_engine = ContinuousDecodeEngine(self)
        # Maintain the same inflight accounting as the baseline path.
        self._inflight_request_count += 1
        try:
            return await self._continuous_engine.enqueue(
                prompt,
                max_new_tokens=max_new_tokens,
                return_metrics=return_metrics,
            )
        finally:
            self._inflight_request_count = max(0, self._inflight_request_count - 1)


class ContinuousDecodeEngine:
    """Step-batched decode engine.

    Each decode step forms a batch of active requests and iterates layers in lock-step:
    dense.prepare_attention_batch -> attention.decode_layer_batch -> dense.finish_layer_batch.
    Dense stays KV-free; KV cache lives in the attention backend.
    """

    def __init__(self, scheduler: GlobalScheduler):
        self.scheduler = scheduler
        self.batch_window_s = max(0.0, float(getattr(scheduler.cluster_config, "decode_continuous_batch_window_s", 0.001)))
        self.max_batch_size = max(1, int(getattr(scheduler.cluster_config, "decode_continuous_max_batch_size", 8)))
        self._pending: list[tuple[str, int, bool, float, asyncio.Future]] = []
        self._active: list[dict] = []
        self._active_rr_cursor: int = 0
        self._task: asyncio.Task | None = None
        self._debug_last_stats: dict[str, object] = {}
        # Aggregated counters for debugging/analysis (reset when engine starts).
        self._attention_batch_calls: int = 0
        self._attention_batch_items: int = 0
        self._shutdown_requested = False

    async def enqueue(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None,
        return_metrics: bool = False,
    ):
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        max_tokens = int(max_new_tokens or self.scheduler.model_config.max_new_tokens)
        request_start = time.time()
        self._pending.append((str(prompt), max_tokens, bool(return_metrics), float(request_start), future))
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return await future

    async def shutdown(self) -> None:
        """Stop the engine loop and best-effort finalize/cancel outstanding work."""
        self._shutdown_requested = True
        task = self._task
        if task is not None:
            task.cancel()
            try:
                await task
            except Exception:
                pass
        self._task = None
        # Best-effort: free active requests to release kvslot helper state.
        attention = self.scheduler.attention_nodes[0] if self.scheduler.attention_nodes else None
        if attention is not None:
            for item in list(self._active):
                try:
                    await attention.free_request.remote(item["request_id"])
                except Exception:
                    pass
        for prompt, max_tokens, need_metrics, request_start, future in list(self._pending):
            if not future.done():
                future.cancel()
        self._pending.clear()
        self._active.clear()

    async def _sleep_window(self):
        if self.batch_window_s > 0:
            await asyncio.sleep(self.batch_window_s)

    async def _run(self):
        try:
            # Phase-2: persistent engine loop.
            # Maintain a global active set and execute decode in step-batches, admitting new
            # requests continuously while previous ones are decoding.
            while (not self._shutdown_requested) and (self._pending or self._active):
                await self._sleep_window()
                await self._admit_pending()
                if not self._active:
                    continue
                await self._decode_one_step_batch()
        finally:
            self._task = None
            if self._pending:
                self._task = asyncio.create_task(self._run())

    async def _admit_pending(self) -> None:
        """Move pending requests into the active set (prefill + attention.init_request)."""
        prefill = self.scheduler.prefill_nodes[0]
        attention = self.scheduler.attention_nodes[0]

        if not self._pending:
            return

        stage_timing = _empty_stage_timing()
        rpc_started = time.perf_counter()
        batch = self._pending[: self.max_batch_size]
        del self._pending[: self.max_batch_size]

        async def _prefill_one(prompt: str):
            out = await prefill.process_prompt.remote(prompt)
            return out, time.time()

        # Prefill in parallel so the batch window doesn't serialize the GPU.
        prefill_tasks = [
            asyncio.create_task(_prefill_one(prompt))
            for prompt, _max_tokens, _need_metrics, _request_start, _future in batch
        ]
        prefill_results = await asyncio.gather(*prefill_tasks, return_exceptions=True)
        stage_timing["scheduler"]["prefill_rpc_s"] += time.perf_counter() - rpc_started

        init_tasks: list[asyncio.Task] = []
        init_task_items: list[dict] = []

        for (prompt, max_tokens, need_metrics, request_start, future), result in zip(batch, prefill_results):
            if future.done():
                continue
            if isinstance(result, Exception):
                future.set_exception(result)
                continue
            prefill_out, first_token_time = result
            try:
                request_id = str(prefill_out["request_id"])
                prompt_len = int(prefill_out["prompt_len"])
                first_token = int(prefill_out["first_token_id"])
                item = {
                    "prompt": prompt,
                    "future": future,
                    "need_metrics": bool(need_metrics),
                    "request_start": float(request_start),
                    "first_token_time": float(first_token_time),
                    "request_id": request_id,
                    "prompt_len": prompt_len,
                    "max_tokens": int(max_tokens),
                    "generated_ids": [first_token],
                    "current_token": first_token,
                    "done": bool(max_tokens <= 1),
                    "next_step": 1,  # decode step counter; first token already produced by prefill
                }
                # Ray remote calls return ObjectRef; asyncio needs a coroutine.
                async def _await_ref(ref):
                    return await ref

                init_ref = attention.init_request.remote(
                    request_id,
                    prefill_out["initial_kv"],
                    int(max_tokens),
                )
                init_tasks.append(asyncio.create_task(_await_ref(init_ref)))
                init_task_items.append(item)
            except Exception as exc:
                future.set_exception(exc)

        # Init KV state in parallel.
        if init_tasks:
            rpc_started = time.perf_counter()
            init_results = await asyncio.gather(*init_tasks, return_exceptions=True)
            stage_timing["scheduler"]["attention_init_rpc_s"] += time.perf_counter() - rpc_started
            for item, result in zip(init_task_items, init_results):
                future = item["future"]
                if future.done():
                    continue
                if isinstance(result, Exception):
                    future.set_exception(result)
                    item["done"] = True
                else:
                    self.scheduler._active_decode_requests += 1
                    # Requests with max_tokens <= 1 finalize immediately.
                    if item["done"]:
                        await self._finalize_item(item)
                    else:
                        self._active.append(item)
                # Attach stage timing so later finalize can report something meaningful.
                current_stage = item.get("stage_timing")
                if current_stage is None:
                    item["stage_timing"] = stage_timing
                else:
                    for key in stage_timing.get("scheduler", {}):
                        current_stage["scheduler"][key] = float(current_stage["scheduler"].get(key, 0.0)) + float(
                            stage_timing["scheduler"].get(key, 0.0)
                        )
                    for key in stage_timing.get("actors", {}):
                        current_stage["actors"][key] = float(current_stage["actors"].get(key, 0.0)) + float(
                            stage_timing["actors"].get(key, 0.0)
                        )

        # If we admitted some items, update debug stats to reflect current active set.
        num_layers = int(self.scheduler.runtime_model_spec.get("num_layers", 0))
        if self._active:
            max_tokens_seen = max(int(item.get("max_tokens", 0)) for item in self._active)
        else:
            max_tokens_seen = 0
        self._debug_last_stats.update(
            {
                "engine_batch_size": int(len(self._active)),
                "engine_num_layers": int(num_layers),
                "engine_max_tokens": int(max_tokens_seen),
            }
        )

    async def _finalize_item(self, item: dict) -> None:
        attention = self.scheduler.attention_nodes[0]
        dense = self.scheduler.decode_dense_nodes[0]
        future = item["future"]
        if future.done():
            return
        stage_timing = item.get("stage_timing")
        if stage_timing is None:
            stage_timing = _empty_stage_timing()
            item["stage_timing"] = stage_timing
        attention_debug_before_free = None
        if item.get("need_metrics", False):
            try:
                attention_debug_before_free = await attention.get_info.remote()
            except Exception:
                attention_debug_before_free = None
        try:
            rpc_started = time.perf_counter()
            await attention.free_request.remote(item["request_id"])
            stage_timing["scheduler"]["free_request_rpc_s"] += time.perf_counter() - rpc_started
        except Exception:
            pass
        self.scheduler._active_decode_requests = max(0, self.scheduler._active_decode_requests - 1)
        try:
            rpc_started = time.perf_counter()
            decode_result = await dense.decode_tokens.remote(item["generated_ids"])
            stage_timing["scheduler"]["decode_tokens_rpc_s"] += time.perf_counter() - rpc_started
            stage_timing["actors"]["dense_decode_tokens_compute_s"] += float(
                decode_result.get("profile", {}).get("compute_s", 0.0)
            )
            generated_text = str(decode_result.get("text", ""))
        except Exception as exc:
            future.set_exception(exc)
            return

        if item.get("need_metrics", False):
            request_end = time.time()
            latency = float(request_end - float(item["request_start"]))
            total_tokens = int(len(item["generated_ids"]))
            ttft = float(float(item["first_token_time"]) - float(item["request_start"]))
            tpot = (latency - ttft) / max(total_tokens - 1, 1)
            throughput = (float(total_tokens) / latency) if latency > 0 else 0.0
            engine_stats = dict(self._debug_last_stats or {})
            # Fill counts and totals for compatibility with baseline analysis.
            stage_timing["counts"]["decode_steps"] = int(item.get("next_step", 1))
            stage_timing["counts"]["decode_layers"] = int(item.get("decode_layers", 0))
            scheduler_rpc_total = sum(float(v) for v in stage_timing["scheduler"].values())
            actor_compute_total = sum(float(v) for v in stage_timing["actors"].values())
            stage_timing["scheduler"]["total_rpc_s"] = float(scheduler_rpc_total)
            stage_timing["actors"]["total_compute_s"] = float(actor_compute_total)
            stage_timing["scheduler_overhead_s"] = max(0.0, float(latency - scheduler_rpc_total))
            metrics = {
                "ttft": float(ttft),
                "tpot": float(tpot),
                "latency": float(latency),
                "throughput": float(throughput),
                "total_tokens": int(total_tokens),
                "stage_timing": stage_timing,
                "attention_backend_before_free": attention_debug_before_free or {},
                "attention_backend": attention_debug_before_free or {},
                "continuous_engine": engine_stats,
            }
            future.set_result((generated_text, metrics))
        else:
            future.set_result(generated_text)

    def _pick_step_batch(self) -> list[dict]:
        """Pick up to max_batch_size active items in a round-robin fashion."""
        if not self._active:
            return []
        start = int(self._active_rr_cursor) % len(self._active)
        picked: list[dict] = []
        idx = start
        visited = 0
        while visited < len(self._active) and len(picked) < self.max_batch_size:
            item = self._active[idx]
            if (not item.get("done", False)) and (not item["future"].done()):
                picked.append(item)
            idx = (idx + 1) % len(self._active)
            visited += 1
        self._active_rr_cursor = idx
        return picked

    async def _decode_one_step_batch(self) -> None:
        dense = self.scheduler.decode_dense_nodes[0]
        attention = self.scheduler.attention_nodes[0]
        num_layers = int(self.scheduler.runtime_model_spec.get("num_layers", 0))

        step_active = self._pick_step_batch()
        if not step_active:
            return

        # NOTE: For phase-2, we keep "step-batched" but allow different requests to be at different steps.
        # Each item carries its own next_step counter and context_len/position is derived from it.
        token_ids = [int(item["current_token"]) for item in step_active]
        positions = [int(item["prompt_len"]) + int(item["next_step"]) - 1 for item in step_active]
        rpc_started = time.perf_counter()
        hidden_result = await dense.start_token_batch.remote(token_ids, positions)
        rpc_elapsed = time.perf_counter() - rpc_started
        for item in step_active:
            stage_timing = item.get("stage_timing")
            if stage_timing is None:
                stage_timing = _empty_stage_timing()
                item["stage_timing"] = stage_timing
            stage_timing["scheduler"]["start_token_rpc_s"] += float(rpc_elapsed) / max(len(step_active), 1)
            stage_timing["actors"]["dense_start_token_compute_s"] += float(
                hidden_result.get("profile", {}).get("compute_s", 0.0)
            ) / max(len(step_active), 1)
        hidden = hidden_result["hidden"]

        for layer_idx in range(num_layers):
            context_lens = [int(item["prompt_len"]) + int(item["next_step"]) for item in step_active]
            request_ids = [str(item["request_id"]) for item in step_active]
            rpc_started = time.perf_counter()
            prepared = await dense.prepare_attention_batch.remote(hidden, int(layer_idx), request_ids, context_lens)
            rpc_elapsed = time.perf_counter() - rpc_started
            for item in step_active:
                stage_timing = item["stage_timing"]
                stage_timing["scheduler"]["prepare_attention_rpc_s"] += float(rpc_elapsed) / max(len(step_active), 1)
                stage_timing["actors"]["dense_prepare_attention_compute_s"] += float(
                    prepared.get("profile", {}).get("compute_s", 0.0)
                ) / max(len(step_active), 1)
            q = prepared["query"]
            k_new = prepared["key"]
            v_new = prepared["value"]
            residual = prepared["residual"]
            score_scale = float(prepared.get("score_scale", 1.0))

            payloads = []
            for i, req_id in enumerate(request_ids):
                payloads.append(
                    {
                        "request_id": req_id,
                        "layer_idx": int(layer_idx),
                        "query": q[i],
                        "key": k_new[i],
                        "value": v_new[i],
                        "score_scale": score_scale,
                    }
                )
            rpc_started = time.perf_counter()
            attention_out = await attention.decode_layer_batch.remote(payloads)
            rpc_elapsed = time.perf_counter() - rpc_started
            per_item_compute = 0.0
            if attention_out:
                per_item_compute = float(attention_out[0].get("profile", {}).get("compute_s", 0.0))
            for item in step_active:
                stage_timing = item["stage_timing"]
                stage_timing["scheduler"]["attention_decode_rpc_s"] += float(rpc_elapsed) / max(len(step_active), 1)
                stage_timing["actors"]["attention_decode_compute_s"] += float(per_item_compute)
            # Track attention batching effectiveness over time. The "last batch size"
            # can be misleading because the tail of a workload always collapses to 1.
            self._attention_batch_calls += 1
            self._attention_batch_items += int(len(payloads))
            self._debug_last_stats["engine_last_attention_batch_size"] = int(len(payloads))
            self._debug_last_stats["engine_attention_batch_calls"] = int(self._attention_batch_calls)
            self._debug_last_stats["engine_attention_batch_items"] = int(self._attention_batch_items)
            self._debug_last_stats["engine_attention_batch_avg_size"] = float(self._attention_batch_items) / float(
                max(self._attention_batch_calls, 1)
            )
            contexts = [item_out["context"] for item_out in attention_out]
            context_tensor = torch.cat([ctx for ctx in contexts], dim=0)
            finish = await dense.finish_layer_batch.remote(residual, context_tensor, int(layer_idx))
            for item in step_active:
                stage_timing = item["stage_timing"]
                stage_timing["scheduler"]["finish_layer_rpc_s"] += 0.0  # remote overhead folded into RPC time below
                stage_timing["actors"]["dense_finish_layer_compute_s"] += float(
                    finish.get("profile", {}).get("compute_s", 0.0)
                ) / max(len(step_active), 1)
            hidden = finish["hidden"]
            for item in step_active:
                item["decode_layers"] = int(item.get("decode_layers", 0)) + 1

        rpc_started = time.perf_counter()
        sample = await dense.sample_next_token_batch.remote(hidden)
        rpc_elapsed = time.perf_counter() - rpc_started
        for item in step_active:
            stage_timing = item["stage_timing"]
            stage_timing["scheduler"]["sample_next_token_rpc_s"] += float(rpc_elapsed) / max(len(step_active), 1)
            stage_timing["actors"]["dense_sample_next_token_compute_s"] += float(
                sample.get("profile", {}).get("compute_s", 0.0)
            ) / max(len(step_active), 1)
        next_tokens = [int(x) for x in sample["token_ids"]]
        for item, next_token in zip(step_active, next_tokens):
            item["generated_ids"].append(next_token)
            item["current_token"] = next_token
            item["next_step"] = int(item["next_step"]) + 1
            if next_token == 2:
                item["done"] = True
            elif int(item["next_step"]) >= int(item["max_tokens"]):
                item["done"] = True

        # Retire finished items.
        done_now = [item for item in step_active if item.get("done", False) and (not item["future"].done())]
        for item in done_now:
            await self._finalize_item(item)
        self._active = [item for item in self._active if (not item.get("done", False)) and (not item["future"].done())]


# Backwards-compatible alias for Ray deserialization / older pickles.
_ContinuousDecodeEngine = ContinuousDecodeEngine
