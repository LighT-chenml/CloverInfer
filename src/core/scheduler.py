from __future__ import annotations

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

    async def initialize_cluster(self):
        gpu_prefill = 1 if self.cluster_config.use_gpu_for_prefill else 0
        gpu_dense = 1 if self.cluster_config.use_gpu_for_decode_dense else 0
        attention_backend_kwargs = {}
        if self.cluster_config.attention_backend == "pim_naive":
            attention_backend_kwargs = {
                "num_dpus": int(self.cluster_config.pim_num_dpus),
                "length": int(self.cluster_config.pim_length),
                "qk_mixed_enabled": bool(self.cluster_config.pim_qk_mixed_enabled),
                "qk_mixed_heads": int(self.cluster_config.pim_qk_mixed_heads),
                "qk_mixed_window": int(self.cluster_config.pim_qk_mixed_window),
            }

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

        rpc_started = time.perf_counter()
        init_result = await attention.init_request.remote(request_id, prefill_out["initial_kv"])
        stage_timing["scheduler"]["attention_init_rpc_s"] += time.perf_counter() - rpc_started
        stage_timing["actors"]["attention_init_compute_s"] += float(
            init_result.get("profile", {}).get("compute_s", 0.0)
        )

        generated_ids: List[int] = [first_token]
        current_token = first_token
        max_tokens = int(max_new_tokens or self.model_config.max_new_tokens)

        try:
            for step in range(1, max_tokens):
                stage_timing["counts"]["decode_steps"] += 1
                position = prompt_len + step - 1
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
                    prepared = await dense.prepare_attention.remote(hidden, layer_idx, request_id)
                    stage_timing["scheduler"]["prepare_attention_rpc_s"] += time.perf_counter() - rpc_started
                    stage_timing["actors"]["dense_prepare_attention_compute_s"] += float(
                        prepared.get("profile", {}).get("compute_s", 0.0)
                    )

                    rpc_started = time.perf_counter()
                    attention_result = await attention.decode_layer.remote(prepared)
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
            metrics["attention_backend"] = await attention.get_info.remote()
            return generated_text, metrics

        return generated_text
