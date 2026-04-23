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
        prefill = self.prefill_nodes[0]
        attention = self.attention_nodes[0]
        dense = self.decode_dense_nodes[0]

        prefill_out = await prefill.process_prompt.remote(prompt)
        request_id = prefill_out["request_id"]
        prompt_len = int(prefill_out["prompt_len"])
        first_token = int(prefill_out["first_token_id"])
        first_token_time = time.time()

        await attention.init_request.remote(request_id, prefill_out["initial_kv"])

        generated_ids: List[int] = [first_token]
        current_token = first_token
        max_tokens = int(max_new_tokens or self.model_config.max_new_tokens)

        try:
            for step in range(1, max_tokens):
                position = prompt_len + step - 1
                hidden = await dense.start_token.remote(current_token, position)

                for layer_idx in range(self.runtime_model_spec["num_layers"]):
                    prepared = await dense.prepare_attention.remote(hidden, layer_idx, request_id)
                    context = await attention.decode_layer.remote(prepared)
                    hidden = await dense.finish_layer.remote(prepared["residual"], context, layer_idx)

                next_token = await dense.sample_next_token.remote(hidden)
                generated_ids.append(next_token)
                current_token = next_token

                if next_token == 2:
                    break
        finally:
            await attention.free_request.remote(request_id)

        generated_text = await dense.decode_tokens.remote(generated_ids)

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
            metrics["attention_backend"] = await attention.get_info.remote()
            return generated_text, metrics

        return generated_text
