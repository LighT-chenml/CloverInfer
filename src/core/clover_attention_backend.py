from __future__ import annotations

import time
from typing import Dict, List

import torch

from .attention_backend import PimNaiveAttentionBackend


class CloverInferAttentionBackend(PimNaiveAttentionBackend):
    """Independent CloverInfer backend.

    This backend intentionally keeps `PimNaiveAttentionBackend` unchanged so it
    can continue serving as the baseline. CloverInfer starts from the same
    resident-KV primitives, but adds:

    - CloverInfer-specific profiling on hot-path sub-operations
    - optional CPU shadow KV maintenance
    - optional shadow validation separated from the default fast path
    """

    def __init__(
        self,
        *args,
        cpu_shadow_enabled: bool = True,
        shadow_checks_enabled: bool = True,
        op_profiling_enabled: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.cpu_shadow_enabled = bool(cpu_shadow_enabled)
        self.shadow_checks_enabled = bool(shadow_checks_enabled)
        self.op_profiling_enabled = bool(op_profiling_enabled)
        self.backend_variant = "cloverinfer"
        self.op_timing_totals: Dict[str, float] = {
            "prepare_decode_record_s": 0.0,
            "normalize_decode_tensors_s": 0.0,
            "resident_append_s": 0.0,
            "cpu_shadow_append_s": 0.0,
            "resident_materialize_s": 0.0,
            "resident_shadow_check_s": 0.0,
            "host_score_compute_s": 0.0,
            "qk_full_batch_s": 0.0,
            "qk_mixed_batch_s": 0.0,
            "finalize_decode_records_s": 0.0,
            "softmax_av_s": 0.0,
            "resident_av_s": 0.0,
            "context_reassemble_s": 0.0,
        }
        self.op_timing_counts: Dict[str, int] = {key: 0 for key in self.op_timing_totals}

    def _timed(self, name: str):
        class _Timer:
            def __init__(self, backend: CloverInferAttentionBackend, key: str):
                self.backend = backend
                self.key = key
                self.started_at = 0.0

            def __enter__(self):
                self.started_at = time.perf_counter()
                return self

            def __exit__(self, exc_type, exc, tb):
                if not self.backend.op_profiling_enabled:
                    return False
                elapsed = time.perf_counter() - self.started_at
                self.backend.op_timing_totals[self.key] += float(elapsed)
                self.backend.op_timing_counts[self.key] += 1
                return False

        return _Timer(self, name)

    def _normalize_decode_tensors(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._timed("normalize_decode_tensors_s"):
            return super()._normalize_decode_tensors(query, key, value)

    def _compute_host_scores(self, record: Dict[str, object]) -> torch.Tensor:
        with self._timed("host_score_compute_s"):
            return super()._compute_host_scores(record)

    def _prepare_decode_record(self, item: Dict[str, object]) -> Dict[str, object]:
        with self._timed("prepare_decode_record_s"):
            request_id = str(item["request_id"])
            layer_idx = int(item["layer_idx"])
            score_scale = float(item.get("score_scale", 1.0))
            if request_id not in self.request_states:
                raise KeyError(f"Missing resident metadata for request {request_id}")
            if self.cpu_shadow_enabled and request_id not in self.cpu_backend.k_cache:
                raise KeyError(f"Unknown request {request_id}")

            q, k_new, v_new = self._normalize_decode_tensors(
                item["query"],
                item["key"],
                item["value"],
            )
            request_state = self.request_states[request_id]
            if layer_idx >= request_state.num_layers:
                raise IndexError(
                    f"layer_idx {layer_idx} out of range for request {request_id} "
                    f"with {request_state.num_layers} layers"
                )

            with self._timed("resident_append_s"):
                self._append_resident_kv(request_state, layer_idx, k_new, v_new)

            if self.cpu_shadow_enabled:
                with self._timed("cpu_shadow_append_s"):
                    self.cpu_backend.k_cache[request_id][layer_idx] = torch.cat(
                        [self.cpu_backend.k_cache[request_id][layer_idx], k_new.unsqueeze(0)], dim=0
                    )
                    self.cpu_backend.v_cache[request_id][layer_idx] = torch.cat(
                        [self.cpu_backend.v_cache[request_id][layer_idx], v_new.unsqueeze(0)], dim=0
                    )

            use_resident_av = self.resident_compute_enabled and self.resident_av_enabled
            need_materialized_kv = self.resident_compute_enabled and (
                (not use_resident_av) or self.shadow_checks_enabled
            )
            if need_materialized_kv:
                with self._timed("resident_materialize_s"):
                    keys, values = self._materialize_layer_kv(request_state, layer_idx)
                if self.shadow_checks_enabled and self.cpu_shadow_enabled:
                    with self._timed("resident_shadow_check_s"):
                        self._update_resident_shadow_diff(request_id, layer_idx, keys, values)
            elif self.cpu_shadow_enabled:
                keys = self.cpu_backend.k_cache[request_id][layer_idx]
                values = self.cpu_backend.v_cache[request_id][layer_idx]
            else:
                keys = None
                values = None

            q_fp32 = q if q.dtype == torch.float32 and q.is_contiguous() else q.to(torch.float32).contiguous()
            scores = None
            if not (self.resident_compute_enabled and self.qk_full_enabled):
                if keys is None:
                    with self._timed("resident_materialize_s"):
                        keys, values = self._materialize_layer_kv(request_state, layer_idx)
                scores = torch.einsum("hd,lhd->hl", q_fp32, keys.float()) * score_scale
            return {
                "request_id": request_id,
                "request_state": request_state,
                "layer_idx": layer_idx,
                "query_dtype": q.dtype,
                "q_fp32": q_fp32,
                "keys": keys,
                "values": values,
                "scores": scores,
                "score_scale": score_scale,
                "use_resident_av": use_resident_av,
            }

    def _apply_qk_full_batch(self, records: List[Dict[str, object]]) -> None:
        with self._timed("qk_full_batch_s"):
            return super()._apply_qk_full_batch(records)

    def _apply_qk_mixed_batch(self, records: List[Dict[str, object]]) -> None:
        with self._timed("qk_mixed_batch_s"):
            return super()._apply_qk_mixed_batch(records)

    def _finalize_decode_records(self, records: List[Dict[str, object]]) -> List[torch.Tensor]:
        with self._timed("finalize_decode_records_s"):
            outputs: List[torch.Tensor] = []
            flat_slot_weights: list[tuple[str, str, torch.Tensor]] = []
            slot_weight_refs: list[tuple[int, int]] = []
            flat_slot_scores: list[tuple[str, str, torch.Tensor]] = []
            slot_score_refs: list[tuple[int, int]] = []

            for record_idx, record in enumerate(records):
                use_fused_softmax_av = bool(record["use_resident_av"] and self.softmax_av_fused_enabled)
                if use_fused_softmax_av:
                    layer_state = record["request_state"].layer_states[record["layer_idx"]]
                    slot_scores = [
                        (
                            group.k_slot,
                            group.v_slot,
                            record["scores"][group.head_start:group.head_end, :].contiguous(),
                        )
                        for group in layer_state.head_groups
                    ]
                    flat_slot_scores.extend(slot_scores)
                    slot_score_refs.append((record_idx, len(slot_scores)))
                    if self.shadow_checks_enabled and record["values"] is not None:
                        with self._timed("softmax_av_s"):
                            record["weights"] = torch.softmax(record["scores"], dim=-1)
                else:
                    with self._timed("softmax_av_s"):
                        weights = torch.softmax(record["scores"], dim=-1)
                    record["weights"] = weights
                    if record["use_resident_av"]:
                        layer_state = record["request_state"].layer_states[record["layer_idx"]]
                        slot_weights = [
                            (
                                group.k_slot,
                                group.v_slot,
                                weights[group.head_start:group.head_end, :].contiguous(),
                            )
                            for group in layer_state.head_groups
                        ]
                        flat_slot_weights.extend(slot_weights)
                        slot_weight_refs.append((record_idx, len(slot_weights)))
                    else:
                        if record["values"] is None:
                            raise RuntimeError("CloverInfer host AV fallback requires materialized values")
                        with self._timed("softmax_av_s"):
                            record["context"] = torch.einsum(
                                "hl,lhd->hd", weights, record["values"].float()
                            ).to(record["query_dtype"])

            if flat_slot_scores:
                with self._timed("resident_av_s"):
                    group_contexts = self.resident_store.softmax_weighted_value_sum_batch(flat_slot_scores)
                self.softmax_av_fused_batch_calls += 1
                offset = 0
                for record_idx, group_count in slot_score_refs:
                    record = records[record_idx]
                    with self._timed("context_reassemble_s"):
                        context = torch.cat(group_contexts[offset : offset + group_count], dim=0).to(record["query_dtype"])
                    offset += group_count
                    if self.shadow_checks_enabled and record.get("weights") is not None and record["values"] is not None:
                        cpu_context = torch.einsum(
                            "hl,lhd->hd", record["weights"], record["values"].float()
                        ).to(record["query_dtype"])
                        av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                        self.softmax_av_fused_shadow_max_abs_diff = max(
                            self.softmax_av_fused_shadow_max_abs_diff, av_diff
                        )
                    self.softmax_av_fused_ops += 1
                    record["context"] = context

            if flat_slot_weights:
                with self._timed("resident_av_s"):
                    group_contexts = self.resident_store.weighted_value_sum_batch(flat_slot_weights)
                self.resident_av_batch_calls += 1
                offset = 0
                for record_idx, group_count in slot_weight_refs:
                    record = records[record_idx]
                    with self._timed("context_reassemble_s"):
                        context = torch.cat(group_contexts[offset : offset + group_count], dim=0).to(record["query_dtype"])
                    offset += group_count
                    if self.shadow_checks_enabled and record["values"] is not None:
                        cpu_context = torch.einsum(
                            "hl,lhd->hd", record["weights"], record["values"].float()
                        ).to(record["query_dtype"])
                        av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                        self.resident_av_shadow_max_abs_diff = max(self.resident_av_shadow_max_abs_diff, av_diff)
                    self.resident_av_ops += 1
                    record["context"] = context

            for record in records:
                if self.cpu_shadow_enabled:
                    if record["layer_idx"] == len(self.cpu_backend.k_cache[record["request_id"]]) - 1:
                        self.cpu_backend.context_lens[record["request_id"]] += 1
                elif record["layer_idx"] == record["request_state"].num_layers - 1:
                    record["request_state"].context_len = int(record["request_state"].context_len)
                outputs.append(record["context"].unsqueeze(0))
            return outputs

    def init_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int = 0,
    ) -> int:
        if self.cpu_shadow_enabled:
            seq_len = self.cpu_backend.init_request(request_id, initial_kv)
        else:
            seq_len = int(initial_kv[0]["key"].shape[0])
            self.cpu_backend.context_lens[request_id] = seq_len
        self.request_states[request_id] = self._build_request_state(
            request_id,
            initial_kv,
            decode_reserve_tokens,
        )
        return seq_len

    def get_context_len(self, request_id: str) -> int:
        if self.cpu_shadow_enabled:
            return self.cpu_backend.get_context_len(request_id)
        request_state = self.request_states.get(request_id)
        if request_state is None:
            raise KeyError(f"Unknown request {request_id}")
        return int(request_state.context_len)

    def free_request(self, request_id: str) -> None:
        if self.cpu_shadow_enabled:
            self.cpu_backend.free_request(request_id)
        else:
            self.cpu_backend.context_lens.pop(request_id, None)
        request_state = self.request_states.pop(request_id, None)
        if request_state is not None:
            for layer_state in request_state.layer_states:
                for group in layer_state.head_groups:
                    self.resident_store.free_group(group.k_slot, group.v_slot)
            self.last_freed_request_id = request_id

    def get_debug_info(self) -> Dict[str, object]:
        debug = super().get_debug_info()
        debug["backend_variant"] = self.backend_variant
        debug["clover_cpu_shadow_enabled"] = self.cpu_shadow_enabled
        debug["clover_shadow_checks_enabled"] = self.shadow_checks_enabled
        debug["clover_op_profiling_enabled"] = self.op_profiling_enabled
        debug["clover_op_timing_totals_s"] = dict(self.op_timing_totals)
        debug["clover_op_timing_counts"] = dict(self.op_timing_counts)
        return debug
