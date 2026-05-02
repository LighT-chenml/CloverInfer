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
        shadow_check_token_interval: int = 1,
        shadow_check_layer_interval: int = 1,
        host_qk_mixed_enabled: bool = False,
        pim_attention_enabled: bool = False,
        pim_context_fused_experimental_enabled: bool = False,
        pim_rank_spread_alloc_experimental_enabled: bool = False,
        fine_head_grouping_experimental_enabled: bool = False,
        target_heads_per_group_experimental: int = 0,
        **kwargs,
    ):
        # These Clover-only knobs are plumbed from the scheduler but are not
        # consumed by the current backend yet. Pop them so the base class
        # constructor doesn't error on unexpected kwargs.
        _ = bool(pim_rank_spread_alloc_experimental_enabled)
        _ = bool(fine_head_grouping_experimental_enabled)
        _ = int(target_heads_per_group_experimental)
        super().__init__(*args, **kwargs)
        self.cpu_shadow_enabled = bool(cpu_shadow_enabled)
        self.shadow_checks_enabled = bool(shadow_checks_enabled)
        self.op_profiling_enabled = bool(op_profiling_enabled)
        self.shadow_check_token_interval = max(1, int(shadow_check_token_interval))
        self.shadow_check_layer_interval = max(1, int(shadow_check_layer_interval))
        self.host_qk_mixed_enabled = bool(host_qk_mixed_enabled)
        self.pim_attention_enabled = bool(pim_attention_enabled)
        self.pim_context_fused_experimental_enabled = bool(pim_context_fused_experimental_enabled)
        self.backend_variant = "cloverinfer"
        self.shadow_k_buffers: Dict[str, List[torch.Tensor]] = {}
        self.shadow_v_buffers: Dict[str, List[torch.Tensor]] = {}
        self.shadow_layer_lens: Dict[str, List[int]] = {}
        self.op_timing_totals: Dict[str, float] = {
            "prepare_decode_record_s": 0.0,
            "normalize_decode_tensors_s": 0.0,
            "resident_append_s": 0.0,
            "cpu_shadow_append_s": 0.0,
            "cpu_shadow_grow_s": 0.0,
            "resident_materialize_s": 0.0,
            "resident_shadow_check_s": 0.0,
            "host_score_compute_s": 0.0,
            "qk_full_batch_s": 0.0,
            "qk_mixed_batch_s": 0.0,
            "qk_mixed_build_queries_s": 0.0,
            "resident_qk_batch_s": 0.0,
            "qk_mixed_apply_scores_s": 0.0,
            "finalize_decode_records_s": 0.0,
            "softmax_av_s": 0.0,
            "host_context_compute_s": 0.0,
            "resident_av_s": 0.0,
            "context_reassemble_s": 0.0,
        }
        self.op_timing_counts: Dict[str, int] = {key: 0 for key in self.op_timing_totals}
        self.shadow_check_invocations = 0
        self.shadow_check_skips = 0
        if self.pim_attention_enabled:
            self.qk_full_enabled = True
            self.softmax_av_fused_enabled = True
            if not self.resident_av_enabled:
                raise ValueError(
                    "CloverInfer PIM attention requires a resident store with PIM AV support; "
                    "use pim_resident_store_backend='upmem_kvslot'"
                )

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
            return torch.einsum(
                "hd,lhd->hl",
                record["q_fp32"],
                record["keys"].float(),
            ) * float(record["score_scale"])

    def _init_cpu_shadow_request(
        self,
        request_id: str,
        initial_kv: List[Dict[str, torch.Tensor]],
        decode_reserve_tokens: int,
    ) -> int:
        if request_id in self.shadow_k_buffers:
            raise ValueError(f"Request {request_id} already exists")
        if not initial_kv:
            raise ValueError("initial_kv must contain at least one layer")

        shadow_k_layers: List[torch.Tensor] = []
        shadow_v_layers: List[torch.Tensor] = []
        shadow_lens: List[int] = []

        for layer in initial_kv:
            layer_k = layer["key"].detach().cpu().contiguous()
            layer_v = layer["value"].detach().cpu().contiguous()
            seq_len, num_heads, head_dim = (int(dim) for dim in layer_k.shape)
            capacity = max(seq_len + max(0, int(decode_reserve_tokens)), seq_len, 1)
            k_buf = torch.empty((capacity, num_heads, head_dim), dtype=layer_k.dtype)
            v_buf = torch.empty((capacity, num_heads, head_dim), dtype=layer_v.dtype)
            k_buf[:seq_len].copy_(layer_k)
            v_buf[:seq_len].copy_(layer_v)
            shadow_k_layers.append(k_buf)
            shadow_v_layers.append(v_buf)
            shadow_lens.append(seq_len)

        self.shadow_k_buffers[request_id] = shadow_k_layers
        self.shadow_v_buffers[request_id] = shadow_v_layers
        self.shadow_layer_lens[request_id] = shadow_lens
        self.cpu_backend.context_lens[request_id] = int(shadow_lens[0])
        return int(shadow_lens[0])

    def _cpu_shadow_active_kv(
        self,
        request_id: str,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = int(self.shadow_layer_lens[request_id][layer_idx])
        return (
            self.shadow_k_buffers[request_id][layer_idx][:seq_len],
            self.shadow_v_buffers[request_id][layer_idx][:seq_len],
        )

    def _append_cpu_shadow_kv(
        self,
        request_id: str,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> None:
        k_buf = self.shadow_k_buffers[request_id][layer_idx]
        v_buf = self.shadow_v_buffers[request_id][layer_idx]
        current_len = int(self.shadow_layer_lens[request_id][layer_idx])
        append_len = int(k_new.shape[0])
        target_len = current_len + append_len

        if target_len > int(k_buf.shape[0]):
            with self._timed("cpu_shadow_grow_s"):
                new_capacity = max(int(k_buf.shape[0]) * 2, target_len, 1)
                new_k_buf = torch.empty(
                    (new_capacity, int(k_buf.shape[1]), int(k_buf.shape[2])),
                    dtype=k_buf.dtype,
                )
                new_v_buf = torch.empty(
                    (new_capacity, int(v_buf.shape[1]), int(v_buf.shape[2])),
                    dtype=v_buf.dtype,
                )
                new_k_buf[:current_len].copy_(k_buf[:current_len])
                new_v_buf[:current_len].copy_(v_buf[:current_len])
                self.shadow_k_buffers[request_id][layer_idx] = new_k_buf
                self.shadow_v_buffers[request_id][layer_idx] = new_v_buf
                k_buf = new_k_buf
                v_buf = new_v_buf

        k_buf[current_len:target_len].copy_(k_new)
        v_buf[current_len:target_len].copy_(v_new)
        self.shadow_layer_lens[request_id][layer_idx] = target_len

    def _update_resident_shadow_diff(
        self,
        request_id: str,
        layer_idx: int,
        resident_keys: torch.Tensor,
        resident_values: torch.Tensor,
    ) -> None:
        cpu_keys, cpu_values = self._cpu_shadow_active_kv(request_id, layer_idx)
        key_diff = float(torch.max(torch.abs(resident_keys.float() - cpu_keys.float())).item())
        value_diff = float(torch.max(torch.abs(resident_values.float() - cpu_values.float())).item())
        self.resident_shadow_max_abs_diff = max(self.resident_shadow_max_abs_diff, key_diff, value_diff)

    def _prepare_decode_record(self, item: Dict[str, object]) -> Dict[str, object]:
        with self._timed("prepare_decode_record_s"):
            request_id = str(item["request_id"])
            layer_idx = int(item["layer_idx"])
            score_scale = float(item.get("score_scale", 1.0))
            if request_id not in self.request_states:
                raise KeyError(f"Missing resident metadata for request {request_id}")
            if self.cpu_shadow_enabled and request_id not in self.shadow_k_buffers:
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
                    self._append_cpu_shadow_kv(
                        request_id,
                        layer_idx,
                        k_new.unsqueeze(0),
                        v_new.unsqueeze(0),
                    )

            use_resident_av = self.resident_compute_enabled and self.resident_av_enabled
            cpu_keys = None
            cpu_values = None
            if self.cpu_shadow_enabled:
                cpu_keys, cpu_values = self._cpu_shadow_active_kv(request_id, layer_idx)

            should_shadow_check = False
            if self.shadow_checks_enabled and self.cpu_shadow_enabled:
                current_token_idx = max(0, int(request_state.context_len) - 1)
                token_match = (current_token_idx % self.shadow_check_token_interval) == 0
                layer_match = (layer_idx % self.shadow_check_layer_interval) == 0
                should_shadow_check = bool(token_match and layer_match)

            resident_keys = None
            resident_values = None
            if self.resident_compute_enabled and should_shadow_check:
                with self._timed("resident_materialize_s"):
                    resident_keys, resident_values = self._materialize_layer_kv(request_state, layer_idx)
                with self._timed("resident_shadow_check_s"):
                    self._update_resident_shadow_diff(
                        request_id,
                        layer_idx,
                        resident_keys,
                        resident_values,
                    )
                self.shadow_check_invocations += 1
            elif self.shadow_checks_enabled and self.cpu_shadow_enabled:
                self.shadow_check_skips += 1

            use_pim_attention = bool(self.pim_attention_enabled and self.resident_compute_enabled)

            # CloverInfer keeps CPU shadow KV as the reference cache for
            # sampled validation. In the PIM-first path, resident PIM performs
            # the main attention computation, but the host-side shadow checks
            # still compare against the CPU shadow tensors.
            keys = cpu_keys
            values = cpu_values

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
                "should_shadow_check": should_shadow_check,
                "use_pim_attention": use_pim_attention,
            }

    def _apply_qk_context_fused_batch(self, records: List[Dict[str, object]]) -> None:
        with self._timed("qk_full_batch_s"):
            flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor, float]] = []
            slot_query_refs: list[tuple[Dict[str, object], int, int]] = []

            for record in records:
                layer_state = record["request_state"].layer_states[record["layer_idx"]]
                record["fused_group_contexts"] = []
                for group in layer_state.head_groups:
                    local_head_indices = list(range(group.group_heads))
                    flat_slot_queries.append(
                        (
                            group.k_slot,
                            group.v_slot,
                            local_head_indices,
                            int(group.seq_len),
                            record["q_fp32"][group.head_start:group.head_end].contiguous(),
                            float(record["score_scale"]),
                        )
                    )
                    slot_query_refs.append((record, int(group.head_start), int(group.head_end)))
                    self.qk_full_count += int(group.group_heads)

            if flat_slot_queries:
                with self._timed("resident_av_s"):
                    group_contexts = self.resident_store.qk_softmax_weighted_value_sum_batch(flat_slot_queries)
                self.qk_full_batch_calls += 1
                self.softmax_av_fused_batch_calls += 1
                for (record, head_start, head_end), context in zip(slot_query_refs, group_contexts):
                    record["fused_group_contexts"].append((head_start, head_end, context))

            self.qk_full_shadow_last_max_abs_diff = 0.0
            need_qk_shadow = bool(self.qk_full_shadow_check)
            need_context_shadow = bool(self.softmax_av_shadow_check)

            shadow_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]] = []
            shadow_slot_refs: list[tuple[Dict[str, object], int, int]] = []
            if need_qk_shadow:
                for record in records:
                    if not record.get("should_shadow_check", False):
                        continue
                    layer_state = record["request_state"].layer_states[record["layer_idx"]]
                    for group in layer_state.head_groups:
                        shadow_slot_queries.append(
                            (
                                group.k_slot,
                                group.v_slot,
                                list(range(group.group_heads)),
                                int(group.seq_len),
                                record["q_fp32"][group.head_start:group.head_end].contiguous(),
                            )
                        )
                        shadow_slot_refs.append((record, int(group.head_start), int(group.head_end)))
            if shadow_slot_queries:
                with self._timed("resident_qk_batch_s"):
                    slot_score_mats = self.resident_store.qk_slot_scores_batch(shadow_slot_queries)
                self.qk_batch_calls += 1
                for (record, head_start, head_end), score_mat in zip(shadow_slot_refs, slot_score_mats):
                    record.setdefault("shadow_group_scores", []).append((head_start, head_end, score_mat))

            for record in records:
                group_contexts = sorted(record.pop("fused_group_contexts", []), key=lambda item: item[0])
                if not group_contexts:
                    record["context"] = torch.empty_like(record["q_fp32"])
                    continue
                with self._timed("context_reassemble_s"):
                    record["context"] = torch.cat([ctx for _, _, ctx in group_contexts], dim=0).to(
                        record["query_dtype"]
                    )
                self.softmax_av_fused_ops += 1

                should_shadow_check = bool(record.get("should_shadow_check", False))
                host_scores = None
                if should_shadow_check and (need_qk_shadow or need_context_shadow):
                    host_scores = self._compute_host_scores(record)

                if should_shadow_check and need_qk_shadow:
                    group_scores = sorted(record.pop("shadow_group_scores", []), key=lambda item: item[0])
                    if group_scores:
                        scores = torch.cat([score_mat for _, _, score_mat in group_scores], dim=0).to(torch.float32)
                        scores = scores * float(record["score_scale"])
                        diff = float(torch.max(torch.abs(scores - host_scores)).item()) if scores.numel() > 0 else 0.0
                        self.qk_full_shadow_checks += 1
                        self.qk_full_shadow_last_max_abs_diff = max(self.qk_full_shadow_last_max_abs_diff, diff)
                        self.qk_full_shadow_max_abs_diff = max(self.qk_full_shadow_max_abs_diff, diff)

                if should_shadow_check and need_context_shadow and record["values"] is not None:
                    with self._timed("softmax_av_s"):
                        weights = torch.softmax(host_scores, dim=-1)
                    cpu_context = torch.einsum("hl,lhd->hd", weights, record["values"].float()).to(
                        record["query_dtype"]
                    )
                    av_diff = float(torch.max(torch.abs(record["context"].float() - cpu_context.float())).item())
                    self.softmax_av_fused_shadow_max_abs_diff = max(self.softmax_av_fused_shadow_max_abs_diff, av_diff)

    def decode_layer_batch(self, items: List[Dict[str, object]]) -> List[torch.Tensor]:
        if not items:
            return []
        self.decode_batch_calls += 1
        self.decode_batch_items += len(items)
        records = [self._prepare_decode_record(item) for item in items]
        use_qk_context_fused = (
            self.pim_attention_enabled
            and self.pim_context_fused_experimental_enabled
            and self.qk_full_enabled
            and self.resident_compute_enabled
            and self.resident_av_enabled
            and self.softmax_av_fused_enabled
        )
        if use_qk_context_fused:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            self._apply_qk_context_fused_batch(records)
            return self._finalize_ready_context_records(records)
        if self.qk_full_enabled and self.resident_compute_enabled:
            self.qk_mixed_last_head_diffs = []
            self.qk_mixed_last_max_abs_diff = 0.0
            self.qk_mixed_last_diag = {}
            self.qk_mixed_last_diag_path = ""
            self._apply_qk_full_batch(records)
        else:
            self.qk_full_shadow_last_max_abs_diff = 0.0
            self._apply_qk_mixed_batch(records)
        return self._finalize_decode_records(records)

    def _finalize_ready_context_records(self, records: List[Dict[str, object]]) -> List[torch.Tensor]:
        with self._timed("finalize_decode_records_s"):
            outputs: List[torch.Tensor] = []
            for record in records:
                if self.cpu_shadow_enabled:
                    if record["layer_idx"] == record["request_state"].num_layers - 1:
                        self.cpu_backend.context_lens[record["request_id"]] += 1
                outputs.append(record["context"].unsqueeze(0))
            return outputs

    def _apply_qk_full_batch(self, records: List[Dict[str, object]]) -> None:
        with self._timed("qk_full_batch_s"):
            if not self.qk_full_enabled:
                self.qk_full_shadow_last_max_abs_diff = 0.0
                return

            flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]] = []
            slot_query_refs: list[tuple[Dict[str, object], str, str]] = []

            for record in records:
                layer_state = record["request_state"].layer_states[record["layer_idx"]]
                record["resident_slot_scores"] = []
                for group in layer_state.head_groups:
                    flat_slot_queries.append(
                        (
                            group.k_slot,
                            group.v_slot,
                            list(range(group.group_heads)),
                            int(group.seq_len),
                            record["q_fp32"][group.head_start:group.head_end].contiguous(),
                        )
                    )
                    slot_query_refs.append((record, group.k_slot, group.v_slot))

            if flat_slot_queries:
                with self._timed("resident_qk_batch_s"):
                    slot_score_mats = self.resident_store.qk_slot_scores_batch(flat_slot_queries)
                self.qk_batch_calls += 1
                self.qk_full_batch_calls += 1
                for (record, k_slot, v_slot), score_mat in zip(slot_query_refs, slot_score_mats):
                    scaled_scores = score_mat.to(torch.float32) * float(record["score_scale"])
                    record["resident_slot_scores"].append((k_slot, v_slot, scaled_scores))

            self.qk_full_shadow_last_max_abs_diff = 0.0
            for record in records:
                slot_scores = record.get("resident_slot_scores", [])
                if not slot_scores:
                    if record["scores"] is None:
                        record["scores"] = self._compute_host_scores(record)
                    continue

                self.qk_full_count += sum(int(score_mat.shape[0]) for _, _, score_mat in slot_scores)
                needs_full_scores = (
                    not bool(record["use_resident_av"] and self.softmax_av_fused_enabled)
                    or bool(record.get("should_shadow_check", False))
                )

                if needs_full_scores:
                    scores = torch.cat([score_mat for _, _, score_mat in slot_scores], dim=0).to(torch.float32)
                    record["scores"] = scores
                else:
                    record["scores"] = None

                if self.qk_full_shadow_check and record.get("should_shadow_check", False):
                    host_scores = self._compute_host_scores(record)
                    compare_scores = record["scores"]
                    if compare_scores is None:
                        compare_scores = torch.cat([score_mat for _, _, score_mat in slot_scores], dim=0).to(
                            torch.float32
                        )
                    diff = (
                        float(torch.max(torch.abs(compare_scores - host_scores)).item())
                        if compare_scores.numel() > 0
                        else 0.0
                    )
                    self.qk_full_shadow_checks += 1
                    self.qk_full_shadow_last_max_abs_diff = max(self.qk_full_shadow_last_max_abs_diff, diff)
                    self.qk_full_shadow_max_abs_diff = max(self.qk_full_shadow_max_abs_diff, diff)

    def _apply_qk_mixed_batch(self, records: List[Dict[str, object]]) -> None:
        with self._timed("qk_mixed_batch_s"):
            use_host_resident_store = str(self.resident_store_backend) == "host"
            if use_host_resident_store and not self.host_qk_mixed_enabled:
                self.qk_mixed_last_head_diffs = []
                self.qk_mixed_last_max_abs_diff = 0.0
                self.qk_mixed_last_diag = {
                    "skipped": True,
                    "reason": "clover_host_qk_mixed_disabled",
                }
                self.qk_mixed_last_diag_path = ""
                return

            if not self.qk_mixed_enabled:
                self.qk_mixed_last_head_diffs = []
                self.qk_mixed_last_max_abs_diff = 0.0
                self.qk_mixed_last_diag = {}
                self.qk_mixed_last_diag_path = ""
                return

            flat_slot_queries: list[tuple[str, str, list[int], int, torch.Tensor]] = []
            slot_query_refs: list[tuple[Dict[str, object], tuple[str, str]]] = []
            total_mixed_heads = 0

            with self._timed("qk_mixed_build_queries_s"):
                for record in records:
                    scores = record["scores"]
                    keys = record["keys"]
                    mixed_heads = min(self.qk_mixed_heads, int(scores.shape[0]))
                    record["mixed_head_diffs"] = []
                    record["head_to_slot_row"] = []
                    record["slot_score_map"] = {}
                    if mixed_heads <= 0:
                        continue

                    total_mixed_heads += mixed_heads
                    layer_state = record["request_state"].layer_states[record["layer_idx"]]
                    window = min(self.qk_mixed_window, int(keys.shape[0]))
                    grouped_slot_queries: Dict[tuple[str, str], Dict[str, object]] = {}
                    head_to_slot_row: list[tuple[tuple[str, str], int]] = []
                    for head in range(mixed_heads):
                        group = self._head_group_for_head(layer_state, head)
                        slot_key = (group.k_slot, group.v_slot)
                        if slot_key not in grouped_slot_queries:
                            grouped_slot_queries[slot_key] = {
                                "local_head_indices": [],
                                "head_rows": [],
                                "window": int(window),
                            }
                        slot_entry = grouped_slot_queries[slot_key]
                        slot_entry["local_head_indices"].append(int(head - group.head_start))
                        slot_entry["head_rows"].append(int(head))
                        head_to_slot_row.append((slot_key, len(slot_entry["local_head_indices"]) - 1))

                    record["head_to_slot_row"] = head_to_slot_row
                    for slot_key, entry in grouped_slot_queries.items():
                        slot_query = (
                            slot_key[0],
                            slot_key[1],
                            list(entry["local_head_indices"]),
                            int(entry["window"]),
                            record["q_fp32"][list(entry["head_rows"])].contiguous(),
                        )
                        flat_slot_queries.append(slot_query)
                        slot_query_refs.append((record, slot_key))

            if total_mixed_heads <= 0:
                self.qk_mixed_last_head_diffs = []
                self.qk_mixed_last_max_abs_diff = 0.0
                self.qk_mixed_last_diag = {}
                self.qk_mixed_last_diag_path = ""
                return

            try:
                if flat_slot_queries:
                    with self._timed("resident_qk_batch_s"):
                        slot_score_mats = self.resident_store.qk_slot_scores_batch(flat_slot_queries)
                    self.qk_batch_calls += 1
                    for (record, slot_key), score_mat in zip(slot_query_refs, slot_score_mats):
                        record["slot_score_map"][slot_key] = score_mat.to(record["scores"].dtype) * float(
                            record["score_scale"]
                        )

                with self._timed("qk_mixed_apply_scores_s"):
                    head_diffs: list[float] = []
                    self.qk_mixed_last_diag = {}
                    self.qk_mixed_last_diag_path = ""
                    for record in records:
                        for head, (slot_key, row_idx) in enumerate(record["head_to_slot_row"]):
                            head_scores = record["slot_score_map"][slot_key][row_idx]
                            cpu_window_scores = record["scores"][head, -int(head_scores.shape[0]) :].clone()
                            diff = (
                                float(torch.max(torch.abs(head_scores - cpu_window_scores)).item())
                                if head_scores.numel() > 0
                                else 0.0
                            )
                            record["mixed_head_diffs"].append(diff)
                            head_diffs.append(diff)
                            if not self.qk_mixed_last_diag and head_scores.numel() > 0:
                                dpu_has_nan = bool(torch.isnan(head_scores).any().item())
                                cpu_has_nan = bool(torch.isnan(cpu_window_scores).any().item())
                                dpu_has_inf = bool(torch.isinf(head_scores).any().item())
                                cpu_has_inf = bool(torch.isinf(cpu_window_scores).any().item())
                                if dpu_has_nan or cpu_has_nan or dpu_has_inf or cpu_has_inf:
                                    self.qk_mixed_last_diag = {
                                        "request_id": str(record["request_id"]),
                                        "layer_idx": int(record["layer_idx"]),
                                        "head": int(head),
                                        "slot_key": [str(slot_key[0]), str(slot_key[1])],
                                        "row_idx": int(row_idx),
                                        "window": int(head_scores.shape[0]),
                                        "dpu_has_nan": dpu_has_nan,
                                        "cpu_has_nan": cpu_has_nan,
                                        "dpu_has_inf": dpu_has_inf,
                                        "cpu_has_inf": cpu_has_inf,
                                        "dpu_preview": head_scores[: min(8, head_scores.shape[0])].detach().cpu().tolist(),
                                        "cpu_preview": cpu_window_scores[
                                            : min(8, cpu_window_scores.shape[0])
                                        ].detach().cpu().tolist(),
                                    }
                            if head_scores.numel() > 0:
                                record["scores"][head, -int(head_scores.shape[0]) :] = head_scores

                    self.qk_mixed_last_head_diffs = head_diffs
                    self.qk_mixed_last_max_abs_diff = max(head_diffs) if head_diffs else 0.0
                    self.qk_mixed_count += total_mixed_heads
            except Exception:
                self.qk_check_failures += 1
                raise

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
                    slot_scores = [
                        (k_slot, v_slot, score_mat.contiguous())
                        for k_slot, v_slot, score_mat in record.get("resident_slot_scores", [])
                    ]
                    flat_slot_scores.extend(slot_scores)
                    slot_score_refs.append((record_idx, len(slot_scores)))
                    if record.get("should_shadow_check", False) and record["values"] is not None:
                        if record["scores"] is None:
                            record["scores"] = torch.cat(
                                [score_mat for _, _, score_mat in record.get("resident_slot_scores", [])],
                                dim=0,
                            ).to(torch.float32)
                        with self._timed("softmax_av_s"):
                            record["weights"] = torch.softmax(record["scores"], dim=-1)
                else:
                    with self._timed("softmax_av_s"):
                        weights = torch.softmax(record["scores"], dim=-1)
                    if record["use_resident_av"]:
                        record["weights"] = weights
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
                        with self._timed("host_context_compute_s"):
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
                    if record.get("should_shadow_check", False) and record.get("weights") is not None and record["values"] is not None:
                        cpu_context = torch.einsum(
                            "hl,lhd->hd", record["weights"], record["values"].float()
                        ).to(record["query_dtype"])
                        av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                        self.softmax_av_fused_shadow_max_abs_diff = max(
                            self.softmax_av_fused_shadow_max_abs_diff, av_diff
                        )
                    self.softmax_av_fused_ops += 1
                    record["context"] = context
                    record.pop("resident_slot_scores", None)

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
                    if record.get("should_shadow_check", False) and record["values"] is not None:
                        cpu_context = torch.einsum(
                            "hl,lhd->hd", record["weights"], record["values"].float()
                        ).to(record["query_dtype"])
                        av_diff = float(torch.max(torch.abs(context.float() - cpu_context.float())).item())
                        self.resident_av_shadow_max_abs_diff = max(self.resident_av_shadow_max_abs_diff, av_diff)
                    self.resident_av_ops += 1
                    record["context"] = context
                    record.pop("resident_slot_scores", None)

            for record in records:
                record.pop("resident_slot_scores", None)
                if self.cpu_shadow_enabled:
                    if record["layer_idx"] == record["request_state"].num_layers - 1:
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
            seq_len = self._init_cpu_shadow_request(
                request_id,
                initial_kv,
                decode_reserve_tokens,
            )
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
            self.shadow_k_buffers.pop(request_id, None)
            self.shadow_v_buffers.pop(request_id, None)
            self.shadow_layer_lens.pop(request_id, None)
            self.cpu_backend.context_lens.pop(request_id, None)
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
        debug["clover_shadow_check_token_interval"] = self.shadow_check_token_interval
        debug["clover_shadow_check_layer_interval"] = self.shadow_check_layer_interval
        debug["clover_host_qk_mixed_enabled"] = self.host_qk_mixed_enabled
        debug["clover_pim_attention_enabled"] = self.pim_attention_enabled
        debug["clover_pim_context_fused_experimental_enabled"] = self.pim_context_fused_experimental_enabled
        debug["clover_shadow_check_invocations"] = self.shadow_check_invocations
        debug["clover_shadow_check_skips"] = self.shadow_check_skips
        debug["clover_op_timing_totals_s"] = dict(self.op_timing_totals)
        debug["clover_op_timing_counts"] = dict(self.op_timing_counts)
        return debug
