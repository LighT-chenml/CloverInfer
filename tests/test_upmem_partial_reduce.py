import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.core.resident_kv_store as resident_kv_store
from src.core.resident_kv_store import UpmemKVSlotStore, _KVSlotHelperClient


def _merge_reference(entries, outputs):
    merged_numerators = {}
    merged_row_max = {}
    merged_row_sum = {}
    for entry, (segment_context, segment_row_max, segment_row_sum) in zip(entries, outputs):
        logical_idx = int(entry["logical_idx"])
        score_scale = float(entry["payload"][4])
        segment_numerator = segment_context.to(torch.float32)
        segment_row_max = segment_row_max.to(torch.float32) * score_scale
        segment_row_sum = segment_row_sum.to(torch.float32)
        if logical_idx not in merged_numerators:
            merged_numerators[logical_idx] = segment_numerator
            merged_row_max[logical_idx] = segment_row_max
            merged_row_sum[logical_idx] = segment_row_sum
            continue

        prev_row_max = merged_row_max[logical_idx]
        prev_row_sum = merged_row_sum[logical_idx]
        prev_numerator = merged_numerators[logical_idx]
        combined_row_max = torch.maximum(prev_row_max, segment_row_max)
        prev_scale = torch.exp(prev_row_max - combined_row_max)
        seg_scale = torch.exp(segment_row_max - combined_row_max)
        combined_row_sum = prev_row_sum * prev_scale + segment_row_sum * seg_scale
        merged_numerators[logical_idx] = (
            prev_numerator * prev_scale.unsqueeze(1)
        ) + (segment_numerator * seg_scale.unsqueeze(1))
        merged_row_max[logical_idx] = combined_row_max
        merged_row_sum[logical_idx] = combined_row_sum
    return {
        logical_idx: numerator / torch.clamp(merged_row_sum[logical_idx], min=1e-12).unsqueeze(1)
        for logical_idx, numerator in merged_numerators.items()
    }


def test_partial_reduce_helper_matches_legacy_merge():
    store = object.__new__(UpmemKVSlotStore)
    entries = [
        {
            "logical_idx": 0,
            "payload": (10, [0], 4, torch.tensor([[1.0, 0.0]]), 0.5),
        },
        {
            "logical_idx": 0,
            "payload": (11, [0], 3, torch.tensor([[1.0, 0.0]]), 0.5),
        },
        {
            "logical_idx": 1,
            "payload": (12, [0], 2, torch.tensor([[0.0, 1.0]]), 1.0),
        },
        {
            "logical_idx": 1,
            "payload": (13, [0], 1, torch.tensor([[0.0, 1.0]]), 1.0),
        },
    ]
    outputs = [
        (
            torch.tensor([[10.000002, 12.130615]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[11.032656, 4.032655]], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[5.0, 10.0]], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
            torch.tensor([2.5], dtype=torch.float32),
        ),
        (
            torch.tensor([[6.0, 0.75]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([0.75], dtype=torch.float32),
        ),
    ]

    merged = UpmemKVSlotStore._merge_partial_contexts(store, entries, outputs)
    reference = _merge_reference(entries, outputs)
    assert sorted(merged.keys()) == sorted(reference.keys())
    for logical_idx in merged:
        assert torch.allclose(merged[logical_idx], reference[logical_idx], atol=1e-5, rtol=1e-5)


def test_partial_reduce_python_fallback_merges_unnormalized_numerators():
    store = object.__new__(UpmemKVSlotStore)
    entries = [
        {
            "logical_idx": 0,
            "payload": (10, [0, 1], 3, torch.ones(2, 4), 0.5),
        },
        {
            "logical_idx": 0,
            "payload": (11, [0, 1], 2, torch.ones(2, 4), 0.5),
        },
    ]
    outputs = [
        (
            torch.tensor([[3.0, 6.0], [8.0, 2.0]], dtype=torch.float32),
            torch.tensor([1.0, 2.0], dtype=torch.float32),
            torch.tensor([1.5, 2.0], dtype=torch.float32),
        ),
        (
            torch.tensor([[5.0, -1.0], [1.0, 7.0]], dtype=torch.float32),
            torch.tensor([4.0, -1.0], dtype=torch.float32),
            torch.tensor([1.25, 0.75], dtype=torch.float32),
        ),
    ]

    old_module = resident_kv_store._HOST_REDUCTION_MODULE
    old_attempted = resident_kv_store._HOST_REDUCTION_IMPORT_ATTEMPTED
    resident_kv_store._HOST_REDUCTION_MODULE = None
    resident_kv_store._HOST_REDUCTION_IMPORT_ATTEMPTED = True
    try:
        merged = UpmemKVSlotStore._merge_partial_contexts(store, entries, outputs)
    finally:
        resident_kv_store._HOST_REDUCTION_MODULE = old_module
        resident_kv_store._HOST_REDUCTION_IMPORT_ATTEMPTED = old_attempted

    reference = _merge_reference(entries, outputs)
    assert sorted(merged.keys()) == [0]
    assert torch.allclose(merged[0], reference[0], atol=1e-5, rtol=1e-5)


def test_int8_kv_quantization_round_trips_with_scales_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.kv_dtype = "int8"

    keys = torch.tensor(
        [[[0.0, 0.25, -0.5, 1.0], [1.5, -2.0, 0.75, -1.25]]],
        dtype=torch.float32,
    )
    values = keys * 0.5

    encoded_k, encoded_v, k_scale, v_scale = UpmemKVSlotStore._encode_kv_pair(store, keys, values)
    assert encoded_k.dtype == torch.int8
    assert encoded_v.dtype == torch.int8
    assert k_scale > 0.0
    assert v_scale > 0.0

    decoded_k = UpmemKVSlotStore._decode_tensor(store, encoded_k, scale=k_scale)
    decoded_v = UpmemKVSlotStore._decode_tensor(store, encoded_v, scale=v_scale)
    assert torch.allclose(decoded_k, keys, atol=max(k_scale, 1e-6), rtol=0.0)
    assert torch.allclose(decoded_v, values, atol=max(v_scale, 1e-6), rtol=0.0)


def test_int8_slot_elem_count_uses_packed_aligned_words_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.kv_dtype = "int8"

    assert UpmemKVSlotStore._slot_elem_count(store, capacity=1, group_heads=1, head_dim=1) == 2
    assert UpmemKVSlotStore._slot_elem_count(store, capacity=1, group_heads=1, head_dim=8) == 2
    assert UpmemKVSlotStore._slot_elem_count(store, capacity=1, group_heads=1, head_dim=9) == 4


def test_mixed_int8_fp16_kv_encodes_k_and_v_separately_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.kv_dtype = "mixed_int8_fp16"

    keys = torch.tensor(
        [[[0.0, 0.25, -0.5, 1.0], [1.5, -2.0, 0.75, -1.25]]],
        dtype=torch.float32,
    )
    values = keys * 0.5

    encoded_k, encoded_v, k_scale, v_scale = UpmemKVSlotStore._encode_kv_pair(store, keys, values)
    assert encoded_k.dtype == torch.int8
    assert encoded_v.dtype == torch.int16
    assert k_scale > 0.0
    assert v_scale == 1.0

    decoded_k = UpmemKVSlotStore._decode_tensor_for_dtype(
        store,
        encoded_k,
        resident_kv_store.KVSLOT_DTYPE_INT8,
        scale=k_scale,
    )
    decoded_v = UpmemKVSlotStore._decode_tensor_for_dtype(
        store,
        encoded_v,
        resident_kv_store.KVSLOT_DTYPE_FP16,
        scale=v_scale,
    )
    assert torch.allclose(decoded_k, keys, atol=max(k_scale, 1e-6), rtol=0.0)
    assert torch.allclose(decoded_v, values, atol=1e-3, rtol=1e-3)


def test_mixed_int8_fp16_slot_elem_count_reserves_larger_kv_packing_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.kv_dtype = "mixed_int8_fp16"

    assert UpmemKVSlotStore._slot_elem_count(store, capacity=1, group_heads=1, head_dim=8) == 4
    assert UpmemKVSlotStore._slot_elem_count(store, capacity=1, group_heads=1, head_dim=9) == 6


def test_mixed_int8_int16_kv_encodes_k_and_v_separately_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.kv_dtype = "mixed_int8_int16"

    keys = torch.tensor(
        [[[0.0, 0.25, -0.5, 1.0], [1.5, -2.0, 0.75, -1.25]]],
        dtype=torch.float32,
    )
    values = keys * 0.5

    encoded_k, encoded_v, k_scale, v_scale = UpmemKVSlotStore._encode_kv_pair(store, keys, values)
    assert encoded_k.dtype == torch.int8
    assert encoded_v.dtype == torch.int16
    assert k_scale > 0.0
    assert v_scale > 0.0

    decoded_k = UpmemKVSlotStore._decode_tensor_for_dtype(
        store,
        encoded_k,
        resident_kv_store.KVSLOT_DTYPE_INT8,
        scale=k_scale,
    )
    decoded_v = UpmemKVSlotStore._decode_tensor_for_dtype(
        store,
        encoded_v,
        resident_kv_store.KVSLOT_DTYPE_INT16,
        scale=v_scale,
    )
    assert torch.allclose(decoded_k, keys, atol=max(k_scale, 1e-6), rtol=0.0)
    assert torch.allclose(decoded_v, values, atol=max(v_scale, 1e-6), rtol=0.0)


def test_mixed_int8_int16_aliases_normalize_without_upmem():
    assert resident_kv_store.normalize_resident_kv_dtype("int8_int16") == "mixed_int8_int16"
    assert resident_kv_store.normalize_resident_kv_dtype("int8-int16") == "mixed_int8_int16"
    assert resident_kv_store.normalize_resident_kv_dtype("k_int8_v_int16") == "mixed_int8_int16"


def test_partial_reduce_flag_can_be_toggled_without_helper():
    store = object.__new__(UpmemKVSlotStore)
    store.host_partial_reduce_enabled = True
    UpmemKVSlotStore.set_experimental_flags(store, host_partial_reduce_enabled=False)
    assert store.host_partial_reduce_enabled is False
    UpmemKVSlotStore.set_experimental_flags(store, host_partial_reduce_enabled=True)
    assert store.host_partial_reduce_enabled is True


def test_partial_reduce_uses_cpp_module_when_available():
    module = resident_kv_store._load_host_reduction_module()
    if module is None:
        return

    store = object.__new__(UpmemKVSlotStore)
    entries = [
        {
            "logical_idx": 0,
            "payload": (10, [0], 4, torch.tensor([[1.0, 0.0]]), 0.5),
        },
        {
            "logical_idx": 0,
            "payload": (11, [0], 3, torch.tensor([[1.0, 0.0]]), 0.5),
        },
    ]
    outputs = [
        (
            torch.tensor([[10.000002, 12.130615]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[11.032656, 4.032655]], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
    ]
    merged = UpmemKVSlotStore._merge_partial_contexts(store, entries, outputs)
    assert 0 in merged
    assert merged[0].shape == (1, 2)


def test_segmented_sparse_qk_softmax_av_uses_partial_fused_path_without_two_stage():
    class _FakeHelper:
        def __init__(self):
            self.payloads = []

        def qk_softmax_weighted_value_sum_partial_batch(self, payloads):
            self.payloads = list(payloads)
            outputs = []
            for idx, (_slot_id, local_heads, _window, queries, _score_scale) in enumerate(payloads):
                outputs.append(
                    (
                        torch.full((len(local_heads), int(queries.shape[1])), float(idx + 1), dtype=torch.float32),
                        torch.zeros(len(local_heads), dtype=torch.float32),
                        torch.ones(len(local_heads), dtype=torch.float32),
                    )
                )
            return outputs

    store = object.__new__(UpmemKVSlotStore)
    store.helper = _FakeHelper()
    store.host_partial_reduce_enabled = False
    store.slot_mapping = {
        ("k", "v"): {
            "backend": "dpu_segmented",
            "seq_len": 10,
            "group_heads": 2,
            "head_dim": 4,
            "blocks": [
                {"slot_id": 100, "physical_dpu": 0, "seq_len": 4},
                {"slot_id": 101, "physical_dpu": 1, "seq_len": 6},
            ],
        }
    }
    store.op_timing_totals_s = {
        "qk_softmax_weighted_value_sum_batch_total": 0.0,
        "qk_softmax_weighted_value_sum_batch_dpu": 0.0,
        "qk_softmax_weighted_value_sum_batch_host_reduce": 0.0,
    }
    store.op_timing_counts = {key: 0 for key in store.op_timing_totals_s}
    store.batch_item_totals = {
        "qk_softmax_weighted_value_sum_batch_total": 0,
        "qk_softmax_weighted_value_sum_batch_blocked_logical_items": 0,
        "qk_softmax_weighted_value_sum_batch_segmented_logical_items": 0,
        "qk_softmax_weighted_value_sum_batch_dpu_items": 0,
        "qk_softmax_weighted_value_sum_batch_host_fallback_items": 0,
        "qk_softmax_weighted_value_sum_batch_host_reduce_items": 0,
    }
    store._helper_submit_sort_key = lambda **kwargs: (
        int(kwargs["logical_idx"]),
        int(kwargs["segment_ordinal"]),
    )
    store.qk_slot_scores_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("segmented sparse fused path should not use two-stage QK")
    )
    store.softmax_weighted_value_sum_batch = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("segmented sparse fused path should not use two-stage AV")
    )

    queries = torch.ones(2, 4, dtype=torch.float32)
    outputs = UpmemKVSlotStore.qk_softmax_weighted_value_sum_batch(
        store,
        [("k", "v", [0, 1], 8, queries, 0.5)],
    )

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.full((2, 4), 1.5, dtype=torch.float32))
    assert [(int(payload[0]), int(payload[2])) for payload in store.helper.payloads] == [
        (100, 2),
        (101, 6),
    ]
    assert store.batch_item_totals["qk_softmax_weighted_value_sum_batch_dpu_items"] == 2


def test_grouped_av_splits_oversized_helper_groups_without_upmem():
    class _FakeHelper:
        MAX_GROUP_SEGMENTS = 2
        MAX_BATCH_ITEMS = 32

        def weighted_value_sum_grouped_batch(self, grouped_slot_weights):
            assert all(0 < len(group) <= self.MAX_GROUP_SEGMENTS for group in grouped_slot_weights)
            outputs = []
            for group in grouped_slot_weights:
                total = sum(float(weights.sum().item()) for _, _, weights in group)
                outputs.append(torch.tensor([[total]], dtype=torch.float32))
            return outputs

    fake = _FakeHelper()
    oversized_group = [
        (slot_id, 1, torch.tensor([[float(slot_id)]], dtype=torch.float32))
        for slot_id in range(5)
    ]
    normal_group = [(10, 1, torch.tensor([[10.0]], dtype=torch.float32))]

    outputs = _KVSlotHelperClient.weighted_value_sum_grouped_batch(
        fake,
        [oversized_group, normal_group],
    )

    assert len(outputs) == 2
    assert torch.allclose(outputs[0], torch.tensor([[10.0]], dtype=torch.float32))
    assert torch.allclose(outputs[1], torch.tensor([[10.0]], dtype=torch.float32))


def test_grouped_av_splits_groups_by_total_dpu_capacity_without_upmem():
    class _FakeHelper:
        MAX_GROUP_SEGMENTS = 8
        MAX_DPU_CAPACITY = 4
        MAX_BATCH_ITEMS = 32

        def weighted_value_sum_grouped_batch(self, grouped_slot_weights):
            assert all(0 < len(group) <= self.MAX_GROUP_SEGMENTS for group in grouped_slot_weights)
            assert all(sum(segment_len for _, segment_len, _ in group) <= self.MAX_DPU_CAPACITY for group in grouped_slot_weights)
            outputs = []
            for group in grouped_slot_weights:
                total = sum(float(weights.sum().item()) for _, _, weights in group)
                outputs.append(torch.tensor([[total]], dtype=torch.float32))
            return outputs

    fake = _FakeHelper()
    capacity_oversized_group = [
        (0, 2, torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        (1, 2, torch.tensor([[3.0, 4.0]], dtype=torch.float32)),
        (2, 2, torch.tensor([[5.0, 6.0]], dtype=torch.float32)),
    ]

    outputs = _KVSlotHelperClient.weighted_value_sum_grouped_batch(
        fake,
        [capacity_oversized_group],
    )

    assert len(outputs) == 1
    assert torch.allclose(outputs[0], torch.tensor([[21.0]], dtype=torch.float32))


def test_slot_capacity_choice_spills_outside_full_allowed_stripe():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [64, 64, 0, 0]
    store._next_slot_seq_by_dpu = [64, 64, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = True
    store.slot_spill_allocations = 0

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    chosen = UpmemKVSlotStore._choose_physical_dpu_with_slot_capacity(
        store,
        preferred_dpu=0,
        elem_count=1,
        allowed_dpus=[0, 1],
    )

    assert chosen in {2, 3}
    assert store.slot_spill_allocations == 1


def test_slot_capacity_choice_prefers_lower_slot_pressure_inside_allowed_stripe():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [50, 20, 0, 0]
    store._next_slot_seq_by_dpu = [50, 20, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = False
    store.slot_spill_allocations = 0
    store.slot_pressure_soft_limit = 48
    store.slot_pressure_aware_alloc_enabled = True

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    chosen = UpmemKVSlotStore._choose_physical_dpu_with_slot_capacity(
        store,
        preferred_dpu=0,
        elem_count=1,
        allowed_dpus=[0, 1],
    )

    assert chosen == 1


def test_slot_capacity_choice_fails_cleanly_when_allowed_stripe_is_full():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [64, 64, 0, 0]
    store._next_slot_seq_by_dpu = [64, 64, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = False
    store.slot_spill_allocations = 0
    store.emergency_slot_spill_enabled = False
    store.emergency_slot_spill_allocations = 0

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    try:
        UpmemKVSlotStore._choose_physical_dpu_with_slot_capacity(
            store,
            preferred_dpu=0,
            elem_count=1,
            allowed_dpus=[0, 1],
        )
    except RuntimeError as exc:
        assert "No DPU KV slot capacity remains" in str(exc)
    else:
        raise AssertionError("expected full allowed stripe to fail before slot assignment")

    assert store._next_slot_seq_by_dpu == [64, 64, 0, 0]
    assert store._free_slot_ids_by_dpu == [[], [], [], []]
    assert store.slot_spill_allocations == 0


def test_choose_physical_dpu_skips_full_preferred_dpu_before_host_fallback():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [64, 4, 0, 0]
    store._next_slot_seq_by_dpu = [64, 4, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = False
    store.slot_spill_allocations = 0
    store.slot_capacity_reroutes = 0
    store.placement_policy = "rank_spread"

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    chosen = UpmemKVSlotStore.choose_physical_dpu(
        store,
        elem_count=1,
        preferred_dpu=0,
        placement_policy="rank_spread",
        allowed_dpus=[0, 1],
    )

    assert chosen == 1
    assert store.slot_capacity_reroutes == 1


def test_choose_physical_dpu_can_globally_spill_when_enabled():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [64, 64, 0, 0]
    store._next_slot_seq_by_dpu = [64, 64, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = True
    store.slot_spill_allocations = 0
    store.slot_capacity_reroutes = 0
    store.placement_policy = "rank_spread"

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    chosen = UpmemKVSlotStore.choose_physical_dpu(
        store,
        elem_count=1,
        preferred_dpu=0,
        placement_policy="rank_spread",
        allowed_dpus=[0, 1],
    )

    assert chosen in {2, 3}
    assert store.slot_spill_allocations == 1
    assert store.slot_capacity_reroutes == 1


def test_slot_capacity_choice_uses_emergency_spill_for_full_allowed_stripe():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.POOL_CAPACITY_ELEMS = 1024
    store.dpu_live_elems_by_dpu = [0, 0, 0, 0]
    store.dpu_live_slot_counts_by_dpu = [64, 64, 0, 0]
    store._next_slot_seq_by_dpu = [64, 64, 0, 0]
    store._free_slot_ids_by_dpu = [[], [], [], []]
    store._helper_topology_cache = {}
    store.slot_spill_alloc_enabled = False
    store.slot_spill_allocations = 0
    store.emergency_slot_spill_enabled = True
    store.emergency_slot_spill_allocations = 0
    store.slot_capacity_reroutes = 0

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    chosen = UpmemKVSlotStore._choose_physical_dpu_with_slot_capacity(
        store,
        preferred_dpu=0,
        elem_count=1,
        allowed_dpus=[0, 1],
    )

    assert chosen in {2, 3}
    assert store.slot_spill_allocations == 0
    assert store.emergency_slot_spill_allocations == 1
    assert store.slot_capacity_reroutes == 1


def test_assign_slot_id_rejects_full_dpu_without_polluting_free_list():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 2
    store._slot_id_map = {}
    store._next_slot_seq_by_dpu = [64, 0]
    store._free_slot_ids_by_dpu = [[128], []]

    class _Helper:
        MAX_SLOTS_PER_DPU = 64

    store.helper = _Helper()

    try:
        UpmemKVSlotStore._assign_slot_id(store, ("k", "v"), preferred_dpu=0)
    except RuntimeError as exc:
        assert "No DPU KV slot capacity remains" in str(exc)
    else:
        raise AssertionError("expected exhausted DPU to reject new slot assignment")

    assert ("k", "v") not in store._slot_id_map
    assert store._free_slot_ids_by_dpu[0] == []
    assert store._next_slot_seq_by_dpu == [64, 0]


def test_segmented_base_capacity_reserves_decode_growth_on_tail_blocks():
    store = object.__new__(UpmemKVSlotStore)
    store.block_tokens = 256

    capacities = UpmemKVSlotStore._reserve_tail_block_capacities(
        store,
        [27, 26, 26, 26],
        128,
    )

    assert capacities == [27, 26, 26, 49]


def test_segmented_base_capacity_can_limit_tail_reserve_tokens():
    store = object.__new__(UpmemKVSlotStore)
    store.block_tokens = 256

    capacities = UpmemKVSlotStore._reserve_tail_block_capacities(
        store,
        [27, 26, 26, 26],
        128,
        max_reserve_tokens=8,
    )

    assert capacities == [27, 26, 26, 34]


def test_segmented_base_allocation_splits_segments_by_helper_capacity_without_upmem():
    store = object.__new__(UpmemKVSlotStore)
    store.num_dpus = 4
    store.block_tokens = 512
    store.growth_block_tokens = 128
    store.max_dpu_capacity = 256
    store.reserve_segment_tail_capacity_enabled = True
    store.reserve_segment_tail_capacity_tokens = 8

    allocated = []

    def _fake_allocate_block_append_only(**kwargs):
        block_k = kwargs["block_k"]
        block_capacity = int(kwargs["block_capacity_override"])
        segment_meta = dict(kwargs["segment_meta"])
        assert int(block_k.shape[0]) <= store.max_dpu_capacity
        assert block_capacity <= store.max_dpu_capacity
        allocated.append(
            {
                "physical_dpu": int(kwargs["physical_dpu_override"]),
                "seq_len": int(block_k.shape[0]),
                "capacity": block_capacity,
                "logical_segment_index": int(segment_meta["logical_segment_index"]),
                "token_range_start": int(segment_meta["token_range_start"]),
                "token_range_end": int(segment_meta["token_range_end"]),
            }
        )
        return {
            "slot_id": len(allocated),
            "physical_dpu": int(kwargs["physical_dpu_override"]),
            "seq_len": int(block_k.shape[0]),
            "capacity": block_capacity,
            "group_heads": int(kwargs["group_heads"]),
            "head_dim": int(kwargs["head_dim"]),
            "token_range_start": int(segment_meta["token_range_start"]),
            "token_range_end": int(segment_meta["token_range_end"]),
            "logical_segment_index": int(segment_meta["logical_segment_index"]),
        }

    store._allocate_block_append_only = _fake_allocate_block_append_only

    initial_k = torch.zeros(515, 1, 8, dtype=torch.float32)
    initial_v = torch.zeros_like(initial_k)
    slot_info = UpmemKVSlotStore._allocate_segmented_group(
        store,
        key=("k", "v"),
        initial_k=initial_k,
        initial_v=initial_v,
        capacity=520,
        physical_dpu=0,
        group_heads=1,
        head_dim=8,
        allowed_dpus=[0, 1],
        segment_plan=[
            {"physical_dpu": 0, "token_range_start": 0, "token_range_end": 258},
            {"physical_dpu": 1, "token_range_start": 258, "token_range_end": 515},
        ],
    )

    assert [(item["token_range_start"], item["token_range_end"]) for item in allocated] == [
        (0, 256),
        (256, 258),
        (258, 514),
        (514, 515),
    ]
    assert max(item["seq_len"] for item in allocated) == 256
    assert max(item["capacity"] for item in allocated) == 256
    assert sum(item["seq_len"] for item in allocated) == 515
    assert int(slot_info["seq_len"]) == 515
