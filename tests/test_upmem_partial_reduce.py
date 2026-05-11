import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.core.resident_kv_store as resident_kv_store
from src.core.resident_kv_store import UpmemKVSlotStore, _KVSlotHelperClient


def _merge_reference(entries, outputs):
    merged_contexts = {}
    merged_row_max = {}
    merged_row_sum = {}
    for entry, (segment_context, segment_row_max, segment_row_sum) in zip(entries, outputs):
        logical_idx = int(entry["logical_idx"])
        score_scale = float(entry["payload"][4])
        segment_context = segment_context.to(torch.float32)
        segment_row_max = segment_row_max.to(torch.float32) * score_scale
        segment_row_sum = segment_row_sum.to(torch.float32)
        if logical_idx not in merged_contexts:
            merged_contexts[logical_idx] = segment_context
            merged_row_max[logical_idx] = segment_row_max
            merged_row_sum[logical_idx] = segment_row_sum
            continue

        prev_row_max = merged_row_max[logical_idx]
        prev_row_sum = merged_row_sum[logical_idx]
        prev_context = merged_contexts[logical_idx]
        combined_row_max = torch.maximum(prev_row_max, segment_row_max)
        prev_scale = torch.exp(prev_row_max - combined_row_max)
        seg_scale = torch.exp(segment_row_max - combined_row_max)
        combined_row_sum = prev_row_sum * prev_scale + segment_row_sum * seg_scale
        safe_sum = torch.clamp(combined_row_sum, min=1e-12)
        prev_weight = (prev_row_sum * prev_scale / safe_sum).unsqueeze(1)
        seg_weight = (segment_row_sum * seg_scale / safe_sum).unsqueeze(1)
        merged_contexts[logical_idx] = (prev_context * prev_weight) + (segment_context * seg_weight)
        merged_row_max[logical_idx] = combined_row_max
        merged_row_sum[logical_idx] = combined_row_sum
    return merged_contexts


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
            torch.tensor([[6.224593, 7.550813]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[6.867378, 2.510163]], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[2.0, 4.0]], dtype=torch.float32),
            torch.tensor([3.0], dtype=torch.float32),
            torch.tensor([2.5], dtype=torch.float32),
        ),
        (
            torch.tensor([[8.0, 1.0]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([0.75], dtype=torch.float32),
        ),
    ]

    merged = UpmemKVSlotStore._merge_partial_contexts(store, entries, outputs)
    reference = _merge_reference(entries, outputs)
    assert sorted(merged.keys()) == sorted(reference.keys())
    for logical_idx in merged:
        assert torch.allclose(merged[logical_idx], reference[logical_idx], atol=1e-5, rtol=1e-5)


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
            torch.tensor([[6.224593, 7.550813]], dtype=torch.float32),
            torch.tensor([1.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
        (
            torch.tensor([[6.867378, 2.510163]], dtype=torch.float32),
            torch.tensor([2.0], dtype=torch.float32),
            torch.tensor([1.606531], dtype=torch.float32),
        ),
    ]
    merged = UpmemKVSlotStore._merge_partial_contexts(store, entries, outputs)
    assert 0 in merged
    assert merged[0].shape == (1, 2)


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
