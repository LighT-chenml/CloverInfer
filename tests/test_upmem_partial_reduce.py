import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import src.core.resident_kv_store as resident_kv_store
from src.core.resident_kv_store import UpmemKVSlotStore


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
