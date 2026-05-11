import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.attention_backend import PimNaiveAttentionBackend
from src.core.resident_kv_store import HostResidentKVStore


class _PlannerOnlyBackend(PimNaiveAttentionBackend):
    def _run_dot_smoke_test(self) -> None:
        self.smoke_test_ok = True
        self.smoke_test_output = "skipped"


def _dummy_initial_kv(seq_len: int, num_heads: int, head_dim: int, num_layers: int = 2):
    return [
        {
            "key": torch.randn(seq_len, num_heads, head_dim),
            "value": torch.randn(seq_len, num_heads, head_dim),
        }
        for _ in range(num_layers)
    ]


def test_request_state_uses_planner_groups_when_dpus_are_fewer_than_heads():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    state = backend._build_request_state(
        "req0",
        _dummy_initial_kv(seq_len=6, num_heads=12, head_dim=8),
        decode_reserve_tokens=2,
    )

    assert state.sharding_plan is not None
    assert state.sharding_plan["metadata"]["planner_mode"] == "single_dpu_multi_head_group"
    assert sorted(state.preferred_dpu_stripe) == [0, 1, 2, 3]

    layer0 = state.layer_states[0]
    assert len(layer0.head_groups) == 4
    assert [(group.head_start, group.head_end) for group in layer0.head_groups] == [
        (0, 3),
        (3, 6),
        (6, 9),
        (9, 12),
    ]
    assert sorted(group.dpu_id for group in layer0.head_groups) == [0, 1, 2, 3]


def test_auto_head_grouping_policy_is_accepted_for_runtime_configs():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="auto",
        dpu_placement_policy="auto",
    )
    assert backend.head_grouping_policy == "balanced"
    assert backend.dpu_placement_policy == "rotated"


class _RecordingHostStore(HostResidentKVStore):
    def __init__(self):
        super().__init__()
        self.allocate_calls = []

    def allocate_group(
        self,
        k_slot,
        v_slot,
        initial_k,
        initial_v,
        capacity,
        preferred_dpu=None,
        force_host_fallback=False,
        allowed_dpus=None,
        segment_plan=None,
    ):
        self.allocate_calls.append(
            {
                "k_slot": str(k_slot),
                "allowed_dpus": [] if allowed_dpus is None else [int(dpu) for dpu in allowed_dpus],
                "segment_plan": [] if segment_plan is None else [dict(item) for item in segment_plan],
                "seq_len": int(initial_k.shape[0]),
                "group_heads": int(initial_k.shape[1]),
            }
        )
        return super().allocate_group(
            k_slot,
            v_slot,
            initial_k,
            initial_v,
            capacity,
            preferred_dpu=preferred_dpu,
            force_host_fallback=force_host_fallback,
            allowed_dpus=allowed_dpus,
            segment_plan=segment_plan,
        )


def test_request_state_materializes_planner_token_segments_for_multi_dpu_group():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    backend.resident_store = _RecordingHostStore()
    state = backend._build_request_state(
        "req1",
        _dummy_initial_kv(seq_len=10, num_heads=2, head_dim=8),
        decode_reserve_tokens=2,
    )

    assert state.sharding_plan is not None
    assert state.sharding_plan["metadata"]["planner_mode"] == "multi_dpu_per_head_group"

    layer0 = state.layer_states[0]
    assert len(layer0.head_groups) == 1
    group = layer0.head_groups[0]
    assert sorted(group.physical_dpus or []) == [0, 1, 2, 3]
    assert [(segment["token_range_start"], segment["token_range_end"]) for segment in (group.token_segments or [])] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]

    assert backend.resident_store.allocate_calls
    first_call = backend.resident_store.allocate_calls[0]
    assert sorted(first_call["allowed_dpus"]) == [0, 1, 2, 3]
    assert [(item["token_range_start"], item["token_range_end"]) for item in first_call["segment_plan"]] == [
        (0, 3),
        (3, 6),
        (6, 8),
        (8, 10),
    ]


def test_request_state_compacts_short_planner_token_segments_when_enabled():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
        compact_short_segments_enabled=True,
        compact_short_segment_min_tokens=32,
    )
    backend.resident_store = _RecordingHostStore()
    state = backend._build_request_state(
        "req_short",
        _dummy_initial_kv(seq_len=10, num_heads=2, head_dim=8),
        decode_reserve_tokens=2,
    )

    layer0 = state.layer_states[0]
    assert len(layer0.head_groups) == 1
    group = layer0.head_groups[0]
    assert len(group.physical_dpus or []) == 1
    assert group.token_segments == [
        {
            "physical_dpu": int(group.dpu_id),
            "token_range_start": 0,
            "token_range_end": 10,
        }
    ]

    first_call = backend.resident_store.allocate_calls[0]
    assert sorted(first_call["allowed_dpus"]) == [0, 1, 2, 3]
    assert first_call["segment_plan"] == []
    assert backend.planner_segment_plan_count == 2
    assert backend.planner_segment_materialized_count == 0
    assert backend.planner_segment_compacted_count == 2
    assert backend.planner_segment_last_decision.startswith("short_segments_compacted")


def test_request_state_coarsens_mid_length_short_segments_when_enabled():
    backend = _PlannerOnlyBackend(
        num_dpus=16,
        resident_store_backend="host",
        head_grouping_policy="balanced",
        compact_short_segments_enabled=True,
        compact_short_segment_min_tokens=16,
    )
    backend.resident_store = _RecordingHostStore()
    state = backend._build_request_state(
        "req_mid",
        _dummy_initial_kv(seq_len=60, num_heads=2, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    group = state.layer_states[0].head_groups[0]
    assert len(group.physical_dpus or []) == 3
    assert [(segment["token_range_start"], segment["token_range_end"]) for segment in (group.token_segments or [])] == [
        (0, 16),
        (16, 32),
        (32, 60),
    ]

    first_call = backend.resident_store.allocate_calls[0]
    assert len(first_call["segment_plan"]) == 3
    assert backend.planner_segment_materialized_count == 1
    assert backend.planner_segment_compacted_count == 1
    assert backend.planner_segment_last_decision.startswith("short_segments_coarsened")


def test_request_state_keeps_large_planner_token_segments_when_compaction_enabled():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
        compact_short_segments_enabled=True,
        compact_short_segment_min_tokens=32,
    )
    backend.resident_store = _RecordingHostStore()
    state = backend._build_request_state(
        "req_large",
        _dummy_initial_kv(seq_len=128, num_heads=2, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    group = state.layer_states[0].head_groups[0]
    assert sorted(group.physical_dpus or []) == [0, 1, 2, 3]
    assert [(segment["token_range_start"], segment["token_range_end"]) for segment in (group.token_segments or [])] == [
        (0, 32),
        (32, 64),
        (64, 96),
        (96, 128),
    ]
    first_call = backend.resident_store.allocate_calls[0]
    assert [(item["token_range_start"], item["token_range_end"]) for item in first_call["segment_plan"]] == [
        (0, 32),
        (32, 64),
        (64, 96),
        (96, 128),
    ]
    assert backend.planner_segment_materialized_count == 1
    assert backend.planner_segment_compacted_count == 0


def test_request_state_does_not_compact_normal_length_segments_by_default():
    backend = _PlannerOnlyBackend(
        num_dpus=32,
        resident_store_backend="host",
        head_grouping_policy="balanced",
        compact_short_segments_enabled=True,
        compact_short_segment_min_tokens=16,
    )
    backend.resident_store = _RecordingHostStore()
    state = backend._build_request_state(
        "req_norm",
        _dummy_initial_kv(seq_len=143, num_heads=12, head_dim=64),
        decode_reserve_tokens=2,
    )

    group0 = state.layer_states[0].head_groups[0]
    assert group0.token_segments
    assert backend.planner_segment_materialized_count > 0
    assert backend.planner_segment_compacted_count == 0
    assert backend.resident_store.allocate_calls[0]["segment_plan"]


def test_append_refreshes_group_token_segments_from_resident_store():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    state = backend._build_request_state(
        "req2",
        _dummy_initial_kv(seq_len=6, num_heads=12, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    layer0 = state.layer_states[0]
    group0 = layer0.head_groups[0]
    assert group0.token_segments == [
        {
            "physical_dpu": int(group0.dpu_id),
            "token_range_start": 0,
            "token_range_end": 6,
        }
    ]

    k_new = torch.randn(12, 8)
    v_new = torch.randn(12, 8)
    backend._append_resident_kv(state, 0, k_new, v_new)

    group0 = state.layer_states[0].head_groups[0]
    assert group0.seq_len == 7
    assert group0.token_segments == [
        {
            "physical_dpu": int(group0.dpu_id),
            "token_range_start": 0,
            "token_range_end": 7,
        }
    ]
    resident_slot = backend.resident_store.slot_debug(group0.k_slot, group0.v_slot)
    assert [(item["token_range_start"], item["token_range_end"]) for item in resident_slot["segments"]] == [(0, 7)]
