import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.attention_backend import PimNaiveAttentionBackend
from src.core.clover_attention_backend import CloverInferAttentionBackend
from src.core.resident_kv_store import HostResidentKVStore


class _PlannerOnlyBackend(PimNaiveAttentionBackend):
    def _run_dot_smoke_test(self) -> None:
        self.smoke_test_ok = True
        self.smoke_test_output = "skipped"


class _PlannerOnlyCloverBackend(CloverInferAttentionBackend):
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


class _RankedRecordingHostStore(_RecordingHostStore):
    def __init__(self, rank_groups):
        super().__init__()
        self.rank_groups = [[int(dpu) for dpu in group] for group in rank_groups]
        self.rank_by_dpu = {
            int(dpu): int(rank_idx)
            for rank_idx, group in enumerate(self.rank_groups)
            for dpu in group
        }

    def get_rank_groups(self):
        return [list(group) for group in self.rank_groups]

    def _ensure_topology_cache(self):
        return None

    def _topology_rank_index(self, physical_dpu):
        return self.rank_by_dpu.get(int(physical_dpu))


class _RecoveringHostStore(HostResidentKVStore):
    def __init__(self):
        super().__init__()
        self.failed_once = False
        self.migrated_slots = []

    def append_group(self, k_slot, v_slot, k_new, v_new):
        key = self._slot_key(k_slot, v_slot)
        if "group1" in str(k_slot) and key not in self.migrated_slots and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("synthetic append failure")
        return super().append_group(k_slot, v_slot, k_new, v_new)

    def migrate_group_to_host_fallback(self, k_slot, v_slot):
        key = self._slot_key(k_slot, v_slot)
        self.migrated_slots.append(key)
        return {
            "backend": "host_fallback",
            "storage": "host_fallback",
            "migrated": True,
        }


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


def test_resident_append_recovers_group_failure_without_seq_len_skew():
    backend = _PlannerOnlyBackend(
        num_dpus=4,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    recovering_store = _RecoveringHostStore()
    backend.resident_store = recovering_store
    state = backend._build_request_state(
        "req_recover",
        _dummy_initial_kv(seq_len=4, num_heads=8, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    backend._append_resident_kv(
        state,
        0,
        torch.randn(8, 8),
        torch.randn(8, 8),
    )

    assert recovering_store.failed_once is True
    assert len(recovering_store.migrated_slots) == 1
    assert state.context_len == 5
    assert {group.seq_len for group in state.layer_states[0].head_groups} == {5}


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


def test_rank_spread_alloc_keeps_rank_local_initial_stripe_by_default():
    backend = _PlannerOnlyBackend(
        num_dpus=8,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    backend.pim_rank_spread_alloc_experimental_enabled = True
    backend.pim_layer_rank_rotation_experimental_enabled = True
    backend.resident_store = _RankedRecordingHostStore(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    state = backend._build_request_state(
        "req_rank_spread",
        _dummy_initial_kv(seq_len=64, num_heads=2, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    rank_by_dpu = {
        int(dpu): int(rank_idx)
        for rank_idx, group in enumerate(backend.resident_store.rank_groups)
        for dpu in group
    }
    stripe_ranks = {rank_by_dpu[int(dpu)] for dpu in state.preferred_dpu_stripe}
    assert len(stripe_ranks) == 1
    assert backend.init_rank_last_reason != "cross_rank_stripe_experimental"


def test_cross_rank_stripe_experiment_prefers_cross_rank_initial_stripe():
    backend = _PlannerOnlyBackend(
        num_dpus=8,
        resident_store_backend="host",
        head_grouping_policy="balanced",
    )
    backend.pim_rank_spread_alloc_experimental_enabled = True
    backend.pim_cross_rank_stripe_experimental_enabled = True
    backend.resident_store = _RankedRecordingHostStore(
        [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
        ]
    )

    state = backend._build_request_state(
        "req_rank_spread",
        _dummy_initial_kv(seq_len=64, num_heads=2, head_dim=8, num_layers=1),
        decode_reserve_tokens=2,
    )

    rank_by_dpu = {
        int(dpu): int(rank_idx)
        for rank_idx, group in enumerate(backend.resident_store.rank_groups)
        for dpu in group
    }
    stripe_ranks = {rank_by_dpu[int(dpu)] for dpu in state.preferred_dpu_stripe}
    assert len(stripe_ranks) > 1
    assert backend.init_rank_last_reason == "cross_rank_stripe_experimental"


def test_sparse_tail_resident_init_does_not_expand_cross_rank_stripe_to_full_prompt_width():
    backend = _PlannerOnlyBackend(
        num_dpus=32,
        length=128,
        block_tokens=256,
        resident_store_backend="host",
        head_grouping_policy="coarse",
        attention_sparse_window=512,
    )
    backend.pim_cross_rank_stripe_experimental_enabled = True
    backend.resident_store = _RankedRecordingHostStore(
        [
            list(range(0, 8)),
            list(range(8, 16)),
            list(range(16, 24)),
            list(range(24, 32)),
        ]
    )

    state = backend._build_request_state(
        "req_sparse_tail_stripe",
        _dummy_initial_kv(seq_len=512, num_heads=12, head_dim=64, num_layers=1),
        decode_reserve_tokens=4,
        logical_context_len=1536,
    )

    assert len(state.preferred_dpu_stripe) == 16
    assert state.logical_context_len == 1536
    assert state.context_len == 512

    state.context_len = 520
    state.logical_context_len = 1544
    backend._maybe_expand_request_stripe(state)

    assert len(state.preferred_dpu_stripe) == 16
    assert state.stripe_expand_count == 0


def test_sparse_tail_stripe_width_can_be_overridden_for_sweeps():
    previous = os.environ.get("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH")
    os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = "8"
    try:
        backend = _PlannerOnlyBackend(
            num_dpus=32,
            length=128,
            block_tokens=256,
            resident_store_backend="host",
            head_grouping_policy="coarse",
            attention_sparse_window=512,
        )
        backend.pim_cross_rank_stripe_experimental_enabled = True
        backend.resident_store = _RankedRecordingHostStore(
            [
                list(range(0, 8)),
                list(range(8, 16)),
                list(range(16, 24)),
                list(range(24, 32)),
            ]
        )

        state = backend._build_request_state(
            "req_sparse_tail_stripe_override",
            _dummy_initial_kv(seq_len=512, num_heads=12, head_dim=64, num_layers=1),
            decode_reserve_tokens=4,
            logical_context_len=1536,
        )

        assert len(state.preferred_dpu_stripe) == 8
    finally:
        if previous is None:
            os.environ.pop("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH", None)
        else:
            os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = previous


def test_sparse_tail_stripe_width_tracks_expected_decode_batch_size():
    previous = os.environ.get("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH")
    os.environ.pop("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH", None)
    try:
        backend = _PlannerOnlyBackend(
            num_dpus=32,
            length=128,
            block_tokens=256,
            resident_store_backend="host",
            head_grouping_policy="coarse",
            attention_sparse_window=128,
            expected_decode_batch_max_size=4,
        )
        backend.pim_cross_rank_stripe_experimental_enabled = True
        backend.resident_store = _RankedRecordingHostStore(
            [[rank_idx, rank_idx + 16] for rank_idx in range(16)]
        )

        state = backend._build_request_state(
            "req_sparse_tail_auto_c4",
            _dummy_initial_kv(seq_len=128, num_heads=12, head_dim=64, num_layers=1),
            decode_reserve_tokens=4,
            logical_context_len=1536,
        )

        assert len(state.preferred_dpu_stripe) == 8
        assert backend.sparse_tail_last_stripe_policy["mode"] == "auto"
        assert backend.sparse_tail_last_stripe_policy["target_width"] == 8

        backend.expected_decode_batch_max_size = 8
        assert backend._sparse_tail_stripe_target_width(128) == 4
        assert backend.sparse_tail_last_stripe_policy["concurrency_cap"] == 4
    finally:
        if previous is None:
            os.environ.pop("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH", None)
        else:
            os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = previous


def test_cross_rank_sparse_stripes_avoid_live_request_overlap():
    previous = os.environ.get("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH")
    os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = "16"
    try:
        backend = _PlannerOnlyBackend(
            num_dpus=32,
            length=128,
            block_tokens=256,
            resident_store_backend="host",
            head_grouping_policy="coarse",
            attention_sparse_window=128,
        )
        backend.pim_cross_rank_stripe_experimental_enabled = True
        backend.resident_store = _RankedRecordingHostStore(
            [[rank_idx, rank_idx + 16] for rank_idx in range(16)]
        )

        first = backend._build_request_state(
            "req_sparse_overlap_a",
            _dummy_initial_kv(seq_len=128, num_heads=12, head_dim=64, num_layers=1),
            decode_reserve_tokens=4,
            logical_context_len=1536,
        )
        backend.request_states[first.request_id] = first
        second = backend._build_request_state(
            "req_sparse_overlap_b",
            _dummy_initial_kv(seq_len=128, num_heads=12, head_dim=64, num_layers=1),
            decode_reserve_tokens=4,
            logical_context_len=1536,
        )

        assert len(first.preferred_dpu_stripe) == 16
        assert len(second.preferred_dpu_stripe) == 16
        assert set(first.preferred_dpu_stripe).isdisjoint(set(second.preferred_dpu_stripe))
    finally:
        if previous is None:
            os.environ.pop("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH", None)
        else:
            os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = previous


def test_cross_rank_sparse_stripes_pack_into_aligned_rank_blocks():
    previous = os.environ.get("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH")
    os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = "8"
    try:
        backend = _PlannerOnlyBackend(
            num_dpus=32,
            length=128,
            block_tokens=256,
            resident_store_backend="host",
            head_grouping_policy="coarse",
            attention_sparse_window=128,
        )
        backend.pim_cross_rank_stripe_experimental_enabled = True
        backend.resident_store = _RankedRecordingHostStore(
            [[rank_idx, rank_idx + 16] for rank_idx in range(16)]
        )

        states = []
        for request_idx in range(4):
            state = backend._build_request_state(
                f"req_sparse_block_{request_idx}",
                _dummy_initial_kv(seq_len=128, num_heads=12, head_dim=64, num_layers=1),
                decode_reserve_tokens=4,
                logical_context_len=1536,
            )
            backend.request_states[state.request_id] = state
            states.append(state)

        stripe_sets = [{int(dpu) for dpu in state.preferred_dpu_stripe} for state in states]
        expected_blocks = [
            {0, 1, 2, 3, 16, 17, 18, 19},
            {4, 5, 6, 7, 20, 21, 22, 23},
            {8, 9, 10, 11, 24, 25, 26, 27},
            {12, 13, 14, 15, 28, 29, 30, 31},
        ]

        assert sorted(stripe_sets, key=lambda item: min(item)) == expected_blocks
    finally:
        if previous is None:
            os.environ.pop("CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH", None)
        else:
            os.environ["CLOVER_PIM_SPARSE_TAIL_STRIPE_WIDTH"] = previous


def test_coarse_rank_spread_rotates_layers_across_rank_local_stripes():
    backend = _PlannerOnlyBackend(
        num_dpus=8,
        resident_store_backend="host",
        head_grouping_policy="coarse",
        dpu_placement_policy="rank_spread",
    )
    backend.pim_rank_spread_alloc_experimental_enabled = True
    backend.pim_layer_rank_rotation_experimental_enabled = True
    backend.resident_store = _RankedRecordingHostStore(
        [
            [0, 1],
            [2, 3],
            [4, 5],
            [6, 7],
        ]
    )

    state = backend._build_request_state(
        "req_layer_rank_rotation",
        _dummy_initial_kv(seq_len=64, num_heads=4, head_dim=8, num_layers=4),
        decode_reserve_tokens=2,
    )

    rank_by_dpu = {
        int(dpu): int(rank_idx)
        for rank_idx, group in enumerate(backend.resident_store.rank_groups)
        for dpu in group
    }
    layer_ranks = []
    for layer_state in state.layer_states:
        group_ranks = {
            rank_by_dpu[int(dpu)]
            for group in layer_state.head_groups
            for dpu in list(group.physical_dpus or [group.dpu_id])
        }
        assert len(group_ranks) == 1
        layer_ranks.append(next(iter(group_ranks)))

    assert len(set(layer_ranks)) > 1
    assert backend.layer_rank_rotation_count > 0


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


def test_clover_perf_guard_routes_compressed_kv_to_cpu_fast_path():
    backend = _PlannerOnlyCloverBackend(
        num_dpus=4,
        resident_store_backend="host",
        resident_kv_dtype="int8",
        pim_attention_enabled=True,
        pim_perf_guard_enabled=True,
        cpu_shadow_enabled=False,
        shadow_checks_enabled=False,
    )
    assert backend.cpu_shadow_enabled is True
    assert backend.pim_perf_guard_triggered is True
    assert backend.pim_perf_guard_reason == "compressed_resident_kv_dtype:int8"

    seq_len = backend.init_request(
        "req_guard_int8",
        _dummy_initial_kv(seq_len=8, num_heads=2, head_dim=4, num_layers=1),
        decode_reserve_tokens=2,
    )

    assert seq_len == 8
    assert backend._use_cpu_fast_path_for_request("req_guard_int8")
    assert "req_guard_int8" not in backend.request_states
    debug = backend.get_debug_info()
    assert debug["clover_pim_perf_guard_triggered"] is True
    assert debug["clover_pim_perf_guard_forced_request_count"] == 1


def test_clover_perf_guard_can_trigger_from_observed_slowdown():
    backend = _PlannerOnlyCloverBackend(
        num_dpus=4,
        resident_store_backend="host",
        pim_attention_enabled=False,
        pim_perf_guard_enabled=True,
        pim_perf_guard_slowdown_threshold=1.1,
        shadow_checks_enabled=False,
    )
    backend.pim_attention_enabled = True
    backend.init_request(
        "req_guard_probe",
        _dummy_initial_kv(seq_len=8, num_heads=2, head_dim=4, num_layers=1),
        decode_reserve_tokens=2,
    )
    record = {
        "request_id": "req_guard_probe",
        "keys": torch.randn(8, 2, 4),
        "values": torch.randn(8, 2, 4),
        "q_fp32": torch.randn(2, 4),
        "score_scale": 1.0,
    }

    backend._maybe_trigger_pim_perf_guard(
        [record],
        cpu_probe_s=0.001,
        pim_observed_s=0.002,
    )

    assert backend.pim_perf_guard_triggered is True
    assert backend._use_cpu_fast_path_for_request("req_guard_probe")
    assert backend.pim_perf_guard_decode_observations == 1
    assert "observed_pim_attention_slowdown" in backend.pim_perf_guard_reason
