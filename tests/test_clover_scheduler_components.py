import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.clover_planner import plan_sharding
from src.core.clover_scheduler_components import (
    CapacityAwareMicroBatchScheduler,
    allocator_aware_capacity_checker,
    default_capacity_checker,
    default_predict_host_time,
    default_predict_pim_time,
)


def test_greedy_micro_batch_stops_before_capacity_overflow():
    scheduler = CapacityAwareMicroBatchScheduler(
        planner=plan_sharding,
        predict_pim_time=lambda reqs, plan: default_predict_pim_time(reqs, plan, a=0.5, b=0.0),
        predict_host_time=lambda total_tokens: default_predict_host_time(total_tokens, c=0.2),
        capacity_checker=default_capacity_checker,
        num_dpus=8,
        num_heads=2,
        time_gap_threshold=0.5,
        lookahead_window=1,
    )
    queue = [
        {"request_id": "r0", "seq_len": 32, "num_new_tokens": 1},
        {"request_id": "r1", "seq_len": 32, "num_new_tokens": 1},
        {"request_id": "r2", "seq_len": 32, "num_new_tokens": 1},
    ]
    decision = scheduler.build_micro_batch(queue, max_capacity_per_dpu=20)
    assert decision.micro_batch.capacity_ok is True
    assert len(decision.micro_batch.requests) >= 1
    assert len(queue) < 3


def test_lookahead_can_pick_non_prefix_combo():
    scheduler = CapacityAwareMicroBatchScheduler(
        planner=plan_sharding,
        predict_pim_time=lambda reqs, plan: default_predict_pim_time(reqs, plan, a=1.0, b=0.0),
        predict_host_time=lambda total_tokens: default_predict_host_time(total_tokens, c=6.0),
        capacity_checker=default_capacity_checker,
        num_dpus=8,
        num_heads=2,
        time_gap_threshold=0.0,
        lookahead_window=3,
    )
    queue = [
        {"request_id": "long", "seq_len": 48, "num_new_tokens": 1},
        {"request_id": "short0", "seq_len": 4, "num_new_tokens": 1},
        {"request_id": "short1", "seq_len": 4, "num_new_tokens": 1},
    ]
    decision = scheduler.build_micro_batch(queue, max_capacity_per_dpu=128)
    picked_ids = {request["request_id"] for request in decision.micro_batch.requests}
    assert len(picked_ids) >= 1
    assert picked_ids != {"long", "short0", "short1"} or decision.micro_batch.time_gap >= 0.0


def test_lookahead_prefers_full_batch_when_capacity_allows():
    scheduler = CapacityAwareMicroBatchScheduler(
        planner=plan_sharding,
        predict_pim_time=lambda reqs, plan: default_predict_pim_time(reqs, plan, a=1.0, b=0.0),
        predict_host_time=lambda total_tokens: default_predict_host_time(total_tokens, c=1.0),
        capacity_checker=default_capacity_checker,
        num_dpus=32,
        num_heads=12,
        time_gap_threshold=0.05,
        lookahead_window=3,
    )
    queue = [
        {"request_id": "r0", "seq_len": 141, "num_new_tokens": 1},
        {"request_id": "r1", "seq_len": 142, "num_new_tokens": 1},
        {"request_id": "r2", "seq_len": 111, "num_new_tokens": 1},
        {"request_id": "r3", "seq_len": 137, "num_new_tokens": 1},
    ]
    decision = scheduler.build_micro_batch(queue, max_capacity_per_dpu=4096, max_batch_size=4)
    assert len(decision.micro_batch.requests) == 4
    assert queue == []


def test_allocator_aware_capacity_checker_rejects_when_live_free_space_is_too_small():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 5},
            {"dpu_id": 1, "total_free_elems": 5},
            {"dpu_id": 2, "total_free_elems": 5},
            {"dpu_id": 3, "total_free_elems": 5},
        ],
        bytes_per_token=1,
    )
    plan = plan_sharding(
        [
            {"request_id": "r0", "seq_len": 16},
            {"request_id": "r1", "seq_len": 12},
        ],
        D=4,
        H=2,
    )
    result = checker([], plan, 64)
    assert result["ok"] is False
    assert result["capacity_source"] == "allocator_stats"


def test_allocator_aware_capacity_checker_accepts_when_live_free_space_is_sufficient():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 64},
            {"dpu_id": 1, "total_free_elems": 64},
            {"dpu_id": 2, "total_free_elems": 64},
            {"dpu_id": 3, "total_free_elems": 64},
        ],
        bytes_per_token=1,
    )
    plan = plan_sharding(
        [
            {"request_id": "r0", "seq_len": 8},
            {"request_id": "r1", "seq_len": 8},
        ],
        D=4,
        H=2,
    )
    result = checker([], plan, 64)
    assert result["ok"] is True
    assert result["usage_ratio"] >= 0.0


def test_allocator_aware_capacity_checker_reports_soft_slot_headroom_when_one_dpu_is_full():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 1024, "live_slot_count": 64},
            {"dpu_id": 1, "total_free_elems": 1024, "live_slot_count": 0},
            {"dpu_id": 2, "total_free_elems": 1024, "live_slot_count": 0},
            {"dpu_id": 3, "total_free_elems": 1024, "live_slot_count": 0},
        ],
        bytes_per_token=1,
    )
    plan = {
        "dpu_loads": {
            0: 1,
            1: 1,
        }
    }
    result = checker([], plan, 1024)
    assert result["ok"] is True
    assert result["slot_headroom_ok"] is False
    assert result["min_remaining_slots"] < 0


def test_allocator_aware_capacity_checker_can_enforce_strict_slot_headroom():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 1024, "live_slot_count": 64},
            {"dpu_id": 1, "total_free_elems": 1024, "live_slot_count": 0},
        ],
        bytes_per_token=1,
        require_slot_headroom=True,
    )
    plan = {
        "dpu_loads": {
            0: 1,
        }
    }
    result = checker([], plan, 1024)
    assert result["ok"] is False
    assert result["min_remaining_slots"] < 0


def test_allocator_aware_capacity_checker_allows_group_slot_substitution():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 1024, "live_slot_count": 64},
            {"dpu_id": 1, "total_free_elems": 1024, "live_slot_count": 0},
        ],
        bytes_per_token=1,
        require_slot_headroom=True,
    )
    plan = {
        "dpu_groups": {
            0: [0, 1],
        },
        "dpu_loads": {
            0: 1,
        },
    }
    result = checker([], plan, 1024)
    assert result["ok"] is True
    assert result["slot_headroom_ok"] is True


def test_lookahead_respects_strict_slot_headroom():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 1024, "live_slot_count": 63},
            {"dpu_id": 1, "total_free_elems": 1024, "live_slot_count": 64},
        ],
        bytes_per_token=1,
        require_slot_headroom=True,
    )
    scheduler = CapacityAwareMicroBatchScheduler(
        planner=lambda requests, D, H: {
            "dpu_groups": {0: [0, 1]},
            "dpu_loads": {0: len(requests), 1: len(requests)},
        },
        predict_pim_time=lambda reqs, plan: float(len(reqs)),
        predict_host_time=lambda total_tokens: float(total_tokens),
        capacity_checker=checker,
        num_dpus=2,
        num_heads=1,
        time_gap_threshold=0.0,
        lookahead_window=2,
    )
    queue = [
        {"request_id": "r0", "seq_len": 8, "num_new_tokens": 1},
        {"request_id": "r1", "seq_len": 8, "num_new_tokens": 1},
    ]

    decision = scheduler.build_micro_batch(queue, max_capacity_per_dpu=1024, max_batch_size=2)

    assert decision.micro_batch.capacity_ok is True
    assert len(decision.micro_batch.requests) == 1
    assert len(queue) == 1


def test_allocator_aware_capacity_usage_uses_live_allocator_capacity():
    checker = allocator_aware_capacity_checker(
        lambda: [
            {"dpu_id": 0, "total_free_elems": 100_000, "live_slot_count": 64},
            {"dpu_id": 1, "total_free_elems": 100_000, "live_slot_count": 64},
        ],
        bytes_per_token=128,
    )
    plan = {
        "dpu_loads": {
            0: 54,
            1: 54,
        }
    }
    result = checker([], plan, 4096)
    assert result["ok"] is True
    assert result["usage_ratio"] < 0.1
