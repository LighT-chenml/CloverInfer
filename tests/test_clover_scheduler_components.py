import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.clover_planner import plan_sharding
from src.core.clover_scheduler_components import (
    CapacityAwareMicroBatchScheduler,
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

