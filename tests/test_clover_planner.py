import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.core.clover_planner import plan_sharding, update_sharding


def test_plan_sharding_balances_loads():
    requests = [
        {"request_id": "r0", "seq_len": 16},
        {"request_id": "r1", "seq_len": 8},
        {"request_id": "r2", "seq_len": 4},
    ]
    plan = plan_sharding(requests, D=8, H=2)
    assert plan["num_dpus"] == 8
    assert plan["num_heads"] == 2
    assert len(plan["dpu_groups"]) == 2
    assert len(plan["dpu_loads"]) == 8
    assert plan["load_stddev"] >= 0.0

    for head_id, request_map in plan["per_head_shards"].items():
        for request in requests:
            shards = request_map[request["request_id"]]
            covered = sum(int(item["token_count"]) for item in shards)
            assert covered == int(request["seq_len"])


def test_update_sharding_replans_with_new_requests():
    base = plan_sharding([{"request_id": "base", "seq_len": 12}], D=4, H=2)
    updated = update_sharding(base, [{"request_id": "new", "seq_len": 6}])
    assert updated["metadata"]["updated_from_existing"] is True
    for head_id, request_map in updated["per_head_shards"].items():
        assert "base" in request_map
        assert "new" in request_map

