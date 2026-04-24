#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/cml/anaconda3/envs/clover_infer/bin/python}"

cleanup() {
  bash "$REPO_ROOT/scripts/cluster_stop.sh" || true
}
trap cleanup EXIT

bash "$REPO_ROOT/scripts/cluster_start.sh"
sleep 8

"$PYTHON_BIN" "$REPO_ROOT/tests/benchmark_attention_sweep.py" "$@"
