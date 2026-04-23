#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

echo "Stopping any existing Ray processes..."
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} stop --force" >/dev/null 2>&1 || true
ssh_remote "${PREFILL_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${PREFILL_RAY_CMD} stop --force >/dev/null 2>&1 || true"
ssh_remote "${ATTENTION_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${ATTENTION_RAY_CMD} stop --force >/dev/null 2>&1 || true"

echo "Starting Ray head on ${HEAD_IP}..."
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} start --head \
  --node-ip-address=${HEAD_IP} \
  --port=${RAY_PORT} \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=${DASHBOARD_PORT} \
  --num-gpus=1 \
  --resources='{\"decode_dense_gpu\": 1}'"

echo "Starting prefill GPU worker on ${PREFILL_IP}..."
ssh_remote "${PREFILL_IP}" "cd ${PROJECT_DIR} && RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${PREFILL_RAY_CMD} start \
  --address=${RAY_ADDRESS} \
  --node-ip-address=${PREFILL_IP} \
  --num-gpus=1 \
  --resources='{\"prefill_gpu\": 1}'"

echo "Starting attention CPU/PIM worker on ${ATTENTION_IP}..."
ssh_remote "${ATTENTION_IP}" "cd ${PROJECT_DIR} && RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${ATTENTION_RAY_CMD} start \
  --address=${RAY_ADDRESS} \
  --node-ip-address=${ATTENTION_IP} \
  --num-gpus=0 \
  --resources='{\"attention_pim\": 1}'"

echo "Ray cluster started. Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} status --address=${RAY_ADDRESS}"
