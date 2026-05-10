#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

HEAD_START_MODE="${HEAD_START_MODE:-auto}"

start_head_local() {
  RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} start --head \
    --node-ip-address=${HEAD_IP} \
    --port=${RAY_PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --num-gpus=1 \
    --resources='{\"prefill_gpu\": 1}'"
}

start_head_via_ssh() {
  ssh_remote "${HEAD_IP}" "cd ${PROJECT_DIR} && RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${RAY_CMD} start --head \
    --node-ip-address=${HEAD_IP} \
    --port=${RAY_PORT} \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=${DASHBOARD_PORT} \
    --num-gpus=1 \
    --resources='{\"prefill_gpu\": 1}'"
}

wait_for_gcs() {
  local attempt
  for attempt in $(seq 1 15); do
    if "${PYTHON_BIN}" - "${HEAD_IP}" "${RAY_PORT}" <<'PY'
import socket
import sys

host = sys.argv[1]
port = int(sys.argv[2])
sock = socket.socket()
sock.settimeout(1.0)
try:
    sock.connect((host, port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
PY
    then
      return 0
    fi
    sleep 1
  done

  echo "Timed out waiting for Ray GCS on ${RAY_ADDRESS}" >&2
  return 1
}

echo "Stopping any existing Ray processes..."
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} stop --force" >/dev/null 2>&1 || true
if [[ "${DECODE_DENSE_IP}" != "${HEAD_IP}" ]]; then
  ssh_remote "${DECODE_DENSE_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${DECODE_DENSE_RAY_CMD} stop --force >/dev/null 2>&1 || true"
fi
if [[ "${PREFILL_IP}" != "${HEAD_IP}" && "${PREFILL_IP}" != "${DECODE_DENSE_IP}" ]]; then
  ssh_remote "${PREFILL_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${PREFILL_RAY_CMD} stop --force >/dev/null 2>&1 || true"
fi
ssh_remote "${ATTENTION_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${ATTENTION_RAY_CMD} stop --force >/dev/null 2>&1 || true"

echo "Starting Ray head on ${HEAD_IP}..."
head_started_via="local"
if [[ "${HEAD_START_MODE}" == "ssh" ]]; then
  start_head_via_ssh
  head_started_via="ssh"
elif [[ "${HEAD_START_MODE}" == "local" ]]; then
  start_head_local
else
  if ssh_remote "${HEAD_IP}" "true" >/dev/null 2>&1; then
    start_head_via_ssh
    head_started_via="ssh"
  else
    start_head_local
  fi
fi

wait_for_gcs

echo "Starting decode dense GPU worker on ${DECODE_DENSE_IP}..."
ssh_remote "${DECODE_DENSE_IP}" "cd ${PROJECT_DIR} && RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${DECODE_DENSE_RAY_CMD} start \
  --address=${RAY_ADDRESS} \
  --node-ip-address=${DECODE_DENSE_IP} \
  --num-gpus=1 \
  --resources='{\"decode_dense_gpu\": 1}'"

echo "Starting attention CPU/PIM worker on ${ATTENTION_IP}..."
ssh_remote "${ATTENTION_IP}" "cd ${PROJECT_DIR} && RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${ATTENTION_RAY_CMD} start \
  --address=${RAY_ADDRESS} \
  --node-ip-address=${ATTENTION_IP} \
  --num-gpus=0 \
  --resources='{\"attention_pim\": 1}'"

echo "Ray cluster started via head=${head_started_via}. Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} status --address=${RAY_ADDRESS}" || true
