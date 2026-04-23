#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../cluster_env.sh"

NUM_DPUS="${NUM_DPUS:-2}"
HEAD_DIM="${HEAD_DIM:-64}"
KEYS_PER_DPU="${KEYS_PER_DPU:-8}"
REMOTE_DIR="${PROJECT_DIR}/src/pim/upmem_qk"

ssh_remote "${ATTENTION_IP}" "cd ${REMOTE_DIR} && make clean >/dev/null && make run NUM_DPUS=${NUM_DPUS} HEAD_DIM=${HEAD_DIM} KEYS_PER_DPU=${KEYS_PER_DPU}"
