#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../cluster_env.sh"

NUM_DPUS="${NUM_DPUS:-4}"
LENGTH="${LENGTH:-128}"
REMOTE_DIR="${PROJECT_DIR}/src/pim/upmem_dot"

ssh_remote "${ATTENTION_IP}" "cd ${REMOTE_DIR} && make clean >/dev/null && make run NUM_DPUS=${NUM_DPUS} LENGTH=${LENGTH}"
