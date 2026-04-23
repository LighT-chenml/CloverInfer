#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

echo "Stopping Ray on ${PREFILL_IP}..."
ssh_remote "${PREFILL_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${PREFILL_RAY_CMD} stop --force >/dev/null 2>&1 || true"

echo "Stopping Ray on ${ATTENTION_IP}..."
ssh_remote "${ATTENTION_IP}" "RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=${RAY_VERSION_MATCH_LEVEL} ${ATTENTION_RAY_CMD} stop --force >/dev/null 2>&1 || true"

echo "Stopping Ray head on ${HEAD_IP}..."
RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} stop --force" >/dev/null 2>&1 || true

echo "Ray cluster stopped."
