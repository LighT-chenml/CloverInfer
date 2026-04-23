#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

echo "Stopping Ray on ${PREFILL_IP}..."
ssh_remote "${PREFILL_IP}" "${PREFILL_RAY_BIN} stop --force >/dev/null 2>&1 || true"

echo "Stopping Ray on ${ATTENTION_IP}..."
ssh_remote "${ATTENTION_IP}" "${ATTENTION_RAY_BIN} stop --force >/dev/null 2>&1 || true"

echo "Stopping Ray head on ${HEAD_IP}..."
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true

echo "Ray cluster stopped."
