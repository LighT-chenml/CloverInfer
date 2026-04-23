#!/usr/bin/env bash

HEAD_IP="${HEAD_IP:-192.168.123.4}"
PREFILL_IP="${PREFILL_IP:-192.168.123.3}"
ATTENTION_IP="${ATTENTION_IP:-192.168.123.7}"
USER_NAME="${USER_NAME:-cml}"
PROJECT_DIR="${PROJECT_DIR:-/home/cml/CloverInfer}"
CONDA_PREFIX_REMOTE="${CONDA_PREFIX_REMOTE:-/home/cml/miniconda3/envs/clover_infer}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"

RAY_BIN="${CONDA_PREFIX_REMOTE}/bin/ray"
PYTHON_BIN="${CONDA_PREFIX_REMOTE}/bin/python"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

ssh_remote() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${USER_NAME}@${host}" "$@"
}
