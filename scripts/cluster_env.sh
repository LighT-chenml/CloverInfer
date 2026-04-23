#!/usr/bin/env bash

HEAD_IP="${HEAD_IP:-192.168.123.4}"
PREFILL_IP="${PREFILL_IP:-192.168.123.3}"
ATTENTION_IP="${ATTENTION_IP:-192.168.123.7}"
USER_NAME="${USER_NAME:-cml}"
PROJECT_DIR="${PROJECT_DIR:-/home/cml/CloverInfer}"
CONDA_PREFIX_HEAD="${CONDA_PREFIX_HEAD:-/home/cml/anaconda3/envs/clover_infer}"
CONDA_PREFIX_PREFILL="${CONDA_PREFIX_PREFILL:-/home/cml/miniconda3/envs/clover_infer}"
CONDA_PREFIX_ATTENTION="${CONDA_PREFIX_ATTENTION:-/home/cml/miniconda3/envs/clover_infer}"
RAY_PORT="${RAY_PORT:-26379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
RAY_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL:-minor}"

PYTHON_BIN="${CONDA_PREFIX_HEAD}/bin/python"
PREFILL_PYTHON_BIN="${CONDA_PREFIX_PREFILL}/bin/python"
ATTENTION_PYTHON_BIN="${CONDA_PREFIX_ATTENTION}/bin/python"
RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

ssh_remote() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${USER_NAME}@${host}" "$@"
}

resolve_ray_bin() {
  local conda_prefix="$1"
  local python_bin="${conda_prefix}/bin/python"
  local ray_bin="${conda_prefix}/bin/ray"

  if [[ -x "${python_bin}" ]] && "${python_bin}" -m ray.scripts.scripts --version >/dev/null 2>&1; then
    printf '%q -m ray.scripts.scripts' "${python_bin}"
  elif [[ -x "${ray_bin}" ]]; then
    printf '%q' "${ray_bin}"
  elif command -v ray >/dev/null 2>&1; then
    printf '%q' "$(command -v ray)"
  elif [[ -x "${HOME}/.local/bin/ray" ]]; then
    printf '%q' "${HOME}/.local/bin/ray"
  else
    echo "Unable to find a Ray CLI for conda prefix ${conda_prefix}" >&2
    return 1
  fi
}

resolve_remote_ray_bin() {
  local host="$1"
  local conda_prefix="$2"
  ssh_remote "${host}" "bash -lc '
    set -e
    ray_bin=${conda_prefix@Q}/bin/ray
    python_bin=${conda_prefix@Q}/bin/python
    if [[ -x \"\${python_bin}\" ]] && \"\${python_bin}\" -m ray.scripts.scripts --version >/dev/null 2>&1; then
      printf \"%q -m ray.scripts.scripts\" \"\${python_bin}\"
    elif [[ -x \"\${ray_bin}\" ]]; then
      printf \"%q\" \"\${ray_bin}\"
    elif command -v ray >/dev/null 2>&1; then
      printf \"%q\" \"\$(command -v ray)\"
    elif [[ -x \"\${HOME}/.local/bin/ray\" ]]; then
      printf \"%q\" \"\${HOME}/.local/bin/ray\"
    else
      echo \"Unable to find a Ray CLI for conda prefix ${conda_prefix}\" >&2
      exit 1
    fi
  '"
}

RAY_CMD="${RAY_CMD:-$(resolve_ray_bin "${CONDA_PREFIX_HEAD}")}"
PREFILL_RAY_CMD="${PREFILL_RAY_CMD:-$(resolve_remote_ray_bin "${PREFILL_IP}" "${CONDA_PREFIX_PREFILL}")}"
ATTENTION_RAY_CMD="${ATTENTION_RAY_CMD:-$(resolve_remote_ray_bin "${ATTENTION_IP}" "${CONDA_PREFIX_ATTENTION}")}"
