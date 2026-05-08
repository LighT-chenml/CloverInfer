#!/usr/bin/env bash

HEAD_IP="${HEAD_IP:-192.168.123.4}"
PREFILL_IP="${PREFILL_IP:-192.168.123.4}"
DECODE_DENSE_IP="${DECODE_DENSE_IP:-192.168.123.3}"
ATTENTION_IP="${ATTENTION_IP:-192.168.123.7}"
USER_NAME="${USER_NAME:-cml}"
PROJECT_DIR="${PROJECT_DIR:-/home/cml/CloverInfer}"

# Keep all cluster-side Python imports inside the resolved conda envs. This
# avoids silently mixing in ~/.local site-packages on the head node.
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"

# These may be set explicitly by the caller. When unset, we auto-detect a
# usable clover_infer env on each machine instead of assuming one global path.
CONDA_PREFIX_HEAD="${CONDA_PREFIX_HEAD:-}"
CONDA_PREFIX_PREFILL="${CONDA_PREFIX_PREFILL:-}"
CONDA_PREFIX_DECODE_DENSE="${CONDA_PREFIX_DECODE_DENSE:-}"
CONDA_PREFIX_ATTENTION="${CONDA_PREFIX_ATTENTION:-}"

RAY_PORT="${RAY_PORT:-26379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"
RAY_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL:-minor}"

RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}"

ssh_remote() {
  local host="$1"
  shift
  ssh -o BatchMode=yes -o ConnectTimeout=8 "${USER_NAME}@${host}" "$@"
}

resolve_local_conda_prefix() {
  local requested_prefix="${1:-}"
  shift || true
  local candidates=()
  if [[ -n "${requested_prefix}" ]]; then
    candidates+=("${requested_prefix}")
  fi
  candidates+=("$@")

  local prefix
  for prefix in "${candidates[@]}"; do
    if [[ -x "${prefix}/bin/python" ]]; then
      printf '%s' "${prefix}"
      return 0
    fi
  done

  echo "Unable to find a local clover_infer env among: ${candidates[*]}" >&2
  return 1
}

resolve_remote_conda_prefix() {
  local host="$1"
  local requested_prefix="${2:-}"
  shift 2 || true

  local candidates=()
  if [[ -n "${requested_prefix}" ]]; then
    candidates+=("${requested_prefix}")
  fi
  candidates+=("$@")
  local quoted_candidates=()
  local prefix
  for prefix in "${candidates[@]}"; do
    quoted_candidates+=("$(printf '%q' "${prefix}")")
  done

  ssh_remote "${host}" "bash -s -- ${quoted_candidates[*]}" <<'EOF'
set -e
for prefix in "$@"; do
  if [[ -x "${prefix}/bin/python" ]]; then
    printf '%s' "${prefix}"
    exit 0
  fi
done
echo "Unable to find a remote clover_infer env among: $*" >&2
exit 1
EOF
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
  local quoted_prefix
  quoted_prefix="$(printf '%q' "${conda_prefix}")"
  ssh_remote "${host}" "bash -s -- ${quoted_prefix}" <<'EOF'
set -e
conda_prefix="$1"
ray_bin="${conda_prefix}/bin/ray"
python_bin="${conda_prefix}/bin/python"
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
  exit 1
fi
EOF
}

CONDA_PREFIX_HEAD="$(
  resolve_local_conda_prefix \
    "${CONDA_PREFIX_HEAD}" \
    "/home/${USER_NAME}/anaconda3/envs/clover_infer" \
    "/home/${USER_NAME}/miniconda3/envs/clover_infer"
)"
if [[ "${PREFILL_IP}" == "${HEAD_IP}" ]]; then
  CONDA_PREFIX_PREFILL="${CONDA_PREFIX_HEAD}"
else
  CONDA_PREFIX_PREFILL="$(
    resolve_remote_conda_prefix \
      "${PREFILL_IP}" \
      "${CONDA_PREFIX_PREFILL}" \
      "/home/${USER_NAME}/anaconda3/envs/clover_infer" \
      "/home/${USER_NAME}/miniconda3/envs/clover_infer"
  )"
fi

if [[ "${DECODE_DENSE_IP}" == "${HEAD_IP}" ]]; then
  CONDA_PREFIX_DECODE_DENSE="${CONDA_PREFIX_HEAD}"
else
  CONDA_PREFIX_DECODE_DENSE="$(
    resolve_remote_conda_prefix \
      "${DECODE_DENSE_IP}" \
      "${CONDA_PREFIX_DECODE_DENSE}" \
      "/home/${USER_NAME}/anaconda3/envs/clover_infer" \
      "/home/${USER_NAME}/miniconda3/envs/clover_infer"
  )"
fi

if [[ "${ATTENTION_IP}" == "${HEAD_IP}" ]]; then
  CONDA_PREFIX_ATTENTION="${CONDA_PREFIX_HEAD}"
else
  CONDA_PREFIX_ATTENTION="$(
    resolve_remote_conda_prefix \
      "${ATTENTION_IP}" \
      "${CONDA_PREFIX_ATTENTION}" \
      "/home/${USER_NAME}/anaconda3/envs/clover_infer" \
      "/home/${USER_NAME}/miniconda3/envs/clover_infer"
  )"
fi

PYTHON_BIN="${CONDA_PREFIX_HEAD}/bin/python"
PREFILL_PYTHON_BIN="${CONDA_PREFIX_PREFILL}/bin/python"
DECODE_DENSE_PYTHON_BIN="${CONDA_PREFIX_DECODE_DENSE}/bin/python"
ATTENTION_PYTHON_BIN="${CONDA_PREFIX_ATTENTION}/bin/python"

RAY_CMD="${RAY_CMD:-$(resolve_ray_bin "${CONDA_PREFIX_HEAD}")}"
if [[ "${PREFILL_IP}" == "${HEAD_IP}" ]]; then
  PREFILL_RAY_CMD="${PREFILL_RAY_CMD:-${RAY_CMD}}"
else
  PREFILL_RAY_CMD="${PREFILL_RAY_CMD:-$(resolve_remote_ray_bin "${PREFILL_IP}" "${CONDA_PREFIX_PREFILL}")}"
fi

if [[ "${DECODE_DENSE_IP}" == "${HEAD_IP}" ]]; then
  DECODE_DENSE_RAY_CMD="${DECODE_DENSE_RAY_CMD:-${RAY_CMD}}"
else
  DECODE_DENSE_RAY_CMD="${DECODE_DENSE_RAY_CMD:-$(resolve_remote_ray_bin "${DECODE_DENSE_IP}" "${CONDA_PREFIX_DECODE_DENSE}")}"
fi

if [[ "${ATTENTION_IP}" == "${HEAD_IP}" ]]; then
  ATTENTION_RAY_CMD="${ATTENTION_RAY_CMD:-${RAY_CMD}}"
else
  ATTENTION_RAY_CMD="${ATTENTION_RAY_CMD:-$(resolve_remote_ray_bin "${ATTENTION_IP}" "${CONDA_PREFIX_ATTENTION}")}"
fi
