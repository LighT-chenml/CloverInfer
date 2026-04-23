#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-clover_infer}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda or initialize conda first." >&2
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -n "${ENV_NAME}" python=3.10 -y
fi

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install "ray[default]" torch transformers pydantic numpy ninja

echo "Environment '${ENV_NAME}' is ready."
