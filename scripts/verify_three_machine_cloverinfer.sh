#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

MODEL_PATH="${MODEL_PATH:-/home/cml/CloverInfer/model/opt-125m}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2}"
PIM_NUM_DPUS="${PIM_NUM_DPUS:-4}"
PIM_LENGTH="${PIM_LENGTH:-128}"
PIM_MAX_TOKENS_PER_DPU="${PIM_MAX_TOKENS_PER_DPU:-256}"

"${PYTHON_BIN}" "${REPO_ROOT}/tests/verify_cluster_placement.py" \
  --address "${RAY_ADDRESS}" \
  --model "${MODEL_PATH}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --attention-backend cloverinfer \
  --pim-num-dpus "${PIM_NUM_DPUS}" \
  --pim-resident-store-backend upmem_kvslot \
  --pim-length "${PIM_LENGTH}" \
  --clover-predictive-scheduling-enabled \
  --clover-capacity-aware-batching-enabled \
  --clover-capacity-aware-lookahead-window 4 \
  --clover-capacity-aware-time-gap-threshold 0.01 \
  --clover-capacity-aware-max-tokens-per-dpu "${PIM_MAX_TOKENS_PER_DPU}" \
  --expected-prefill-ip "${PREFILL_IP}" \
  --expected-dense-ip "${DECODE_DENSE_IP}" \
  --expected-attention-ip "${ATTENTION_IP}"
