#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL="${RAY_VERSION_MATCH_LEVEL}" bash -lc "${RAY_CMD} status --address=${RAY_ADDRESS}"
