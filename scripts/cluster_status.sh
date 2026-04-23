#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/cluster_env.sh"

"${RAY_BIN}" status --address="${RAY_ADDRESS}"
