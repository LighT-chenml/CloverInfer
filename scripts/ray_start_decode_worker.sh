#!/usr/bin/env bash
set -euo pipefail

source /home/cml/miniconda3/etc/profile.d/conda.sh
conda activate clover_infer
cd /home/cml/CloverInfer

RAY_DEFAULT_PYTHON_VERSION_MATCH_LEVEL=minor \
python -m ray.scripts.scripts start \
  --address=192.168.123.4:26379 \
  --node-ip-address=192.168.123.3 \
  --num-gpus=1 \
  --resources='{"decode_dense_gpu": 1}'
