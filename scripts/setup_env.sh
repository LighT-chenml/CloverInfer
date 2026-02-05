#!/bin/bash
# setup_env.sh

# Create Conda environment
conda create -n clover_infer python=3.10 -y

# Activate
source activates clover_infer

# Install dependencies
# Ray for distributed scheduling
# PyTorch for tensor ops
# Ninja for JIT compilation of C++ extensions
pip install ray[default] torch ninja pydantic numpy

echo "Environment 'clover_infer' created."
