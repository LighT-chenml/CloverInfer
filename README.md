# CloverInfer 🍀

CloverInfer is a research-oriented LLM inference framework for experimenting with disaggregated serving. It separates prefill from decoding, and splits decoding into a dense GPU node and an attention node that can initially run on CPU and later be replaced by a PIM backend.

The current correctness-first path uses **Ray** for orchestration and transport. RDMA and PIM integration are planned optimization layers after the Ray path is stable.

## Key Features 🚀

### 1. Disaggregated Architecture
*   **Prefill-Decoding Separation**: Dedicated `PrefillNode` actors handle prompt processing and initial KV generation.
*   **Dense-Attention Decode Split**: `DecodeDenseNode` handles embeddings, QKV projection, output projection, FFN, and LM head. `AttentionNode` owns KV cache and attention computation.
*   **PIM-Ready Attention Backend**: The attention node currently uses a CPU reference backend. A PIM backend can be added behind the same node interface.

### 2. High-Performance Kernels
*   **PagedAttention**: Includes a custom CUDA implementation of PagedAttention (inspired by vLLM) to minimize memory fragmentation and support large batch sizes.
*   **Memory Management**: A centralized `KVCacheManager` handles block allocation, ensuring efficient GPU memory utilization.

### 3. Advanced Networking
*   **RDMA Integration**: RDMA code remains in `src/network`, but the current end-to-end path intentionally uses Ray transport until correctness and instrumentation are stable.

### 4. Graph Compilation (Experimental)
*   **Automatic Splitter**: Includes a graph compiler (`src/core/graph_compiler.py`) capable of analyzing and splitting PyTorch models into constituent sub-graphs (Attention vs FFN) for distributed execution.

## Directory Structure 📂

```
CloverInfer/
├── src/
│   ├── core/           # Main logic: Scheduler, Nodes, Memory Manager
│   │   ├── nodes.py           # Ray Actor definitions (PrefillNode, AttentionNode, DecodeDenseNode)
│   │   ├── scheduler.py       # Global Scheduler & Orchestrator
│   │   ├── memory_manager.py  # KV Cache management & PagedAttention wrapper
│   │   └── graph_compiler.py  # Model partitioning logic
│   ├── kernels/        # Custom CUDA Kernels
│   │   └── csrc/              # C++/CUDA source for PagedAttention
│   └── network/        # RDMA Networking
│   │   └── csrc/              # C++ source for RDMA transport
├── scripts/            # Helper scripts
└── tests/              # Unit tests and benchmarks
```

## Installation 🛠️

### Prerequisites
*   OS: Linux
*   GPU: NVIDIA GPU with CUDA support (Tested on V100/A100)
*   Drivers: Mellanox OFED (for RDMA support)
*   Software: Python 3.8+, CUDA Toolkit 11.x/12.x

### Build from Source

1.  **Set up Environment**
    ```bash
    conda create -n clover_infer python=3.10 -y
    conda activate clover_infer
    pip install "ray[default]" torch transformers pydantic numpy
    ```

2.  **Build Custom Kernels**
    ```bash
    cd src/kernels
    pip install .
    ```

3.  **Build Network Extensions**
    ```bash
    cd src/network
    pip install .
    ```

## Usage 💡

To use the framework, initialize the `GlobalScheduler` with your cluster configuration.

```python
import ray
from src.core.scheduler import GlobalScheduler
from src.core.config import ClusterConfig, ModelConfig

ray.init()

# Configure Cluster
cluster_conf = ClusterConfig(
    num_prefill_workers=1,
    num_attention_nodes=1,
    num_decode_dense_nodes=1,
)
model_conf = ModelConfig(model_path="/home/cml/CloverInfer/model/opt-125m")

# Initialize Scheduler
scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
ray.get(scheduler.initialize_cluster.remote())

# Submit Request
ray.get(scheduler.submit_request.remote("Hello, CloverInfer!"))
```

## Status ⚠️
This project is currently in a **Research Preview** state.
*   The current default path targets correctness over speed.
*   CPU attention is the reference backend for future PIM integration.
*   RDMA verification scripts are temporarily disabled during the correctness-first refactor.
