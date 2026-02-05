# CloverInfer 🍀

CloverInfer is a high-performance, research-oriented LLM inference framework designed for next-generation distributed serving. It features a disaggregated architecture that separates prefill and decoding phases, and further decomposes decoding into fine-grained Attention and Feed-Forward Network (FFN) micro-services.

This framework leverages **Ray** for orchestration, **RDMA** for low-latency interconnects, and custom **CUDA PagedAttention kernels** for efficient memory management.

## Key Features 🚀

### 1. Disaggregated Architecture
*   **Prefill-Decoding Separation**: Dedicated workers (`PrefillWorker`) handle the compute-intensive prefill phase, while specialized nodes handle the memory-bound decoding phase.
*   **Fine-Grained Deployment**: Unlike traditional monolithic pipelines, CloverInfer breaks down the Transformer block. Key components like Attention and FFN are deployed as independent Ray actors (`AttnNode`, `FFNNode`), allowing for flexible resource allocation and pipeline parallelism.

### 2. High-Performance Kernels
*   **PagedAttention**: Includes a custom CUDA implementation of PagedAttention (inspired by vLLM) to minimize memory fragmentation and support large batch sizes.
*   **Memory Management**: A centralized `KVCacheManager` handles block allocation, ensuring efficient GPU memory utilization.

### 3. Advanced Networking
*   **RDMA Integration**: Utilizing IB/RoCE, the framework supports direct peer-to-peer tensor transport between distinct Ray actors (e.g., from `AttnNode` to `FFNNode`), bypassing the overhead of the Ray Object Store for critical data paths.

### 4. Graph Compilation (Experimental)
*   **Automatic Splitter**: Includes a graph compiler (`src/core/graph_compiler.py`) capable of analyzing and splitting PyTorch models into constituent sub-graphs (Attention vs FFN) for distributed execution.

## Directory Structure 📂

```
CloverInfer/
├── src/
│   ├── core/           # Main logic: Scheduler, Nodes, Memory Manager
│   │   ├── nodes.py           # Ray Actor definitions (AttnNode, FFNNode)
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
    conda create -n clover python=3.9
    conda activate clover
    pip install torch torchvision ray
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
cluster_conf = ClusterConfig(num_prefill_workers=1, num_decode_nodes=2)
model_conf = ModelConfig(model_name="llama-2-7b")

# Initialize Scheduler
scheduler = GlobalScheduler.remote(cluster_conf, model_conf)
ray.get(scheduler.initialize_cluster.remote())

# Submit Request
ray.get(scheduler.submit_request.remote("Hello, CloverInfer!"))
```

## Status ⚠️
This project is currently in a **Research Preview** state.
*   Some configurations regarding RDMA devices and memory blocks are hardcoded for specific testbeds (V100).
*   The Graph Splitter uses heuristic partitioning.
