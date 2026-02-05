import torch
import clover_net
import sys

def test_rdma():
    print("Testing RDMA Transport...")
    
    device_name = "mlx5_0" # Found in previous step
    try:
        ctx = clover_net.RDMAContext(device_name)
    except Exception as e:
        print(f"Skipping test: {e}")
        return

    # Endpoint A
    ep_a = clover_net.RDMAEndpoint(ctx)
    info_a = ep_a.get_info()
    print(f"Endpoint A: QPN={info_a[0]} LID={info_a[1]}")

    # Endpoint B
    ep_b = clover_net.RDMAEndpoint(ctx)
    info_b = ep_b.get_info()
    print(f"Endpoint B: QPN={info_b[0]} LID={info_b[1]}")

    # Connect Loopback
    ep_a.connect(info_b[0], info_b[1])
    ep_b.connect(info_a[0], info_a[1])

    # Buffers
    size = 1024
    tensor_a = torch.arange(size, dtype=torch.float32, device='cpu')
    tensor_b = torch.zeros(size, dtype=torch.float32, device='cpu')

    # Register MR
    mr_a = ep_a.register_mr(tensor_a)
    mr_b = ep_b.register_mr(tensor_b)
    
    print("Memory Registered.")

    # 1. Post Recv on B
    ep_b.post_recv(mr_b, tensor_b)
    
    # 2. Post Send on A
    ep_a.post_send(mr_a, tensor_a)

    # 3. Poll
    print("Polling...")
    ep_a.poll() # Send Completion
    ep_b.poll() # Recv Completion
    
    # 4. Verify
    print("Transfer Complete. Checking data...")
    if torch.allclose(tensor_a, tensor_b):
        print("SUCCESS: Data matches via RDMA Loopback.")
    else:
        print("FAILURE: Mismatch.")
        print("A:", tensor_a[:10])
        print("B:", tensor_b[:10])

if __name__ == "__main__":
    test_rdma()
