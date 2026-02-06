import time
import asyncio
import ray
from .config import ClusterConfig, ModelConfig
from .nodes import AttnNode, FFNNode, PrefillWorker

@ray.remote
class GlobalScheduler:
    def __init__(self, cluster_config: ClusterConfig, model_config: ModelConfig):
        self.cluster_config = cluster_config
        self.model_config = model_config
        self.prefill_workers = []
        self.attn_nodes = []
        self.ffn_nodes = []
    
    async def initialize_cluster(self):
        print("Initializing Cluster...")
        
        # Check available resources
        resources = ray.available_resources()
        num_gpus_available = resources.get("GPU", 0)
        gpu_per_worker = 0.1 if num_gpus_available > 0 else 0
        
        # Start Prefill Worker
        self.prefill_workers = [
            PrefillWorker.options(num_gpus=gpu_per_worker).remote(i, self.model_config) 
            for i in range(self.cluster_config.num_prefill_workers)
        ]
        
        # Start Decode Nodes (Attn + FFN)
        # Using fractions to share GPU on single node
        self.attn_nodes = [AttnNode.options(num_gpus=gpu_per_worker).remote(0, self.model_config)]
        self.ffn_nodes = [FFNNode.options(num_gpus=gpu_per_worker).remote(0, self.model_config)]
        
        # 1. Initialize RDMA Contexts
        print("Initializing RDMA Contexts...")
        for worker in self.prefill_workers: await worker.init_infra.remote()
        for node in self.attn_nodes: await node.init_infra.remote()
        for node in self.ffn_nodes: await node.init_infra.remote()
            
        # 2. Setup Prefill <-> Attn Mesh (Persistent)
        # For V1: Connect Prefill[0] <-> Attn[0]
        print("Establishing RDMA Connection: Prefill <-> Attn...")
        
        # Create Endpoints
        p_info = await self.prefill_workers[0].create_endpoint.remote() # (qpn, lid)
        a_info_p = await self.attn_nodes[0].create_endpoint.remote()   # (qpn, lid) for Prefill link
        
        if p_info and a_info_p:
            p_qpn, p_lid = p_info
            a_qpn, a_lid = a_info_p
            
            # Connect
            await self.prefill_workers[0].connect_endpoint.remote(p_qpn, a_qpn, a_lid)
            await self.attn_nodes[0].connect_endpoint.remote(a_qpn, p_qpn, p_lid)
            
            # Store Info for Request Dispatch
            # We need to know: When using Prefill[0] and Attn[0], 
            # Prefill uses local QP p_qpn (to send to a_qpn) ?? 
            # Wait, send_kv_rdma takes DEST QPN. 
            # So Prefill needs to know a_qpn.
            # prepare_recv_kv takes SRC QPN.
            # So Attn needs to know p_qpn.
            
            self.rdma_map = {
                "prefill_0_attn_0": {
                    "prefill_qpn": p_qpn, # Source QPN
                    "attn_qpn": a_qpn     # Dest QPN
                }
            }
            print(f"Connected Prefill(QP:{p_qpn}) <-> Attn(QP:{a_qpn})")
        else:
            print("Failed to create endpoints for Prefill-Attn.")
            self.rdma_map = {}

        # 3. Setup Attn <-> FFN Mesh (Persistent)
        # Attn[0] <-> FFN[0]
        print("Establishing RDMA Connection: Attn <-> FFN...")
        a_info_f = await self.attn_nodes[0].create_endpoint.remote() # QP for FFN
        f_info = await self.ffn_nodes[0].create_endpoint.remote()    # QP for Attn
        
        if a_info_f and f_info:
             a_qpn, a_lid = a_info_f
             f_qpn, f_lid = f_info
             
             await self.attn_nodes[0].connect_endpoint.remote(a_qpn, f_qpn, f_lid)
             await self.ffn_nodes[0].connect_endpoint.remote(f_qpn, a_qpn, a_lid)
             print(f"Connected Attn(QP:{a_qpn}) <-> FFN(QP:{f_qpn})")
             
             # Store if needed (AttnNode/FFNNode store peers internally?)
             # They store by remote QPN.
             # AttnNode needs to know FFN QPN to send? 
             # Current code logic for Attn-FFN is not fully visible here but assuming nodes handle it?
             # Actually `AttnNode` doesn't have `send_rdma` to FFN yet?
             # Wait, the task was about KV Transfer. FFN connection was pre-existing?
             # The existing code had:
             # await attn_ref.connect_peer(ffn_info...)
             # My refactor replaced connect_peer with connect_endpoint.
             # So this setup is correct.
        else:
             print("Failed to create endpoints for Attn-FFN.")

        print(f"Started {len(self.prefill_workers)} Prefill, {len(self.attn_nodes)} Attn, {len(self.ffn_nodes)} FFN nodes.")

    async def submit_request(self, prompt: str, return_metrics: bool = False):
        print(f" Scheduler: Rcvd request '{prompt}'")
        
        req_start_time = time.time()
        
        # 1. Dispatch to Prefill
        worker_idx = 0 
        worker = self.prefill_workers[worker_idx]
        
        # Exec Prefill
        prefill_out = await worker.process_prompt.remote(prompt)
        
        req_id = prefill_out["req_id"]
        seq_len = prefill_out["seq_len"]
        first_token = prefill_out["first_token_id"]
        
        print(f"Prefill Done. ReqID: {req_id}, SeqLen: {seq_len}, First Token: {first_token}")
        
        # 2. Handoff to AttnNode via RDMA
        attn_idx = 0
        attn_node = self.attn_nodes[attn_idx]
        
        link_key = f"prefill_{worker_idx}_attn_{attn_idx}"
        
        if link_key in self.rdma_map:
            conn_info = self.rdma_map[link_key]
            p_qpn = conn_info["prefill_qpn"]
            a_qpn = conn_info["attn_qpn"]
            
            print(f"Initiating RDMA Transfer. Src:{p_qpn} -> Dst:{a_qpn}")
            
            # A. Prepare Recv on Destination (Expect from p_qpn)
            ready = await attn_node.prepare_recv_kv.remote(req_id, seq_len, p_qpn, 0) # LID unused if connected
            if not ready: raise RuntimeError("AttnNode failed to prepare recv")
            
            # B. Send from Source (Send to a_qpn)
            sent = await worker.send_kv_rdma.remote(req_id, a_qpn, 0) # LID unused if connected
            if not sent: raise RuntimeError("PrefillWorker failed to send")
            
            # C. Finalize (Poll & Copy)
            finalized = await attn_node.finalize_rdma_transfer.remote(req_id)
            if not finalized: raise RuntimeError("AttnNode failed to finalize RDMA")
            
            print("RDMA Transfer & Cache Loading Complete.")
            
        else:
            print("WARNING: RDMA link not found. Using Ray Fallback.")
            k_ref = prefill_out["k_ref"]
            v_ref = prefill_out["v_ref"]
            await attn_node.init_kv_from_prefill.remote(req_id, k_ref, v_ref)
        
        # 3. Decoding Loop
        generated_text, first_token_time, num_gen_tokens = await self.run_generation_loop(req_id, first_token, seq_len)
        
        if return_metrics:
            req_end_time = time.time()
            ttft = first_token_time - req_start_time if first_token_time else 0
            latency = req_end_time - req_start_time
            tpot = (latency - ttft) / (num_gen_tokens - 1) if num_gen_tokens > 1 else 0
            throughput = num_gen_tokens / latency if latency > 0 else 0
            
            metrics = {
                "ttft": ttft,
                "tpot": tpot,
                "latency": latency,
                "throughput": throughput,
                "total_tokens": num_gen_tokens
            }
            return generated_text, metrics
            
        return generated_text

    async def run_generation_loop(self, req_id: str, start_token: int, start_len):
        import torch
        
        current_token_id = start_token
        generated_tokens = []
        first_token_time = None
        
        # Record time just before first decode step logic effectively starts or after prefill
        # But TTFT is usually end of prefill + first decode
        # We will capture timestamp when the first token is generated by decoding loop (or reused from prefill if that counts)
        # Actually start_token IS the first generated token from prefill. 
        # So we mark first_token_time NOW? 
        # Definition: TTFT is time from arrival to first token being available.
        # prefill_out gave us first_token. So that is already done.
        
        # However, usually we might want to see how fast we get the NEXT token?
        # Standard TTFT: Time to First Token. 
        # If prefill returns the first token, then `time.time()` right after prefill is the time we got the first token.
        
        # Let's say we count the prefill output as the first token. 
        # Then first_token_time = time.time() happens before this function is called? 
        # Or we treat the first *decoded* token as the first token for streaming purposes?
        # Usually TTFT includes prefill time. 
        
        # Let's consider the token form prefill as the first token.
        first_token_time = time.time()
        
        for step in range(20): # Gen 20 tokens max
            # Initial Input for Layer 0 is Token ID
            # Subsequent Layers input is Hidden State
            
            # We need to maintain state across layers?
            # Or just pass output of L(i) to L(i+1)
            
            # Layer 0 starts with Token ID
            current_input = {
                "input_ids": [current_token_id],
                "request_ids": [req_id],
                "seq_lens": [start_len + step],
                "layer_idx": 0
            }
            
            hidden_state = None
            
            # Iterate All Layers
            for layer_idx in range(self.model_config.num_layers):
                attn_node = self.attn_nodes[0] # Assuming single node holds all layers for now
                ffn_node = self.ffn_nodes[0]
                
                # 1. Attn Step
                if layer_idx == 0:
                     # Layer 0 takes Token ID
                     inputs = current_input
                else:
                     # Subsequent layers take Hidden State (Embedding/activation)
                     inputs = {
                        "embeddings": hidden_state,
                        "request_ids": [req_id],
                        "seq_lens": [start_len + step],
                        "layer_idx": layer_idx
                     }

                # Exec Attn
                attn_out, _ = await attn_node.process_step.remote(inputs)
                
                # 2. FFN Step
                # FFN takes Attn output
                ffn_inputs = (attn_out, [req_id], layer_idx)
                ffn_out = await ffn_node.process_step.remote(ffn_inputs)
                
                # FFN out is Hidden State (except last layer returns TokenID)
                if layer_idx == self.model_config.num_layers - 1:
                    # Last Layer -> Token ID
                    token_id = ffn_out
                else:
                    # Middle Layer -> Hidden State
                    hidden_state = ffn_out
            
            # EOS Check (OPT EOS=2)
            if token_id == 2:
                generated_tokens.append("<EOS>")
                break
                
            generated_tokens.append(f"{token_id}")
            current_token_id = token_id
            
        print(f"Generation Complete. Tokens: {generated_tokens}")
        # We need Tokenizer to decode? Or just return IDs.
        # Let's return IDs string for now.
        return " ".join(generated_tokens), first_token_time, len(generated_tokens)
