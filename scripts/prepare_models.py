import argparse
import os
from huggingface_hub import snapshot_download

# Defined model mappings
MODELS = {
    "llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-hf",
    "opt-125m": "facebook/opt-125m",
    "opt-6.7b": "facebook/opt-6.7b",
    "opt-13b": "facebook/opt-13b",
    "qwen-1.8b": "Qwen/Qwen-1_8B", # Note: HuggingFace ID uses 1_8B
    "qwen-7b": "Qwen/Qwen-7B",
    "qwen-14b": "Qwen/Qwen-14B",
}

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from Hugging Face.")
    parser.add_argument("--output-dir", type=str, default="model", help="Base directory to save models.")
    parser.add_argument("--models", nargs="+", default="all", 
                        help=f"Specific models to download. Choices: {list(MODELS.keys())} or 'all'.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional if logged in).")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    targets = []
    if "all" in args.models:
        targets = list(MODELS.keys())
    else:
        for m in args.models:
            if m.lower() in MODELS:
                targets.append(m.lower())
            else:
                print(f"Warning: Model '{m}' not found in presets. Skipping.")

    print(f"Downloading models to {args.output_dir}...")
    
    for key in targets:
        hf_id = MODELS[key]
        model_name = hf_id.split("/")[-1] # e.g., Llama-2-7b-hf
        local_dir = os.path.join(args.output_dir, model_name)
        
        print(f"\nProcessing {key} -> {hf_id}...")
        try:
            snapshot_download(
                repo_id=hf_id,
                local_dir=local_dir,
                token=args.token,
                local_dir_use_symlinks=False, # Standard for model weights
                resume_download=True
            )
            print(f"Successfully downloaded {hf_id} to {local_dir}")
        except Exception as e:
            print(f"Failed to download {hf_id}: {e}")
            if "llama" in key:
                print("Note: Llama-2 models require a Hugging Face token with granted access.")

if __name__ == "__main__":
    main()
