import argparse
import os
from typing import Dict, Iterable, List, Tuple

from huggingface_hub import HfApi, snapshot_download

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


def _paths_exist(base_dir: str, candidates: Iterable[str]) -> bool:
    return any(os.path.exists(os.path.join(base_dir, path)) for path in candidates)


def _required_assets(model_key: str) -> Dict[str, List[str]]:
    lower = model_key.lower()
    assets = {
        "config": ["config.json"],
        "tokenizer": [
            "tokenizer.model",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
        ],
        "weights": [
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "pytorch_model.bin",
        ],
    }
    if lower.startswith("qwen-"):
        assets["tokenizer"].extend(["qwen.tiktoken", "merges.txt"])
    return assets


def _validate_local_dir(model_key: str, local_dir: str) -> Tuple[bool, List[str]]:
    missing = []
    for asset_name, candidates in _required_assets(model_key).items():
        if not _paths_exist(local_dir, candidates):
            missing.append(
                f"{asset_name} missing (expected one of: {', '.join(candidates)})"
            )
    return not missing, missing


def _check_remote_access(repo_id: str, token: str | None) -> None:
    api = HfApi(token=token)
    api.model_info(repo_id)

def main():
    parser = argparse.ArgumentParser(description="Download LLM models from Hugging Face.")
    parser.add_argument("--output-dir", type=str, default="model", help="Base directory to save models.")
    parser.add_argument("--models", nargs="+", default="all", 
                        help=f"Specific models to download. Choices: {list(MODELS.keys())} or 'all'.")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token (optional if logged in).")
    parser.add_argument(
        "--skip-remote-check",
        action="store_true",
        help="Skip explicit Hugging Face access verification before downloading.",
    )
    
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
            is_complete, missing = _validate_local_dir(key, local_dir)
            if is_complete:
                print(f"Local model directory already looks complete: {local_dir}")
                continue
            if os.path.isdir(local_dir):
                print(f"Local model directory is incomplete: {local_dir}")
                for item in missing:
                    print(f"  - {item}")
            if not args.skip_remote_check:
                print(f"Checking remote access for {hf_id}...")
                _check_remote_access(hf_id, args.token)
            snapshot_download(
                repo_id=hf_id,
                local_dir=local_dir,
                token=args.token,
                local_dir_use_symlinks=False, # Standard for model weights
                resume_download=True
            )
            is_complete, missing = _validate_local_dir(key, local_dir)
            if not is_complete:
                raise RuntimeError(
                    "Download finished but local directory is still incomplete: "
                    + "; ".join(missing)
                )
            print(f"Successfully downloaded {hf_id} to {local_dir}")
        except Exception as e:
            print(f"Failed to download {hf_id}: {e}")
            if "llama" in key:
                print("Note: Llama-2 models require a Hugging Face token with granted access.")

if __name__ == "__main__":
    main()
