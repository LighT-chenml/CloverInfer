import argparse
import json
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="dataset/longbench")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # URL for the data.zip from Hugging Face mirror (direct file access)
    # Original: https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip
    url = "https://hf-mirror.com/datasets/THUDM/LongBench/resolve/main/data.zip"
    zip_path = os.path.join(args.output_dir, "data.zip")

    print(f"Downloading LongBench data from {url}...")
    try:
        # Use wget or requests. Here we use os.system for simplicity in this environment
        # or we could use requests if available. Let's use wget.
        ret = os.system(f"wget {url} -O {zip_path}")
        if ret != 0:
            raise RuntimeError("Download failed.")
            
        print("Extracting data...")
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(args.output_dir)
            
        # The zip extracts to a 'data' subdirectory. Move files up.
        data_subdir = os.path.join(args.output_dir, "data")
        if os.path.exists(data_subdir):
            for filename in os.listdir(data_subdir):
                src = os.path.join(data_subdir, filename)
                dst = os.path.join(args.output_dir, filename)
                os.rename(src, dst)
            os.rmdir(data_subdir)
            
        # Clean up zip
        os.remove(zip_path)
        print(f"LongBench data prepared in {args.output_dir}")

    except Exception as e:
        print(f"Failed to prepare LongBench: {e}")

if __name__ == "__main__":
    main()
