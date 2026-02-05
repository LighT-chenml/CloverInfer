import argparse
import json
import random
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dataset/sharegpt_processed.json")
    parser.add_argument("--num-samples", type=int, default=None, help="Number of samples to keeping. If None, keep all.")
    args = parser.parse_args()

    print("Loading ShareGPT...")
    # Using a known clean version or the original if available
    # For speed/reliability we often use a smaller subset or similar if ShareGPT is huge/gated
    # 'anon8231489123/ShareGPT_Vicuna_unfiltered' is a common hugginface mirror
    # We specify the JSON file explicitly because the repo structure might not be auto-detected
    dataset = load_dataset("anon8231489123/ShareGPT_Vicuna_unfiltered", data_files="ShareGPT_V3_unfiltered_cleaned_split.json", split="train")

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    samples = []
    # Simple filtering: take conversations starting with human
    count = 0
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for idx in indices:
        if args.num_samples is not None and count >= args.num_samples:
            break
            
        item = dataset[idx]
        convs = item.get("conversations", [])
        if not convs:
            continue
            
        # Extract first prompt
        first_sender = convs[0].get("from", "")
        # ShareGPT varies: 'human', 'user'
        if first_sender in ["human", "user"]:
            prompt = convs[0].get("value", "")
            if prompt:
                samples.append({"id": item.get("id", str(idx)), "prompt": prompt})
                count += 1

    print(f"Collected {len(samples)} samples. Saving to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(samples, f, indent=2)
    print("Done.")

if __name__ == "__main__":
    main()
