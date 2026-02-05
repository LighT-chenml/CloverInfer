import argparse
import json
import os
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="dataset/humaneval.jsonl")
    args = parser.parse_args()

    print("Loading HumanEval...")
    try:
        # openai_humaneval is the standard name on HF
        dataset = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print(f"Failed to load via datasets: {e}")
        return

    print(f"Dataset loaded. Total samples: {len(dataset)}")
    
    # Save as JSONL
    with open(args.output, "w") as f:
        for item in dataset:
            # HumanEval has 'task_id', 'prompt', 'canonical_solution', 'test', 'entry_point'
            f.write(json.dumps(item) + "\n")
            
    print(f"Saved to {args.output}")

if __name__ == "__main__":
    main()
