from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

from transformers import AutoTokenizer

from src.core.model_adapter import LocalQwenTokenizer


DATASET_FORMAT_CHOICES = [
    "auto",
    "prompt_jsonl",
    "humaneval_jsonl",
    "sharegpt_json",
    "sharegpt_jsonl",
    "longbench_jsonl",
]


def load_tokenizer_for_benchmark(model_path: str):
    qwen_vocab = os.path.join(model_path, "qwen.tiktoken")
    if os.path.exists(qwen_vocab):
        return LocalQwenTokenizer(qwen_vocab)

    try:
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            use_fast=False,
        )


def encode_prompt(tokenizer, prompt: str) -> List[int]:
    if hasattr(tokenizer, "encode"):
        try:
            return tokenizer.encode(prompt, add_special_tokens=False)
        except TypeError:
            return tokenizer.encode(prompt)
    raise TypeError("tokenizer does not provide an encode method")


def decode_prompt(tokenizer, token_ids: List[int]) -> str:
    if hasattr(tokenizer, "decode"):
        try:
            return tokenizer.decode(token_ids, clean_up_tokenization_spaces=False)
        except TypeError:
            return tokenizer.decode(token_ids)
    raise TypeError("tokenizer does not provide a decode method")


def build_prompt_for_target_tokens(
    tokenizer,
    target_tokens: int,
    base_prompt: str | None = None,
) -> Tuple[str, int]:
    if target_tokens <= 0:
        raise ValueError("target token length must be positive")

    filler_ids = encode_prompt(tokenizer, " clover")
    if not filler_ids:
        raise ValueError("failed to build filler tokens for prompt resizing")

    if base_prompt:
        token_ids = list(encode_prompt(tokenizer, base_prompt))
    else:
        prefix_ids = encode_prompt(tokenizer, "Context:")
        if not prefix_ids:
            raise ValueError("failed to build prefix tokens for prompt resizing")
        token_ids = list(prefix_ids)

    if len(token_ids) >= target_tokens:
        token_ids = token_ids[:target_tokens]
    else:
        while len(token_ids) < target_tokens:
            token_ids.extend(filler_ids)
        token_ids = token_ids[:target_tokens]

    prompt = decode_prompt(tokenizer, token_ids)
    actual_ids = encode_prompt(tokenizer, prompt)
    return prompt, len(actual_ids)


def _read_jsonl(path: str) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _normalize_prompt_record(
    record: Dict[str, object],
    idx: int,
    source_dataset: str,
) -> Dict[str, object]:
    prompt = str(record.get("prompt", ""))
    if not prompt:
        raise ValueError(f"record {idx} from {source_dataset} does not contain a non-empty prompt")
    return {
        "task_id": str(record.get("task_id", record.get("id", f"{source_dataset}/{idx}"))),
        "prompt": prompt,
        "source_dataset": source_dataset,
        "raw_record": record,
    }


def _build_longbench_prompt(record: Dict[str, object]) -> str:
    dataset_name = str(record.get("dataset", "longbench"))
    context = str(record.get("context", "")).strip()
    question = str(record.get("input", "")).strip()
    answers = record.get("answers")

    sections = [f"Dataset: {dataset_name}"]
    if context:
        sections.append(f"Context:\n{context}")
    if question:
        sections.append(f"Question:\n{question}")
    if answers:
        sections.append("Answer the question using the provided context.")

    prompt = "\n\n".join(section for section in sections if section.strip())
    if not prompt:
        raise ValueError(f"LongBench record for dataset {dataset_name} could not be converted to a prompt")
    return prompt


def _normalize_longbench_record(
    record: Dict[str, object],
    idx: int,
    source_dataset: str,
) -> Dict[str, object]:
    task_id = str(record.get("_id", f"{source_dataset}/{idx}"))
    return {
        "task_id": task_id,
        "prompt": _build_longbench_prompt(record),
        "source_dataset": str(record.get("dataset", source_dataset)),
        "raw_record": record,
    }


def infer_dataset_format(path: str) -> str:
    if path.endswith(".json"):
        return "sharegpt_json"

    rows = _read_jsonl(path)
    if not rows:
        raise ValueError(f"dataset {path} is empty")

    first = rows[0]
    if "prompt" in first:
        if "task_id" in first:
            return "humaneval_jsonl"
        return "prompt_jsonl"
    if "input" in first or "context" in first:
        return "longbench_jsonl"
    raise ValueError(f"could not infer dataset format for {path}")


def load_benchmark_samples(
    path: str,
    dataset_format: str = "auto",
    limit: int | None = None,
    tokenizer=None,
    prompt_token_length: int = 0,
) -> List[Dict[str, object]]:
    if dataset_format not in DATASET_FORMAT_CHOICES:
        raise ValueError(f"unsupported dataset format: {dataset_format}")

    resolved_format = infer_dataset_format(path) if dataset_format == "auto" else dataset_format
    source_dataset = os.path.splitext(os.path.basename(path))[0]

    if resolved_format == "sharegpt_json":
        with open(path, "r", encoding="utf-8") as f:
            raw_rows = json.load(f)
    else:
        raw_rows = _read_jsonl(path)

    if limit is not None:
        raw_rows = raw_rows[:limit]

    samples: List[Dict[str, object]] = []
    for idx, record in enumerate(raw_rows):
        if resolved_format in {"prompt_jsonl", "humaneval_jsonl", "sharegpt_json", "sharegpt_jsonl"}:
            sample = _normalize_prompt_record(record, idx, source_dataset)
        elif resolved_format == "longbench_jsonl":
            sample = _normalize_longbench_record(record, idx, source_dataset)
        else:
            raise ValueError(f"unsupported dataset format: {resolved_format}")

        if tokenizer is not None and prompt_token_length > 0:
            resized_prompt, actual_tokens = build_prompt_for_target_tokens(
                tokenizer,
                prompt_token_length,
                sample["prompt"],
            )
            sample["prompt"] = resized_prompt
            sample["prompt_token_length"] = int(actual_tokens)
        elif tokenizer is not None:
            sample["prompt_token_length"] = int(len(encode_prompt(tokenizer, sample["prompt"])))

        samples.append(sample)

    if not samples:
        raise ValueError(f"no samples loaded from {path}")
    return samples
