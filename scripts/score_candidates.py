from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from neural_data import INSTRUCTION, digest, dump, load, prompt


def read_candidates(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len({row["candidate_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate candidate IDs")
    if any(row.get("schema_version") != 1 for row in rows):
        raise ValueError("Unsupported neural DP candidate schema")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score legal monotonic-DP candidates with a Qwen LoRA adapter")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.batch_size < 1 or args.max_length < 1 or (args.stop_after is not None and args.stop_after < 1):
        raise ValueError("Numeric options must be positive")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A BF16-capable CUDA GPU is required")

    manifest = load(args.data / "manifest.json")
    if manifest.get("schema_version") != 1 or digest(args.data / "candidates.jsonl") != manifest["files"]["candidates.jsonl"]:
        raise ValueError("Neural DP candidate manifest mismatch")
    config = load(args.checkpoint / "run_config.json")
    if config["instruction"] != INSTRUCTION or args.max_length != config["max_length"]:
        raise ValueError("Prompt or token budget differs from adapter training")

    args.output.mkdir(parents=True, exist_ok=True)
    score_path = args.output / "scores.jsonl"
    completed = set()
    if score_path.exists():
        with score_path.open(encoding="utf-8") as handle:
            for line in handle:
                completed.add(json.loads(line)["candidate_id"])
    elif any(args.output.iterdir()):
        raise FileExistsError("Output contains files but no resumable scores.jsonl")

    rows = read_candidates(args.data / "candidates.jsonl")
    rows.sort(key=lambda row: len(row["ja_text"]) + len(row["zh_text"]))
    pending = [row for row in rows if row["candidate_id"] not in completed]
    if len(completed) + len(pending) != len(rows):
        raise ValueError("Existing output contains unknown or duplicate candidate IDs")

    tokenizer = AutoTokenizer.from_pretrained(config["model"], revision=config["revision"],
                                              padding_side="left", trust_remote_code=False)
    yes_id, no_id = [tokenizer.convert_tokens_to_ids(value) for value in ("yes", "no")]
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], revision=config["revision"], torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(model, args.checkpoint)
    model.eval()

    started = time.perf_counter()
    written = 0
    max_tokens = 0
    token_sum = 0
    with score_path.open("a", encoding="utf-8") as handle, torch.no_grad():
        for start in range(0, len(pending), args.batch_size):
            if args.stop_after is not None and written >= args.stop_after:
                break
            batch_rows = pending[start:start + args.batch_size]
            if args.stop_after is not None:
                batch_rows = batch_rows[:args.stop_after - written]
            texts = [prompt(row["ja_text"], row["zh_text"]) for row in batch_rows]
            encoded = tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
            lengths = list(map(len, encoded))
            if max(lengths) > args.max_length:
                raise ValueError(f"Overlength candidate encountered: {max(lengths)} > {args.max_length}")
            max_tokens = max(max_tokens, max(lengths))
            token_sum += sum(lengths)
            batch = tokenizer.pad(
                {"input_ids": encoded, "attention_mask": [[1] * length for length in lengths]},
                padding=True, return_tensors="pt",
            )
            batch = {name: value.to("cuda:0") for name, value in batch.items()}
            logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            values = (logits[:, yes_id] - logits[:, no_id]).cpu().tolist()
            for row, score in zip(batch_rows, values, strict=True):
                handle.write(json.dumps({
                    "candidate_id": row["candidate_id"], "window_id": row["window_id"],
                    "ja_start": row["ja_start"], "ja_end": row["ja_end"],
                    "zh_start": row["zh_start"], "zh_end": row["zh_end"],
                    "raw_logit_difference": score,
                }) + "\n")
            handle.flush()
            written += len(batch_rows)
            total = len(completed) + written
            if total % 4096 < len(batch_rows) or total == len(rows):
                elapsed = time.perf_counter() - started
                print(f"Scored {total}/{len(rows)}; this-run {written / elapsed:.1f} pairs/s", flush=True)

    total_complete = len(completed) + written
    progress = {
        "status": "complete" if total_complete == len(rows) else "paused",
        "pairs_total": len(rows), "pairs_complete": total_complete,
        "pairs_this_run": written, "seconds_this_run": time.perf_counter() - started,
        "max_tokens_this_run": max_tokens,
        "mean_tokens_this_run": token_sum / written if written else None,
        "batch_size": args.batch_size,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "candidate_manifest_sha256": digest(args.data / "manifest.json"),
        "checkpoint_sha256": digest(args.checkpoint / "adapter_model.safetensors"),
        "checkpoint": str(args.checkpoint), "model": config["model"], "revision": config["revision"],
        "instruction": INSTRUCTION, "score": "yes logit minus no logit; not a probability",
    }
    dump(args.output / "progress.json", progress)
    print(json.dumps(progress, indent=2), flush=True)


if __name__ == "__main__":
    main()
