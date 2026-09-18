from __future__ import annotations

import hashlib
import json
from pathlib import Path

PREFIX = (
    '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
INSTRUCTION = (
    "Judge whether the Chinese document is a translation equivalent of the Japanese query. "
    "Allow literary paraphrase and sentence splitting or merging. Prefer complete coverage of the "
    "same content over partial coverage, unrelated additions, or local drift."
)
MODEL = "Qwen/Qwen3-Reranker-8B"
REVISION = "77d193c791ed757ca307ee72715aa132723da912"


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def prompt(japanese: str, chinese: str, instruction: str = INSTRUCTION) -> str:
    return PREFIX + f"<Instruct>: {instruction}\n<Query>: {japanese}\n<Document>: {chinese}" + SUFFIX


def read_groups(path: Path):
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("schema_version") != 2 or row["query_id"] in seen:
                raise ValueError("Only unique, schema-v2 fixed-query candidate groups are accepted")
            seen.add(row["query_id"])
            candidates = row["candidates"]
            if len(candidates) < 2 or candidates[0]["label"] != 1 or any(item["label"] != 0 for item in candidates[1:]):
                raise ValueError(f"Invalid candidate labels: {row['query_id']}")
            if len({tuple(item["zh"]) for item in candidates}) != len(candidates):
                raise ValueError("Duplicate target indices")
            yield row


def verify_dataset(directory: Path) -> dict:
    manifest = load(directory / "manifest.json")
    if manifest.get("schema_version") != 2:
        raise ValueError("Old structured candidates are unsafe and not supported")
    for name, expected in manifest["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"Candidate file hash mismatch: {name}")
    return manifest


def select_candidates(group: dict, negatives: int):
    if negatives <= 0:
        return group["candidates"]
    return group["candidates"][:negatives + 1]


def tokenize_groups(tokenizer, path: Path, max_length: int, negatives: int = 0, limit: int | None = None):
    accepted, lengths, skipped = [], [], []
    for group in read_groups(path):
        if limit is not None and len(accepted) >= limit:
            break
        candidates = select_candidates(group, negatives)
        texts = [prompt(group["ja_text"], row["zh_text"]) for row in candidates]
        tokens = tokenizer(texts, padding=False, truncation=False, add_special_tokens=False)["input_ids"]
        lengths.extend(map(len, tokens))
        if any(len(sequence) > max_length for sequence in tokens):
            skipped.append(group["query_id"])
            continue
        accepted.append({"query_id": group["query_id"], "tokens": tokens,
                         "candidate_ids": [row["zh"] for row in candidates]})
    if not accepted:
        raise ValueError("No complete groups fit the token budget")
    return accepted, {
        "accepted_groups": len(accepted), "accepted_sequences": sum(len(row["tokens"]) for row in accepted),
        "skipped_groups": skipped, "sequences_examined": len(lengths),
        "overlength_sequences": sum(length > max_length for length in lengths),
        "truncated_sequences": 0, "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths), "policy": "skip whole overlength group; never truncate prompt or bilingual evidence",
    }
