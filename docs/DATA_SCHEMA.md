# Private training-data contract / 非公開学習データ仕様

The repository intentionally excludes JCFPC novel text, annotations, candidate
scores, databases, and model checkpoints. Bring your own licensed data in a
private directory ignored by Git.

本リポジトリには、JCFPC の小説本文、対訳注釈、候補スコア、データベース、
チェックポイントを含めません。学習時には、利用権限を確認したデータを
Git 管理外の非公開ディレクトリに配置してください。

## Ranking groups

`train.jsonl` and `dev.jsonl` contain one fixed Japanese query per line. The first
Chinese candidate is the positive item; following items are hard negatives.

```json
{
  "schema_version": 2,
  "query_id": "synthetic:0001",
  "work_id": "synthetic",
  "ja": [1, 2],
  "ja_text": "Synthetic Japanese query",
  "candidates": [
    {"zh": [1], "zh_text": "Positive target", "kind": "positive", "label": 1},
    {"zh": [2], "zh_text": "Hard negative", "kind": "negative", "label": 0}
  ]
}
```

`manifest.json` pins the SHA-256 digest of both files. The loader rejects duplicate
queries, duplicate target spans, malformed labels, and hash mismatches.

## DP candidates

Inference uses `candidates.jsonl`. Each row contains one legal non-gap monotonic
move and its Japanese/Chinese text. `score_candidates.py` writes only IDs, spans,
and raw `yes - no` logits. The logit difference is not a probability.

## Required isolation checks

Before training or evaluation, verify at minimum:

- zero sentence-ID overlap between protected evaluation contexts and training;
- zero normalized exact-text overlap between protected contexts and training;
- zero train/development sentence overlap;
- no jointly shifted bilingual pair mislabeled as a negative;
- no truncation of bilingual evidence during tokenization.
