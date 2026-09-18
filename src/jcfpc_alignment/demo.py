from __future__ import annotations

import json
from pathlib import Path

from .alignment import align_lengths
from .metrics import exact_unit_metrics


def main() -> None:
    path = Path(__file__).resolve().parents[2] / "examples" / "synthetic_demo.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    score_table = {
        tuple(map(int, key.split(","))): value
        for key, value in data["scores"].items()
    }

    def similarity(source_start: int, source_end: int, target_start: int, target_end: int) -> float:
        return score_table.get((source_start, source_end, target_start, target_end), 0.0)

    steps = align_lengths(
        [len(sentence) for sentence in data["japanese"]],
        [len(sentence) for sentence in data["chinese"]],
        similarity=similarity,
        semantic_weight=1.2,
        semantic_baseline=0.5,
    )
    prediction = [(step.source, step.target) for step in steps]
    reference = [(tuple(unit["ja"]), tuple(unit["zh"])) for unit in data["gold"]]
    print(json.dumps({
        "notice": "Synthetic sentences only; no JCFPC novel text is distributed.",
        "prediction": [{"ja": list(ja), "zh": list(zh)} for ja, zh in prediction],
        "metrics": exact_unit_metrics(reference, prediction),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
