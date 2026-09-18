from __future__ import annotations

from collections.abc import Iterable


Unit = tuple[tuple[int, ...], tuple[int, ...]]


def exact_unit_metrics(reference: Iterable[Unit], prediction: Iterable[Unit]) -> dict[str, float | int]:
    reference_set = set(reference)
    prediction_set = set(prediction)
    matches = len(reference_set & prediction_set)
    precision = matches / len(prediction_set) if prediction_set else 0.0
    recall = matches / len(reference_set) if reference_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference_units": len(reference_set),
        "predicted_units": len(prediction_set),
        "matched_units": matches,
        "unit_precision": precision,
        "unit_recall": recall,
        "unit_f1": f1,
    }
