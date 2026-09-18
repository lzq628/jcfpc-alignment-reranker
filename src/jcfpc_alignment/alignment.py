from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AlignmentStep:
    source: tuple[int, ...]
    target: tuple[int, ...]
    cost: float


WIDE_MOVES = (
    (1, 1, 0.00),
    (1, 2, 0.18), (2, 1, 0.18), (2, 2, 0.25),
    (1, 3, 0.45), (3, 1, 0.45), (2, 3, 0.42), (3, 2, 0.42),
    (3, 3, 0.52), (1, 4, 0.75), (4, 1, 0.75), (2, 4, 0.65),
    (4, 2, 0.65), (3, 4, 0.70), (4, 3, 0.70), (4, 4, 0.85),
    (1, 5, 1.05), (5, 1, 1.05), (2, 5, 0.95), (5, 2, 0.95),
    (3, 5, 0.90), (5, 3, 0.90), (1, 6, 1.35), (6, 1, 1.35),
    (2, 6, 1.20), (6, 2, 1.20), (1, 7, 1.65), (7, 1, 1.65),
    (1, 8, 1.95), (8, 1, 1.95), (1, 0, 2.70), (0, 1, 2.70),
)


def _group_cost(source_length: int, target_length: int, ratio: float, penalty: float) -> float:
    if source_length == 0 or target_length == 0:
        return penalty + math.log1p(source_length + target_length) * 0.08
    expected = max(source_length * ratio, 1.0)
    observed = max(target_length, 1.0)
    return abs(math.log(observed / expected)) + penalty


def align_lengths(
    source_lengths: list[int],
    target_lengths: list[int],
    *,
    ratio: float | None = None,
    band: int = 250,
    similarity: Callable[[int, int, int, int], float | None] | None = None,
    semantic_weight: float = 0.0,
    semantic_baseline: float = 0.42,
    semantic_empty_penalty: float = 0.0,
    moves: Sequence[tuple[int, int, float]] = WIDE_MOVES,
) -> list[AlignmentStep]:
    """Find the minimum-cost monotonic path through two sentence sequences.

    The scorer receives half-open source and target spans. It may provide frozen
    embedding cosine, a transformed reranker score, or a score-level fusion.
    """
    source_count = len(source_lengths)
    target_count = len(target_lengths)
    if source_count == 0 and target_count == 0:
        return []
    if ratio is None:
        ratio = (sum(target_lengths) + 1) / (sum(source_lengths) + 1)

    source_prefix = [0]
    target_prefix = [0]
    for value in source_lengths:
        source_prefix.append(source_prefix[-1] + value)
    for value in target_lengths:
        target_prefix.append(target_prefix[-1] + value)

    def in_band(source_index: int, target_index: int) -> bool:
        if source_index in (0, source_count) or source_count == 0:
            return True
        expected = round(source_index * target_count / source_count)
        return abs(target_index - expected) <= band

    costs: dict[tuple[int, int], float] = {(0, 0): 0.0}
    back: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int, float]]] = {}

    for source_index in range(source_count + 1):
        expected = round(source_index * target_count / source_count) if source_count else 0
        minimum_target = 0 if source_index == 0 else max(0, expected - band)
        maximum_target = target_count if source_index == source_count else min(target_count, expected + band)
        for target_index in range(minimum_target, maximum_target + 1):
            state = (source_index, target_index)
            base_cost = costs.get(state)
            if base_cost is None:
                continue
            for source_size, target_size, penalty in moves:
                next_source = source_index + source_size
                next_target = target_index + target_size
                if next_source > source_count or next_target > target_count or not in_band(next_source, next_target):
                    continue
                source_length = source_prefix[next_source] - source_prefix[source_index]
                target_length = target_prefix[next_target] - target_prefix[target_index]
                step_cost = _group_cost(source_length, target_length, ratio, penalty)
                if similarity is not None and semantic_weight > 0:
                    if source_size and target_size:
                        score = similarity(source_index, next_source, target_index, next_target)
                        if score is not None:
                            score = max(-1.0, min(1.0, score))
                            step_cost = max(0.0, step_cost + semantic_weight * (semantic_baseline - score))
                    else:
                        step_cost += semantic_empty_penalty
                candidate_cost = base_cost + step_cost
                next_state = (next_source, next_target)
                if candidate_cost < costs.get(next_state, float("inf")):
                    costs[next_state] = candidate_cost
                    back[next_state] = (state, (source_size, target_size, step_cost))

    end = (source_count, target_count)
    if end not in costs:
        if band < max(source_count, target_count) + 2:
            return align_lengths(
                source_lengths,
                target_lengths,
                ratio=ratio,
                band=max(source_count, target_count) + 2,
                similarity=similarity,
                semantic_weight=semantic_weight,
                semantic_baseline=semantic_baseline,
                semantic_empty_penalty=semantic_empty_penalty,
                moves=moves,
            )
        raise RuntimeError("Alignment failed to reach the final state")

    steps: list[AlignmentStep] = []
    state = end
    while state != (0, 0):
        previous, (source_size, target_size, step_cost) = back[state]
        source_index, target_index = previous
        steps.append(AlignmentStep(
            source=tuple(range(source_index, source_index + source_size)),
            target=tuple(range(target_index, target_index + target_size)),
            cost=step_cost,
        ))
        state = previous
    return list(reversed(steps))
