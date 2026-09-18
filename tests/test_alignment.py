from jcfpc_alignment.alignment import align_lengths
from jcfpc_alignment.metrics import exact_unit_metrics


def test_empty_alignment():
    assert align_lengths([], []) == []


def test_monotonic_coverage():
    steps = align_lengths([8, 10, 7], [9, 8, 7], band=10)
    assert [index for step in steps for index in step.source] == [0, 1, 2]
    assert [index for step in steps for index in step.target] == [0, 1, 2]


def test_exact_unit_metrics():
    reference = [((0, 1), (0,)), ((2,), (1,))]
    prediction = [((0, 1), (0,)), ((2,), (1,))]
    assert exact_unit_metrics(reference, prediction)["unit_f1"] == 1.0
