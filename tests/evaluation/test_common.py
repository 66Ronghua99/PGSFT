from __future__ import annotations

import pytest

from pgsft.evaluation.common import EvaluationSummary, TaskResult


def test_fixed_denominator_summary_keeps_failures_and_subjects() -> None:
    ids = ["a", "b", "c", "d"]
    results = [
        TaskResult("a", "correct", True, "A", {}),
        TaskResult("b", "wrong_answer", False, "B", {}),
        TaskResult("c", "format_error", False, "", {}),
        TaskResult("d", "infrastructure_error", None, "", {}),
    ]
    summary = EvaluationSummary.build(
        "fixture",
        ids,
        results,
        subjects={"a": "x", "b": "x", "c": "y", "d": "y"},
    )
    assert summary.denominator == 4
    assert summary.correct == 1
    assert summary.explicit_answer_coverage == 2
    assert summary.by_subject == {
        "x": {"correct": 1, "total": 2},
        "y": {"correct": 0, "total": 2},
    }


def test_result_contract_and_coverage_fail_closed() -> None:
    with pytest.raises(ValueError, match="reserved"):
        TaskResult("x", "model_error", None, "", {})
    with pytest.raises(ValueError, match="coverage"):
        EvaluationSummary.build(
            "fixture", ["a", "b"], [TaskResult("a", "correct", True, "", {})]
        )
    with pytest.raises(ValueError, match="duplicate"):
        EvaluationSummary.build(
            "fixture",
            ["a", "b"],
            [
                TaskResult("a", "correct", True, "", {}),
                TaskResult("a", "correct", True, "", {}),
            ],
        )
