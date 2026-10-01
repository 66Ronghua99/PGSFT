from __future__ import annotations

from pathlib import Path

import pytest

from pgsft.cli.evaluate import load_evaluation_recipe
from pgsft.evaluation.gpqa import parse_gpqa_broad, run_gpqa


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("final", "label"),
    [
        ("Answer: A", "A"),
        ("The answer is (B).", "B"),
        (r"Therefore, \boxed{C}", "C"),
        ("some final explanation\nD", "D"),
        ("The exact option text!", "A"),
        ("Answer: A\nAnswer: A", "A"),
    ],
)
def test_broad_parser_accepts_frozen_explicit_forms(final: str, label: str) -> None:
    observed, status = parse_gpqa_broad(
        "private reasoning</think>\n" + final,
        options={"A": "The exact option text", "B": "second", "C": "third", "D": "fourth"},
    )
    assert observed == label and status == "explicit"


def test_broad_parser_rejects_conflicts_missing_and_unfinished_thinking() -> None:
    label, status = parse_gpqa_broad("reason</think>Answer: A\n\\boxed{B}")
    assert label is None and status == "conflicting_answers"
    label, status = parse_gpqa_broad("reason</think>maybe A because B")
    assert label is None and status == "missing_answer"
    label, status = parse_gpqa_broad("Answer: A")
    assert label is None and status == "unfinished_thinking"


class FakeEndpoint:
    def __init__(self):
        self.index = 0

    def complete(self, messages, **kwargs):
        label = "ABCD"[self.index % 4]
        self.index += 1
        return {"content": f"reason</think>Answer: {label}", "finish_reason": "stop"}


def test_gpqa_runs_exactly_198_once_at_64k_and_reports_present_subjects() -> None:
    rows = [
        {
            "task_id": f"task-{index:03d}",
            "prompt": f"question {index}",
            "options": {label: f"option {label} {index}" for label in "ABCD"},
            "correct_label": "ABCD"[index % 4],
            "subject": "biology" if index < 20 else "physics",
        }
        for index in range(198)
    ]
    summary = run_gpqa(
        rows,
        FakeEndpoint(),
        context_length=65_536,
        temperature=1.0,
        max_output_tokens=32_768,
        seed=None,
    )
    assert summary.denominator == 198
    assert summary.correct == 198
    assert summary.explicit_answer_coverage == 198
    assert summary.by_subject == {
        "biology": {"correct": 20, "total": 20},
        "physics": {"correct": 178, "total": 178},
    }


def test_formal_recipes_bind_64k_and_have_no_rescue_mode() -> None:
    for backbone in ("qwen3", "qwen35"):
        recipe = load_evaluation_recipe(
            ROOT / "configs" / "evaluation" / backbone / "gpqa.yaml", root=ROOT
        )
        assert recipe["context_length"] == 65_536
        assert "rescue" not in str(recipe).lower()


def test_non_64k_gpqa_is_rejected() -> None:
    with pytest.raises(ValueError, match="65536"):
        run_gpqa(
            [],
            FakeEndpoint(),
            context_length=32_768,
            temperature=1.0,
            max_output_tokens=1,
            seed=None,
        )
