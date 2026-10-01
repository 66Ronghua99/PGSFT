"""Shared fail-closed task results and fixed-denominator summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Literal, Sequence


TaskStatus = Literal[
    "correct",
    "wrong_answer",
    "format_error",
    "context_overflow",
    "model_error",
    "infrastructure_error",
    "runtime_error",
    "syntax_error",
]
STATUSES = frozenset(
    {
        "correct",
        "wrong_answer",
        "format_error",
        "context_overflow",
        "model_error",
        "infrastructure_error",
        "runtime_error",
        "syntax_error",
    }
)


@dataclass(frozen=True)
class TaskResult:
    task_id: str
    status: TaskStatus
    correct: bool | None
    final_text: str
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.task_id or self.status not in STATUSES:
            raise ValueError("task result has invalid identity or status")
        if (self.status == "infrastructure_error") != (self.correct is None):
            raise ValueError("correct=None is reserved for infrastructure errors")
        if self.status == "correct" and self.correct is not True:
            raise ValueError("correct status requires correct=True")
        if self.status not in {"correct", "infrastructure_error"} and self.correct is not False:
            raise ValueError("benchmark failures require correct=False")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationSummary:
    benchmark: str
    denominator: int
    correct: int
    explicit_answer_coverage: int
    status_counts: dict[str, int]
    by_subject: dict[str, dict[str, int]]
    results: tuple[TaskResult, ...]

    @classmethod
    def build(
        cls,
        benchmark: str,
        inventory_ids: Sequence[str],
        results: Sequence[TaskResult],
        *,
        subjects: dict[str, str] | None = None,
    ) -> "EvaluationSummary":
        if len(inventory_ids) != len(set(inventory_ids)):
            raise ValueError("inventory contains duplicate task IDs")
        observed = [result.task_id for result in results]
        if len(observed) != len(set(observed)):
            raise ValueError("evaluation results contain duplicate task IDs")
        if set(observed) != set(inventory_ids) or len(observed) != len(inventory_ids):
            raise ValueError("evaluation result coverage differs from inventory")
        by_id = {result.task_id: result for result in results}
        ordered = tuple(by_id[task_id] for task_id in inventory_ids)
        counts = Counter(result.status for result in ordered)
        for status in STATUSES:
            counts.setdefault(status, 0)
        coverage = sum(
            result.status in {"correct", "wrong_answer"} for result in ordered
        )
        by_subject: dict[str, dict[str, int]] = {}
        if subjects:
            if set(subjects) != set(inventory_ids):
                raise ValueError("subject mapping differs from inventory")
            for subject in sorted(set(subjects.values())):
                group = [result for result in ordered if subjects[result.task_id] == subject]
                by_subject[subject] = {
                    "correct": sum(result.status == "correct" for result in group),
                    "total": len(group),
                }
        return cls(
            benchmark=benchmark,
            denominator=len(inventory_ids),
            correct=counts["correct"],
            explicit_answer_coverage=coverage,
            status_counts=dict(sorted(counts.items())),
            by_subject=by_subject,
            results=ordered,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "denominator": self.denominator,
            "correct": self.correct,
            "accuracy": self.correct / self.denominator if self.denominator else 0.0,
            "explicit_answer_coverage": self.explicit_answer_coverage,
            "status_counts": self.status_counts,
            "by_subject": self.by_subject,
            "results": [result.to_dict() for result in self.results],
        }
