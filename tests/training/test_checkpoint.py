from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgsft.training.checkpoint import load_selection, select_checkpoint


def _checkpoints(root: Path, *steps: int) -> None:
    for step in steps:
        (root / f"checkpoint-{step}").mkdir(parents=True)


def test_fixed_final_selection_requires_declared_checkpoint(tmp_path: Path) -> None:
    _checkpoints(tmp_path, 8, 16, 24)
    selected = select_checkpoint(tmp_path, {"rule": "fixed_final", "step": 24})
    assert selected["checkpoint"] == "checkpoint-24"
    assert load_selection(tmp_path) == selected


def test_min_metric_uses_earliest_step_tie_break(tmp_path: Path) -> None:
    _checkpoints(tmp_path, 8, 16, 24)
    selected = select_checkpoint(
        tmp_path,
        {"rule": "min_metric", "metric": "eval_loss"},
        metric_history=[
            {"step": 8, "eval_loss": 0.4},
            {"step": 16, "eval_loss": 0.3},
            {"step": 24, "eval_loss": 0.3},
        ],
    )
    assert selected["step"] == 16


def test_selection_rejects_missing_checkpoint_or_metric(tmp_path: Path) -> None:
    _checkpoints(tmp_path, 8)
    with pytest.raises(ValueError, match="missing"):
        select_checkpoint(tmp_path, {"rule": "fixed_final", "step": 16})
    with pytest.raises(ValueError, match="no checkpoint"):
        select_checkpoint(
            tmp_path,
            {"rule": "min_metric", "metric": "eval_loss"},
            metric_history=[{"step": 8, "other": 1.0}],
        )
