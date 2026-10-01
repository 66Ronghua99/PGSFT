"""Deterministic checkpoint selection with explicit provenance."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence

from pgsft.provenance.manifest import write_once_json


CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")


def checkpoint_steps(run_dir: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    for path in run_dir.iterdir():
        match = CHECKPOINT_RE.fullmatch(path.name)
        if match and path.is_dir():
            result[int(match.group(1))] = path
    return result


def select_checkpoint(
    run_dir: Path,
    selection: dict[str, Any],
    *,
    metric_history: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    checkpoints = checkpoint_steps(run_dir)
    if not checkpoints:
        raise ValueError("run contains no checkpoint-N directories")
    rule = selection.get("rule")
    if rule == "fixed_final":
        step = selection.get("step")
        if not isinstance(step, int) or step not in checkpoints:
            raise ValueError(f"fixed checkpoint is missing: checkpoint-{step}")
        metric = None
    elif rule == "min_metric":
        metric = selection.get("metric")
        if not isinstance(metric, str) or not metric:
            raise ValueError("min_metric selection requires a metric name")
        candidates = []
        for record in metric_history:
            step = record.get("step")
            value = record.get(metric)
            if isinstance(step, int) and step in checkpoints and isinstance(value, (int, float)):
                candidates.append((float(value), step))
        if not candidates:
            raise ValueError(f"no checkpoint has selection metric {metric!r}")
        _, step = min(candidates, key=lambda item: (item[0], item[1]))
    else:
        raise ValueError(f"unsupported checkpoint selection rule: {rule!r}")
    result = {
        "schema_version": 1,
        "rule": rule,
        "metric": metric,
        "step": step,
        "checkpoint": f"checkpoint-{step}",
    }
    write_once_json(run_dir / "selected_checkpoint.json", result)
    return result


def load_selection(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "selected_checkpoint.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("checkpoint") != f"checkpoint-{value.get('step')}":
        raise ValueError("selected checkpoint provenance is inconsistent")
    if not (run_dir / value["checkpoint"]).is_dir():
        raise ValueError("selected checkpoint directory is missing")
    return value
