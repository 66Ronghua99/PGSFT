"""Frozen 64K GPQA Diamond broad-answer evaluation."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Mapping, Sequence

from pgsft.evaluation.common import EvaluationSummary, TaskResult
from pgsft.evaluation.serving import (
    ContextOverflowError,
    EndpointError,
    InfrastructureEndpointError,
    OpenAIEndpoint,
)


LABELS = "ABCD"


def _normalize_option(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def parse_gpqa_broad(
    raw: str, *, options: Mapping[str, str] | None = None
) -> tuple[str | None, str]:
    """Return one explicit final label and status; never inspect open thinking."""
    thinking, marker, final = raw.rpartition("</think>")
    if not marker:
        return None, "unfinished_thinking"
    candidates: list[str] = []
    patterns = (
        r"(?im)^\s*Answer\s*:\s*\(?([ABCD])\)?\s*[.!]?\s*$",
        r"(?im)^\s*The\s+answer\s+is\s*\(?([ABCD])\)?\s*[.!]?\s*$",
        r"\\boxed\s*\{\s*([ABCD])\s*\}",
    )
    for pattern in patterns:
        candidates.extend(match.upper() for match in re.findall(pattern, final))
    lines = [line.strip() for line in final.splitlines() if line.strip()]
    if lines:
        match = re.fullmatch(r"\(?([ABCD])\)?[.!]?", lines[-1], flags=re.I)
        if match:
            candidates.append(match.group(1).upper())
    if options:
        normalized_final = _normalize_option(final)
        for label, option in options.items():
            if label not in LABELS:
                raise ValueError("GPQA option labels must be A/B/C/D")
            if normalized_final == _normalize_option(option):
                candidates.append(label)
    unique = sorted(set(candidates))
    if len(unique) == 1:
        return unique[0], "explicit"
    return None, "conflicting_answers" if unique else "missing_answer"


def load_gpqa_inventory(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    ids = [row.get("task_id") for row in rows]
    if len(rows) != 198 or len(ids) != len(set(ids)) or not all(isinstance(item, str) and item for item in ids):
        raise ValueError("formal GPQA inventory must contain exactly 198 unique IDs")
    for row in rows:
        if row.get("correct_label") not in LABELS or set(row.get("options", {})) != set(LABELS):
            raise ValueError(f"invalid GPQA scoring row: {row.get('task_id')}")
    return rows


def run_gpqa(
    rows: Sequence[dict],
    endpoint: OpenAIEndpoint,
    *,
    context_length: int,
    temperature: float,
    max_output_tokens: int,
    seed: int | None,
) -> EvaluationSummary:
    if context_length != 65_536:
        raise ValueError("formal GPQA evaluation requires context_length=65536")
    results = []
    for row in rows:
        task_id = row["task_id"]
        try:
            completion = endpoint.complete(
                [{"role": "user", "content": row["prompt"]}],
                temperature=temperature,
                max_tokens=max_output_tokens,
                seed=seed,
            )
            raw = completion["content"]
            if completion["finish_reason"] == "length":
                result = TaskResult(task_id, "context_overflow", False, raw, completion)
            else:
                label, parse_status = parse_gpqa_broad(raw, options=row["options"])
                if label is None:
                    result = TaskResult(task_id, "format_error", False, raw, {"parse_status": parse_status})
                elif label == row["correct_label"]:
                    result = TaskResult(task_id, "correct", True, raw, {"label": label})
                else:
                    result = TaskResult(task_id, "wrong_answer", False, raw, {"label": label})
        except ContextOverflowError as error:
            result = TaskResult(task_id, "context_overflow", False, "", {"error": str(error)})
        except InfrastructureEndpointError as error:
            result = TaskResult(task_id, "infrastructure_error", None, "", {"error": str(error)})
        except EndpointError as error:
            result = TaskResult(task_id, "model_error", False, "", {"error": str(error)})
        results.append(result)
    return EvaluationSummary.build(
        "gpqa",
        [row["task_id"] for row in rows],
        results,
        subjects={row["task_id"]: row["subject"] for row in rows},
    )
