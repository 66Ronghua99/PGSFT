"""BFCL v4 multi-turn adapter for an OpenAI-compatible local model.

The official ``bfcl-eval`` package owns task construction, state transitions and
grading.  This module supplies the small model adapter needed by the paper: it
keeps reasoning separate and recognizes Qwen ``<tool_call>`` fallback blocks.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Sequence

from pgsft.evaluation.common import EvaluationSummary, TaskResult


CATEGORIES = (
    "multi_turn_base",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_miss_func",
)


def fallback_tool_calls(content: str) -> list[dict[str, Any]]:
    """Convert only complete, schema-valid Qwen fallback tool calls."""
    calls: list[dict[str, Any]] = []
    for raw in re.findall(
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>", content, flags=re.DOTALL
    ):
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(payload, dict)
            or not isinstance(payload.get("name"), str)
            or not isinstance(payload.get("arguments", {}), dict)
        ):
            continue
        calls.append(
            {
                "id": f"call_{len(calls)}",
                "type": "function",
                "function": {
                    "name": payload["name"],
                    "arguments": json.dumps(
                        payload.get("arguments", {}), ensure_ascii=False
                    ),
                },
            }
        )
    return calls


def split_reasoning(content: str, reasoning: str | None = None) -> tuple[str, str]:
    """Return reasoning and visible assistant content without inventing either."""
    if reasoning is not None:
        return reasoning, content
    before, marker, after = content.rpartition("</think>")
    if marker:
        return before.removeprefix("<think>"), after.lstrip()
    return "", content


def load_bfcl_inventory(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["tasks"] if isinstance(value, dict) else value
    ids = [row["task_id"] for row in rows]
    counts = Counter(row["category"] for row in rows)
    if len(ids) != 200 or len(set(ids)) != 200:
        raise ValueError("BFCL inventory must contain 200 unique tasks")
    if counts != Counter({category: 50 for category in CATEGORIES}):
        raise ValueError("BFCL inventory must contain 50 tasks in every category")
    return rows


def summarize_official_bfcl(
    inventory: Sequence[dict[str, str]], official_records: Sequence[dict[str, Any]]
) -> EvaluationSummary:
    """Normalize one official BFCL result per task into the shared report."""
    by_id = {row["id"]: row for row in official_records}
    results = []
    for task in inventory:
        task_id = task["task_id"]
        row = by_id.get(task_id)
        if row is None:
            results.append(TaskResult(task_id, "infrastructure_error", None, "", {"error": "missing official result"}))
        elif row.get("valid") is True or row.get("success") is True:
            results.append(TaskResult(task_id, "correct", True, row.get("model_output", ""), row))
        else:
            results.append(TaskResult(task_id, "wrong_answer", False, row.get("model_output", ""), row))
    return EvaluationSummary.build(
        "bfcl",
        [row["task_id"] for row in inventory],
        results,
        subjects={row["task_id"]: row["category"] for row in inventory},
    )


def run_bfcl(recipe: dict[str, Any], endpoint: Any, *, root: Path, output_dir: Path | None = None) -> EvaluationSummary:
    """Run the official bridge and summarize its JSON output.

    The bridge is intentionally a separate process in ``envs/bfcl`` so its
    dependency closure cannot perturb the training environment.
    """
    import subprocess
    import tempfile

    if recipe["context_length"] != 65_536:
        raise ValueError("formal BFCL evaluation requires 65536 context")
    inventory_path = root / recipe["inventory"]
    inventory = load_bfcl_inventory(inventory_path)
    target = output_dir or Path(tempfile.mkdtemp(prefix="pgsft-bfcl-"))
    target.mkdir(parents=True, exist_ok=True)
    command = [
        "uv", "run", "--directory", str(root / "envs/bfcl"),
        "python", str(root / "scripts/bfcl_official.py"),
        "--inventory", str(inventory_path), "--output", str(target),
        "--base-url", endpoint.base_url, "--served-model", endpoint.served_name,
        "--model-root", str(endpoint.model_root), "--context-length", "65536",
    ]
    subprocess.run(command, check=True, cwd=root)
    records = [json.loads(line) for line in (target / "official-results.jsonl").open()]
    return summarize_official_bfcl(inventory, records)
