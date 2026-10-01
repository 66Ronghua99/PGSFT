"""Compact SWE-bench tools-v2 agent runner and official-harness bridge."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

from pgsft.evaluation.common import EvaluationSummary, TaskResult
from pgsft.evaluation.swebench_protocol import extract_tool_calls


MAX_ITERATIONS = 150
SYSTEM_PROMPT = """You are a software engineer fixing a bug in /testbed.
Think first, then emit exactly one tool call:
<tool_call>{"name":"execute_bash","arguments":{"command":"..."}}</tool_call>
Tools: execute_bash, str_replace_editor (view/create/str_replace/insert/undo_edit), and
finish. Each shell call starts fresh in /testbed. Use finish only after editing and testing.
"""


def load_swebench_inventory(path: Path) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    ids = value["instance_ids"]
    if len(ids) != 89 or len(set(ids)) != 89:
        raise ValueError("SWE-bench inventory must contain 89 unique tasks")
    return ids


def _docker(*args: str, input_text: str | None = None, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["docker", *args], input=input_text, text=True, capture_output=True, check=check)


def capture_workspace_patch(container: str) -> str:
    """Capture tracked and non-ignored untracked changes without touching HEAD/index."""
    script = r'''set -e
cd /testbed
idx=$(mktemp)
rm -f "$idx"
trap 'rm -f "$idx"' EXIT
GIT_INDEX_FILE="$idx" git read-tree HEAD
GIT_INDEX_FILE="$idx" git add -A
GIT_INDEX_FILE="$idx" git diff --cached --binary HEAD
'''
    return _docker("exec", container, "bash", "-lc", script).stdout


def _run_instance(row: dict[str, Any], endpoint: Any, *, editor: Path, max_output_tokens: int, temperature: float) -> dict[str, Any]:
    instance_id = row["instance_id"]
    image_id = instance_id.replace("__", "_1776_")
    image = f"swebench/sweb.eval.x86_64.{image_id}:latest"
    container = "pgsft_" + instance_id.replace("__", "_").replace("-", "_")[:50]
    _docker("run", "-d", "--rm", "--name", container, image, "sleep", "infinity")
    try:
        _docker("cp", str(editor), f"{container}:/tmp/pgsft_editor.py")
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["problem_statement"]}]
        repeated_errors = 0
        trajectory = []
        for iteration in range(1, MAX_ITERATIONS + 1):
            completion = endpoint.complete(messages, temperature=temperature, max_tokens=max_output_tokens, seed=None)
            content = completion["content"]
            calls, count, action, error = extract_tool_calls(content, thinking_open=True)
            messages.append({"role": "assistant", "content": content})
            if error or count != 1 or action is None:
                observation = f"Format error: {error or 'emit exactly one complete tool call'}"
                repeated_errors += 1
                if repeated_errors >= 3:
                    return {"instance_id": instance_id, "patch": capture_workspace_patch(container), "status": "repeated_format_error", "trajectory": trajectory}
            else:
                repeated_errors = 0
                name, arguments = action.get("name"), action.get("arguments", {})
                if name == "execute_bash" and isinstance(arguments.get("command"), str):
                    proc = _docker("exec", "-w", "/testbed", container, "bash", "-lc", arguments["command"], check=False)
                    observation = (proc.stdout + proc.stderr)[-20000:]
                elif name == "str_replace_editor" and isinstance(arguments, dict):
                    proc = _docker("exec", "-i", "-w", "/testbed", container, "python", "/tmp/pgsft_editor.py", input_text=json.dumps(arguments), check=False)
                    observation = (proc.stdout + proc.stderr)[-20000:]
                elif name == "finish":
                    patch = capture_workspace_patch(container)
                    if patch.strip():
                        return {"instance_id": instance_id, "patch": patch, "status": "submitted", "trajectory": trajectory}
                    observation = "Submission rejected: make a code change before finish."
                else:
                    observation = "Tool error: unknown tool or invalid arguments."
            trajectory.append({"iteration": iteration, "assistant": content, "observation": observation})
            messages.append({"role": "tool", "content": observation})
        return {"instance_id": instance_id, "patch": capture_workspace_patch(container), "status": "max_iterations", "trajectory": trajectory}
    finally:
        _docker("rm", "-f", container, check=False)


def write_predictions(results: Sequence[dict[str, Any]], path: Path, model_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps({"instance_id": row["instance_id"], "model_name_or_path": model_name, "model_patch": row.get("patch", "")}, ensure_ascii=False) + "\n")


def validate_swebench_run(inventory: Sequence[str], predictions: Sequence[dict[str, Any]], report: dict[str, Any]) -> None:
    ids = [row["instance_id"] for row in predictions]
    if ids != list(inventory) or len(ids) != len(set(ids)):
        raise ValueError("SWE-bench prediction coverage/order differs from inventory")
    reported = set(report.get("resolved_ids", [])) | set(report.get("unresolved_ids", [])) | set(report.get("error_ids", []))
    if reported != set(inventory):
        raise ValueError("official SWE-bench report coverage differs from inventory")


def run_swebench(recipe: dict[str, Any], endpoint: Any, *, root: Path, output_dir: Path) -> EvaluationSummary:
    from datasets import load_dataset

    ids = load_swebench_inventory(root / recipe["inventory"])
    revision = recipe["benchmark_args"]["dataset_revision"]
    dataset = load_dataset("princeton-nlp/SWE-bench_Lite", split="test", revision=revision)
    by_id = {row["instance_id"]: row for row in dataset}
    editor = Path(__file__).with_name("swebench_editor.py")
    results = [_run_instance(by_id[task_id], endpoint, editor=editor, max_output_tokens=int(recipe["max_output_tokens"]), temperature=float(recipe["temperature"])) for task_id in ids]
    predictions_path = output_dir / "predictions.jsonl"
    write_predictions(results, predictions_path, endpoint.served_name)
    subprocess.run(["bash", str(root / "scripts/run_swebench_harness.sh"), str(predictions_path), str(output_dir / "report.json"), recipe["benchmark_args"].get("workers", "4")], check=True, cwd=root)
    predictions = [json.loads(line) for line in predictions_path.open()]
    report = json.loads((output_dir / "report.json").read_text())
    validate_swebench_run(ids, predictions, report)
    resolved = set(report["resolved_ids"])
    normalized = [TaskResult(task_id, "correct" if task_id in resolved else "wrong_answer", task_id in resolved, "", {}) for task_id in ids]
    return EvaluationSummary.build("swebench", ids, normalized)
