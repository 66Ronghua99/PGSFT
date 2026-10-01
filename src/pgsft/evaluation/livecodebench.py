"""LiveCodeBench release-v6 preparation, generation and Docker grading."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Sequence

from pgsft.evaluation.common import EvaluationSummary, TaskResult
from pgsft.evaluation.serving import ContextOverflowError, EndpointError, InfrastructureEndpointError


QUOTAS = {"easy": 37, "medium": 43, "hard": 40}


def load_lcb_inventory(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    rows = value["tasks"] if isinstance(value, dict) else value
    ids = [row["task_id"] for row in rows]
    if len(ids) != 120 or len(set(ids)) != 120:
        raise ValueError("LCB inventory must contain 120 unique tasks")
    if Counter(row["difficulty"] for row in rows) != Counter(QUOTAS):
        raise ValueError("LCB inventory difficulty quotas differ from 37/43/40")
    return rows


def prepare_lcb(all_rows: Sequence[dict[str, Any]], inventory: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Select the frozen IDs from rows produced by the pinned official loader."""
    by_id = {str(row["question_id"]): row for row in all_rows}
    prepared = []
    for item in inventory:
        row = by_id[item["task_id"]]
        prepared.append({
            "task_id": item["task_id"],
            "difficulty": item["difficulty"],
            "prompt": row["user_prompt"],
            "official_evaluation_sample": row["official_evaluation_sample"],
        })
    return prepared


def score_lcb(request: dict[str, Any], *, root: Path, image: str) -> dict[str, Any]:
    """Grade one extracted completion in the isolated official LCB image."""
    with tempfile.TemporaryDirectory(prefix="pgsft-lcb-") as directory:
        work = Path(directory)
        request_path = work / "request.json"
        result_path = work / "result.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        subprocess.run([
            "docker", "run", "--rm", "--network=none", "--memory=4g", "--cpus=2",
            "-v", f"{work}:/work", image, "grade", "/work/request.json", "/work/result.json",
        ], check=True, cwd=root)
        return json.loads(result_path.read_text(encoding="utf-8"))


def run_lcb(recipe: dict[str, Any], endpoint: Any, *, root: Path, output_dir: Path | None = None) -> EvaluationSummary:
    if recipe["context_length"] != 65_536:
        raise ValueError("formal LiveCodeBench evaluation requires 65536 context")
    inventory = load_lcb_inventory(root / recipe["inventory"])
    prepared_path = root / recipe["benchmark_args"]["prepared_data"]
    prepared = [json.loads(line) for line in prepared_path.open(encoding="utf-8")]
    if [row["task_id"] for row in prepared] != [row["task_id"] for row in inventory]:
        raise ValueError("prepared LCB rows differ from frozen inventory")
    results = []
    for row in prepared:
        task_id = row["task_id"]
        try:
            completion = endpoint.complete(
                [{"role": "user", "content": row["prompt"]}],
                temperature=float(recipe["temperature"]),
                max_tokens=int(recipe["max_output_tokens"]), seed=recipe["seed"],
            )
            if completion["finish_reason"] == "length":
                result = TaskResult(task_id, "context_overflow", False, completion["content"], completion)
            else:
                graded = score_lcb(
                    {"task_id": task_id, "final": completion["content"], "official_evaluation_sample": row["official_evaluation_sample"]},
                    root=root, image=recipe["benchmark_args"]["docker_image"],
                )
                status = graded["status"]
                if status == "infrastructure_error":
                    result = TaskResult(task_id, status, None, completion["content"], graded)
                else:
                    correct = bool(graded.get("correct"))
                    result = TaskResult(task_id, "correct" if correct else status, correct, completion["content"], graded)
        except ContextOverflowError as error:
            result = TaskResult(task_id, "context_overflow", False, "", {"error": str(error)})
        except InfrastructureEndpointError as error:
            result = TaskResult(task_id, "infrastructure_error", None, "", {"error": str(error)})
        except EndpointError as error:
            result = TaskResult(task_id, "model_error", False, "", {"error": str(error)})
        results.append(result)
    return EvaluationSummary.build(
        "livecodebench", [row["task_id"] for row in inventory], results,
        subjects={row["task_id"]: row["difficulty"] for row in inventory},
    )
