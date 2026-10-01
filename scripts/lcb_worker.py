#!/usr/bin/env python3
"""Official LiveCodeBench extraction and code-generation metric worker."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import sys


def grade(request: dict) -> dict:
    from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    code = extract_code(request["final"], LMStyle.OpenAIChat)
    if not code:
        return {"task_id": request["task_id"], "status": "format_error", "correct": False, "code": ""}
    try:
        ast.parse(code)
    except SyntaxError:
        return {"task_id": request["task_id"], "status": "syntax_error", "correct": False, "code": code}
    raw = codegen_metrics([request["official_evaluation_sample"]], [[code]], k_list=[1], num_process_evaluate=1, timeout=6)
    if not isinstance(raw, list) or len(raw) != 3:
        raise ValueError("malformed official grader output")
    _, results, metadata = raw
    rows = next(iter(results.values()))
    outcomes = rows[0]
    meta = json.loads(metadata[0][0])
    if meta.get("error_code") == -5 or meta.get("error_message") == "TestRunnerError":
        status, correct = "infrastructure_error", None
    elif all(value is True for value in outcomes):
        status, correct = "correct", True
    elif meta.get("error_code") in {-3, -4}:
        status, correct = "runtime_error", False
    else:
        status, correct = "wrong_answer", False
    return {"task_id": request["task_id"], "status": status, "correct": correct, "code": code, "outcomes": outcomes, "metadata": meta}


def main() -> int:
    if len(sys.argv) != 4 or sys.argv[1] != "grade":
        raise SystemExit("usage: lcb_worker.py grade REQUEST RESULT")
    request = json.loads(Path(sys.argv[2]).read_text())
    Path(sys.argv[3]).write_text(json.dumps(grade(request), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
