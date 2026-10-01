#!/usr/bin/env python3
"""Build the frozen LCB120 prompt/test file using the pinned official loader."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset
from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
from lcb_runner.prompts.code_generation import get_generic_question_template_answer
from pgsft.evaluation.livecodebench import load_lcb_inventory, prepare_lcb


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-script", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = load_dataset(str(args.dataset_script), split="test", version_tag="release_v6", trust_remote_code=True)
    rows = []
    for raw in dataset:
        problem = CodeGenerationProblem(**raw)
        rows.append({"question_id": str(problem.question_id), "user_prompt": get_generic_question_template_answer(problem),
                     "official_evaluation_sample": problem.get_evaluation_sample()})
    selected = prepare_lcb(rows, load_lcb_inventory(args.inventory))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(selected)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
