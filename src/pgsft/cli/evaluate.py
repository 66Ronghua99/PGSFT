"""Run one frozen benchmark against a validated existing vLLM endpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from pgsft.config import load_yaml
from pgsft.evaluation.gpqa import load_gpqa_inventory, run_gpqa
from pgsft.evaluation.serving import OpenAIEndpoint
from pgsft.model_profiles import load_model_profile
from pgsft.provenance.manifest import write_once_json


EVAL_KEYS = {
    "schema_version",
    "benchmark",
    "model_profile",
    "inventory",
    "inference_template",
    "served_model_name",
    "context_length",
    "temperature",
    "seed",
    "max_output_tokens",
    "benchmark_args",
}


def load_evaluation_recipe(path: Path, *, root: Path) -> dict[str, Any]:
    value = load_yaml(path)
    if set(value) != EVAL_KEYS or value["schema_version"] != 1:
        raise ValueError("invalid evaluation recipe schema")
    profile = load_model_profile(value["model_profile"], root)
    expected = str(profile.config.inference_template.relative_to(root))
    if value["inference_template"] != expected:
        raise ValueError("evaluation inference template drift")
    if value["context_length"] != 65_536:
        raise ValueError("formal evaluation requires context_length=65536")
    inventory = (root / value["inventory"]).resolve()
    try:
        inventory.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("evaluation inventory escapes standalone root") from error
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    recipe = load_evaluation_recipe(args.config, root=root)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"evaluation output is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    endpoint = OpenAIEndpoint(
        args.endpoint, recipe["served_model_name"], args.model_root
    )
    service = endpoint.validate()
    if recipe["benchmark"] == "gpqa":
        rows = load_gpqa_inventory(root / recipe["inventory"])
        summary = run_gpqa(
            rows,
            endpoint,
            context_length=recipe["context_length"],
            temperature=float(recipe["temperature"]),
            max_output_tokens=int(recipe["max_output_tokens"]),
            seed=recipe["seed"],
        )
    elif recipe["benchmark"] == "bfcl":
        from pgsft.evaluation.bfcl import run_bfcl

        summary = run_bfcl(recipe, endpoint, root=root, output_dir=args.output_dir)
    elif recipe["benchmark"] == "livecodebench":
        from pgsft.evaluation.livecodebench import run_lcb

        summary = run_lcb(recipe, endpoint, root=root, output_dir=args.output_dir)
    elif recipe["benchmark"] == "swebench":
        from pgsft.evaluation.swebench import run_swebench

        summary = run_swebench(recipe, endpoint, root=root, output_dir=args.output_dir)
    else:
        raise ValueError(f"unsupported benchmark: {recipe['benchmark']}")
    write_once_json(args.output_dir / "results.json", summary.to_dict())
    manifest = {
        "schema_version": 1,
        "recipe": recipe,
        "service": service,
        "context_length": 65_536,
        "model_root": args.model_root.name,
    }
    write_once_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps({"correct": summary.correct, "denominator": summary.denominator}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
