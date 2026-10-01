"""Validate a formal recipe and launch the standalone training runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
from typing import Any, Sequence

import torch

from pgsft.config import load_yaml
from pgsft.methods.registry import METHOD_NAMES, get_method
from pgsft.model_profiles import load_model_profile
from pgsft.provenance.files import file_record
from pgsft.provenance.manifest import canonical_json_bytes, write_once_json


RECIPE_KEYS = {
    "schema_version",
    "model_profile",
    "method",
    "data",
    "templates",
    "lora",
    "training",
    "selection",
    "method_args",
    "evidence",
}


def load_recipe(path: Path, *, root: Path) -> dict[str, Any]:
    recipe = load_yaml(path)
    if set(recipe) != RECIPE_KEYS:
        raise ValueError(
            f"invalid recipe keys: missing={sorted(RECIPE_KEYS - set(recipe))} "
            f"unknown={sorted(set(recipe) - RECIPE_KEYS)}"
        )
    if recipe["schema_version"] != 1:
        raise ValueError("unsupported recipe schema_version")
    method = recipe["method"]
    if method not in METHOD_NAMES:
        raise ValueError(f"unsupported recipe method: {method}")
    profile = load_model_profile(recipe["model_profile"], root)
    templates = recipe["templates"]
    expected_templates = {
        "train": str(profile.config.train_template.relative_to(root)),
        "inference": str(profile.config.inference_template.relative_to(root)),
    }
    if templates != expected_templates:
        raise ValueError(f"recipe template binding drift: {templates} != {expected_templates}")
    data = recipe["data"]
    if set(data) != {"train_jsonl", "eval_jsonl", "manifest", "schema"}:
        raise ValueError("recipe data keys are invalid")
    required_schema = get_method(method).required_row_schema
    if data["schema"] != required_schema:
        raise ValueError(
            f"method {method} requires {required_schema} rows, got {data['schema']}"
        )
    for section in ("data", "templates"):
        for value in recipe[section].values():
            if isinstance(value, str) and Path(value).is_absolute():
                raise ValueError(f"recipe {section} contains an absolute path")
    evidence = recipe["evidence"]
    if evidence.get("protocol_status") not in {"paper_reported", "supported_unreported"}:
        raise ValueError("recipe evidence.protocol_status is invalid")
    training = recipe["training"]
    for key in ("learning_rate", "global_batch_size", "num_epochs", "max_steps", "seed"):
        if key not in training:
            raise ValueError(f"recipe training is missing {key}")
    return recipe


def _resolve_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"recipe path escapes standalone root: {value}") from error
    return path


def _validate_row_schema(path: Path, schema: str) -> None:
    with path.open(encoding="utf-8") as handle:
        line = handle.readline()
    if not line:
        raise ValueError(f"training data is empty: {path}")
    row = json.loads(line)
    required = (
        {"id", "input_ids", "labels", "assistant_turn_ids"}
        if schema == "sft"
        else {
            "id",
            "student_input_ids",
            "privileged_input_ids",
            "student_positions",
            "privileged_positions",
            "target_ids",
            "turn_indices",
            "token_indices",
        }
    )
    if not isinstance(row, dict) or not required <= set(row):
        raise ValueError(f"{path} does not satisfy the {schema} row schema")


def resolve_preflight(recipe: dict[str, Any], *, root: Path) -> dict[str, Any]:
    profile = load_model_profile(recipe["model_profile"], root)
    paths = {
        key: _resolve_path(root, recipe["data"][key])
        for key in ("train_jsonl", "eval_jsonl", "manifest")
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    _validate_row_schema(paths["train_jsonl"], recipe["data"]["schema"])
    _validate_row_schema(paths["eval_jsonl"], recipe["data"]["schema"])
    return {
        "schema_version": 1,
        "recipe": recipe,
        "model": {
            "name": profile.config.name,
            "revision": profile.config.revision,
            "train_template": profile.train_template_record.to_dict(),
            "inference_template": profile.inference_template_record.to_dict(),
        },
        "immutable_inputs": {
            key: file_record(path, root=root).to_dict() for key, path in paths.items()
        },
    }


def _environment() -> dict[str, Any]:
    packages = {}
    for name in ("torch", "transformers", "peft", "accelerate", "trl"):
        packages[name] = importlib.metadata.version(name)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
    }


def _prepare_run(
    run_dir: Path, resolved: dict[str, Any], *, resume: bool
) -> None:
    config_path = run_dir / "resolved_config.json"
    if resume:
        if not config_path.is_file():
            raise ValueError("resume run is missing resolved_config.json")
        if config_path.read_bytes() != canonical_json_bytes(resolved):
            raise ValueError("resume resolved-config or immutable-input drift")
        return
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    write_once_json(config_path, resolved)
    write_once_json(
        run_dir / "manifest.json",
        {"schema_version": 1, "status": "preflight_complete", "environment": _environment()},
    )
    (run_dir / "train.log").touch(exist_ok=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/train"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    recipe = load_recipe(args.config, root=root)
    resolved = resolve_preflight(recipe, root=root)
    run_dir = (args.output_root / args.run_id).resolve()
    _prepare_run(run_dir, resolved, resume=args.resume)
    if args.dry_run:
        print(f"preflight_ok run_dir={run_dir}")
        return 0
    from pgsft.training.trainer import run_training

    run_training(resolved, run_dir=run_dir, resume=args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
