"""Strict typed configuration for data, training, and evaluation runs."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml


MethodName = Literal["sft", "dft", "sft_base_kl", "psft", "pg_sft"]
BenchmarkName = Literal["gpqa", "bfcl", "livecodebench", "swebench"]

METHODS = frozenset({"sft", "dft", "sft_base_kl", "psft", "pg_sft"})
BENCHMARKS = frozenset({"gpqa", "bfcl", "livecodebench", "swebench"})


def load_yaml(path: Path) -> dict[str, Any]:
    """Load one YAML mapping without accepting an empty or non-mapping file."""
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"configuration keys must be strings: {path}")
    return value


def _strict_keys(
    value: Mapping[str, Any], *, required: set[str], optional: set[str], label: str
) -> None:
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise ValueError(f"{label} is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown keys: {sorted(unknown)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _finite_float(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (positive and parsed <= 0):
        qualifier = "positive and " if positive else ""
        raise ValueError(f"{label} must be {qualifier}finite")
    return parsed


def _path(value: Any, label: str, root: Path) -> Path:
    raw = Path(_text(value, label))
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (root / raw).resolve()
        try:
            resolved.relative_to(root.resolve())
        except ValueError as error:
            raise ValueError(f"{label} escapes the standalone root") from error
    return resolved


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str
    train_template: Path
    inference_template: Path
    student_max_length: int
    privileged_max_length: int | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any], *, root: Path) -> "ModelConfig":
        _strict_keys(
            value,
            required={
                "name",
                "revision",
                "train_template",
                "inference_template",
                "student_max_length",
            },
            optional={"privileged_max_length"},
            label="model configuration",
        )
        privileged = value.get("privileged_max_length")
        return cls(
            name=_text(value["name"], "model.name"),
            revision=_text(value["revision"], "model.revision"),
            train_template=_path(value["train_template"], "model.train_template", root),
            inference_template=_path(
                value["inference_template"], "model.inference_template", root
            ),
            student_max_length=_positive_int(
                value["student_max_length"], "model.student_max_length"
            ),
            privileged_max_length=(
                None
                if privileged is None
                else _positive_int(privileged, "model.privileged_max_length")
            ),
        )


@dataclass(frozen=True)
class TrainingConfig:
    schema_version: int
    method: MethodName
    model: ModelConfig
    train_jsonl: Path
    eval_jsonl: Path
    data_manifest: Path
    output_root: Path
    seed: int
    global_batch_size: int
    learning_rate: float
    max_steps: int | None
    num_epochs: float
    method_args: dict[str, Any]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, root: Path
    ) -> "TrainingConfig":
        _strict_keys(
            value,
            required={
                "schema_version",
                "method",
                "model",
                "train_jsonl",
                "eval_jsonl",
                "data_manifest",
                "output_root",
                "seed",
                "global_batch_size",
                "learning_rate",
                "num_epochs",
                "method_args",
            },
            optional={"max_steps"},
            label="training configuration",
        )
        method = _text(value["method"], "method")
        if method not in METHODS:
            raise ValueError(f"unsupported training method: {method}")
        model_value = value["model"]
        if not isinstance(model_value, Mapping):
            raise ValueError("model must be a mapping")
        model = ModelConfig.from_mapping(model_value, root=root)
        if method == "pg_sft" and model.privileged_max_length is None:
            raise ValueError("PG-SFT requires model.privileged_max_length")
        method_args = value["method_args"]
        if not isinstance(method_args, dict) or not all(
            isinstance(key, str) for key in method_args
        ):
            raise ValueError("method_args must be a string-keyed mapping")
        max_steps = value.get("max_steps")
        return cls(
            schema_version=_positive_int(value["schema_version"], "schema_version"),
            method=method,  # type: ignore[arg-type]
            model=model,
            train_jsonl=_path(value["train_jsonl"], "train_jsonl", root),
            eval_jsonl=_path(value["eval_jsonl"], "eval_jsonl", root),
            data_manifest=_path(value["data_manifest"], "data_manifest", root),
            output_root=_path(value["output_root"], "output_root", root),
            seed=int(value["seed"]),
            global_batch_size=_positive_int(
                value["global_batch_size"], "global_batch_size"
            ),
            learning_rate=_finite_float(
                value["learning_rate"], "learning_rate", positive=True
            ),
            max_steps=(
                None if max_steps is None else _positive_int(max_steps, "max_steps")
            ),
            num_epochs=_finite_float(value["num_epochs"], "num_epochs", positive=True),
            method_args=dict(method_args),
        )


@dataclass(frozen=True)
class EvaluationConfig:
    schema_version: int
    benchmark: BenchmarkName
    model: ModelConfig
    inventory: Path
    output_root: Path
    context_length: int
    temperature: float
    seed: int | None
    max_output_tokens: int
    benchmark_args: dict[str, Any]

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, root: Path
    ) -> "EvaluationConfig":
        _strict_keys(
            value,
            required={
                "schema_version",
                "benchmark",
                "model",
                "inventory",
                "output_root",
                "context_length",
                "temperature",
                "max_output_tokens",
                "benchmark_args",
            },
            optional={"seed"},
            label="evaluation configuration",
        )
        benchmark = _text(value["benchmark"], "benchmark")
        if benchmark not in BENCHMARKS:
            raise ValueError(f"unsupported benchmark: {benchmark}")
        model_value = value["model"]
        if not isinstance(model_value, Mapping):
            raise ValueError("model must be a mapping")
        args = value["benchmark_args"]
        if not isinstance(args, dict) or not all(isinstance(key, str) for key in args):
            raise ValueError("benchmark_args must be a string-keyed mapping")
        seed = value.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("seed must be an integer or null")
        return cls(
            schema_version=_positive_int(value["schema_version"], "schema_version"),
            benchmark=benchmark,  # type: ignore[arg-type]
            model=ModelConfig.from_mapping(model_value, root=root),
            inventory=_path(value["inventory"], "inventory", root),
            output_root=_path(value["output_root"], "output_root", root),
            context_length=_positive_int(value["context_length"], "context_length"),
            temperature=_finite_float(value["temperature"], "temperature"),
            seed=seed,
            max_output_tokens=_positive_int(
                value["max_output_tokens"], "max_output_tokens"
            ),
            benchmark_args=dict(args),
        )


def load_training_config(path: Path, *, root: Path) -> TrainingConfig:
    return TrainingConfig.from_mapping(load_yaml(path), root=root)


def load_evaluation_config(path: Path, *, root: Path) -> EvaluationConfig:
    return EvaluationConfig.from_mapping(load_yaml(path), root=root)

