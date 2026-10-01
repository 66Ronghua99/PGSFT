from __future__ import annotations

from pathlib import Path

import pytest

from pgsft.config import EvaluationConfig, ModelConfig, TrainingConfig, load_yaml


def model(root: Path, *, privileged: bool = True) -> dict:
    value = {
        "name": "org/model",
        "revision": "abc123",
        "train_template": "templates/train.jinja",
        "inference_template": "templates/inference.jinja",
        "student_max_length": 65536,
    }
    if privileged:
        value["privileged_max_length"] = 84992
    return value


def training(root: Path) -> dict:
    return {
        "schema_version": 1,
        "method": "pg_sft",
        "model": model(root),
        "train_jsonl": "data/train.jsonl",
        "eval_jsonl": "data/eval.jsonl",
        "data_manifest": "data/manifest.json",
        "output_root": "artifacts/train",
        "seed": 42,
        "global_batch_size": 32,
        "learning_rate": 5e-5,
        "max_steps": 32,
        "num_epochs": 1,
        "method_args": {"chunk_size": 128},
    }


def evaluation(root: Path) -> dict:
    return {
        "schema_version": 1,
        "benchmark": "gpqa",
        "model": model(root),
        "inventory": "data/gpqa.json",
        "output_root": "artifacts/eval",
        "context_length": 65536,
        "temperature": 0,
        "seed": 42,
        "max_output_tokens": 8192,
        "benchmark_args": {},
    }


def test_load_yaml_requires_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("- one\n- two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml(path)


def test_model_resolves_paths_inside_root(tmp_path: Path) -> None:
    observed = ModelConfig.from_mapping(model(tmp_path), root=tmp_path)
    assert observed.student_max_length == 65536
    assert observed.train_template == (tmp_path / "templates/train.jinja").resolve()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.update(extra=True), "unknown keys"),
        (lambda value: value.update(method="unknown"), "unsupported training method"),
        (lambda value: value.update(global_batch_size=0), "positive integer"),
        (lambda value: value.update(learning_rate=float("nan")), "positive and finite"),
        (lambda value: value.update(train_jsonl="../escape.jsonl"), "escapes"),
        (
            lambda value: value["model"].pop("privileged_max_length"),
            "requires model.privileged_max_length",
        ),
    ],
)
def test_training_rejects_invalid_contract(tmp_path: Path, mutate, message: str) -> None:
    value = training(tmp_path)
    mutate(value)
    with pytest.raises(ValueError, match=message):
        TrainingConfig.from_mapping(value, root=tmp_path)


def test_evaluation_contract(tmp_path: Path) -> None:
    observed = EvaluationConfig.from_mapping(evaluation(tmp_path), root=tmp_path)
    assert observed.benchmark == "gpqa"
    assert observed.context_length == 65536


def test_evaluation_rejects_unknown_benchmark(tmp_path: Path) -> None:
    value = evaluation(tmp_path)
    value["benchmark"] = "other"
    with pytest.raises(ValueError, match="unsupported benchmark"):
        EvaluationConfig.from_mapping(value, root=tmp_path)

