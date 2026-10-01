from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgsft.cli.train import _prepare_run, _validate_row_schema, load_recipe


ROOT = Path(__file__).resolve().parents[2]
METHODS = {"sft", "dft", "sft_base_kl", "psft", "pg_sft"}


def test_all_ten_recipes_validate_and_bind_release_templates() -> None:
    observed = set()
    for backbone in ("qwen3", "qwen35"):
        for method in METHODS:
            path = ROOT / "configs" / "training" / backbone / f"{method}.yaml"
            recipe = load_recipe(path, root=ROOT)
            observed.add((backbone, method))
            assert recipe["model_profile"] == backbone
            assert recipe["method"] == method
            assert recipe["lora"] == {
                "r": 32,
                "alpha": 64,
                "dropout": 0.0,
                "target_modules": "all-linear",
                "exclude_modules": ["lm_head"],
            }
            assert recipe["training"]["global_batch_size"] == 32
            assert recipe["training"]["precision"] == "bf16"
            assert recipe["evidence"]["protocol_status"] in {
                "paper_reported",
                "supported_unreported",
            }
            serialized = json.dumps(recipe)
            assert "/home/" not in serialized and "AgentTrainer" not in serialized
            assert recipe["data"]["schema"] == (
                "paired" if method == "pg_sft" else "sft"
            )
    assert len(observed) == 10


def test_reported_scales_and_method_specific_arguments_are_frozen() -> None:
    qwen3_pg = load_recipe(
        ROOT / "configs/training/qwen3/pg_sft.yaml", root=ROOT
    )
    assert qwen3_pg["training"]["max_steps"] == 156
    assert qwen3_pg["selection"] == {"rule": "fixed_final", "step": 156}
    assert qwen3_pg["method_args"]["alpha_scope"] == "assistant_turn"
    qwen35_psft = load_recipe(
        ROOT / "configs/training/qwen35/psft.yaml", root=ROOT
    )
    assert qwen35_psft["training"]["learning_rate"] == 1e-6
    assert qwen35_psft["training"]["num_epochs"] == 3
    assert qwen35_psft["training"]["max_steps"] == 96
    assert qwen35_psft["method_args"] == {"epsilon_high": 0.28, "refresh_steps": 8}
    base_kl = load_recipe(
        ROOT / "configs/training/qwen35/sft_base_kl.yaml", root=ROOT
    )
    assert base_kl["method_args"]["base_kl_coefficient"] == 0.1


def test_schema_preflight_nonempty_run_and_resume_drift(tmp_path: Path) -> None:
    good = tmp_path / "sft.jsonl"
    good.write_text(
        '{"id":"x","input_ids":[1],"labels":[1],"assistant_turn_ids":[0]}\n'
    )
    _validate_row_schema(good, "sft")
    with pytest.raises(ValueError, match="paired"):
        _validate_row_schema(good, "paired")

    resolved = {"schema_version": 1, "value": "first"}
    run = tmp_path / "run"
    _prepare_run(run, resolved, resume=False)
    with pytest.raises(FileExistsError, match="not empty"):
        _prepare_run(run, resolved, resume=False)
    _prepare_run(run, resolved, resume=True)
    with pytest.raises(ValueError, match="drift"):
        _prepare_run(run, {"schema_version": 1, "value": "changed"}, resume=True)
