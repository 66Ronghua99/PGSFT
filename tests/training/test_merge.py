from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from pgsft.model_profiles import load_model_profile
from pgsft.provenance.manifest import write_once_json
from pgsft.training.merge import merge_selected_adapter, verify_merged_tensors


ROOT = Path(__file__).resolve().parents[2]


class FakeMerged:
    def __init__(self, finite: bool = True):
        self.finite = finite

    def save_pretrained(self, output: Path, *, safe_serialization: bool):
        value = torch.tensor([1.0 if self.finite else float("inf")])
        save_file({"weight": value}, output / "model.safetensors")
        (output / "config.json").write_text("{}\n")


class FakePeft:
    def __init__(self, finite: bool = True):
        self.finite = finite

    def merge_and_unload(self):
        return FakeMerged(self.finite)


def _run(root: Path) -> Path:
    run = root / "run"
    checkpoint = run / "checkpoint-8"
    checkpoint.mkdir(parents=True)
    (checkpoint / "adapter.safetensors").write_bytes(b"adapter")
    write_once_json(
        run / "selected_checkpoint.json",
        {
            "schema_version": 1,
            "rule": "fixed_final",
            "metric": None,
            "step": 8,
            "checkpoint": "checkpoint-8",
        },
    )
    return run


def test_merge_writes_verified_manifest(monkeypatch, tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "model.safetensors").write_bytes(b"base")
    monkeypatch.setattr(
        "pgsft.training.merge.snapshot_download", lambda *args, **kwargs: str(base)
    )
    output = tmp_path / "merged"
    manifest = merge_selected_adapter(
        run_dir=_run(tmp_path),
        output_dir=output,
        profile=load_model_profile("qwen3", ROOT),
        model_loader=lambda _: object(),
        adapter_loader=lambda _model, _path: FakePeft(),
    )
    assert manifest["tensor_count"] == 1
    assert len(manifest["base_sha256"]) == 64
    assert (output / "merge_manifest.json").is_file()


def test_merge_rejects_nonempty_output(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    (output / "existing").write_text("occupied")
    with pytest.raises(FileExistsError, match="not empty"):
        merge_selected_adapter(
            run_dir=_run(tmp_path),
            output_dir=output,
            profile=load_model_profile("qwen3", ROOT),
        )


def test_nonfinite_merged_tensor_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    save_file({"bad": torch.tensor([float("nan")])}, output / "model.safetensors")
    with pytest.raises(ValueError, match="non-finite"):
        verify_merged_tensors(output)
