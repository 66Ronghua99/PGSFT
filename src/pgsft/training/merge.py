"""Offline selected-adapter merge and tensor-integrity verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from pgsft.model_profiles import ModelProfile
from pgsft.provenance.manifest import write_once_json
from pgsft.training.checkpoint import load_selection


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def verify_merged_tensors(output_dir: Path) -> int:
    paths = sorted(output_dir.glob("*.safetensors"))
    if not paths:
        raise ValueError("merged output contains no safetensors weights")
    tensors = 0
    for path in paths:
        with safe_open(path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                value = handle.get_tensor(key)
                tensors += 1
                if value.dtype.is_floating_point and not bool(torch.isfinite(value).all()):
                    raise ValueError(f"merged tensor is non-finite: {path.name}:{key}")
    return tensors


def merge_selected_adapter(
    *,
    run_dir: Path,
    output_dir: Path,
    profile: ModelProfile,
    model_loader: Callable[[Path], torch.nn.Module] | None = None,
    adapter_loader: Callable[[torch.nn.Module, Path], torch.nn.Module] | None = None,
) -> dict:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"merge output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    selection = load_selection(run_dir)
    adapter = run_dir / selection["checkpoint"]
    base = Path(
        snapshot_download(
            profile.config.name,
            revision=profile.config.revision,
            local_files_only=True,
        )
    ).resolve()
    if model_loader is None:
        if profile.key == "qwen3":
            from transformers import AutoModelForCausalLM

            model_loader = lambda path: AutoModelForCausalLM.from_pretrained(
                path, local_files_only=True, dtype=torch.bfloat16
            )
        else:
            from transformers import AutoModelForImageTextToText

            model_loader = lambda path: AutoModelForImageTextToText.from_pretrained(
                path, local_files_only=True, dtype=torch.bfloat16
            )
    if adapter_loader is None:
        from peft import PeftModel

        adapter_loader = lambda model, path: PeftModel.from_pretrained(
            model, path, local_files_only=True
        )
    model = model_loader(base)
    peft_model = adapter_loader(model, adapter)
    merge = getattr(peft_model, "merge_and_unload", None)
    if not callable(merge):
        raise TypeError("adapter loader did not return a mergeable PEFT model")
    merged = merge()
    merged.save_pretrained(output_dir, safe_serialization=True)
    tensor_count = verify_merged_tensors(output_dir)
    manifest = {
        "schema_version": 1,
        "model": profile.config.name,
        "revision": profile.config.revision,
        "base_sha256": directory_sha256(base),
        "adapter_sha256": directory_sha256(adapter),
        "output_sha256": directory_sha256(output_dir),
        "tensor_count": tensor_count,
        "selection": selection,
    }
    write_once_json(output_dir / "merge_manifest.json", manifest)
    return manifest
