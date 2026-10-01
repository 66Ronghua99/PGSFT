"""Pinned backbone profiles and their immutable chat-template records."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Any

from pgsft.config import ModelConfig, load_yaml
from pgsft.provenance.files import FileRecord, file_record, verify_file_record


ModelKey = Literal["qwen3", "qwen35"]
MODEL_KEYS: tuple[ModelKey, ...] = ("qwen3", "qwen35")
_PROFILE_KEYS = {
    "schema_version",
    "key",
    "model",
    "revision",
    "train_template",
    "inference_template",
    "student_max_length",
    "privileged_max_length",
}


@dataclass(frozen=True)
class ModelProfile:
    """A validated model config paired with content-addressed templates."""

    schema_version: int
    key: ModelKey
    config: ModelConfig
    train_template_record: FileRecord
    inference_template_record: FileRecord

    def verify_templates(self, *, root: Path) -> None:
        verify_file_record(self.train_template_record, root=root)
        verify_file_record(self.inference_template_record, root=root)


def _positive_schema_version(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("model profile schema_version must be a positive integer")
    return value


def _profile_mapping(value: Mapping[str, Any], *, expected_key: ModelKey) -> dict[str, Any]:
    unknown = set(value) - _PROFILE_KEYS
    missing = _PROFILE_KEYS - set(value)
    if missing:
        raise ValueError(f"model profile is missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"model profile has unknown keys: {sorted(unknown)}")
    if value["key"] != expected_key:
        raise ValueError(
            f"model profile key mismatch: expected {expected_key!r}, got {value['key']!r}"
        )
    return {
        "name": value["model"],
        "revision": value["revision"],
        "train_template": value["train_template"],
        "inference_template": value["inference_template"],
        "student_max_length": value["student_max_length"],
        "privileged_max_length": value["privileged_max_length"],
    }


def load_model_profile(name: ModelKey, root: Path) -> ModelProfile:
    """Load one named profile beneath ``root`` and hash both template contracts."""
    if name not in MODEL_KEYS:
        raise ValueError(f"unsupported model profile: {name}")
    root = root.resolve(strict=True)
    raw = load_yaml(root / "configs" / "models" / f"{name}.yaml")
    schema_version = _positive_schema_version(raw.get("schema_version"))
    config = ModelConfig.from_mapping(
        _profile_mapping(raw, expected_key=name), root=root
    )
    profile = ModelProfile(
        schema_version=schema_version,
        key=name,
        config=config,
        train_template_record=file_record(config.train_template, root=root),
        inference_template_record=file_record(config.inference_template, root=root),
    )
    profile.verify_templates(root=root)
    return profile


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=_default_root())
    parser.add_argument("--verify-all", action="store_true")
    args = parser.parse_args(argv)
    if not args.verify_all:
        parser.error("--verify-all is required")
    for key in MODEL_KEYS:
        profile = load_model_profile(key, args.root)
        print(f"{key} train {profile.train_template_record.sha256}")
        print(f"{key} inference {profile.inference_template_record.sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
