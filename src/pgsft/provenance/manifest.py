"""Canonical and atomic manifest serialization."""

from __future__ import annotations

import json
import os
from pathlib import Path


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_once_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing manifest: {path}")
    stage = path.with_name(f".{path.name}.partial")
    if stage.exists():
        raise FileExistsError(f"staging path already exists: {stage}")
    try:
        with stage.open("xb") as handle:
            handle.write(canonical_json_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"manifest appeared during write: {path}")
        stage.replace(path)
    except BaseException:
        stage.unlink(missing_ok=True)
        raise

