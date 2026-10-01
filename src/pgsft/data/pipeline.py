"""Deterministic canonicalization and paper split freezing."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any, Iterable, Sequence

from pgsft.data.canonical import CanonicalTrace, write_canonical_jsonl
from pgsft.data.sources import (
    FILTER_VERSION,
    NEMOTRON_REPO,
    NEMOTRON_REVISION,
    SourceRecordError,
    canonicalize_nemotron,
)
from pgsft.provenance.files import FileRecord, file_record, sha256_file
from pgsft.provenance.manifest import write_once_json


PAPER_SPLITS = {
    "qwen3": (4_992, 120),
    "qwen35": (1_000, 120),
}


def _ordered_id_digest(records: Sequence[CanonicalTrace]) -> str:
    payload = "".join(f"{record.id}\n" for record in records).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _source_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted([*path.rglob("*.jsonl"), *path.rglob("*.parquet")])
        if files:
            return files
    raise FileNotFoundError(f"no JSONL or Parquet source files under {path}")


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    line_number = 0
    for source_file in _source_files(path):
        if source_file.suffix == ".parquet":
            import pyarrow.parquet as parquet

            for batch in parquet.ParquetFile(source_file).iter_batches():
                for value in batch.to_pylist():
                    line_number += 1
                    if not isinstance(value, dict):
                        raise ValueError(f"source row must be a mapping at {source_file}:{line_number}")
                    yield line_number, value
            continue
        with source_file.open(encoding="utf-8") as handle:
            for line in handle:
                line_number += 1
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSON at {source_file}:{line_number}") from error
                if not isinstance(value, dict):
                    raise ValueError(f"source row must be a mapping at {source_file}:{line_number}")
                yield line_number, value


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def canonicalize_source(
    source: Path, output_dir: Path, *, backbone: str
) -> dict[str, Any]:
    """Canonicalize one ordered Nemotron JSONL inventory with a rejection ledger."""
    if backbone not in PAPER_SPLITS:
        raise ValueError(f"unsupported backbone: {backbone}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted: list[CanonicalTrace] = []
    rejections: list[dict[str, str]] = []
    seen: set[str] = set()
    scanned = 0
    for source_index, (line_number, row) in enumerate(_iter_jsonl(source)):
        scanned += 1
        display_id = row.get("uuid")
        row_id = display_id if isinstance(display_id, str) and display_id else f"line:{line_number}"
        if row_id in seen:
            rejections.append({"id": row_id, "stage": "identity", "reason": "duplicate_id"})
            continue
        seen.add(row_id)
        try:
            accepted.append(canonicalize_nemotron(row, source_index=source_index))
        except (SourceRecordError, ValueError) as error:
            rejections.append(
                {"id": row_id, "stage": "filter_or_convert", "reason": str(error)}
            )
    canonical_path = output_dir / "canonical.jsonl"
    rejection_path = output_dir / "rejections.jsonl"
    write_canonical_jsonl(canonical_path, accepted)
    _write_jsonl(rejection_path, rejections)
    rejection_summary = dict(sorted(Counter(row["reason"] for row in rejections).items()))
    source_files = _source_files(source)
    manifest = {
        "schema_version": 1,
        "backbone": backbone,
        "upstream": {"repo": NEMOTRON_REPO, "revision": NEMOTRON_REVISION},
        "source": {
            "path": source.name,
            "files": [file_record(path, root=source if source.is_dir() else source.parent).to_dict() for path in source_files],
        },
        "filter_version": FILTER_VERSION,
        "ordered_ids_sha256": _ordered_id_digest(accepted),
        "rows": {"scanned": scanned, "accepted": len(accepted), "rejected": len(rejections)},
        "duplicate_count": sum(row["reason"] == "duplicate_id" for row in rejections),
        "rejection_summary": rejection_summary,
        "outputs": {
            "canonical": {
                "path": canonical_path.name,
                "sha256": sha256_file(canonical_path),
                "size_bytes": canonical_path.stat().st_size,
            },
            "rejections": {
                "path": rejection_path.name,
                "sha256": sha256_file(rejection_path),
                "size_bytes": rejection_path.stat().st_size,
            },
        },
    }
    write_once_json(output_dir / "manifest.json", manifest)
    return manifest


def freeze_split(
    records: Sequence[CanonicalTrace],
    output_dir: Path,
    *,
    backbone: str,
    train_size: int | None = None,
    holdout_size: int | None = None,
    seed: int = 42,
    source_records: Sequence[FileRecord] = (),
    rejection_summary: dict[str, int] | None = None,
) -> dict[str, Any]:
    """Freeze ordered, disjoint train/holdout membership for one backbone."""
    if backbone not in PAPER_SPLITS:
        raise ValueError(f"unsupported backbone: {backbone}")
    default_train, default_holdout = PAPER_SPLITS[backbone]
    train_size = default_train if train_size is None else train_size
    holdout_size = default_holdout if holdout_size is None else holdout_size
    if train_size <= 0 or holdout_size <= 0:
        raise ValueError("split sizes must be positive")
    ids = [record.id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("cannot freeze duplicate trace IDs")
    if len(records) < train_size + holdout_size:
        raise ValueError(
            f"need {train_size + holdout_size} rows, received {len(records)}"
        )
    ordered = list(records)
    if backbone == "qwen35":
        random.Random(seed).shuffle(ordered)
    selected = ordered[: train_size + holdout_size]
    train = selected[:train_size]
    holdout = selected[train_size:]
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / f"train-{train_size}.jsonl"
    holdout_path = output_dir / f"holdout-{holdout_size}.jsonl"
    write_canonical_jsonl(train_path, train)
    write_canonical_jsonl(holdout_path, holdout)
    membership = [*train, *holdout]
    manifest = {
        "schema_version": 1,
        "backbone": backbone,
        "upstream": {"repo": NEMOTRON_REPO, "revision": NEMOTRON_REVISION},
        "source_files": [record.to_dict() for record in source_records],
        "filter_version": FILTER_VERSION,
        "seed": seed if backbone == "qwen35" else None,
        "ordered_ids_sha256": _ordered_id_digest(membership),
        "row_counts": {"train": len(train), "holdout": len(holdout)},
        "duplicate_count": 0,
        "rejection_summary": dict(sorted((rejection_summary or {}).items())),
        "splits": {
            "train": file_record(train_path, root=output_dir).to_dict(),
            "holdout": file_record(holdout_path, root=output_dir).to_dict(),
        },
    }
    write_once_json(output_dir / "manifest.json", manifest)
    return manifest
