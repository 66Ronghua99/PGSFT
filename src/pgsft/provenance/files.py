"""File identity records with standalone-root confinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FileRecord:
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FileRecord":
        if set(value) != {"path", "sha256", "size_bytes"}:
            raise ValueError("file record fields must be path, sha256, and size_bytes")
        path = value["path"]
        digest = value["sha256"]
        size = value["size_bytes"]
        if not isinstance(path, str) or not path:
            raise ValueError("file record path must be non-empty")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("file record sha256 must be lowercase hexadecimal")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("file record size_bytes must be non-negative")
        return cls(path=path, sha256=digest, size_bytes=size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, *, root: Path) -> FileRecord:
    resolved = path.resolve(strict=True)
    root_resolved = root.resolve(strict=True)
    try:
        display = str(resolved.relative_to(root_resolved))
    except ValueError as error:
        raise ValueError(f"file is outside the standalone root: {resolved}") from error
    if not resolved.is_file():
        raise ValueError(f"not a regular file: {resolved}")
    return FileRecord(display, sha256_file(resolved), resolved.stat().st_size)


def verify_file_record(record: FileRecord, *, root: Path) -> Path:
    raw = Path(record.path)
    if raw.is_absolute():
        raise ValueError("file record path must be relative")
    resolved = (root.resolve() / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("file record path escapes the standalone root") from error
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    observed_size = resolved.stat().st_size
    if observed_size != record.size_bytes:
        raise ValueError(
            f"file size drift for {record.path}: {observed_size} != {record.size_bytes}"
        )
    observed_digest = sha256_file(resolved)
    if observed_digest != record.sha256:
        raise ValueError(
            f"file digest drift for {record.path}: {observed_digest} != {record.sha256}"
        )
    return resolved

