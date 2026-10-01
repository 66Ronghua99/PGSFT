from __future__ import annotations

import json
from pathlib import Path

import pytest

from pgsft.provenance import (
    FileRecord,
    canonical_json_bytes,
    file_record,
    verify_file_record,
    write_once_json,
)


def test_canonical_json_is_sorted_compact_and_finite() -> None:
    assert canonical_json_bytes({"b": 2, "a": "值"}) == b'{"a":"\xe5\x80\xbc","b":2}\n'
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_write_once_is_atomic_and_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "manifest.json"
    write_once_json(path, {"value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}
    assert not path.with_name(".manifest.json.partial").exists()
    with pytest.raises(FileExistsError):
        write_once_json(path, {"value": 2})


def test_file_record_round_trip_and_drift(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("one\n", encoding="utf-8")
    record = file_record(path, root=tmp_path)
    assert FileRecord.from_mapping(record.to_dict()) == record
    assert verify_file_record(record, root=tmp_path) == path.resolve()
    path.write_text("two\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest drift"):
        verify_file_record(record, root=tmp_path)


def test_file_record_rejects_escape_and_outside_file(tmp_path: Path) -> None:
    outside = tmp_path.parent / "outside-pgsft-test.txt"
    outside.write_text("outside", encoding="utf-8")
    try:
        with pytest.raises(ValueError, match="outside"):
            file_record(outside, root=tmp_path)
        record = FileRecord("../outside-pgsft-test.txt", "0" * 64, 7)
        with pytest.raises(ValueError, match="escapes"):
            verify_file_record(record, root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "value",
    [
        {"path": "x", "sha256": "ABC", "size_bytes": 1},
        {"path": "", "sha256": "0" * 64, "size_bytes": 1},
        {"path": "x", "sha256": "0" * 64, "size_bytes": -1},
    ],
)
def test_file_record_validates_fields(value: dict) -> None:
    with pytest.raises(ValueError):
        FileRecord.from_mapping(value)
