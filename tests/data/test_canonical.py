from __future__ import annotations

from pathlib import Path

import pytest

from pgsft.data.canonical import (
    CanonicalTrace,
    Message,
    read_canonical_jsonl,
    write_canonical_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]


def _trace(trace_id: str = "trace-1") -> CanonicalTrace:
    return CanonicalTrace(
        id=trace_id,
        messages=(
            Message("user", "problem"),
            Message("assistant", "action", "reason"),
            Message("tool", "observation"),
            Message("assistant", "finish", "conclusion"),
        ),
        source={"shard": "data/train-000.parquet", "row_index": 1},
    )


def test_license_safe_example_is_valid_and_explicitly_synthetic() -> None:
    records = read_canonical_jsonl(ROOT / "examples" / "canonical_trace.jsonl")
    assert len(records) == 1
    assert records[0].source == {
        "license_note": "synthetic example authored for this release",
        "synthetic": True,
    }
    assert [message.role for message in records[0].messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]


@pytest.mark.parametrize(
    "messages, match",
    [
        ((Message("assistant", "action"),), "first"),
        ((Message("user", "question"),), "at least one assistant"),
        ((Message("user", "question"), Message("assistant", "")), "action content"),
        (
            (
                Message("user", "question"),
                Message("assistant", "one"),
                Message("assistant", "two"),
            ),
            "immediately follow",
        ),
    ],
)
def test_trace_rejects_invalid_assistant_structure(
    messages: tuple[Message, ...], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        CanonicalTrace(id="bad", messages=messages)


def test_source_metadata_rejects_secret_keys_and_absolute_paths() -> None:
    with pytest.raises(ValueError, match="secret"):
        CanonicalTrace(
            id="bad-secret",
            messages=_trace().messages,
            source={"api_token": "do-not-publish"},
        )
    with pytest.raises(ValueError, match="absolute"):
        CanonicalTrace(
            id="bad-path",
            messages=_trace().messages,
            source={"shard": "/private/data.parquet"},
        )


def test_jsonl_round_trip_is_deterministic_and_write_once(tmp_path: Path) -> None:
    path = tmp_path / "traces.jsonl"
    records = [_trace("a"), _trace("b")]
    write_canonical_jsonl(path, records)
    first = path.read_bytes()
    assert read_canonical_jsonl(path) == records
    assert first.endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_canonical_jsonl(path, records)


def test_jsonl_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.jsonl"
    row = '{"id":"same","messages":[{"content":"q","role":"user"},{"content":"a","role":"assistant"}]}\n'
    path.write_text(row + row, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        read_canonical_jsonl(path)
