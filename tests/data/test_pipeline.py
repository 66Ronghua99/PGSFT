from __future__ import annotations

from pathlib import Path

import pytest

from pgsft.data.canonical import CanonicalTrace, Message, read_canonical_jsonl
from pgsft.data.pipeline import canonicalize_source, freeze_split
from pgsft.data.sources import NEMOTRON_REPO, NEMOTRON_REVISION


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "nemotron-mini.jsonl"


def _records(count: int) -> list[CanonicalTrace]:
    return [
        CanonicalTrace(
            id=f"row-{index:03d}",
            messages=(Message("user", "question"), Message("assistant", "answer")),
        )
        for index in range(count)
    ]


def test_fixture_canonicalization_records_every_rejection(tmp_path: Path) -> None:
    output = tmp_path / "canonical"
    manifest = canonicalize_source(FIXTURE, output, backbone="qwen35")
    assert manifest["upstream"] == {
        "repo": NEMOTRON_REPO,
        "revision": NEMOTRON_REVISION,
    }
    assert manifest["rows"] == {"scanned": 5, "accepted": 2, "rejected": 3}
    assert manifest["duplicate_count"] == 1
    assert manifest["rejection_summary"] == {
        "duplicate_id": 1,
        "not_openhands": 1,
        "tail_test_failed": 1,
    }
    traces = read_canonical_jsonl(output / "canonical.jsonl")
    assert [trace.id for trace in traces] == ["valid-001", "valid-002"]
    assert "/workspace" not in (output / "canonical.jsonl").read_text()
    rejection_lines = (output / "rejections.jsonl").read_text().splitlines()
    assert len(rejection_lines) == 3


def test_canonicalization_is_byte_deterministic_across_output_roots(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one"
    second = tmp_path / "two"
    canonicalize_source(FIXTURE, first, backbone="qwen3")
    canonicalize_source(FIXTURE, second, backbone="qwen3")
    for name in ("canonical.jsonl", "rejections.jsonl", "manifest.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_freeze_split_preserves_qwen3_order_and_seeds_qwen35(tmp_path: Path) -> None:
    records = _records(8)
    qwen3 = tmp_path / "qwen3"
    manifest3 = freeze_split(
        records, qwen3, backbone="qwen3", train_size=4, holdout_size=2
    )
    assert [row.id for row in read_canonical_jsonl(qwen3 / "train-4.jsonl")] == [
        "row-000",
        "row-001",
        "row-002",
        "row-003",
    ]
    assert manifest3["seed"] is None

    qwen35a = tmp_path / "qwen35-a"
    qwen35b = tmp_path / "qwen35-b"
    first = freeze_split(
        records, qwen35a, backbone="qwen35", train_size=4, holdout_size=2
    )
    second = freeze_split(
        records, qwen35b, backbone="qwen35", train_size=4, holdout_size=2
    )
    assert first["ordered_ids_sha256"] == second["ordered_ids_sha256"]
    assert (qwen35a / "train-4.jsonl").read_bytes() == (
        qwen35b / "train-4.jsonl"
    ).read_bytes()
    assert first["duplicate_count"] == 0


def test_freeze_split_rejects_duplicates_and_short_inventory(tmp_path: Path) -> None:
    duplicate = [_records(1)[0], _records(1)[0]]
    with pytest.raises(ValueError, match="duplicate"):
        freeze_split(
            duplicate, tmp_path / "duplicate", backbone="qwen3", train_size=1, holdout_size=1
        )
    with pytest.raises(ValueError, match="need"):
        freeze_split(
            _records(2), tmp_path / "short", backbone="qwen35", train_size=2, holdout_size=1
        )
