"""Download pinned source files or construct canonical PG-SFT data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Sequence

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

from pgsft.data.canonical import read_canonical_jsonl
from pgsft.data.pipeline import canonicalize_source, freeze_split
from pgsft.data.sources import NEMOTRON_REPO, NEMOTRON_REVISION
from pgsft.data.tokenization import tokenize_pgsft_trace, tokenize_sft_trace
from pgsft.model_profiles import load_model_profile
from pgsft.provenance.files import sha256_file
from pgsft.provenance.manifest import write_once_json


def _download(args: argparse.Namespace) -> int:
    if args.revision != NEMOTRON_REVISION:
        raise ValueError(
            f"revision must match the release pin {NEMOTRON_REVISION}, got {args.revision}"
        )
    snapshot_download(
        repo_id=NEMOTRON_REPO,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output_dir,
        allow_patterns=["*.parquet"],
    )
    print(f"downloaded {NEMOTRON_REPO}@{args.revision} to {args.output_dir}")
    return 0


def _canonicalize(args: argparse.Namespace) -> int:
    manifest = canonicalize_source(args.source, args.output_dir, backbone=args.backbone)
    rows = manifest["rows"]
    print(
        f"scanned={rows['scanned']} accepted={rows['accepted']} "
        f"rejected={rows['rejected']} manifest={args.output_dir / 'manifest.json'}"
    )
    return 0


def _write_rows(path: Path, rows: list[object]) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(
                    json.dumps(
                        asdict(row),
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


def _tokenize(args: argparse.Namespace) -> int:
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[3]
    profile = load_model_profile(args.backbone, root)
    tokenizer = AutoTokenizer.from_pretrained(
        profile.config.name,
        revision=profile.config.revision,
        local_files_only=True,
        trust_remote_code=False,
    )
    traces = read_canonical_jsonl(args.canonical)
    standard = [
        tokenize_sft_trace(trace, tokenizer=tokenizer, profile=profile)
        for trace in traces
    ]
    paired = [
        tokenize_pgsft_trace(trace, tokenizer=tokenizer, profile=profile)
        for trace in traces
    ]
    standard_path = args.output_dir / "sft.jsonl"
    paired_path = args.output_dir / "pgsft.jsonl"
    _write_rows(standard_path, standard)
    _write_rows(paired_path, paired)
    manifest = {
        "schema_version": 1,
        "backbone": args.backbone,
        "model": profile.config.name,
        "revision": profile.config.revision,
        "rows": len(traces),
        "source": {
            "path": args.canonical.name,
            "sha256": sha256_file(args.canonical),
            "size_bytes": args.canonical.stat().st_size,
        },
        "templates": {
            "train": profile.train_template_record.to_dict(),
            "inference": profile.inference_template_record.to_dict(),
        },
        "outputs": {
            "sft": {
                "path": standard_path.name,
                "sha256": sha256_file(standard_path),
                "size_bytes": standard_path.stat().st_size,
            },
            "pgsft": {
                "path": paired_path.name,
                "sha256": sha256_file(paired_path),
                "size_bytes": paired_path.stat().st_size,
            },
        },
    }
    write_once_json(args.output_dir / "manifest.json", manifest)
    print(f"rows={len(traces)} output={args.output_dir}")
    return 0


def _split(args: argparse.Namespace) -> int:
    records = read_canonical_jsonl(args.canonical)
    manifest = freeze_split(records, args.output_dir, backbone=args.backbone)
    print(f"train={manifest['row_counts']['train']} holdout={manifest['row_counts']['holdout']} output={args.output_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    download = subparsers.add_parser("download")
    download.add_argument("--output-dir", type=Path, required=True)
    download.add_argument("--revision", required=True)
    download.set_defaults(handler=_download)
    canonicalize = subparsers.add_parser("canonicalize")
    canonicalize.add_argument("--source", type=Path, required=True)
    canonicalize.add_argument("--output-dir", type=Path, required=True)
    canonicalize.add_argument("--backbone", choices=("qwen3", "qwen35"), required=True)
    canonicalize.set_defaults(handler=_canonicalize)
    split = subparsers.add_parser("split")
    split.add_argument("--backbone", choices=("qwen3", "qwen35"), required=True)
    split.add_argument("--canonical", type=Path, required=True)
    split.add_argument("--output-dir", type=Path, required=True)
    split.set_defaults(handler=_split)
    tokenize = subparsers.add_parser("tokenize")
    tokenize.add_argument("--backbone", choices=("qwen3", "qwen35"), required=True)
    tokenize.add_argument("--canonical", type=Path, required=True)
    tokenize.add_argument("--output-dir", type=Path, required=True)
    tokenize.set_defaults(handler=_tokenize)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
