"""Merge the selected LoRA checkpoint into its exact pinned Base."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from pgsft.model_profiles import load_model_profile
from pgsft.training.merge import merge_selected_adapter


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--backbone", choices=("qwen3", "qwen35"), required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    profile = load_model_profile(args.backbone, root)
    output = args.output_dir or args.run_dir / "merged"
    manifest = merge_selected_adapter(
        run_dir=args.run_dir, output_dir=output, profile=profile
    )
    print(f"merged={output} sha256={manifest['output_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
