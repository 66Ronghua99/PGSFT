#!/usr/bin/env bash
set -euo pipefail
TARGET=${1:-envs/livecodebench/source}
test ! -e "$TARGET"
git clone --filter=blob:none https://github.com/LiveCodeBench/LiveCodeBench "$TARGET"
git -C "$TARGET" checkout 28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24
rm -rf "$TARGET/.git"
