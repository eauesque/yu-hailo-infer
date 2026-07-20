#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== cargo clippy --workspace =="
CARGO_BUILD_JOBS=1 cargo clippy --workspace --all-targets -- -D warnings

echo "== cargo test --workspace =="
CARGO_BUILD_JOBS=1 cargo test --workspace
