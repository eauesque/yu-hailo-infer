#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

echo "== cargo clippy --workspace =="
CARGO_BUILD_JOBS=1 cargo clippy --workspace --all-targets -- -D warnings

echo "== cargo test --workspace =="
# --test-threads=1 is required, not a preference. run_media_preprocessing draws
# permits from a process-global semaphore shared by the audio route (160 MiB per
# reservation) and every image route, so tests running in parallel compete for it
# and the loser gets a 503 "media_preprocessing_busy" instead of the status it
# asserts. Measured: speech2text_transcribe_rejects_invalid_repetition_penalty
# fails in the default parallel run and passes both alone and with this flag.
CARGO_BUILD_JOBS=1 cargo test --workspace -- --test-threads=1
