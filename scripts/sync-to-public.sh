#!/usr/bin/env bash
#
# Sync this private repository's content to the public mirror.
#
#   ./scripts/sync-to-public.sh           # copy + commit locally, do NOT push
#   ./scripts/sync-to-public.sh --check   # report drift, change nothing
#   ./scripts/sync-to-public.sh --push    # copy + commit + push
#
# The two repositories do NOT share history: the public one was created from a
# content snapshot, so `git merge-base` between them is empty. Syncing therefore
# means copying files and making a fresh commit on the public side — never a
# merge, never a force-push.
#
# --check exits non-zero when the public mirror is behind, so it can be wired
# into a release check. A mirror nobody verifies is worse than no mirror,
# because it looks current.

set -euo pipefail

PRIVATE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PUBLIC_URL="https://github.com/eauesque/yu-hailo-infer"
MODE="${1:-sync}"

# Paths that stay private. Each needs a reason: an unexplained entry here
# silently withholds work from the published crate, which is the failure this
# script exists to prevent.
PRIVATE_ONLY=(
  ".git"          # never copied — the mirror keeps its own history
  ".claude"       # agent configuration, not part of the crate
  ".serena"       # agent configuration (Serena), not part of the crate
  ".mcp.json"     # MCP server registration naming machine-local binaries
  ".yu"           # ai-coreutils workspace config
  "CLAUDE.md"     # instructions to coding agents
  "TODO.md"       # internal work tracking
  "docs/superpowers"  # design plans, written for this project's own process
  "target"        # build output
)

# `.github` is deliberately NOT in that list. The public repository is what
# yu_ai_manager pins, so it is the copy that must be verified; keeping CI
# private would leave the published crate unchecked. The workflows run on
# ubuntu-latest with no HailoRT SDK, which build.rs handles by compiling
# src/hailort/shim_stub.cpp — see .github/workflows/ci.yml.

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
git clone --quiet --depth 1 "$PUBLIC_URL" "$work/public"

# Build the rsync exclusion list from PRIVATE_ONLY so the two never drift apart.
excludes=()
for path in "${PRIVATE_ONLY[@]}"; do
  excludes+=(--exclude="$path")
done

if [ "$MODE" = "--check" ]; then
  # `-c` compares checksums, not timestamps. Without it every file looks
  # changed: the public mirror is a fresh clone, so its mtimes are all "now"
  # and rsync's default size+mtime heuristic reports the entire tree as drift.
  # That noise would hide the handful of files that actually differ.
  drift="$(rsync -rnc --delete --itemize-changes "${excludes[@]}" \
             "$PRIVATE/" "$work/public/" || true)"
  if [ -n "$drift" ]; then
    echo "DRIFT: the public mirror is not up to date with this repository:"
    echo "$drift" | sed 's/^/  /'
    echo
    echo "public head: $(cd "$work/public" && git rev-parse HEAD)"
    echo "run without --check to stage the sync"
    exit 1
  fi
  echo "public mirror is in sync ($(cd "$work/public" && git rev-parse --short HEAD))"
  exit 0
fi

rsync -ac --delete "${excludes[@]}" "$PRIVATE/" "$work/public/"

cd "$work/public"
if git diff --quiet && git diff --cached --quiet; then
  echo "nothing to sync — public mirror already matches ($(git rev-parse --short HEAD))"
  exit 0
fi

git add -A
echo "=== files this sync changes ==="
git diff --cached --stat

# A sync that would delete most of the mirror is far more likely to be a broken
# exclusion list than an intended removal. Refuse rather than publish it.
deleted="$(git diff --cached --diff-filter=D --name-only | wc -l)"
kept="$(git ls-files | wc -l)"
if [ "$deleted" -gt 0 ] && [ "$kept" -lt "$deleted" ]; then
  echo "ERROR: this sync deletes more files ($deleted) than it keeps ($kept)." >&2
  echo "       That is almost certainly a bug in PRIVATE_ONLY, not a real change." >&2
  exit 2
fi

git commit -q -F - <<'MSG'
sync: update from the development repository

Content snapshot from the private development repository. The two histories are
independent by design; this commit carries the file contents, not the upstream
commits.
MSG

new_rev="$(git rev-parse HEAD)"
echo
echo "committed on the public mirror: $new_rev"

if [ "$MODE" = "--push" ]; then
  git push origin HEAD:main
  echo "pushed."
  echo
  echo "Next: pin this rev in yu_ai_manager/crates/Cargo.toml — BOTH entries"
  echo "(infer-core and yu-infer) must share it, or ort/ONNX Runtime builds twice:"
  echo "  rev = \"$new_rev\""
else
  echo "not pushed (pass --push to publish)."
fi
