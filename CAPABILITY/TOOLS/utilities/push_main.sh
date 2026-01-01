#!/usr/bin/env bash
set -euo pipefail

# Idempotent helper for "why did it push to the old branch?"
# - ensures you're on `main`
# - ensures `origin/main` is the upstream for local `main`
# - pushes `main` to `origin/main`

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: Not a git repository: $repo_root" >&2
  exit 2
fi

current_branch="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
if [[ "$current_branch" != "main" ]]; then
  echo "Switching to main (was: ${current_branch})"
  git checkout main
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "ERROR: Missing remote 'origin'." >&2
  exit 3
fi

if [[ -x "CAPABILITY/TOOLS/utilities/ensure_https_remote.sh" ]]; then
  "CAPABILITY/TOOLS/utilities/ensure_https_remote.sh" >/dev/null
fi

echo "Setting upstream: main -> origin/main"
git branch --set-upstream-to=origin/main main >/dev/null 2>&1 || true

echo "Pushing: origin main"
git push origin main

