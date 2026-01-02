#!/usr/bin/env bash
set -euo pipefail

# Enforce HTTPS transport for the current repo's `origin` remote.
# Idempotent and safe to re-run.
#
# Exit codes:
#  0: OK (already HTTPS or fixed)
#  2: Not a git repo
#  3: origin missing
#  4: Unsupported/unknown origin URL (cannot derive HTTPS)

repo_root="${1:-.}"

if ! git -C "$repo_root" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[ensure_https_remote] Not a git repo: $repo_root" >&2
  exit 2
fi

if ! git -C "$repo_root" remote get-url origin >/dev/null 2>&1; then
  echo "[ensure_https_remote] Missing remote: origin" >&2
  exit 3
fi

origin="$(git -C "$repo_root" remote get-url origin)"

derive_https() {
  local url="$1"
  # git@github.com:OWNER/REPO.git
  if [[ "$url" =~ ^git@github\.com:([^/]+)/([^/]+)$ ]]; then
    local owner="${BASH_REMATCH[1]}"
    local repo="${BASH_REMATCH[2]}"
    repo="${repo%.git}"
    echo "https://github.com/${owner}/${repo}.git"
    return 0
  fi
  # ssh://git@github.com/OWNER/REPO.git
  if [[ "$url" =~ ^ssh://git@github\.com/([^/]+)/([^/]+)$ ]]; then
    local owner="${BASH_REMATCH[1]}"
    local repo="${BASH_REMATCH[2]}"
    repo="${repo%.git}"
    echo "https://github.com/${owner}/${repo}.git"
    return 0
  fi
  # https://github.com/OWNER/REPO(.git)
  if [[ "$url" =~ ^https://github\.com/([^/]+)/([^/]+)$ ]]; then
    local owner="${BASH_REMATCH[1]}"
    local repo="${BASH_REMATCH[2]}"
    repo="${repo%.git}"
    echo "https://github.com/${owner}/${repo}.git"
    return 0
  fi
  return 1
}

https_url=""
if https_url="$(derive_https "$origin")"; then
  :
else
  echo "[ensure_https_remote] Cannot derive HTTPS URL from origin: $origin" >&2
  exit 4
fi

if [[ "$origin" == "$https_url" ]]; then
  echo "[ensure_https_remote] OK (origin already HTTPS): $origin"
  exit 0
fi

git -C "$repo_root" remote set-url origin "$https_url"
echo "[ensure_https_remote] Rewrote origin:"
echo "  from: $origin"
echo "  to:   $https_url"
