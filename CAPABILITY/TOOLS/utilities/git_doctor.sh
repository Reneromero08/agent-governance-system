#!/usr/bin/env bash
set -euo pipefail

# GitHub HTTPS auth + remote consistency doctor for WSL/VS Code.
# - Never writes secrets/tokens.
# - Idempotent when used with --apply.
#
# Usage:
#   ./CAPABILITY/TOOLS/utilities/git_doctor.sh [--repo <path>] [--apply]
#
# Exit codes:
#   0: Healthy (or applied successfully)
#   1: Unhealthy and not fixed
#   2: Not a git repo

repo="."
apply="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo) repo="${2:-.}"; shift 2;;
    --apply) apply="true"; shift;;
    -h|--help)
      echo "Usage: $0 [--repo <path>] [--apply]"
      exit 0
      ;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if ! git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "[git_doctor] Not a git repo: $repo" >&2
  exit 2
fi

ensure_https_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ensure_https_remote.sh"

gcm_candidates=(
  "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager.exe"
  "/mnt/c/Program Files/Git/mingw64/bin/git-credential-manager-core.exe"
)

gcm_path=""
for cand in "${gcm_candidates[@]}"; do
  if [[ -x "$cand" ]]; then
    gcm_path="$cand"
    break
  fi
done

origin_url="$(git -C "$repo" remote get-url origin 2>/dev/null || true)"
cred_helper="$(git config --global --get credential.helper 2>/dev/null || true)"
use_http_path="$(git config --global --get credential.https://github.com.useHttpPath 2>/dev/null || true)"

echo "== Git Doctor (WSL) =="
echo "repo:   $(cd "$repo" && pwd)"
echo "git:    $(command -v git) ($(git --version))"
echo "origin: ${origin_url:-<missing>}"
echo "credential.helper (global): ${cred_helper:-<unset>}"
echo "credential.https://github.com.useHttpPath (global): ${use_http_path:-<unset>}"
echo "GCM (Windows) path: ${gcm_path:-<not found>}"

gh_ok="false"
if command -v gh >/dev/null 2>&1; then
  if gh auth status -h github.com >/dev/null 2>&1; then
    gh_ok="true"
  fi
  echo "gh: $(command -v gh) (authenticated=${gh_ok})"
else
  echo "gh: <not installed>"
fi

needs_fix="false"

if [[ -z "$origin_url" ]]; then
  echo "[git_doctor] FAIL: origin remote is missing"
  needs_fix="true"
else
  if [[ "$origin_url" == git@github.com:* || "$origin_url" == ssh://git@github.com/* ]]; then
    echo "[git_doctor] WARN: origin is SSH; must be HTTPS"
    needs_fix="true"
  fi
fi

if [[ -n "$gcm_path" ]]; then
  if [[ "$cred_helper" != "$gcm_path" ]]; then
    echo "[git_doctor] WARN: credential.helper not set to Windows GCM"
    needs_fix="true"
  fi
  if [[ "${use_http_path:-}" != "true" ]]; then
    echo "[git_doctor] WARN: credential.https://github.com.useHttpPath not true"
    needs_fix="true"
  fi
else
  if [[ "$gh_ok" != "true" ]]; then
    echo "[git_doctor] WARN: no Windows GCM detected and gh not authenticated"
    needs_fix="true"
  fi
fi

if [[ "$apply" != "true" ]]; then
  if [[ "$needs_fix" == "true" ]]; then
    echo ""
    echo "Suggested fixes:"
    echo "  1) Enforce HTTPS remote:"
    echo "     bash \"$ensure_https_script\" \"$repo\""
    if [[ -n "$gcm_path" ]]; then
      echo "  2) Configure WSL Git to use Windows GCM:"
      echo "     git config --global credential.helper \"$gcm_path\""
      echo "     git config --global credential.https://github.com.useHttpPath true"
    else
      echo "  2) Fallback: install + configure GitHub CLI (gh) in WSL:"
      echo "     sudo apt-get update && sudo apt-get install -y gh"
      echo "     gh auth login"
      echo "     gh auth setup-git"
    fi
    exit 1
  fi
  echo "[git_doctor] OK"
  exit 0
fi

echo ""
echo "[git_doctor] Applying fixes..."
bash "$ensure_https_script" "$repo" || true

if [[ -n "$gcm_path" ]]; then
  git config --global credential.helper "$gcm_path"
  git config --global credential.https://github.com.useHttpPath true
  echo "[git_doctor] Set credential helper to Windows GCM and enabled useHttpPath"
else
  if command -v gh >/dev/null 2>&1; then
    echo "[git_doctor] Running: gh auth setup-git"
    gh auth setup-git
    echo "[git_doctor] NOTE: if not logged in yet, run: gh auth login"
  else
    echo "[git_doctor] Cannot apply gh fallback automatically (gh not installed)." >&2
    exit 1
  fi
fi

echo "[git_doctor] Done"

