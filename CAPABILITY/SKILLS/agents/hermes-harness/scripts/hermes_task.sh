#!/usr/bin/env bash
set -euo pipefail

TASK="${1:-}"
if [[ -z "$TASK" ]]; then
  echo "Usage: $0 'task for Hermes to delegate'" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/hermes_harness.py" run \
  --task "$TASK" \
  --workspace "${WORKSPACE:-$PWD}" \
  --mode "${HERMES_HARNESS_MODE:-auto}" \
  --max-workers "${HERMES_HARNESS_MAX_WORKERS:-3}"
