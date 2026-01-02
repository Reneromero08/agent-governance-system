#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/CAPABILITY/TOOLS/utilities/ensure_https_remote.sh" "$@"

