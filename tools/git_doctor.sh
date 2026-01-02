#!/usr/bin/env bash
set -euo pipefail

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/CAPABILITY/TOOLS/utilities/git_doctor.sh" "$@"

