#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAMPAIGN_ROOT="${1:-/root/catcas_evidence/phase6b5_t48_d32b1bed_20260619}"
OUTPUT_DIR="${2:-${HERE}/results/phase6b5c_t48_d32b1bed_20260619}"
NULL_SEED="${PHASE6B5C_NULL_SEED:-65005}"

if [[ ! -f "${CAMPAIGN_ROOT}/campaign.json" ]]; then
    echo "missing campaign root: ${CAMPAIGN_ROOT}" >&2
    exit 2
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "refusing to overwrite existing output: ${OUTPUT_DIR}" >&2
    exit 2
fi

mkdir -p "$(dirname "${OUTPUT_DIR}")"
python3 "${HERE}/analyze_transfer_geometry.py" run \
    "${CAMPAIGN_ROOT}" \
    --output "${OUTPUT_DIR}" \
    --null-seed "${NULL_SEED}"

python3 "${HERE}/analyze_transfer_geometry.py" verify "${OUTPUT_DIR}"
sha256sum "${OUTPUT_DIR}/analysis_manifest.json"
printf 'PHASE6B5C_OUTPUT=%s\n' "${OUTPUT_DIR}"
