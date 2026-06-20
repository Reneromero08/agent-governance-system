#!/usr/bin/env python3
"""Verify that the Phase 6 authority stack preserves the current execution order."""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve()
SUBSTRATE = HERE.parents[3]
FRONTIER = SUBSTRATE / "14_noncollapse_frontier"
AUDIT = FRONTIER / "carrier_witness_closure" / "theory_gate_audit"

FILES = {
    "base": FRONTIER / "COURSE_CORRECTION.md",
    "addendum": FRONTIER / "COURSE_CORRECTION_ADDENDUM_2026-06-19.md",
    "frontier": FRONTIER / "CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md",
    "master": SUBSTRATE / "PHASE6_ROADMAP.md",
    "navigation": SUBSTRATE / "PHASE6_NAVIGATION.md",
    "consolidation": AUDIT / "PHASE6B5D_CONSOLIDATION_REPORT.md",
    "deterministic": AUDIT / "PHASE6B5D_DETERMINISTIC_MANIFEST.json",
    "tone_order": AUDIT / "PHASE6B5E_TONE_ORDER_CONTROL_CONTRACT.md",
}

REQUIRED = {
    "base": (
        "**Status:** ACTIVE DIRECTIVE",
        "**Primary doctrine:** `THE ALGORITHM IS DEAD`",
        "**Primary object:** unresolved relational geometry / `OrbitState`",
    ),
    "addendum": (
        "ACTIVE_BINDING_ADDENDUM",
        "Phase 6B.5C — Transfer-Aware Carrier Geometry Analysis",
        "No new physical acquisition is authorized",
        "increase trials until the old conjunction passes",
    ),
    "frontier": (
        "COURSE_CORRECTION_ADDENDUM_2026-06-19.md",
        "TRANSFER_EQUIVARIANCE_SUPPORTED__GATE_R_NEXT",
        "Immediate gate:** Gate R",
        "proposed but unauthorized",
        "strict closure remains `PARTIAL`",
    ),
    "master": (
        "COURSE_CORRECTION_ADDENDUM_2026-06-19.md",
        "PHASE6B5C_TRANSFER_EQUIVARIANCE_SUPPORTED__GATE_R_NEXT",
        "Immediate gate:** Gate R",
        "proposed but not authorized",
        "strict carrier verdict remains `PARTIAL`",
    ),
    "navigation": (
        "COURSE_CORRECTION_ADDENDUM_2026-06-19.md",
        "PHASE6B5C_TRANSFER_EQUIVARIANCE_SUPPORTED__GATE_R_NEXT",
        "No physical acquisition is authorized",
        "Gate R",
        "Blind repetition or trial-count escalation is not an allowed substitute",
    ),
    "consolidation": (
        "COMPLETE__CLAIM_FROZEN_PENDING_GATE_R",
        "Scalar calibration cannot rescue the old gates",
        "SCALAR_GAIN_OUTLIER_WITH_RELATIONAL_INVARIANTS_PRESERVED",
        "Further open-ended analysis of this campaign is not authorized",
        "d11bf9d41c1b9a9195d79d5ba1ab8b591f9c364b3f57435fded958d5a0861f31",
        "Gate R",
    ),
    "deterministic": (
        "CAT_CAS_PHASE6B5D_DETERMINISTIC_BINDING_V1",
        "d11bf9d41c1b9a9195d79d5ba1ab8b591f9c364b3f57435fded958d5a0861f31",
        "sha256:defb733ba811e435068514fabb0e024f1a2492d3151b30491518e68197dd2e92",
        "FROZEN_PENDING_GATE_R",
        "timestamp_variant_repository_snapshot_retained\": false",
    ),
    "tone_order": (
        "PREREGISTERED_NOT_AUTHORIZED",
        "separate spectral tone identity from within-symbol path position",
        "TONE_IDENTITY_EQUIVARIANCE_SUPPORTED",
        "No hidden label, result-dependent permutation, or post-hoc order selection",
    ),
}

FORBIDDEN_CURRENT_ORDER = {
    "frontier": (
        "**Immediate gate:** close the carrier witness",
        "**Immediate gate:** Phase 6B.5C transfer-aware analysis",
    ),
    "master": (
        "**Immediate gate:** carrier-witness closure",
        "**Immediate gate:** Phase 6B.5C transfer-aware analysis",
    ),
    "navigation": ("2. **Carrier witness closure**",),
    "consolidation": (
        "increase trials until the old conjunction passes",
        "open-ended analysis remains authorized",
    ),
}


def verify() -> list[str]:
    errors: list[str] = []
    texts: dict[str, str] = {}
    for name, path in FILES.items():
        if not path.is_file():
            errors.append(f"missing authority file: {path}")
            continue
        texts[name] = path.read_text(encoding="utf-8")

    for name, needles in REQUIRED.items():
        text = texts.get(name, "")
        for needle in needles:
            if needle not in text:
                errors.append(f"{name} missing required authority marker: {needle}")

    for name, needles in FORBIDDEN_CURRENT_ORDER.items():
        for needle in needles:
            if needle in texts.get(name, ""):
                errors.append(f"{name} retains stale or forbidden order: {needle}")

    return errors


def main() -> int:
    errors = verify()
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("PHASE6_AUTHORITY_STACK_ALIGNED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
