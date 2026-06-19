#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
FRONTIER = HERE.parent
RATIFICATION = FRONTIER / "gate_r" / "PROJECT_OWNER_RATIFICATION.json"
GATE_R = FRONTIER / "gate_r" / "GATE_R_DETERMINISTIC_MANIFEST.json"
CONTRACT = HERE / "CAMPAIGN_CONTRACT.md"


def main() -> int:
    errors: list[str] = []
    for path in (RATIFICATION, GATE_R, CONTRACT):
        if not path.is_file():
            errors.append(f"missing authority file: {path}")
    if not errors:
        owner = json.loads(RATIFICATION.read_text())
        review = json.loads(GATE_R.read_text())
        if owner.get("decision") != "RATIFY_AND_AUTHORIZE_COMBINED_TONE_ORDER_OBSERVABILITY_CAMPAIGN":
            errors.append("wrong owner decision")
        for field in ("project_owner_ratified", "campaign_implementation_authorized", "combined_physical_acquisition_authorized_after_preflight"):
            if owner.get(field) is not True:
                errors.append(f"owner authority missing {field}")
        for field in ("restoration_authorized", "target_coupling_authorized", "orientation_recovery_authorized", "small_wall_authorized"):
            if owner.get(field) is not False:
                errors.append(f"forbidden owner authority enabled: {field}")
        decision = review.get("decision", {})
        if decision.get("technical_review_complete") is not True:
            errors.append("Gate R technical review incomplete")
        if decision.get("verdict") != "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED":
            errors.append("Gate R verdict mismatch")
        if decision.get("implementation_authorized") is not False or decision.get("physical_acquisition_authorized") is not False:
            errors.append("technical review improperly grants owner authority")
        text = CONTRACT.read_text(encoding="utf-8")
        for marker in (
            "AUTHORIZED_FOR_IMPLEMENTATION_AND_POST_PREFLIGHT_ACQUISITION",
            "PERSISTENT_STATE_CANDIDATE",
            "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
            "No result authorizes restoration",
        ):
            if marker not in text:
                errors.append(f"contract missing marker: {marker}")
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("COMBINED_CAMPAIGN_AUTHORIZATION_ALIGNED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
