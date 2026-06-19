#!/usr/bin/env python3
"""Guard the Gate R technical-review and authorization boundary."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main() -> int:
    errors: list[str] = []
    manifest_path = HERE / "GATE_R_DETERMINISTIC_MANIFEST.json"
    status_path = HERE / "GATE_R_STATUS.md"
    addendum_path = HERE / "L4B5B0_GATE_R_REPAIR_ADDENDUM.md"
    review_path = HERE / "L4B5B0_GATE_R_TECHNICAL_REVIEW.md"

    for path in (manifest_path, status_path, addendum_path, review_path):
        if not path.is_file():
            errors.append(f"missing Gate R authority file: {path}")

    if not errors:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        decision = manifest.get("decision", {})
        expected_false = (
            "human_review",
            "project_owner_ratified",
            "implementation_authorized",
            "physical_acquisition_authorized",
            "physical_restoration_authorized",
        )
        if decision.get("technical_review_complete") is not True:
            errors.append("technical review is not complete")
        if decision.get("verdict") != "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED":
            errors.append("unexpected Gate R verdict")
        if decision.get("project_owner_ratification_required") is not True:
            errors.append("owner ratification is not required")
        for field in expected_false:
            if decision.get(field) is not False:
                errors.append(f"Gate R improperly sets {field}")
        if manifest.get("gate_r_manifest_sha256") != "0a4d5a479c289658985fcf97e5a1ad04fa786205ec2ac90940e151d3907c654f":
            errors.append("Gate R manifest binding changed")

        status = status_path.read_text(encoding="utf-8")
        for marker in (
            "Technical audit:** `COMPLETE`",
            "Project-owner ratification:** `NEXT`",
            "Physical acquisition authorized:** no",
            "RATIFY_AND_AUTHORIZE_TONE_ORDER_CONTROL_ONLY",
            "Ratification and execution authorization are separate acts",
        ):
            if marker not in status:
                errors.append(f"Gate R status missing marker: {marker}")

        addendum = addendum_path.read_text(encoding="utf-8")
        for marker in (
            "S1_contextual = gauge_normalize",
            "PERSISTENT_STATE_CANDIDATE",
            "DRIVEN_RELATIONAL_TRANSPORT_ONLY",
            "GR5 governance separation",
        ):
            if marker not in addendum:
                errors.append(f"Gate R addendum missing marker: {marker}")

    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    if errors:
        return 1
    print("GATE_R_STATUS_ALIGNED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
