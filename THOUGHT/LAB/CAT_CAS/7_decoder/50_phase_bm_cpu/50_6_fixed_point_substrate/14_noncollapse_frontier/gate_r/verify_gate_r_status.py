#!/usr/bin/env python3
"""Guard technical-review immutability and separate owner authorization."""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent


def main() -> int:
    errors: list[str] = []
    paths = {
        "review": HERE / "GATE_R_DETERMINISTIC_MANIFEST.json",
        "status": HERE / "GATE_R_STATUS.md",
        "addendum": HERE / "L4B5B0_GATE_R_REPAIR_ADDENDUM.md",
        "owner": HERE / "PROJECT_OWNER_RATIFICATION.json",
        "campaign": HERE / "COMBINED_CAMPAIGN_BINDING.json",
    }
    for path in paths.values():
        if not path.is_file():
            errors.append(f"missing Gate R authority file: {path}")

    if not errors:
        review = json.loads(paths["review"].read_text())
        decision = review.get("decision", {})
        if decision.get("technical_review_complete") is not True:
            errors.append("technical review incomplete")
        if decision.get("verdict") != "TECHNICAL_ACCEPT_WITH_REQUIRED_REPAIRS_APPLIED":
            errors.append("unexpected technical verdict")
        for field in ("human_review", "project_owner_ratified", "implementation_authorized", "physical_acquisition_authorized", "physical_restoration_authorized"):
            if decision.get(field) is not False:
                errors.append(f"technical review improperly sets {field}")
        if review.get("gate_r_manifest_sha256") != "0a4d5a479c289658985fcf97e5a1ad04fa786205ec2ac90940e151d3907c654f":
            errors.append("Gate R review binding changed")

        owner = json.loads(paths["owner"].read_text())
        if owner.get("decision") != "RATIFY_AND_AUTHORIZE_COMBINED_TONE_ORDER_OBSERVABILITY_CAMPAIGN":
            errors.append("owner decision mismatch")
        for field in ("project_owner_ratified", "campaign_implementation_authorized", "combined_physical_acquisition_authorized_after_preflight"):
            if owner.get(field) is not True:
                errors.append(f"owner authority missing {field}")
        for field in ("restoration_authorized", "target_coupling_authorized", "orientation_recovery_authorized", "small_wall_authorized"):
            if owner.get(field) is not False:
                errors.append(f"forbidden authority enabled: {field}")

        campaign = json.loads(paths["campaign"].read_text())
        if campaign.get("campaign_plan", {}).get("sha256") != "eb5a46d0a37d66910649467cf0d4e3cf947dee11fab94a36e9bdfed388455e53":
            errors.append("combined campaign plan binding changed")
        auth = campaign.get("authorization", {})
        if auth.get("physical_acquisition_executed") is not False or auth.get("restoration_authorized") is not False:
            errors.append("campaign status exceeds authority")

        status = paths["status"].read_text(encoding="utf-8")
        for marker in (
            "Project-owner ratification:** `COMPLETE`",
            "Campaign implementation authorized:** yes",
            "Physical acquisition authorized:** after executor verification and catcas preflight",
            "Physical acquisition executed:** no",
            "local schedule-driven executor: NEXT",
        ):
            if marker not in status:
                errors.append(f"Gate R status missing marker: {marker}")

        addendum = paths["addendum"].read_text(encoding="utf-8")
        for marker in ("S1_contextual = gauge_normalize", "PERSISTENT_STATE_CANDIDATE", "DRIVEN_RELATIONAL_TRANSPORT_ONLY", "GR5 governance separation"):
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
