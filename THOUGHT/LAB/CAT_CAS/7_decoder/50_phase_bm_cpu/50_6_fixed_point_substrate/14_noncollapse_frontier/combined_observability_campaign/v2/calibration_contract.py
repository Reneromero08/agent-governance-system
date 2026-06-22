"""Non-authorizing Phase 6 V2 spectral-calibration contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FALSE_AUTHORIZATIONS = {
    "acquisition_authorized": False,
    "restoration_authorized": False,
    "target_coupling_authorized": False,
    "small_wall_authorized": False,
}


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def build_plan() -> dict:
    return {
        "schema_id": "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_PLAN_V1",
        "execution_class": "ENGINEERING_QUALIFICATION_NOT_SCIENTIFIC_ACQUISITION",
        "routes": ["v4s5", "v2s3"],
        "tones": list(range(12)),
        "amplitudes": [1, 2, 3],
        "theta_indices": list(range(8)),
        "real_sender_on_and_off_required": True,
        "repeated_sessions_and_reboot_required": True,
        "frequency_settling_required": True,
        "calibration_authorized": False,
        **FALSE_AUTHORIZATIONS,
    }


def validate_authorization(value: dict, expected_plan_sha256: str) -> None:
    if value.get("schema_id") != "CAT_CAS_PHASE6_V2_SPECTRAL_CALIBRATION_AUTHORIZATION_V1":
        raise ValueError("wrong calibration authorization schema")
    if not value.get("calibration_authorized"):
        raise ValueError("calibration is not authorized")
    for key, expected in FALSE_AUTHORIZATIONS.items():
        if value.get(key) is not expected:
            raise ValueError(f"{key} must remain false")
    if value.get("automatic_retry") is not False:
        raise ValueError("automatic_retry must remain false")
    if value.get("campaign_plan_sha256") != expected_plan_sha256:
        raise ValueError("campaign plan digest mismatch")
    required = {
        "executor_commit", "executor_sha256", "source_bundle_sha256",
        "session_ids", "route_cores", "pin_khz", "slot_s", "off_window_s",
        "read_hz", "temperature_veto_c", "authorized_output_root", "authorized_by",
    }
    missing = sorted(required - value.keys())
    if missing:
        raise ValueError(f"missing authorization fields: {missing}")
    if value["route_cores"] != {"v4s5": [4, 5], "v2s3": [2, 3]}:
        raise ValueError("route/core binding mismatch")
    if not value["session_ids"]:
        raise ValueError("exact session IDs required")


def write_immutable(path: Path, value: dict) -> str:
    payload = canonical_bytes(value)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return digest
