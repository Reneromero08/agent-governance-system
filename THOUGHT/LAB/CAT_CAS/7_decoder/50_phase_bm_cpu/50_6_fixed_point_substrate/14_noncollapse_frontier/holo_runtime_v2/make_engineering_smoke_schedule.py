#!/usr/bin/env python3
"""Create the only hardware schedule allowed without acquisition authorization."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SESSION_ID = "ENGINEERING_SMOKE_TEST"
PARTITION = "ENGINEERING_SMOKE_TEST_NOT_SCIENTIFIC_ACQUISITION"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_schedule(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    common = {
        "session_id": SESSION_ID,
        "block_id": SESSION_ID,
        "family": "engineering_smoke",
        "executed_tone_order": "ENGINEERING",
        "declared_tone_order": "ENGINEERING",
    }
    rows: list[dict[str, Any]] = []
    for index, mode in enumerate(("basis", "rotation")):
        rows.append({
            **common,
            "window_index": index,
            "stage": "ENGINEERING_SMOKE_DRIVEN",
            "actual_mode": mode,
            "declared_mode": mode,
            "measurement_mode": "lockin_and_raw_ring",
            "drive_on": True,
            "sender_off_required": False,
            "physical_tone_index": index,
            "receiver_codeword_source_index": index,
            "sender_codeword_source_index": index,
            "receiver_theta_idx": index,
            "sender_theta_idx": index,
            "shared_schedule": True,
            "scramble_key_digest": "0" * 64,
            "amplitude_level": 1,
            "sender_off_control_for_tone_index": None,
            "sender_off_control_theta_idx": None,
        })
    rows.append({
        **common,
        "window_index": 2,
        "stage": "ENGINEERING_SMOKE_SENDER_OFF",
        "actual_mode": None,
        "declared_mode": None,
        "measurement_mode": "raw_ring_sender_off",
        "drive_on": False,
        "sender_off_required": True,
        "physical_tone_index": None,
        "receiver_codeword_source_index": None,
        "sender_codeword_source_index": None,
        "receiver_theta_idx": None,
        "sender_theta_idx": None,
        "shared_schedule": True,
        "scramble_key_digest": "0" * 64,
        "amplitude_level": 0,
        "sender_off_control_for_tone_index": 0,
        "sender_off_control_theta_idx": 0,
    })
    header = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V2",
        "campaign_source_commit": "0" * 40,
        "campaign_plan_sha256": "0" * 64,
        "session_id": SESSION_ID,
        "route": "v4s5",
        "seed": -1,
        "partition": PARTITION,
        "window_count": len(rows),
        "restoration_authorized": False,
    }
    write_json(output / "session.json", header)
    (output / "windows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    manifest = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V2",
        "session_id": SESSION_ID,
        "files": {
            name: {
                "size": (output / name).stat().st_size,
                "sha256": sha256_file(output / name),
            }
            for name in ("session.json", "windows.jsonl")
        },
    }
    write_json(output / "session_manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    make_schedule(args.output)
    print(SESSION_ID)
    print("NOT_SCIENTIFIC_ACQUISITION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
