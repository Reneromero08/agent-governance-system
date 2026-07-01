"""Software-entry explicit-slot runtime with mock hardware custody."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from contracts.contract import AUTHORITY, NOMINAL_SAMPLES_PER_SLOT, digest
from contracts.schedule import campaign_schedule, validate_schedule, write_json
from runtime.state_machine import SenderStateMachine, validate_runtime_events


AUTHORITY_ERROR = "SOFTWARE_ENTRY_ONLY_AUTHORITY: real hardware execution is not authorized"


def reject_real_hardware() -> None:
    raise PermissionError(AUTHORITY_ERROR)


def mock_slot_capture(
    session: dict[str, Any],
    slot: dict[str, Any],
    measured_tsc_hz: int,
    runtime_events: list[dict[str, Any]],
) -> dict[str, Any]:
    start = int(session["session_tsc_origin"] + round(slot["slot_index"] * 0.5 * measured_tsc_hz))
    end = int(session["session_tsc_origin"] + round((slot["slot_index"] + 1) * 0.5 * measured_tsc_hz))
    tone = slot["executed"]["physical_tone_index"]
    drive = 1.0 if slot["executed"]["drive_on"] else 0.0
    tone_term = 0.0 if tone is None else (tone + 1) / 1000.0
    sign = slot["executed"]["sign"] or 0
    return {
        "stage": slot["stage"],
        "packet_id": slot["packet_id"],
        "session_index": session["session_index"],
        "reboot_block": session["reboot_block"],
        "split": session["split"],
        "route": session["route"],
        "sender_core": session["sender_core"],
        "receiver_core": session["receiver_core"],
        "session_chronology": session["session_chronology"],
        "slot_index": slot["slot_index"],
        "requested_start_tsc": start,
        "requested_end_tsc": end,
        "actual_start_tsc": start,
        "actual_end_tsc": end,
        "measured_tsc_hz": measured_tsc_hz,
        "nominal_samples": NOMINAL_SAMPLES_PER_SLOT,
        "measured_sample_count": NOMINAL_SAMPLES_PER_SLOT,
        "empirical_sample_rate": 8000.0,
        "temperature": 42.0,
        "r_t": {
            "lockin_I": round(drive * sign + tone_term, 9),
            "lockin_Q": round(drive * 0.5 * sign - tone_term, 9),
            "ring_osc_period": round(100.0 + tone_term + 0.0001 * slot["slot_index"], 9),
        },
        "u_t": slot["executed"],
        "declared": slot["declared"],
        "runtime_events": runtime_events,
        "c_t": {
            "route": session["route"],
            "sender_core": session["sender_core"],
            "receiver_core": session["receiver_core"],
            "reboot_block": session["reboot_block"],
            "session_chronology": session["session_chronology"],
            "session_tsc_origin": session["session_tsc_origin"],
            "actual_start_tsc": start,
            "actual_end_tsc": end,
            "measured_tsc_hz": measured_tsc_hz,
            "empirical_sample_rate": 8000.0,
            "temperature": 42.0,
            "P_state": "MOCK_PINNED",
            "capture_quality": "MOCK_ACCEPTED",
        },
        "capture_quality": "MOCK_ACCEPTED",
    }


def run_mock(schedule: dict[str, Any]) -> dict[str, Any]:
    validate_schedule(schedule)
    captured_sessions = []
    for session in schedule["sessions"]:
        measured_tsc_hz = 3_200_000_000
        runtime_session = {
            key: session[key]
            for key in ("session_index", "route", "reboot_block", "sender_core", "receiver_core", "split", "session_chronology")
        }
        runtime_session["session_tsc_origin"] = 1_000_000_000_000 + session["session_index"] * 10_000_000_000
        runtime_session["measured_tsc_hz"] = measured_tsc_hz
        machine = SenderStateMachine()
        captured = []
        for slot in session["slots"]:
            events = machine.apply(slot)
            captured.append(mock_slot_capture(runtime_session, slot, measured_tsc_hz, events))
        if captured:
            captured[-1]["runtime_events"].extend(machine.finish(session["slots"][-1]))
        runtime_session["slots"] = captured
        captured_sessions.append(runtime_session)
    custody = {
        "schema_id": "CAT_CAS_PHASE6B6_MOCK_CAPTURE_CUSTODY_V1",
        "authority": AUTHORITY,
        "contract_sha256": schedule["contract"]["contract_sha256"],
        "schedule_sha256": schedule["schedule_sha256"],
        "session_count": len(captured_sessions),
        "total_slots": sum(len(session["slots"]) for session in captured_sessions),
        "sessions": captured_sessions,
    }
    validate_runtime_events(custody)
    custody["custody_sha256"] = digest(custody)
    return custody


def validate_authority(validate_only: bool, mock_hardware: bool, hardware: bool) -> None:
    if hardware:
        reject_real_hardware()
    if not validate_only and not mock_hardware:
        reject_real_hardware()
    for field in (
        "hardware_ran",
        "authorization_artifact_created",
        "calibration_authorized",
        "scientific_acquisition_authorized",
        "restoration_authorized",
        "target_coupling_authorized",
        "small_wall_authorized",
    ):
        if AUTHORITY[field] is not False:
            raise PermissionError(f"authority invariant failed: {field}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--mock-hardware", action="store_true")
    parser.add_argument("--hardware", action="store_true")
    parser.add_argument("--schedule-out", type=Path)
    parser.add_argument("--custody-out", type=Path)
    args = parser.parse_args(argv)
    try:
        validate_authority(args.validate_only, args.mock_hardware, args.hardware)
        schedule = campaign_schedule()
        validate_schedule(schedule)
        if args.schedule_out:
            write_json(args.schedule_out, schedule)
        if args.mock_hardware:
            custody = run_mock(schedule)
            if args.custody_out:
                write_json(args.custody_out, custody)
            print(
                "PHASE6B6_MOCK_RUNTIME_OK "
                f"sessions={custody['session_count']} slots={custody['total_slots']} "
                f"sha256={custody['custody_sha256']}"
            )
        else:
            print(
                "PHASE6B6_VALIDATE_ONLY_OK "
                f"sessions={schedule['session_count']} slots={schedule['total_slots']} "
                f"sha256={schedule['schedule_sha256']}"
            )
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
