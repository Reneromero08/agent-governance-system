"""Deterministic Phase 6B.6 explicit-slot schedule generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from .contract import (
        AUTHORITY,
        NOMINAL_SAMPLES_PER_SLOT,
        ORDER_ARRAYS,
    )
    from .v2_interface import TONE_CODEWORD_TABLE
except ImportError:  # pragma: no cover - direct script execution fallback
    from contract import AUTHORITY, NOMINAL_SAMPLES_PER_SLOT, ORDER_ARRAYS  # type: ignore
    from v2_interface import TONE_CODEWORD_TABLE  # type: ignore

try:
    from .contract import (
        SLOTS_PER_SESSION,
        TONE_COUNT,
        canonical_json,
        contract_manifest,
        declared_and_executed_order,
        digest,
        order_family_sequence,
        route_order_for_block,
        split_for_block,
    )
except ImportError:  # pragma: no cover
    from contract import (  # type: ignore
        SLOTS_PER_SESSION,
        TONE_COUNT,
        canonical_json,
        contract_manifest,
        declared_and_executed_order,
        digest,
        order_family_sequence,
        route_order_for_block,
        split_for_block,
    )


PHASE_ACTIONS = {"positive": "0", "negative": "pi"}


def _base_slot(
    session: dict[str, Any],
    slot_index: int,
    stage: str,
    mode: str,
    drive_on: bool,
    physical_tone_index: int | None,
    amplitude_level: int | None,
    sign: int | None,
    phase_action: str,
    declared_mode: str | None = None,
    declared_order_family: str | None = None,
    executed_order_family: str | None = None,
    executed_order_position: int | None = None,
    analysis_tone_index: int | None = None,
    packet_id: str | None = None,
    sender_epoch_id: str | None = None,
) -> dict[str, Any]:
    declared_tone = physical_tone_index
    v2_mode = "basis" if drive_on else None
    codeword_source = physical_tone_index if drive_on and physical_tone_index is not None else None
    codeword_sign = None
    if codeword_source is not None:
        codeword_sign = TONE_CODEWORD_TABLE["tones"][codeword_source]["mode_signs"][v2_mode]
    return {
        "session_index": session["session_index"],
        "slot_index": slot_index,
        "stage": stage,
        "packet_id": packet_id,
        "requested_start_tick": slot_index,
        "requested_end_tick": slot_index + 1,
        "nominal_samples": NOMINAL_SAMPLES_PER_SLOT,
        "declared": {
            "declared_mode": declared_mode or mode,
            "declared_amplitude_level": amplitude_level,
            "declared_phase_action": phase_action,
            "declared_physical_tone_index": declared_tone,
            "declared_order_family": declared_order_family,
            "declared_order_position": executed_order_position,
            "analysis_tone_index": analysis_tone_index if analysis_tone_index is not None else declared_tone,
        },
        "executed": {
            "drive_on": drive_on,
            "executed_mode": mode,
            "executed_v2_mode": v2_mode,
            "amplitude_level": amplitude_level if drive_on else None,
            "phase_action": phase_action if drive_on else None,
            "physical_tone_index": physical_tone_index if drive_on else None,
            "analysis_tone_index": analysis_tone_index,
            "tone_execution_order_position": executed_order_position,
            "executed_order_family": executed_order_family,
            "executed_order_position": executed_order_position,
            "executed_codeword_signs": list(TONE_CODEWORD_TABLE["codebook"][v2_mode]) if drive_on else None,
            "codeword_source_index": codeword_source,
            "codeword_sign": codeword_sign,
            "sign": sign if drive_on else None,
            "sender_epoch_id": sender_epoch_id,
        },
    }


def sessions() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    session_index = 0
    for block_index in range(6):
        reboot_block = f"b{block_index}"
        for route in route_order_for_block(block_index):
            sender_core, receiver_core = (4, 5) if route == "v4s5" else (2, 3)
            result.append(
                {
                    "session_index": session_index,
                    "reboot_block": reboot_block,
                    "route": route,
                    "sender_core": sender_core,
                    "receiver_core": receiver_core,
                    "split": split_for_block(reboot_block),
                    "session_chronology": session_index,
                    "order_family_sequence": list(order_family_sequence(reboot_block, route)),
                }
            )
            session_index += 1
    return result


def session_schedule(session: dict[str, Any]) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []

    def append(*args: Any, **kwargs: Any) -> None:
        slots.append(_base_slot(session, len(slots), *args, **kwargs))

    for _ in range(48):
        append("preamble", "SENDER_OFF_IDLE", False, None, None, None, "none")
    for tone in range(TONE_COUNT):
        append("preamble", "CARRIER_OFF", False, tone, None, None, "none", analysis_tone_index=tone)
    for tone in range(TONE_COUNT):
        append(
            "preamble",
            "DECLARATION_SHAM",
            False,
            tone,
            2,
            1,
            "0",
            declared_mode="ANCHOR_DECLARATION",
            analysis_tone_index=tone,
        )
    for tone in range(TONE_COUNT):
        for sign in (1, -1):
            append(
                "preamble",
                "ANCHOR",
                True,
                tone,
                2,
                sign,
                PHASE_ACTIONS["positive" if sign > 0 else "negative"],
                sender_epoch_id=f"s{session['session_index']}:preamble:anchor:{tone}:{sign}",
            )

    for family in session["order_family_sequence"]:
        declared_family, declared_array, executed_family, executed_array = declared_and_executed_order(
            family, session["reboot_block"]
        )
        for order_position, physical_tone in enumerate(executed_array):
            declared_tone = declared_array[order_position]
            for amplitude_level in (0, 1, 2):
                for sign in (-1, 1):
                    row = _base_slot(
                        session,
                        len(slots),
                        "prepared_order",
                        "PREPARED_ORDER",
                        True,
                        physical_tone,
                        amplitude_level,
                        sign,
                        PHASE_ACTIONS["positive" if sign > 0 else "negative"],
                        declared_order_family=declared_family,
                        executed_order_family=executed_family,
                        executed_order_position=order_position,
                        sender_epoch_id=f"s{session['session_index']}:prepared:{family}:{order_position}:{amplitude_level}:{sign}",
                    )
                    row["declared"]["declared_physical_tone_index"] = declared_tone
                    slots.append(row)

    for tone in range(TONE_COUNT):
        sign = 1 if (tone + int(session["reboot_block"][1:]) + session["session_index"]) % 2 == 0 else -1
        phase_shift = "pi/2" if (tone + session["session_index"]) % 2 == 0 else "-pi/2"
        packet_prefix = f"s{session['session_index']}:tone{tone}"
        append(
            "trajectory",
            "IMPULSE",
            True,
            tone,
            2,
            sign,
            "0",
            packet_id=f"{packet_prefix}:impulse",
            sender_epoch_id=f"{packet_prefix}:impulse:drive",
        )
        for off_index in range(7):
            append("trajectory", "IMPULSE_OFF", False, tone, 2, sign, "0", packet_id=f"{packet_prefix}:impulse", analysis_tone_index=tone)

        step_epoch = f"{packet_prefix}:step:epoch"
        for _ in range(4):
            append(
                "trajectory",
                "STEP",
                True,
                tone,
                2,
                sign,
                "0",
                packet_id=f"{packet_prefix}:step",
                sender_epoch_id=step_epoch,
            )
        for _ in range(4):
            append("trajectory", "STEP_OFF", False, tone, 2, sign, "0", packet_id=f"{packet_prefix}:step", analysis_tone_index=tone)

        phase_epoch = f"{packet_prefix}:phase_shift:epoch"
        for _ in range(2):
            append(
                "trajectory",
                "PHASE_SHIFT",
                True,
                tone,
                2,
                sign,
                "0",
                packet_id=f"{packet_prefix}:phase_shift",
                sender_epoch_id=phase_epoch,
            )
        for _ in range(2):
            append(
                "trajectory",
                "PHASE_SHIFT",
                True,
                tone,
                2,
                sign,
                phase_shift,
                packet_id=f"{packet_prefix}:phase_shift",
                sender_epoch_id=phase_epoch,
            )
        for _ in range(4):
            append("trajectory", "PHASE_SHIFT_OFF", False, tone, 2, sign, phase_shift, packet_id=f"{packet_prefix}:phase_shift", analysis_tone_index=tone)

        for _ in range(8):
            append(
                "trajectory",
                "CARRIER_OFF_SHAM",
                False,
                tone,
                2,
                sign,
                "0",
                declared_mode="CARRIER_OFF_SHAM_DECLARED_DRIVE",
                packet_id=f"{packet_prefix}:carrier_off_sham",
                analysis_tone_index=tone,
            )

    for _ in range(12):
        append("tail_drift", "TAIL_SENDER_OFF", False, None, None, None, "none")
    for tone in range(TONE_COUNT):
        sign = 1 if tone % 2 == 0 else -1
        append(
            "tail_drift",
            "TAIL_ANCHOR",
            True,
            tone,
            2,
            sign,
            PHASE_ACTIONS["positive" if sign > 0 else "negative"],
            sender_epoch_id=f"s{session['session_index']}:tail:anchor:{tone}",
        )

    if len(slots) != SLOTS_PER_SESSION:
        raise AssertionError(f"generated {len(slots)} slots, expected {SLOTS_PER_SESSION}")
    return slots


def campaign_schedule() -> dict[str, Any]:
    session_rows = sessions()
    payload = {
        "schema_id": "CAT_CAS_PHASE6B6_EXPLICIT_SLOT_SCHEDULE_V1",
        "contract": contract_manifest(),
        "sessions": [],
    }
    for session in session_rows:
        slots = session_schedule(session)
        payload["sessions"].append({**session, "slots": slots, "slot_count": len(slots)})
    payload["session_count"] = len(payload["sessions"])
    payload["total_slots"] = sum(session["slot_count"] for session in payload["sessions"])
    payload["schedule_sha256"] = digest(payload)
    return payload


def validate_schedule(schedule: dict[str, Any]) -> None:
    if schedule["session_count"] != 12 or schedule["total_slots"] != 10368:
        raise ValueError("campaign geometry mismatch")
    for session in schedule["sessions"]:
        if session["slot_count"] != SLOTS_PER_SESSION:
            raise ValueError("session slot count mismatch")
        last_end = 0
        for slot in session["slots"]:
            if slot["slot_index"] != last_end:
                raise ValueError("slot chronology gap or reorder")
            if slot["requested_start_tick"] != last_end or slot["requested_end_tick"] != last_end + 1:
                raise ValueError("non-contiguous requested slot boundary")
            if not slot["executed"]["drive_on"] and slot["executed"]["sender_epoch_id"] is not None:
                raise ValueError("sender epoch present in sender-off slot")
            last_end += 1
    if digest({key: value for key, value in schedule.items() if key != "schedule_sha256"}) != schedule["schedule_sha256"]:
        raise ValueError("schedule digest mismatch")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    schedule = campaign_schedule()
    validate_schedule(schedule)
    write_json(args.out, schedule)
    print(f"PHASE6B6_SCHEDULE_OK sessions={schedule['session_count']} slots={schedule['total_slots']} sha256={schedule['schedule_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
