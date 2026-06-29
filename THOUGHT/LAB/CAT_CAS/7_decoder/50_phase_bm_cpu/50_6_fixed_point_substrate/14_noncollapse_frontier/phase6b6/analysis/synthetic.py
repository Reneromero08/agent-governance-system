"""Deterministic synthetic custody fixtures for the full software pipeline."""

from __future__ import annotations

import math
from typing import Any

from contracts.schedule import campaign_schedule
from runtime.explicit_slot_runtime import run_mock


SCENARIOS = (
    "shared_persistent",
    "shared_driven",
    "route_local",
    "confounded",
    "rejected",
    "session_lookup_dominates",
)


def _control(row: dict[str, Any]) -> float:
    tone = row["u_t"].get("physical_tone_index")
    analysis_tone = row["declared"].get("analysis_tone_index")
    tone_term = 0.0 if tone is None else (tone + 1) * 0.015
    if tone is None and analysis_tone is not None:
        tone_term = (analysis_tone + 1) * 0.004
    sign = row["u_t"].get("sign") or row["u_t"].get("codeword_sign") or 0
    phase = row["u_t"].get("phase_action")
    phase_term = {"0": 0.0, "pi": -0.10, "pi/2": 0.18, "-pi/2": -0.18}.get(phase, 0.0)
    drive = 1.0 if row["u_t"]["drive_on"] else 0.0
    return drive * (0.45 + tone_term + 0.08 * float(sign) + phase_term)


def _write_response(row: dict[str, Any], signal: float) -> None:
    row["r_t"]["lockin_I"] = round(signal, 9)
    row["r_t"]["lockin_Q"] = round(0.35 * signal + 0.02 * math.sin(signal), 9)
    row["r_t"]["ring_osc_period"] = round(100.0 + 0.25 * signal, 9)


def synthetic_custody(scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic scenario: {scenario}")
    schedule = campaign_schedule()
    custody = run_mock(schedule)
    custody["synthetic_scenario"] = scenario
    for session in custody["sessions"]:
        state = 0.05 * (session["session_index"] + 1)
        route_sign = 1.0 if session["route"] == "v4s5" else -1.0
        for row in session["slots"]:
            ctrl = _control(row)
            if scenario in ("shared_persistent", "shared_driven", "route_local") and "SHAM" in str(row["u_t"].get("executed_mode")):
                signal = 0.02
            elif scenario == "shared_persistent":
                if row["u_t"]["drive_on"]:
                    state = 0.58 * state + ctrl
                else:
                    state = 0.94 * state + 0.02 * ((row["declared"].get("analysis_tone_index") or 0) + 1)
                signal = state
            elif scenario == "shared_driven":
                if row["u_t"]["drive_on"]:
                    state = 0.58 * state + ctrl
                else:
                    state = 0.01 * state
                signal = state
            elif scenario == "route_local":
                state = 0.50 * state + route_sign * ctrl
                signal = state
            elif scenario == "confounded":
                state = 0.005 * row["slot_index"]
                signal = state
            elif scenario == "session_lookup_dominates":
                state = float(session["session_index"])
                signal = state
            else:
                state = 0.001 * ((row["slot_index"] * 17 + session["session_index"] * 5) % 23 - 11)
                signal = state
            _write_response(row, signal)
    return custody
