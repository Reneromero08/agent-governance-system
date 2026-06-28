"""Deterministic synthetic custody fixtures for the full software pipeline."""

from __future__ import annotations

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


def synthetic_custody(scenario: str) -> dict[str, Any]:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown synthetic scenario: {scenario}")
    schedule = campaign_schedule()
    custody = run_mock(schedule)
    custody["synthetic_scenario"] = scenario
    for session in custody["sessions"]:
        for row in session["slots"]:
            tone = row["u_t"].get("physical_tone_index")
            tone_term = 0.0 if tone is None else (tone + 1) * 0.01
            drive = 1.0 if row["u_t"]["drive_on"] else 0.0
            if scenario == "rejected":
                signal = 0.001 * ((row["slot_index"] % 7) - 3)
            elif scenario == "confounded":
                signal = row["slot_index"] * 0.002
            elif scenario == "route_local":
                signal = (1.0 if row["route"] == "v4s5" else -1.0) * (drive + tone_term)
            elif scenario == "session_lookup_dominates":
                signal = float(row["session_index"])
            else:
                signal = drive + tone_term
                if scenario == "shared_persistent" and row["stage"] == "trajectory" and not row["u_t"]["drive_on"]:
                    signal += 0.3
            row["r_t"]["lockin_I"] = round(signal, 9)
            row["r_t"]["lockin_Q"] = round(signal * 0.5, 9)
            row["r_t"]["ring_osc_period"] = round(100.0 + signal, 9)
    return custody
