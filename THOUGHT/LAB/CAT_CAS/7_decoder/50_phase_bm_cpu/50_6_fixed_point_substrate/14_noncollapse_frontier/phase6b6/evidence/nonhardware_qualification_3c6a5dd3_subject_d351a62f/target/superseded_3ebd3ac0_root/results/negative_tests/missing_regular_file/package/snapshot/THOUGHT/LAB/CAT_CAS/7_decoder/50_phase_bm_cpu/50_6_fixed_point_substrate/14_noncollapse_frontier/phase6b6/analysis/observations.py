"""Canonical runtime custody observation loader."""

from __future__ import annotations

from typing import Any


REQUIRED_OBSERVATION_FIELDS = (
    "stage",
    "packet_id",
    "session_index",
    "reboot_block",
    "split",
    "route",
    "sender_core",
    "receiver_core",
    "session_chronology",
    "slot_index",
    "declared",
    "u_t",
    "r_t",
    "c_t",
)


def flatten_custody(custody: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for session in custody["sessions"]:
        for slot in session["slots"]:
            row = dict(slot)
            for field in REQUIRED_OBSERVATION_FIELDS:
                if field not in row:
                    raise ValueError(f"custody row missing {field}")
            rows.append(row)
    rows.sort(key=lambda row: (row["session_index"], row["slot_index"]))
    return rows


def split_rows(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["split"] == split]


def assert_test_sealed(rows: list[dict[str, Any]], allow_test: bool) -> None:
    if not allow_test and any(row["split"] == "test" for row in rows):
        raise PermissionError("TEST_SET_SEALED_UNTIL_ANALYSIS_CHOICE_MANIFEST")
