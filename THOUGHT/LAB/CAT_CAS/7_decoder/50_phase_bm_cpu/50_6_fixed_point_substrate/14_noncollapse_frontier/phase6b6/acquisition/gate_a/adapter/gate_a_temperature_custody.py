#!/usr/bin/env python3
"""Fail-closed k10temp discovery and evidence custody for Gate A.

The frozen target exposes the AMD CPU sensor through Linux hwmon.  This module
selects exactly one ``k10temp`` device and exactly ``temp1_input``.  It never
falls back to GPU sensors, ACPI thermal zones, or a best-effort maximum.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Callable

SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_TEMPERATURE_RECEIPT_V1"
HWMON_ROOT = Path("/sys/class/hwmon")
DRIVER_NAME = "k10temp"
TEMPERATURE_INPUT = "temp1_input"
MILLIDEGREES_PER_C = 1000
VETO_C = 68.0
MIN_PLAUSIBLE_C = -40.0
MAX_PLAUSIBLE_C = 125.0
RECEIPT_KEYS = frozenset({
    "schema_id",
    "phase",
    "hwmon_root",
    "required_driver_name",
    "required_temperature_input",
    "millidegrees_per_c",
    "enumerated_entries",
    "selected_hwmon_entry",
    "selected_driver_name",
    "selected_temperature_path",
    "raw_temperature_text",
    "raw_temperature_sha256",
    "raw_millidegrees_c",
    "normalized_temperature_c",
    "veto_temperature_c",
    "observation_complete",
    "veto_passed",
    "failure",
})
ENTRY_KEYS = frozenset({"entry", "name_path", "driver_name", "driver_name_sha256"})


class TemperatureCustodyError(RuntimeError):
    pass


ReadText = Callable[[Path], str]
StatPath = Callable[[Path], os.stat_result]
ListEntries = Callable[[Path], list[Path]]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TemperatureCustodyError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def receipt_sha256(receipt: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_bytes(receipt)).hexdigest()


def _default_read_text(path: Path) -> str:
    return path.read_text(encoding="ascii")


def _default_stat(path: Path) -> os.stat_result:
    return path.stat()


def _default_list_entries(root: Path) -> list[Path]:
    return list(root.iterdir())


def _identity(stat_result: os.stat_result) -> tuple[int, int, int]:
    return (stat_result.st_dev, stat_result.st_ino, stat_result.st_mode)


def _stable_text(path: Path, *, read_text: ReadText, stat_path: StatPath) -> str:
    before = _identity(stat_path(path))
    first = read_text(path)
    middle = _identity(stat_path(path))
    second = read_text(path)
    after = _identity(stat_path(path))
    require(before == middle == after, f"temperature path identity changed: {path}")
    require(first == second, f"temperature path changed between reads: {path}")
    require(first != "", f"temperature path is empty: {path}")
    return first


def _base_receipt(phase: str, root: Path) -> dict[str, Any]:
    return {
        "schema_id": SCHEMA_ID,
        "phase": phase,
        "hwmon_root": str(root),
        "required_driver_name": DRIVER_NAME,
        "required_temperature_input": TEMPERATURE_INPUT,
        "millidegrees_per_c": MILLIDEGREES_PER_C,
        "enumerated_entries": [],
        "selected_hwmon_entry": None,
        "selected_driver_name": None,
        "selected_temperature_path": None,
        "raw_temperature_text": None,
        "raw_temperature_sha256": None,
        "raw_millidegrees_c": None,
        "normalized_temperature_c": None,
        "veto_temperature_c": VETO_C,
        "observation_complete": False,
        "veto_passed": False,
        "failure": None,
    }


def observe_temperature(
    phase: str,
    *,
    hwmon_root: Path = HWMON_ROOT,
    read_text: ReadText = _default_read_text,
    stat_path: StatPath = _default_stat,
    list_entries: ListEntries = _default_list_entries,
) -> dict[str, Any]:
    """Return a closed receipt on both success and fail-closed observation."""

    receipt = _base_receipt(phase, hwmon_root)
    try:
        require(phase in {"pre_runtime", "pre_capture", "post_capture"}, "temperature phase is not closed")
        require(hwmon_root.is_dir(), "hwmon root unavailable")
        entries = sorted(
            (entry for entry in list_entries(hwmon_root) if re.fullmatch(r"hwmon[0-9]+", entry.name)),
            key=lambda entry: entry.name,
        )
        require(entries, "hwmon enumeration is empty")
        candidates: list[Path] = []
        for entry in entries:
            name_path = entry / "name"
            driver_raw = _stable_text(name_path, read_text=read_text, stat_path=stat_path)
            driver_name = driver_raw.strip()
            require(driver_name != "", f"hwmon driver name is empty: {name_path}")
            receipt["enumerated_entries"].append({
                "entry": str(entry),
                "name_path": str(name_path),
                "driver_name": driver_name,
                "driver_name_sha256": hashlib.sha256(driver_raw.encode("ascii")).hexdigest(),
            })
            if driver_name == DRIVER_NAME:
                candidates.append(entry)
        require(len(candidates) == 1, f"expected exactly one usable k10temp entry, found {len(candidates)}")
        selected = candidates[0]
        temperature_path = selected / TEMPERATURE_INPUT
        raw_text = _stable_text(temperature_path, read_text=read_text, stat_path=stat_path)
        stripped = raw_text.strip()
        require(re.fullmatch(r"[+-]?[0-9]+", stripped) is not None, "temperature value is not a complete integer")
        raw_millidegrees = int(stripped, 10)
        require(abs(raw_millidegrees) <= 1_000_000, "temperature integer is implausibly large")
        normalized = raw_millidegrees / MILLIDEGREES_PER_C
        require(math.isfinite(normalized), "temperature is nonfinite")
        require(MIN_PLAUSIBLE_C <= normalized <= MAX_PLAUSIBLE_C, "temperature is physically implausible")
        receipt.update({
            "selected_hwmon_entry": str(selected),
            "selected_driver_name": DRIVER_NAME,
            "selected_temperature_path": str(temperature_path),
            "raw_temperature_text": raw_text,
            "raw_temperature_sha256": hashlib.sha256(raw_text.encode("ascii")).hexdigest(),
            "raw_millidegrees_c": raw_millidegrees,
            "normalized_temperature_c": normalized,
            "observation_complete": True,
            "veto_passed": normalized < VETO_C,
        })
        if normalized >= VETO_C:
            receipt["failure"] = "TEMPERATURE_VETO"
    except (TemperatureCustodyError, OSError, UnicodeError, ValueError, TypeError) as exc:
        receipt["failure"] = f"TEMPERATURE_UNOBSERVABLE:{type(exc).__name__}:{exc}"
        receipt["observation_complete"] = False
        receipt["veto_passed"] = False
    validate_temperature_receipt(receipt, expected_phase=phase, require_pass=False)
    return receipt


def validate_temperature_receipt(
    receipt: dict[str, Any],
    *,
    expected_phase: str,
    require_pass: bool,
) -> None:
    require(isinstance(receipt, dict) and set(receipt) == RECEIPT_KEYS, "temperature receipt key set mismatch")
    require(receipt["schema_id"] == SCHEMA_ID, "temperature receipt schema mismatch")
    require(receipt["phase"] == expected_phase, "temperature receipt phase mismatch")
    require(receipt["required_driver_name"] == DRIVER_NAME, "temperature driver contract mismatch")
    require(receipt["required_temperature_input"] == TEMPERATURE_INPUT, "temperature input contract mismatch")
    require(receipt["millidegrees_per_c"] == MILLIDEGREES_PER_C, "temperature scale mismatch")
    require(receipt["veto_temperature_c"] == VETO_C, "temperature veto threshold mismatch")
    entries = receipt["enumerated_entries"]
    require(isinstance(entries, list), "temperature enumeration is not a list")
    previous = ""
    for entry in entries:
        require(isinstance(entry, dict) and set(entry) == ENTRY_KEYS, "temperature enumeration entry mismatch")
        require(entry["entry"] > previous, "temperature enumeration order is not deterministic")
        require(entry["name_path"] == str(Path(entry["entry"]) / "name"), "temperature name path mismatch")
        require(isinstance(entry["driver_name"], str) and entry["driver_name"] != "", "temperature driver name malformed")
        require(isinstance(entry["driver_name_sha256"], str) and len(entry["driver_name_sha256"]) == 64, "temperature driver-name digest malformed")
        previous = entry["entry"]
    if receipt["observation_complete"]:
        require(receipt["selected_driver_name"] == DRIVER_NAME, "selected temperature driver mismatch")
        selected = Path(receipt["selected_hwmon_entry"])
        require(receipt["selected_temperature_path"] == str(selected / TEMPERATURE_INPUT), "selected temperature path mismatch")
        raw_text = receipt["raw_temperature_text"]
        require(isinstance(raw_text, str) and raw_text != "", "raw temperature text missing")
        require(hashlib.sha256(raw_text.encode("ascii")).hexdigest() == receipt["raw_temperature_sha256"], "raw temperature digest mismatch")
        stripped = raw_text.strip()
        require(re.fullmatch(r"[+-]?[0-9]+", stripped) is not None, "raw temperature integer malformed")
        raw_millidegrees = int(stripped, 10)
        require(raw_millidegrees == receipt["raw_millidegrees_c"], "raw temperature value mismatch")
        normalized = raw_millidegrees / MILLIDEGREES_PER_C
        require(math.isclose(normalized, receipt["normalized_temperature_c"], rel_tol=0.0, abs_tol=1e-12), "normalized temperature mismatch")
        require(MIN_PLAUSIBLE_C <= normalized <= MAX_PLAUSIBLE_C, "normalized temperature implausible")
        require(receipt["veto_passed"] is (normalized < VETO_C), "temperature veto result mismatch")
        if receipt["veto_passed"]:
            require(receipt["failure"] is None, "passing temperature receipt contains failure")
        else:
            require(receipt["failure"] == "TEMPERATURE_VETO", "temperature veto failure mismatch")
    else:
        require(receipt["veto_passed"] is False, "incomplete temperature observation passed veto")
        require(isinstance(receipt["failure"], str) and receipt["failure"], "incomplete temperature observation lacks failure")
    if require_pass:
        require(receipt["observation_complete"] is True, "temperature observation incomplete")
        require(receipt["veto_passed"] is True, "temperature veto")


def normalized_temperature_c(receipt: dict[str, Any]) -> float:
    validate_temperature_receipt(receipt, expected_phase=receipt.get("phase"), require_pass=True)
    value = receipt["normalized_temperature_c"]
    require(isinstance(value, (int, float)) and math.isfinite(value), "temperature value malformed")
    return float(value)
