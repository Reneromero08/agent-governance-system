#!/usr/bin/env python3
"""Expand one frozen campaign session into explicit acquisition windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def emit_symbol(stage: str, block_id: str, symbol_index: int, row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    order = row["tone_execution_indices"]
    perm = row["codeword_bin_permutation"]
    for position, tone_index in enumerate(order):
        yield {
            "stage": stage,
            "block_id": block_id,
            "symbol_index": symbol_index,
            "window_in_symbol": position,
            "physical_tone_index": tone_index,
            "codeword_source_index": perm[tone_index],
            "drive_on": row["drive_on"],
            "shared_schedule": row["shared_schedule"],
            "family": row["family"],
            "actual_mode": row["actual_mode"],
            "declared_mode": row["declared_mode"],
            "theta_idx": row["theta_idx"],
            "executed_tone_order": row["executed_tone_order"],
            "declared_tone_order": row["declared_tone_order"],
            "measurement_mode": "lockin_and_raw_ring",
            "sender_off_required": False,
        }


def compile_session(plan: dict[str, Any], session_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    session = next((item for item in plan["sessions"] if item["session_id"] == session_id), None)
    if session is None:
        raise ValueError(f"unknown session {session_id}")
    windows: list[dict[str, Any]] = []
    symbol_index = 0
    for row in session["blocks"]["gauge_preamble"]:
        windows.extend(emit_symbol("A_GAUGE", "gauge_preamble", symbol_index, row))
        symbol_index += 1
    for block in session["blocks"]["tone_order"]:
        block_id = f"tone_{block['order']}"
        for row in block["symbols"]:
            windows.extend(emit_symbol("B_TONE_ORDER", block_id, symbol_index, row))
            symbol_index += 1
    for event in session["blocks"]["persistence"]:
        order = event["tone_execution_indices"]
        for index in range(event["prepare_windows"]):
            tone_index = order[index % len(order)]
            windows.append({
                "stage": "C_PERSISTENCE_PREPARE", "block_id": event["event_id"],
                "event_window": index, "physical_tone_index": tone_index,
                "codeword_source_index": tone_index, "drive_on": True,
                "family": event["input_type"], "actual_mode": event["actual_mode"],
                "declared_mode": event["actual_mode"], "theta_idx": event["theta_idx"],
                "executed_tone_order": event["executed_tone_order"],
                "declared_tone_order": event["executed_tone_order"],
                "measurement_mode": "lockin_and_raw_ring", "sender_off_required": False,
            })
        for index in range(event["sender_off_windows"]):
            windows.append({
                "stage": "C_PERSISTENCE_OFF", "block_id": event["event_id"],
                "event_window": index, "physical_tone_index": None,
                "codeword_source_index": None, "drive_on": False,
                "family": event["input_type"], "actual_mode": event["actual_mode"],
                "declared_mode": event["actual_mode"], "theta_idx": event["theta_idx"],
                "executed_tone_order": event["executed_tone_order"],
                "declared_tone_order": event["executed_tone_order"],
                "measurement_mode": "raw_ring_sender_off", "sender_off_required": True,
            })
    for block in session["blocks"]["trajectories"]:
        order = plan["orders"][block["order"]]
        for step in block["steps"]:
            for position, tone_index in enumerate(order):
                windows.append({
                    "stage": "D_TRAJECTORY", "block_id": f"trajectory_{block['order']}",
                    "trajectory_step": step["step"], "window_in_step": position,
                    "physical_tone_index": tone_index, "codeword_source_index": tone_index,
                    "drive_on": step["drive_on"], "amplitude_level": step["amplitude_level"],
                    "family": "trajectory", "actual_mode": step["actual_mode"],
                    "declared_mode": step["actual_mode"], "theta_idx": step["theta_idx"],
                    "executed_tone_order": block["order"], "declared_tone_order": block["order"],
                    "measurement_mode": "lockin_and_raw_ring", "sender_off_required": not step["drive_on"],
                })
    for index, window in enumerate(windows):
        window["window_index"] = index
        window["session_id"] = session_id
        window["route"] = session["route"]
        window["seed"] = session["seed"]
        window["partition"] = session["partition"]
    header = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_SCHEDULE_V1",
        "campaign_source_commit": plan["source_commit"],
        "session_id": session_id,
        "route": session["route"],
        "seed": session["seed"],
        "partition": session["partition"],
        "order_sequence": session["order_sequence"],
        "window_count": len(windows),
        "restoration_authorized": False,
    }
    return header, windows


def validate(header: dict[str, Any], windows: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if header.get("window_count") != len(windows): errors.append("window count mismatch")
    if [row.get("window_index") for row in windows] != list(range(len(windows))): errors.append("window indices not contiguous")
    if any(row.get("stage") == "C_PERSISTENCE_OFF" and (row.get("drive_on") or not row.get("sender_off_required")) for row in windows): errors.append("sender-off violation")
    if any(row.get("family") == "order_sham" and row.get("executed_tone_order") == row.get("declared_tone_order") for row in windows): errors.append("order sham collapsed")
    if any(row.get("family") == "silent" and row.get("drive_on") for row in windows): errors.append("silent drive enabled")
    if header.get("restoration_authorized") is not False: errors.append("restoration enabled")
    return errors


def write_session(plan_path: Path, session_id: str, output: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text())
    header, windows = compile_session(plan, session_id)
    header["campaign_plan_sha256"] = sha256_file(plan_path)
    errors = validate(header, windows)
    if errors: raise ValueError("; ".join(errors))
    output.mkdir(parents=True, exist_ok=False)
    header_path = output / "session.json"
    windows_path = output / "windows.jsonl"
    header_path.write_text(json.dumps(header, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with windows_path.open("w", encoding="utf-8") as f:
        for row in windows: f.write(json.dumps(row, sort_keys=True) + "\n")
    manifest = {
        "schema_id": "CAT_CAS_PHASE6_COMBINED_SESSION_MANIFEST_V1",
        "session_id": session_id,
        "files": {
            "session.json": {"size": header_path.stat().st_size, "sha256": sha256_file(header_path)},
            "windows.jsonl": {"size": windows_path.stat().st_size, "sha256": sha256_file(windows_path)},
        },
    }
    (output / "session_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("plan", type=Path); p.add_argument("session_id"); p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(); print(json.dumps(write_session(args.plan.resolve(), args.session_id, args.output.resolve()), indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
