#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP50 Phase 5.10 aggregator.

Rolls the per-subphase analysis outputs (5.10A instrumentation status, 5.10B
basin-scan selection CSV, 5.10C selection CSV) into one master verdict file,
phase5_10_master_verdict.csv, using the gate logic and verdict labels frozen
in the SPEC (PHASE5_10_GATES_AND_VERDICTS.md / GATE_QUESTIONS_5_10.md). This
file owns none of the .md; it only implements the verdict mapping the spec
defines.

Verdict labels (from the spec, NOT invented here):
  EXP50_PHASE5_10_BOUNDARY_STATE_PREPARATION_CONFIRMED
  EXP50_PHASE5_10_BASIN_STRUCTURE_CONFIRMED_SELECTION_PARTIAL
  EXP50_PHASE5_10_INSTRUMENTATION_BLOCKED
  EXP50_PHASE5_10_NO_REPRODUCIBLE_BASIN
  EXP50_PHASE5_10_ARTIFACT_DOMINANT
  EXP50_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT   (Gate-8 PASS only)

Discipline: the aggregator NEVER upgrades a verdict it cannot justify from the
subphase evidence. Missing instrumentation caps the claim (carried explicitly).
Outputs via Path(__file__) (M-7).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

THIS_DIR = Path(__file__).resolve().parent
# src/ is one level deeper than the phase dir root post-merge; _generated lives at the
# phase dir root (50_5_10_encoding_wall/_generated/), so anchor outputs on THIS_DIR.parent.
OUTPUT_DIR = THIS_DIR.parent / "_generated"

# Verdict labels (verbatim from the spec).
V_CONFIRMED = "EXP50_PHASE5_10_BOUNDARY_STATE_PREPARATION_CONFIRMED"
V_PARTIAL = "EXP50_PHASE5_10_BASIN_STRUCTURE_CONFIRMED_SELECTION_PARTIAL"
V_INSTR_BLOCKED = "EXP50_PHASE5_10_INSTRUMENTATION_BLOCKED"
V_NO_SELECTION = "EXP50_PHASE5_10_NO_REPRODUCIBLE_BASIN"
V_ARTIFACT = "EXP50_PHASE5_10_ARTIFACT_DOMINANT"
V_READY = "EXP50_PHASE5_10_READY_FOR_PHASE6_FIXED_POINT"


@dataclass
class SubphaseStatus:
    """One subphase's roll-up evidence feeding the master verdict."""

    name: str
    status: str
    selection_csv: Optional[Path] = None
    # 5.10A instrumentation confidence: pass | partial | blocked | failed
    instrumentation: str = "partial"
    # parametric-scaling verdict from analyze (G-4): SCALES_PHYSICAL |
    # INVARIANT_LOGICAL | UNDERPOWERED | not_run
    parametric_scaling: str = "not_run"
    notes: str = ""


def _read_selection_csv(path: Path) -> List[dict]:
    if not path or not Path(path).exists():
        return []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _has_surviving_selection(rows: Sequence[dict]) -> bool:
    """True iff some selector's selection survives null AND Holm correction."""
    for row in rows:
        if row.get("verdict") == "SELECTION_SURVIVES_NULL_AND_CORRECTION":
            return True
    return False


def _has_weak_selection(rows: Sequence[dict]) -> bool:
    for row in rows:
        if row.get("verdict") == "SELECTION_WEAK_FAILS_CORRECTION":
            return True
    return False


def _has_restoration_void(rows: Sequence[dict]) -> bool:
    for row in rows:
        if row.get("verdict") == "VOID_RESTORATION_FAILURE":
            return True
        try:
            if int(float(row.get("restoration_failures", "0"))) > 0:
                return True
        except (TypeError, ValueError):
            continue
    return False


@dataclass
class MasterVerdict:
    verdict: str
    gate1_instrumentation: str
    gate3_basin_classification: str
    gate6_selection: str
    gate7_artifact_controls: str
    gate4_parametric_scaling: str  # G-4 substrate-identity (central test)
    gate8_phase6_ready: str
    instrumentation_caveat: str
    rationale: str

    def to_row(self) -> dict:
        return {
            "phase": "5.10",
            "verdict": self.verdict,
            "gate1_instrumentation": self.gate1_instrumentation,
            "gate3_basin_classification": self.gate3_basin_classification,
            "gate6_selection": self.gate6_selection,
            "gate7_artifact_controls": self.gate7_artifact_controls,
            "gate4_parametric_scaling": self.gate4_parametric_scaling,
            "gate8_phase6_ready": self.gate8_phase6_ready,
            "instrumentation_caveat": self.instrumentation_caveat,
            "rationale": self.rationale,
        }


def derive_master_verdict(subphases: Sequence[SubphaseStatus]) -> MasterVerdict:
    """Map subphase evidence -> the spec's master verdict (no upgrades).

    Gate ordering follows the spec: instrumentation (G-1), basin classification
    (G-2/Gate-3), selection (G-3/Gate-6), artifact controls (Gate-7), and the
    central parametric-scaling substrate-identity test (G-4). Gate-8 (Phase 6
    readiness) PASSES only when every prior gate passes.
    """
    by_name = {s.name.lower(): s for s in subphases}
    a = by_name.get("5.10a") or by_name.get("instrumentation")
    c = by_name.get("5.10c") or by_name.get("selection")

    instr = (a.instrumentation if a else "partial").lower()
    sel_rows = _read_selection_csv(c.selection_csv) if (c and c.selection_csv) else []

    # --- Gate 1: instrumentation witness ---
    if instr in ("blocked", "failed"):
        gate1 = "BLOCKED" if instr == "blocked" else "FAIL"
    elif instr == "pass":
        gate1 = "PASS"
    else:
        gate1 = "PARTIAL"

    instr_caveat = {
        "pass": "physical witness present",
        "partial": "PARTIAL instrumentation: only decoded VID / P-state / temp / "
                   "wall power available; basin claim strength capped",
        "blocked": "BLOCKED: only decoded-VID register moved; substrate state "
                   "not witnessed",
        "failed": "FAILED: only requested values exist; substrate state unproven",
    }.get(instr, "PARTIAL instrumentation")

    # --- Gate 3 (basin classification): PASS if frozen-threshold rows exist ---
    gate3 = "PASS" if sel_rows else "FAIL"

    # --- Restoration integrity (Gate 2) gates everything ---
    if _has_restoration_void(sel_rows):
        return MasterVerdict(
            verdict=V_ARTIFACT,
            gate1_instrumentation=gate1,
            gate3_basin_classification=gate3,
            gate6_selection="FAIL",
            gate7_artifact_controls="FAIL",
            gate4_parametric_scaling="n/a",
            gate8_phase6_ready="FAIL",
            instrumentation_caveat=instr_caveat,
            rationale="restoration failure on a non-control run (Gate-2 FAIL): "
                      "run VOID, no catalysis, selection uninterpretable",
        )

    surviving = _has_surviving_selection(sel_rows)
    weak = _has_weak_selection(sel_rows)

    # --- Gate 6: selection ---
    if surviving:
        gate6 = "PASS"
    elif weak:
        gate6 = "PARTIAL"
    else:
        gate6 = "FAIL"

    # --- Gate 7: artifact controls (shuffled / no-prelude must NOT survive) ---
    control_names = ("shuffled", "no_prelude", "quiet", "wrong", "control")
    control_survived = any(
        row.get("verdict") == "SELECTION_SURVIVES_NULL_AND_CORRECTION"
        and any(cn in row.get("selector", "").lower() for cn in control_names)
        for row in sel_rows
    )
    gate7 = "FAIL" if control_survived else ("PASS" if sel_rows else "FAIL")

    # --- Gate 4: parametric-scaling substrate-identity (central) ---
    pscale = (c.parametric_scaling if c else "not_run").lower()
    if pscale == "scales_physical":
        gate4 = "PASS"
    elif pscale == "invariant_logical":
        gate4 = "FAIL"  # logical artifact: NOT a substrate basin
    else:
        gate4 = "PARTIAL"

    # --- Verdict mapping (no upgrade past the weakest controlling gate) ---
    if control_survived:
        verdict = V_ARTIFACT
        gate8 = "FAIL"
        rationale = "an artifact control reproduced the selection (Gate-7 FAIL)"
    elif gate1 in ("BLOCKED",) and not surviving:
        verdict = V_INSTR_BLOCKED
        gate8 = "FAIL"
        rationale = "instrumentation blocked and no surviving selection to interpret"
    elif surviving and gate7 == "PASS" and gate4 == "PASS" and gate1 == "PASS":
        verdict = V_READY
        gate8 = "PASS"
        rationale = ("selection survives null + Holm, controls fail, basin scales "
                     "parametrically, witness present: Gate-8 PASS")
    elif surviving and gate7 == "PASS":
        # Real selection but witness/parametric not at full strength -> CONFIRMED
        # at the preparation level, NOT Phase-6-ready.
        verdict = V_CONFIRMED
        gate8 = "FAIL"
        rationale = ("boundary state preparation confirmed (selection survives "
                     "null + correction, controls fail); Phase-6 readiness blocked "
                     "by instrumentation/parametric caveat (G-1/G-4 not full PASS)")
    elif weak:
        verdict = V_PARTIAL
        gate8 = "FAIL"
        rationale = ("basin structure present but selection too weak to survive "
                     "correction (Gate-6 PARTIAL)")
    else:
        verdict = V_NO_SELECTION
        gate8 = "FAIL"
        rationale = ("no selector biases a target basin above the reshuffle null "
                     "after correction (Gate-6 FAIL)")

    return MasterVerdict(
        verdict=verdict,
        gate1_instrumentation=gate1,
        gate3_basin_classification=gate3,
        gate6_selection=gate6,
        gate7_artifact_controls=gate7,
        gate4_parametric_scaling=gate4,
        gate8_phase6_ready=gate8,
        instrumentation_caveat=instr_caveat,
        rationale=rationale,
    )


def write_master_verdict_csv(verdict: MasterVerdict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    row = verdict.to_row()
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def aggregate(
    out_dir: Path,
    instrumentation: str = "partial",
    selection_csv: Optional[Path] = None,
    parametric_scaling: str = "not_run",
) -> MasterVerdict:
    """Roll subphase evidence into the master verdict and write the CSV."""
    subphases = [
        SubphaseStatus(name="5.10A", status="ingested", instrumentation=instrumentation),
        SubphaseStatus(
            name="5.10C", status="ingested",
            selection_csv=selection_csv, parametric_scaling=parametric_scaling,
        ),
    ]
    verdict = derive_master_verdict(subphases)
    write_master_verdict_csv(verdict, out_dir / "phase5_10_master_verdict.csv")
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EXP50 Phase 5.10 aggregator")
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument(
        "--selection-csv", type=str, default=None,
        help="5.10C selection CSV from analyze_phase5_10.py "
             "(defaults to the self-test's synthetic_selection.csv if present)",
    )
    parser.add_argument(
        "--instrumentation", type=str, default="partial",
        choices=["pass", "partial", "blocked", "failed"],
        help="5.10A instrumentation lock status (G-1)",
    )
    parser.add_argument(
        "--parametric-scaling", type=str, default="not_run",
        choices=["scales_physical", "invariant_logical", "underpowered", "not_run"],
        help="G-4 parametric-scaling verdict from analyze",
    )
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    sel_csv = Path(args.selection_csv) if args.selection_csv else None
    if sel_csv is None:
        default_sel = out_dir / "synthetic_selection.csv"
        if default_sel.exists():
            sel_csv = default_sel

    verdict = aggregate(
        out_dir=out_dir,
        instrumentation=args.instrumentation,
        selection_csv=sel_csv,
        parametric_scaling=args.parametric_scaling,
    )
    print(json.dumps(verdict.to_row(), indent=2, sort_keys=True))
    print(f"\n[aggregate] wrote {out_dir / 'phase5_10_master_verdict.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
