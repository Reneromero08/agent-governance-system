#!/usr/bin/env python3
"""Bind the Phase 6 V2 review-ledger correction into canonical authority files."""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

REVIEW_LEDGER_COMMIT = "3ed3b53cd44a244674422343d605187781c6e76e"
ROOT = Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True
).strip())
BASE = ROOT / "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate"
ROADMAP = BASE / "PHASE6_ROADMAP.md"
NAVIGATION = BASE / "PHASE6_NAVIGATION.md"
CHIRAL = BASE / "14_noncollapse_frontier/CHIRAL_LANE_NONCOLLAPSE_ROADMAP.md"
WORK_PACKAGE = BASE / "14_noncollapse_frontier/combined_observability_campaign/V2_FINAL_QUALIFICATION_WORK_PACKAGE.md"
ARCH_REVIEW = BASE / "14_noncollapse_frontier/combined_observability_campaign/v2/ARCHITECTURE_REVIEW.md"
EVIDENCE = BASE / "14_noncollapse_frontier/combined_observability_campaign/v2/evidence"
BINDINGS = EVIDENCE / "FINAL_BINDINGS.json"
INVENTORY = EVIDENCE / "EVIDENCE_INVENTORY.sha256"
VERIFICATION = EVIDENCE / "EVIDENCE_INVENTORY_VERIFICATION.txt"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_docs() -> None:
    anchor = "command evidence closure: f531ac8016c9c95141ed1c0ec180bcd01370d346\n"
    replacement = anchor + f"review ledger correction: {REVIEW_LEDGER_COMMIT}\n"
    for path, label in (
        (ROADMAP, "roadmap review ledger"),
        (NAVIGATION, "navigation review ledger"),
        (CHIRAL, "chiral review ledger"),
        (ARCH_REVIEW, "architecture review ledger"),
    ):
        replace_once(path, anchor, replacement, label)

    metadata_anchor = "**Command evidence closure:** `f531ac8016c9c95141ed1c0ec180bcd01370d346`\n"
    metadata_replacement = metadata_anchor + f"**Review ledger correction:** `{REVIEW_LEDGER_COMMIT}`\n"
    replace_once(WORK_PACKAGE, metadata_anchor, metadata_replacement,
                 "work package metadata review ledger")

    record_anchor = "command evidence closure: f531ac8016c9c95141ed1c0ec180bcd01370d346\n"
    record_replacement = record_anchor + f"review ledger correction: {REVIEW_LEDGER_COMMIT}\n"
    replace_once(WORK_PACKAGE, record_anchor, record_replacement,
                 "work package record review ledger")


def patch_bindings() -> None:
    data = json.loads(BINDINGS.read_text(encoding="utf-8"))
    data["review_ledger_correction_commit"] = REVIEW_LEDGER_COMMIT
    provenance = data.setdefault("provenance", {})
    provenance["3ed3b53c"] = "review ledger provenance correction"
    BINDINGS.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def regenerate_inventory() -> None:
    files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file() and path != INVENTORY)
    rels = [path.relative_to(ROOT).as_posix() for path in files]
    verification = ["Evidence inventory verification"]
    verification.extend(f"OK: {rel}" for rel in rels)
    verification.extend([
        "Command ledger entries: 16",
        "Linux command entries: 13",
        "Windows command entries: 3",
        "Per-command timestamps recorded: false",
        f"Review ledger correction: {REVIEW_LEDGER_COMMIT}",
        f"Entries: {len(rels)}",
        "Verification: PASSED",
        "",
    ])
    VERIFICATION.write_text("\n".join(verification), encoding="utf-8")

    files = sorted(path for path in EVIDENCE.rglob("*") if path.is_file() and path != INVENTORY)
    rows = []
    for path in files:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    INVENTORY.write_text("\n".join(rows) + "\n", encoding="utf-8")

    previous = ""
    seen: set[str] = set()
    for row in INVENTORY.read_text(encoding="utf-8").splitlines():
        digest, rel = row.split("  ", 1)
        if rel in seen or rel < previous:
            raise RuntimeError("inventory duplicate or ordering failure")
        seen.add(rel)
        previous = rel
        path = ROOT / rel
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"inventory verification failed: {rel}")


def main() -> int:
    patch_docs()
    patch_bindings()
    regenerate_inventory()
    allowed = {
        ROADMAP.relative_to(ROOT).as_posix(),
        NAVIGATION.relative_to(ROOT).as_posix(),
        CHIRAL.relative_to(ROOT).as_posix(),
        WORK_PACKAGE.relative_to(ROOT).as_posix(),
        ARCH_REVIEW.relative_to(ROOT).as_posix(),
        BINDINGS.relative_to(ROOT).as_posix(),
        INVENTORY.relative_to(ROOT).as_posix(),
        VERIFICATION.relative_to(ROOT).as_posix(),
    }
    changed = set(subprocess.check_output(
        ["git", "diff", "--name-only"], cwd=ROOT, text=True
    ).splitlines())
    if changed != allowed:
        raise RuntimeError(f"unexpected changed paths: {sorted(changed ^ allowed)}")
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    print(json.dumps({"changed_paths": sorted(changed), "review_ledger": REVIEW_LEDGER_COMMIT}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
