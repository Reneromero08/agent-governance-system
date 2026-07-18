#!/usr/bin/env python3
"""Copy authored P0 repository context into this research bundle without modifying the source directory."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = ['P0_COMPONENT_DOCUMENTS.json', 'P0_BUILD_READINESS_COMPONENT_SEED.md', 'P0_CARRIER_AND_ACCESS_SELECTION.md', 'P0_BOM_SAFETY_AND_SILICON_TRANSLATION.md', 'P0_MEASUREMENT_AND_SOURCE_OFF_PLAN.md', 'P0_CONTROL_KILL_AND_ADJUDICATION.md', 'P0_REVIEW_REPORTS.md', 'P0_BUILD_READINESS_AUTHORITY.md', 'PHYSICAL_PHASE_CARRIER_P0_CONTRACT.md', 'P0_BUILD_READINESS_FINDINGS.json', 'P0_BUILD_READINESS_PACKET.md', 'P0_FINAL_NETLIST.json', 'P0_NONPURCHASING_BOM.json', 'P0_PCB_FABRICATION_RELEASE.json', 'P0_BUILD_READINESS_SCHEMAS.json', 'P0_SCIENTIFIC_FIXTURES.json', 'P0_ANALYZER_REFERENCE_RESULTS.json', 'p0_scientific_analyzer.py', 'p0_build_readiness_design.py', 'p0_build_readiness_validator.py']

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--p0-dir", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    src = args.p0_dir.resolve()
    if not src.is_dir():
        p.error(f"not a directory: {src}")
    dest = ROOT / "repo_context"
    dest.mkdir(exist_ok=True)
    results = []
    for name in FILES:
        source = src / name
        target = dest / name
        if not source.is_file():
            results.append({"path": name, "status": "MISSING_IN_REPO"})
            continue
        if target.exists() and not args.force:
            results.append({"path": name, "status": "ALREADY_PRESENT", "sha256": sha256(target), "bytes": target.stat().st_size})
            continue
        shutil.copy2(source, target)
        results.append({"path": name, "status": "COPIED", "sha256": sha256(target), "bytes": target.stat().st_size})
    receipt = {"schema": "p0.repo-context-import-receipt.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
               "source_directory": str(src), "results": results}
    (ROOT / "REPO_CONTEXT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"copied": sum(r["status"] == "COPIED" for r in results),
                      "missing": sum(r["status"] == "MISSING_IN_REPO" for r in results)}, indent=2))
    return 0 if not any(r["status"] == "MISSING_IN_REPO" for r in results) else 1
if __name__ == "__main__":
    raise SystemExit(main())
