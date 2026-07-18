#!/usr/bin/env python3
"""Verify downloaded/manual source bytes against the P0 research manifest."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
REPORT = ROOT / "VERIFICATION_REPORT.json"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def source_path(record: dict) -> Path:
    group = "supplemental" if record["collection"] != "core_component_document" else "official"
    return ROOT / "sources" / group / record["local_filename"]

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--require-all-core", action="store_true")
    p.add_argument("--strict-legacy", action="store_true")
    args = p.parse_args()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    results = []
    missing_core = 0
    legacy_mismatches = 0
    for record in manifest["records"]:
        path = source_path(record)
        item = {"id": record["id"], "path": str(path.relative_to(ROOT)), "present": path.is_file()}
        if not path.is_file():
            item["status"] = "MISSING"
            if record["collection"] == "core_component_document":
                missing_core += 1
        else:
            actual = sha256(path)
            item.update(bytes=path.stat().st_size, sha256=actual)
            legacy_hash = record.get("legacy_expected_sha256")
            legacy_bytes = record.get("legacy_expected_bytes")
            if legacy_hash:
                match = actual == legacy_hash and (legacy_bytes is None or path.stat().st_size == legacy_bytes)
                item["legacy_match"] = match
                item["status"] = "PRESENT_LEGACY_MATCH" if match else "PRESENT_CURRENT_BYTES_LEGACY_DIFFERS"
                if not match:
                    legacy_mismatches += 1
            else:
                item["status"] = "PRESENT_NO_LEGACY_HASH"
        results.append(item)
    report = {"schema": "p0.research-verification-report.v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "missing_core": missing_core, "legacy_mismatches": legacy_mismatches,
              "results": results}
    REPORT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"present": sum(x["present"] for x in results), "missing_core": missing_core,
                      "legacy_mismatches": legacy_mismatches, "report": str(REPORT)}, indent=2))
    if (args.require_all_core and missing_core) or (args.strict_legacy and legacy_mismatches):
        return 2
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
