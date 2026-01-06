#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:  # scanner: read only
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        inputs = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        # Use simple error return if we can't parse input
        # Firewall forbids raw write, so we must use GuardedWriter to write error if possible
        pass 

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(
        project_root=PROJECT_ROOT,
        durable_roots=[
            "LAW/CONTRACTS/_runs",
            "NAVIGATION/CORTEX/_generated",
            "MEMORY/LLM_PACKER/_packs",
            "BUILD"
        ]
    )
    writer.open_commit_gate()

    # Re-handle input error with writer
    try:
        inputs = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        writer.write_durable(str(output_path), json.dumps({"status": "failure", "error": f"INPUT_INVALID: {exc}"}, indent=2) + "\n")
        return 0

    cas_root_str = inputs.get("cas_root")
    if not isinstance(cas_root_str, str) or not cas_root_str.strip():
        writer.write_durable(str(output_path), json.dumps({"status": "failure", "error": "MISSING_CAS_ROOT"}, indent=2) + "\n")
        return 0

    cas_root = Path(cas_root_str)
    if not cas_root.is_absolute():
        cas_root = (PROJECT_ROOT / cas_root).resolve()

    if not cas_root.exists():
        writer.write_durable(
            str(output_path),
            json.dumps({"status": "failure", "error": f"CAS_ROOT_NOT_FOUND: {cas_root}", "cas_root": str(cas_root)}, indent=2) + "\n"
        )
        return 0

    corrupt_blobs = []
    total_blobs = 0

    for file_path in cas_root.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue

        total_blobs += 1
        expected_hash = file_path.name
        if len(expected_hash) != 64:
            corrupt_blobs.append({"path": str(file_path), "reason": "invalid_filename_format"})
            continue

        try:
            actual_hash = _sha256_file(file_path)
        except Exception as exc:
            corrupt_blobs.append({"path": str(file_path), "reason": f"read_error: {exc}"})
            continue

        if actual_hash != expected_hash:
            corrupt_blobs.append({"path": str(file_path), "expected": expected_hash, "actual": actual_hash, "reason": "hash_mismatch"})

    status = "success" if not corrupt_blobs else "failure"
    result = {"status": status, "total_blobs": total_blobs, "corrupt_blobs": corrupt_blobs, "cas_root": str(cas_root)}

    writer.mkdir_durable(str(output_path.parent))
    writer.write_durable(str(output_path), json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

