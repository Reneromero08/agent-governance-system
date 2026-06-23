#!/usr/bin/env python3
"""Apply only the analyzer and analyzer-test hunks from the audited repair payload."""
from __future__ import annotations

import ast
import base64
import hashlib
import subprocess
import zlib
from pathlib import Path

SOURCE_PATCH_SHA256 = "95c0988727881c23c690d876caf66a45b4aad42fe97513c1c675f658f1a257d1"
WORKFLOW_MARKER = b"--- a/.github/workflows/phase6-v2-strict-qualification.yml\n"
FILES = {
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/analyze_spectral_calibration_v2.py": (
        "b4947af2a0c93bd7d3e817904b5a3a7ceaf98c92e526baf19797da6261ee7b27",
        "87e6043633812411941cce67267f2ef6790b71cbbcd1f97bb99073b00084f9d3",
    ),
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/50_6_fixed_point_substrate/14_noncollapse_frontier/combined_observability_campaign/v2/test_spectral_calibration_analyzer.py": (
        "33b0aa51d0b818188bba9ba5725357f406311e481a78b933ba230428f57944a0",
        "236aced8eacdbd92ab292ad97b76f47158e8b856e1ee52c3b6ccbc9ddb172626",
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_payload(installer: Path) -> bytes:
    tree = ast.parse(installer.read_text(encoding="utf-8"), filename=str(installer))
    encoded = None
    declared_digest = None
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id == "PATCH_ZLIB_B64":
            encoded = ast.literal_eval(node.value)
        elif target.id == "PATCH_SHA256":
            declared_digest = ast.literal_eval(node.value)
    if not isinstance(encoded, str) or not isinstance(declared_digest, str):
        raise RuntimeError("audited repair payload constants are missing")
    full_patch = zlib.decompress(base64.b64decode(encoded))
    if hashlib.sha256(full_patch).hexdigest() != declared_digest:
        raise RuntimeError("audited full repair payload digest mismatch")
    if WORKFLOW_MARKER not in full_patch:
        raise RuntimeError("workflow split marker absent from audited repair payload")
    source_patch = full_patch.split(WORKFLOW_MARKER, 1)[0]
    if hashlib.sha256(source_patch).hexdigest() != SOURCE_PATCH_SHA256:
        raise RuntimeError("audited source-only repair digest mismatch")
    return source_patch


def main() -> int:
    root = Path(subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip())
    current = {name: digest(root / name) for name in FILES}
    if all(current[name] == final for name, (_before, final) in FILES.items()):
        print("SAME_BYTE_CUSTODY_SOURCE_REPAIR_ALREADY_APPLIED")
        return 0
    mismatches = {
        name: {"expected": before, "actual": current[name]}
        for name, (before, _final) in FILES.items()
        if current[name] != before
    }
    if mismatches:
        raise RuntimeError(f"source repair preimage mismatch: {mismatches}")
    installer = Path(__file__).with_name("APPLY_SAME_BYTE_CUSTODY_REPAIR.py")
    patch = extract_payload(installer)
    checked = subprocess.run(
        ["git", "apply", "--check", "--whitespace=error-all", "-"],
        cwd=root, input=patch, capture_output=True,
    )
    if checked.returncode:
        raise RuntimeError(checked.stderr.decode("utf-8", errors="replace"))
    applied = subprocess.run(
        ["git", "apply", "--whitespace=error-all", "-"],
        cwd=root, input=patch, capture_output=True,
    )
    if applied.returncode:
        raise RuntimeError(applied.stderr.decode("utf-8", errors="replace"))
    final = {name: digest(root / name) for name in FILES}
    failures = {
        name: {"expected": expected, "actual": final[name]}
        for name, (_before, expected) in FILES.items()
        if final[name] != expected
    }
    if failures:
        raise RuntimeError(f"source repair final digest mismatch: {failures}")
    print("SAME_BYTE_CUSTODY_SOURCE_REPAIR_APPLIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
