#!/usr/bin/env python3
"""Verify the source-only frequency-preparation authority/transport boundary."""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import build_gate_a_frequency_preparation_bundle as builder
import gate_a_frequency_preparation_authority as authority
import gate_a_frequency_preparation_bundle as target_bundle

HERE = Path(__file__).resolve().parent
ACTIVE_AUTHORITY = HERE / "GATE_A_FREQUENCY_PREPARATION_AUTHORITY.json"
SCHEMA = HERE / "schemas" / "gate_a_frequency_preparation_authority.schema.json"
EXECUTION_MANIFEST = HERE.parent / "adapter" / "GATE_A_EXECUTION_BUNDLE_MANIFEST.json"
EXPECTED_GATE_A_EXECUTION_BUNDLE = "353f7e2d865508ebc018cb72648d3d3f227dc1c1128681fd9b4e99d81c9aa47f"


class VerifyError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise VerifyError(message)


def verify_constants() -> dict[str, Any]:
    require(authority.REQUIRED_FREQUENCY_KHZ == 1_600_000, "required frequency changed")
    require(authority.SAMPLE_COUNT == 200 and authority.SAMPLE_INTERVAL_MS == 10, "observation window changed")
    require(authority.MAXIMUM_WRITE_ATTEMPT_COUNT == 8, "write cap changed")
    require(authority.MAXIMUM_TRANSACTION_COUNT == 1, "transaction count changed")
    require(authority.EXPECTED_SYSFS_ROOT == "/sys", "sysfs root changed")
    return {"status": "PREPARATION_CONSTANTS_EXACT"}


def verify_ast() -> dict[str, Any]:
    files = [
        HERE / "gate_a_frequency_preparation_authority.py",
        HERE / "gate_a_frequency_preparation_bundle.py",
        HERE / "gate_a_frequency_preparation_live.py",
        HERE / "gate_a_frequency_preparation_target.py",
        HERE / "gate_a_frequency_preparation_transport.py",
        HERE / "gate_a_frequency_preparation_adapter.py",
    ]
    forbidden_calls = {"eval", "exec", "compile", "os.system", "popen"}
    for path in files:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    require(node.func.id not in forbidden_calls, f"forbidden call in {path.name}: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    dotted = node.func.attr
                    require(dotted not in forbidden_calls, f"forbidden method in {path.name}: {dotted}")
            if isinstance(node, ast.keyword) and node.arg == "shell":
                require(not (isinstance(node.value, ast.Constant) and node.value.value is True), f"shell=True forbidden in {path.name}")
    adapter_source = (HERE / "gate_a_frequency_preparation_adapter.py").read_text(encoding="utf-8")
    require("if __name__ == \"__main__\"" in adapter_source, "adapter main guard missing")
    return {"status": "PREPARATION_SOURCE_AST_CLOSED", "file_count": len(files)}


def verify_authority_resting_state() -> dict[str, Any]:
    require(not ACTIVE_AUTHORITY.exists(), "active preparation authority artifact exists")
    require(not any(HERE.glob("GATE_A_FREQUENCY_PREPARATION_AUTHORITY_CONSUMED_*.json")), "unexpected consumed preparation authority archive")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    require(schema["$id"] == "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PREPARATION_AUTHORITY_SCHEMA_V1", "authority schema ID mismatch")
    require(schema["additionalProperties"] is False, "authority schema not closed")
    return {"status": "NO_ACTIVE_PREPARATION_AUTHORITY"}


def verify_manifest() -> dict[str, Any]:
    manifest = builder.validate_committed_manifest_exact("HEAD")
    target_bundle.validate_manifest_shape(manifest)
    first = builder.deployment_archive("HEAD")
    second = builder.deployment_archive("HEAD")
    require(first == second, "deployment archive not deterministic")
    return {
        "status": "PREPARATION_BUNDLE_EXACT",
        "bundle_sha256": manifest["bundle_sha256"],
        "deterministic_archive_sha256": manifest["deterministic_archive_sha256"],
        "deployment_archive_sha256": hashlib.sha256(first).hexdigest(),
        "file_count": len(manifest["files"]),
    }


def verify_gate_a_bundle_unchanged() -> dict[str, Any]:
    manifest = json.loads(EXECUTION_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["execution_bundle_sha256"] == EXPECTED_GATE_A_EXECUTION_BUNDLE, "Gate A execution bundle changed")
    return {"status": "GATE_A_EXECUTION_BUNDLE_UNCHANGED", "execution_bundle_sha256": manifest["execution_bundle_sha256"]}


def verify_adapter_default() -> dict[str, Any]:
    result = subprocess.run([sys.executable, "-B", str(HERE / "gate_a_frequency_preparation_adapter.py")], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    require(result.returncode == 2, "adapter default did not refuse")
    require("FREQUENCY_PREPARATION_AUTHORITY_REQUIRED" in result.stderr, "adapter refusal status missing")
    return {"status": "ADAPTER_DEFAULT_FAIL_CLOSED"}


def main() -> int:
    try:
        value = {
            "status": "GATE_A_FREQUENCY_PREPARATION_AUTHORITY_TRANSPORT_EXACT",
            "constants": verify_constants(),
            "ast": verify_ast(),
            "authority_state": verify_authority_resting_state(),
            "bundle": verify_manifest(),
            "gate_a_bundle": verify_gate_a_bundle_unchanged(),
            "adapter": verify_adapter_default(),
            "target_contacts": 0,
            "frequency_writes": 0,
            "smoke_executions": 0,
            "authority_artifact_created": False,
            "preparation_qualification_authorized": False,
            "third_smoke_authorized": False,
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.SubprocessError, VerifyError, builder.BuildError, target_bundle.BundleError, authority.AuthorityError) as exc:
        print(f"GATE_A_FREQUENCY_PREPARATION_AUTHORITY_TRANSPORT_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
