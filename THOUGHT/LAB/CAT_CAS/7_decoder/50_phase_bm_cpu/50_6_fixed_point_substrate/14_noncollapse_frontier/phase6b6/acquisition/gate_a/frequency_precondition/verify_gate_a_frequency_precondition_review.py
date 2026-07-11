#!/usr/bin/env python3
"""Verify the Gate A frequency-precondition review remains read-only."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[10]
PROBE = HERE / "gate_a_frequency_precondition_probe.py"
README = HERE / "README.md"
MERGED_GATE_A_CLOSURE = "26576400cb10c3dfb2968f44cc7066f4b143463a"
ADAPTER = HERE.parent / "adapter"

sys.path.insert(0, str(ADAPTER))
import build_gate_a_execution_bundle as bundle  # noqa: E402

EXPECTED_CONSTANTS = {
    "SCHEMA_ID": "CAT_CAS_PHASE6B6_GATE_A_FREQUENCY_PRECONDITION_OBSERVATION_V1",
    "OBSERVATION_MODE": "IDLE_READ_ONLY",
    "DEFAULT_CORES": (4, 5),
    "REQUIRED_FREQUENCY_KHZ": 1_600_000,
    "DEFAULT_SAMPLE_COUNT": 200,
    "DEFAULT_INTERVAL_MS": 10,
}
FORBIDDEN_IMPORT_ROOTS = {
    "asyncio",
    "http",
    "paramiko",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "tempfile",
    "urllib",
}
FORBIDDEN_METHODS = {
    "chmod",
    "chown",
    "mkdir",
    "rename",
    "replace",
    "rmdir",
    "touch",
    "unlink",
    "write_bytes",
    "write_text",
}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile"}


class VerifyError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise VerifyError(message)


def run(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(args),
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=check,
    )


def rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def load_probe_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "gate_a_frequency_precondition_probe",
        PROBE,
    )
    require(spec is not None and spec.loader is not None, "probe module spec missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_ast() -> dict[str, Any]:
    source = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROBE))
    imported: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                require(
                    node.func.id not in FORBIDDEN_NAMES,
                    f"forbidden call in read-only probe: {node.func.id}",
                )
            elif isinstance(node.func, ast.Attribute):
                require(
                    node.func.attr not in FORBIDDEN_METHODS,
                    f"forbidden mutation method in read-only probe: {node.func.attr}",
                )

    forbidden_imports = sorted(imported & FORBIDDEN_IMPORT_ROOTS)
    require(not forbidden_imports, f"forbidden imports: {forbidden_imports}")
    require("os.open" not in source, "os.open is forbidden")
    require("/dev/cpu/" not in source, "MSR device path is forbidden")
    require("ssh" not in source.casefold(), "SSH surface is forbidden")
    require("scp" not in source.casefold(), "SCP surface is forbidden")
    return {
        "status": "READ_ONLY_AST_CLOSED",
        "imports": sorted(imported),
    }


def verify_constants() -> dict[str, Any]:
    module = load_probe_module()
    observed = {name: getattr(module, name) for name in EXPECTED_CONSTANTS}
    require(observed == EXPECTED_CONSTANTS, f"frozen constants changed: {observed}")
    return {
        "status": "FREQUENCY_PRECONDITION_CONSTANTS_EXACT",
        "constants": observed,
    }


def verify_documentation() -> dict[str, Any]:
    text = README.read_text(encoding="utf-8")
    required_phrases = (
        "READ_ONLY_QUALIFICATION_IMPLEMENTED__TARGET_OBSERVATION_NOT_AUTHORIZED",
        "third Gate A attempt authorized = false",
        "frequency preparation writes authorized = false",
        "must **already** equal",
        "PASS_STATIC_PRECONDITION_OBSERVED",
        "INCONCLUSIVE_DYNAMIC_PRECONDITION",
        "FAIL_REQUIRED_FREQUENCY_NOT_OBSERVED",
        "FAILED_CLOSED_UNOBSERVABLE",
    )
    for phrase in required_phrases:
        require(phrase in text, f"review documentation missing: {phrase}")
    return {"status": "REVIEW_BOUNDARY_DOCUMENTED"}


def verify_execution_bundle_unchanged() -> dict[str, Any]:
    require(
        run(
            "git",
            "merge-base",
            "--is-ancestor",
            MERGED_GATE_A_CLOSURE,
            "HEAD",
            check=False,
        ).returncode
        == 0,
        "merged Gate A closure is not an ancestor",
    )
    protected = tuple(
        sorted(
            {
                *(bundle.rel(source) for _package, source, _role in bundle.PACKAGE_FILES),
                bundle.rel(bundle.MANIFEST_PATH),
            }
        )
    )
    require(
        run(
            "git",
            "diff",
            "--quiet",
            MERGED_GATE_A_CLOSURE,
            "HEAD",
            "--",
            *protected,
            check=False,
        ).returncode
        == 0,
        "Gate A execution bundle changed",
    )
    manifest = json.loads(bundle.MANIFEST_PATH.read_text(encoding="utf-8"))
    exact = bundle.validate_committed_manifest_exact(manifest, "HEAD")
    require(
        exact["execution_bundle_sha256"]
        == "353f7e2d865508ebc018cb72648d3d3f227dc1c1128681fd9b4e99d81c9aa47f",
        "execution bundle digest changed",
    )
    return {
        "status": "GATE_A_EXECUTION_BUNDLE_UNCHANGED",
        "protected_path_count": len(protected),
        "execution_bundle_sha256": exact["execution_bundle_sha256"],
    }


def verify_authority_resting_state() -> dict[str, Any]:
    active = ADAPTER / "GATE_A_EXECUTION_AUTHORITY.json"
    consumed = sorted(ADAPTER.glob("GATE_A_EXECUTION_AUTHORITY_CONSUMED_*.json"))
    require(not active.exists(), "active Gate A authority exists")
    require(
        [path.name for path in consumed]
        == [
            "GATE_A_EXECUTION_AUTHORITY_CONSUMED_1dabfc7b.json",
            "GATE_A_EXECUTION_AUTHORITY_CONSUMED_7e1e8835.json",
        ],
        "consumed authority archive set changed",
    )
    return {
        "status": "NO_ACTIVE_AUTHORITY__TWO_CONSUMED_ARCHIVES",
        "archives": [path.name for path in consumed],
    }


def main() -> int:
    try:
        result = {
            "status": "GATE_A_FREQUENCY_PRECONDITION_REVIEW_EXACT",
            "ast": verify_ast(),
            "constants": verify_constants(),
            "documentation": verify_documentation(),
            "execution_bundle": verify_execution_bundle_unchanged(),
            "authority_state": verify_authority_resting_state(),
            "target_contacts": 0,
            "hardware_executions": 0,
            "frequency_writes": 0,
            "voltage_writes": 0,
            "msr_reads": 0,
            "msr_writes": 0,
            "third_attempt_authorized": False,
        }
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        VerifyError,
    ) as exc:
        print(f"GATE_A_FREQUENCY_PRECONDITION_REVIEW_INVALID: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
