#!/usr/bin/env python3
"""Verify the Gate A frequency preparation/restoration source boundary.

The integrated read-only observation and synthetic fixtures are the null
baseline. This verifier opens no network connection and performs no target or
frequency write.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[10]
MODULE = HERE / "gate_a_frequency_preparation.py"
README = HERE / "README.md"
OBSERVATION_MAIN = "8726b3d92e80fe4c3047f0b9708248d0da0c92ea"
ADAPTER = HERE.parent / "adapter"

sys.path.insert(0, str(ADAPTER))
import build_gate_a_execution_bundle as bundle  # noqa: E402

EXPECTED = {
    "CORES": (4, 5),
    "REQUIRED_DRIVER": "acpi-cpufreq",
    "REQUIRED_GOVERNOR": "schedutil",
    "REQUIRED_FREQUENCY_KHZ": 1_600_000,
    "EXPECTED_BASELINE_MIN_KHZ": 800_000,
    "EXPECTED_BASELINE_MAX_KHZ": 3_200_000,
    "MAX_WRITE_ATTEMPT_COUNT": 8,
    "POLICY_RELATIVE_PATHS": {
        4: "devices/system/cpu/cpufreq/policy4",
        5: "devices/system/cpu/cpufreq/policy5",
    },
    "WRITABLE_SUFFIXES": ("scaling_max_freq", "scaling_min_freq"),
}
FORBIDDEN_IMPORTS = {
    "asyncio",
    "http",
    "paramiko",
    "requests",
    "socket",
    "subprocess",
    "urllib",
}
FORBIDDEN_TEXT = ("/dev/cpu/", "rdmsr", "wrmsr")


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


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gate_a_frequency_preparation", MODULE
    )
    require(spec is not None and spec.loader is not None, "module spec missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_ast() -> dict[str, Any]:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(MODULE))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".", 1)[0])
    require(
        not (imports & FORBIDDEN_IMPORTS),
        f"forbidden imports: {sorted(imports & FORBIDDEN_IMPORTS)}",
    )
    lowered = source.casefold()
    require(
        "ssh" not in lowered and "scp" not in lowered,
        "network transport surface found",
    )
    for text in FORBIDDEN_TEXT:
        require(
            text.casefold() not in lowered,
            f"forbidden control surface found: {text}",
        )
    require(
        "LIVE_FREQUENCY_PREPARATION_NOT_AUTHORIZED" in source,
        "live CLI refusal missing",
    )
    require("allow_live_sysfs" not in source, "live sysfs bypass is forbidden")
    return {
        "status": "PREPARATION_AST_BOUNDARY_CLOSED",
        "imports": sorted(imports),
    }


def verify_constants() -> dict[str, Any]:
    module = load_module()
    observed = {name: getattr(module, name) for name in EXPECTED}
    require(observed == EXPECTED, f"preparation constants changed: {observed}")
    return {"status": "PREPARATION_CONSTANTS_EXACT", "constants": observed}


def verify_documentation() -> dict[str, Any]:
    text = README.read_text(encoding="utf-8")
    for phrase in (
        "SOURCE_IMPLEMENTED__LIVE_WRITES_NOT_AUTHORIZED",
        "live frequency preparation authorized = false",
        "third Gate A attempt authorized = false",
        "Restoration is attempted after every write-bearing failure",
        "policy4/scaling_max_freq",
        "policy5/scaling_min_freq",
    ):
        require(phrase in text, f"documentation missing phrase: {phrase}")
    return {"status": "PREPARATION_BOUNDARY_DOCUMENTED"}


def verify_execution_bundle_unchanged() -> dict[str, Any]:
    require(
        run(
            "git",
            "merge-base",
            "--is-ancestor",
            OBSERVATION_MAIN,
            "HEAD",
            check=False,
        ).returncode
        == 0,
        "observation main is not an ancestor",
    )
    protected = tuple(
        sorted(
            {
                *(
                    bundle.rel(source)
                    for _package, source, _role in bundle.PACKAGE_FILES
                ),
                bundle.rel(bundle.MANIFEST_PATH),
            }
        )
    )
    require(
        run(
            "git",
            "diff",
            "--quiet",
            OBSERVATION_MAIN,
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
        "execution_bundle_sha256": exact["execution_bundle_sha256"],
    }


def verify_authority_state() -> dict[str, Any]:
    active = ADAPTER / "GATE_A_EXECUTION_AUTHORITY.json"
    require(not active.exists(), "active Gate A authority exists")
    return {
        "status": "NO_ACTIVE_AUTHORITY",
        "live_frequency_preparation_authorized": False,
        "preparation_qualification_authorized": False,
        "third_gate_a_attempt_authorized": False,
        "gate_b_authorized": False,
    }


def main() -> int:
    try:
        result = {
            "status": "GATE_A_FREQUENCY_PREPARATION_RESTORATION_SOURCE_EXACT",
            "ast": verify_ast(),
            "constants": verify_constants(),
            "documentation": verify_documentation(),
            "execution_bundle": verify_execution_bundle_unchanged(),
            "authority_state": verify_authority_state(),
            "target_contacts": 0,
            "hardware_executions": 0,
        }
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
        VerifyError,
    ) as exc:
        print(
            f"GATE_A_FREQUENCY_PREPARATION_RESTORATION_INVALID: {exc}",
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
