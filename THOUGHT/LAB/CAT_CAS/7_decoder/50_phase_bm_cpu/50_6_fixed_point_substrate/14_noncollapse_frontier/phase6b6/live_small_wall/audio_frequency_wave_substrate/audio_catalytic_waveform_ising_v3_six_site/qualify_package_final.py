from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


sys.dont_write_bytecode = True


PACKAGE_DIR = Path(__file__).resolve().parent
VERIFIERS = (
    "development_qualifier.py",
    "control_qualifier.py",
    "freeze_builder.py",
    "preoracle_runner.py",
    "oracle_adjudicator.py",
    "independent_verifier.py",
    "resource_accounting_final.py",
    "finalize_package.py",
)


def repository_root() -> Path:
    for candidate in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError("repository root not found")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_environment() -> dict[str, str]:
    freeze = json.loads(
        (PACKAGE_DIR / "SIX_SITE_FREEZE.json").read_text(encoding="utf-8")
    )
    expected = freeze["machine_contract"]["execution_environment"]
    observed = {
        "numpy_version": np.__version__,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }
    if observed != expected:
        raise RuntimeError(
            f"frozen execution environment required: {expected}; observed: {observed}"
        )
    return observed


def compile_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        source = path.read_bytes()
        compile(source, str(path), "exec", dont_inherit=True, optimize=0)
        records.append(
            {
                "bytes": len(source),
                "path": path.name,
                "sha256": hashlib.sha256(source).hexdigest(),
            }
        )
    return records


def verify_fresh_processes() -> list[dict[str, Any]]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    records: list[dict[str, Any]] = []
    for name in VERIFIERS:
        path = PACKAGE_DIR / name
        completed = subprocess.run(
            [sys.executable, str(path), "verify"],
            cwd=repository_root(),
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=180,
        )
        record = {
            "exit_code": completed.returncode,
            "path": name,
            "source_sha256": sha256_file(path),
            "stderr": completed.stderr.strip(),
            "stdout": completed.stdout.strip(),
        }
        records.append(record)
        if completed.returncode != 0:
            raise RuntimeError(json.dumps(record, sort_keys=True))
    return records


def assert_closed_package_tree() -> None:
    nested = sorted(
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file() and path.parent != PACKAGE_DIR
    )
    if nested:
        raise RuntimeError(
            "nested package inputs are forbidden: " + ", ".join(nested)
        )


def main() -> int:
    assert_closed_package_tree()
    environment = verify_execution_environment()
    sources = compile_sources()
    verifiers = verify_fresh_processes()
    print(
        json.dumps(
            {
                "compiled_source_count": len(sources),
                "decision": (
                    "CATALYTIC_WAVEFORM_ISING_V3_SIX_SITE_VERIFIED"
                ),
                "execution_environment": environment,
                "fresh_process_verifier_count": len(verifiers),
                "sources": sources,
                "status": "PASS",
                "verifiers": verifiers,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
