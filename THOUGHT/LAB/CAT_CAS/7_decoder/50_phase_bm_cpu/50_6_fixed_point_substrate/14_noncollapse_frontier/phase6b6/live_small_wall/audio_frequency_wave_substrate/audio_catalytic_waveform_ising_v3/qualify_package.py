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


PACKAGE_DIR = Path(__file__).resolve().parent
VERIFIERS = (
    "development_qualifier.py",
    "control_qualifier.py",
    "stress_qualifier.py",
    "freeze_builder.py",
    "preoracle_runner.py",
    "oracle_adjudicator.py",
    "independent_verifier_v2.py",
    "finalize_package.py",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_environment() -> dict[str, str]:
    freeze = json.loads(
        (PACKAGE_DIR / "V3_FREEZE.json").read_text(encoding="utf-8")
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
        compile(source, str(path), "exec")
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
            cwd=PACKAGE_DIR,
            env=environment,
            capture_output=True,
            check=False,
            text=True,
            timeout=120,
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


def main() -> int:
    execution_environment = verify_execution_environment()
    source_records = compile_sources()
    verifier_records = verify_fresh_processes()
    print(
        json.dumps(
            {
                "compiled_source_count": len(source_records),
                "decision": "CATALYTIC_WAVEFORM_ISING_V3_VERIFIED",
                "execution_environment": execution_environment,
                "fresh_process_verifier_count": len(verifier_records),
                "sources": source_records,
                "status": "PASS",
                "verifiers": verifier_records,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
