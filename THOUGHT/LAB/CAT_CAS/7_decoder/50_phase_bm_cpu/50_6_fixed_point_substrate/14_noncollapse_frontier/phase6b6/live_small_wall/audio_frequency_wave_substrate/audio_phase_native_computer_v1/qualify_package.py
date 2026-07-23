from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
VERIFIERS = (
    ("development_qualifier.py", "verify"),
    ("prospective_runner.py", "verify-contract"),
    ("prospective_runner.py", "verify-raw"),
    ("adjudicator.py", "verify"),
    ("resource_profiler.py", "verify"),
    ("independent_verifier.py", "verify"),
    ("finalize_package.py", "verify"),
)
DECISION = "PHASE_NATIVE_COMPUTER_REFERENCE_VERIFIED"


def repository_root() -> Path:
    for parent in (PACKAGE_DIR, *PACKAGE_DIR.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("cannot locate worktree root")


REPOSITORY_ROOT = repository_root()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compile_sources() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        source = path.read_text(encoding="utf-8")
        compile(source, str(path), "exec")
        records.append(
            {
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    return records


def run_verifiers() -> list[dict[str, Any]]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    records: list[dict[str, Any]] = []
    for source, command in VERIFIERS:
        completed = subprocess.run(
            [sys.executable, str(PACKAGE_DIR / source), command],
            cwd=REPOSITORY_ROOT,
            env=environment,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"{source} {command} failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        records.append(
            {
                "command": command,
                "source": source,
                "stdout": completed.stdout.strip(),
            }
        )
    return records


def assert_final_semantics() -> dict[str, Any]:
    development = json.loads(
        (PACKAGE_DIR / "DEVELOPMENT_RESULTS.json").read_text(encoding="utf-8")
    )
    raw = json.loads(
        (PACKAGE_DIR / "PROSPECTIVE_RAW_RESULTS.json").read_text(encoding="utf-8")
    )
    final = json.loads(
        (PACKAGE_DIR / "FINAL_RESULTS.json").read_text(encoding="utf-8")
    )
    independent = json.loads(
        (PACKAGE_DIR / "INDEPENDENT_VERIFICATION.json").read_text(encoding="utf-8")
    )
    resources = json.loads(
        (PACKAGE_DIR / "RESOURCE_RESULTS.json").read_text(encoding="utf-8")
    )
    review = (PACKAGE_DIR / "FOCUSED_REVIEW.md").read_text(encoding="utf-8")
    checks = {
        "development": development["all_passed"],
        "factorized": all(
            row["complete_configuration_modes"] == 0
            for row in resources["scaling"]
        ),
        "focused_review": "remaining findings: none" in review
        and "verdict: **PASS**" in review,
        "independent": independent["verdict"] == "PASS",
        "prospective_raw": raw["summary"]["all_boundaries_valid"]
        and raw["summary"]["all_restoration_passed"]
        and raw["summary"]["control_passed"],
        "result": final["decision"] == DECISION
        and final["exact_count"] == final["case_count"]
        and final["incorrect_count"] == 0,
    }
    if not all(checks.values()):
        raise RuntimeError(f"final semantic checks failed: {checks}")
    return checks


def assert_no_transient_files() -> None:
    transient = [
        path
        for path in PACKAGE_DIR.rglob("*")
        if path.is_dir() and path.name == "__pycache__"
        or path.is_file() and path.name.endswith((".pyc", ".tmp"))
    ]
    if transient:
        raise RuntimeError(
            "transient package files present: "
            + ", ".join(str(path) for path in transient)
        )


def main() -> int:
    compiled = compile_sources()
    semantics = assert_final_semantics()
    verifiers = run_verifiers()
    assert_no_transient_files()
    print(
        json.dumps(
            {
                "compiled_sources": len(compiled),
                "decision": DECISION,
                "semantics": semantics,
                "verifiers": len(verifiers),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
