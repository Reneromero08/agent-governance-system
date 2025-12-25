"""
Phase 0 schema testbench (CAT-DPT)

What it does:
- Loads the three Draft-07 JSON schemas.
- Ensures each schema itself is valid.
- Validates all fixtures under:
    CATALYTIC-DPT/FIXTURES/phase0/valid/*.json   (must PASS)
    CATALYTIC-DPT/FIXTURES/phase0/invalid/*.json (must FAIL)

Run (from repo root):
  python -m pip install jsonschema
  python CATALYTIC-DPT/TESTBENCH/test_schemas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from jsonschema import Draft7Validator
from referencing import Registry, Resource


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}")
    raise SystemExit(1)


def _ok(msg: str) -> None:
    print(f"OK: {msg}")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    schemas_dir = repo_root / "CATALYTIC-DPT" / "SCHEMAS"
    fixtures_valid = repo_root / "CATALYTIC-DPT" / "FIXTURES" / "phase0" / "valid"
    fixtures_invalid = repo_root / "CATALYTIC-DPT" / "FIXTURES" / "phase0" / "invalid"

    schema_files = {
        "jobspec": schemas_dir / "jobspec.schema.json",
        "validation_error": schemas_dir / "validation_error.schema.json",
        "ledger": schemas_dir / "ledger.schema.json",
        "proof": schemas_dir / "proof.schema.json",
    }

    for k, p in schema_files.items():
        if not p.exists():
            _fail(f"missing schema file: {k} -> {p}")
    _ok("schema files exist")

    jobspec_schema = _load_json(schema_files["jobspec"])
    validation_error_schema = _load_json(schema_files["validation_error"])
    ledger_schema = _load_json(schema_files["ledger"])
    proof_schema = _load_json(schema_files["proof"])

    for name, schema in [
        ("jobspec", jobspec_schema),
        ("validation_error", validation_error_schema),
        ("ledger", ledger_schema),
        ("proof", proof_schema),
    ]:
        try:
            Draft7Validator.check_schema(schema)
        except Exception as e:
            _fail(f"schema invalid: {name}: {e}")
    _ok("schemas validate (Draft-07)")

    v_jobspec = Draft7Validator(jobspec_schema)
    v_validation_error = Draft7Validator(validation_error_schema)
    v_proof = Draft7Validator(proof_schema)

    registry = Registry().with_resources(
        [
            ("jobspec.schema.json", Resource.from_contents(jobspec_schema)),
            ("validation_error.schema.json", Resource.from_contents(validation_error_schema)),
            ("ledger.schema.json", Resource.from_contents(ledger_schema)),
            ("proof.schema.json", Resource.from_contents(proof_schema)),
        ]
    )
    v_ledger = Draft7Validator(ledger_schema, registry=registry)

    def choose_validator(doc: Dict[str, Any]) -> Draft7Validator:
        if "job_id" in doc and "task_type" in doc:
            return v_jobspec
        if "valid" in doc and "errors" in doc and "warnings" in doc:
            return v_validation_error
        if "RUN_INFO" in doc and "PRE_MANIFEST" in doc and "POST_MANIFEST" in doc:
            return v_ledger
        if "proof_version" in doc and "restoration_result" in doc:
            return v_proof
        _fail("fixture does not match any known schema (cannot choose validator)")
        raise AssertionError

    valid_files: List[Path] = sorted(fixtures_valid.glob("*.json"))
    if not valid_files:
        _fail(f"no valid fixtures found in {fixtures_valid}")
    for p in valid_files:
        doc = _load_json(p)
        errors = sorted(choose_validator(doc).iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            _fail(f"valid fixture FAILED: {p.name}: {errors[0].message}")
    _ok(f"valid fixtures pass ({len(valid_files)})")

    invalid_files: List[Path] = sorted(fixtures_invalid.glob("*.json"))
    if not invalid_files:
        _fail(f"no invalid fixtures found in {fixtures_invalid}")
    for p in invalid_files:
        doc = _load_json(p)
        errors = sorted(choose_validator(doc).iter_errors(doc), key=lambda e: list(e.path))
        if not errors:
            _fail(f"invalid fixture unexpectedly PASSED: {p.name}")
    _ok(f"invalid fixtures fail as expected ({len(invalid_files)})")

    return 0


def test_all_schemas():
    """Pytest entry point."""
    assert main() == 0


if __name__ == "__main__":
    sys.exit(main())
