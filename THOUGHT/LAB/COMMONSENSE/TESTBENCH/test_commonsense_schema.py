"""Phase 0 testbench for CommonSense/Logic entry schema.

Usage:
  python COMMONSENSE/TESTBENCH/test_commonsense_schema.py

Contract:
- Loads SCHEMAS/commonsense_entry.schema.json
- Valid fixtures must validate
- Invalid fixtures must fail (at least 1 error each)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from jsonschema import Draft7Validator


def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    schema_path = root / "SCHEMAS" / "commonsense_entry.schema.json"
    valid_dir = root / "FIXTURES" / "phase0" / "valid"
    invalid_dir = root / "FIXTURES" / "phase0" / "invalid"

    assert schema_path.exists(), f"Missing schema: {schema_path}"
    assert valid_dir.exists(), f"Missing fixtures dir: {valid_dir}"
    assert invalid_dir.exists(), f"Missing fixtures dir: {invalid_dir}"

    schema = _load_json(schema_path)
    validator = Draft7Validator(schema)

    print("OK: schema file exists")

    # Smoke: schema itself is valid draft-07
    Draft7Validator.check_schema(schema)
    print("OK: schema validates (Draft-07)")

    # Valid fixtures must pass
    valid_files = sorted(valid_dir.glob("*.json"))
    assert valid_files, "No valid fixtures found"
    for p in valid_files:
        doc = _load_json(p)
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if errors:
            print(f"FAIL(valid): {p.name}")
            for e in errors[:5]:
                print(f"  - {list(e.path)}: {e.message}")
            return 2
    print(f"OK: valid fixtures pass ({len(valid_files)})")

    # Invalid fixtures must fail
    invalid_files = sorted(invalid_dir.glob("*.json"))
    assert invalid_files, "No invalid fixtures found"
    for p in invalid_files:
        doc = _load_json(p)
        errors = sorted(validator.iter_errors(doc), key=lambda e: list(e.path))
        if not errors:
            print(f"FAIL(invalid): {p.name} unexpectedly validated")
            return 3
    print(f"OK: invalid fixtures fail as expected ({len(invalid_files)})")

    return 0


def test_commonsense_schema():
    """Pytest entry point."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
