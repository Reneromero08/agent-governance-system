"""Phase 1 testbench for deterministic resolver.

Validates:
  - resolver output conforms to resolution_result.schema.json (draft-07)
  - golden expectations for several fact sets (must_select/must_not_select/must_emit)

Usage:
  python COMMONSENSE/TESTBENCH/test_resolver.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from jsonschema import Draft7Validator


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _contains_all(hay: List[Any], needles: List[Any]) -> bool:
    for n in needles:
        if n not in hay:
            return False
    return True


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent

    schema_path = root / "SCHEMAS" / "resolution_result.schema.json"
    db_path = root / "db.example.json"
    valid_dir = root / "FIXTURES" / "phase1" / "valid"

    assert schema_path.exists(), f"Missing schema: {schema_path}"
    assert db_path.exists(), f"Missing example DB: {db_path}"
    assert valid_dir.exists(), f"Missing fixtures dir: {valid_dir}"

    schema = _load_json(schema_path)
    Draft7Validator.check_schema(schema)
    validator = Draft7Validator(schema)

    # Import resolver from repo-relative location
    import importlib.util
    spec = importlib.util.spec_from_file_location("resolver", str(root / "resolver.py"))
    assert spec and spec.loader
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)  # type: ignore

    db = _load_json(db_path)
    print("OK: loaded example DB")

    for i in range(1, 6):
        facts = _load_json(valid_dir / f"facts_{i:02d}.json")["facts"]
        expected = _load_json(valid_dir / f"expected_{i:02d}.json")

        out = resolver.resolve(db, facts, with_trace=False)

        errors = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
        if errors:
            print(f"FAIL(schema): facts_{i:02d}.json")
            for e in errors[:5]:
                print(f"  - {list(e.path)}: {e.message}")
            return 2

        sel = out["selected_ids"]
        em = out["emits"]

        if "must_select" in expected:
            if not _contains_all(sel, expected["must_select"]):
                print(f"FAIL(expect must_select): {i}")
                print("selected_ids:", sel)
                print("expected:", expected["must_select"])
                return 3

        if "must_not_select" in expected:
            if any(x in sel for x in expected["must_not_select"]):
                print(f"FAIL(expect must_not_select): {i}")
                print("selected_ids:", sel)
                print("must_not_select:", expected["must_not_select"])
                return 4

        if "must_emit" in expected:
            if not _contains_all(em, expected["must_emit"]):
                print(f"FAIL(expect must_emit): {i}")
                print("emits:", em)
                print("expected:", expected["must_emit"])
                return 5

    print("OK: resolver outputs schema-valid + expectations pass (5)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
