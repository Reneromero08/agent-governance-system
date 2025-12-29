"""Phase 2 testbench: symbolic handles -> predicate expansion -> deterministic resolve.

Requires:
  - jsonschema
  - COMMONSENSE/db.example.json from Phase 1 package (or your real DB)

Usage:
  python COMMONSENSE/TESTBENCH/test_symbols.py
"""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
from typing import Any, List


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

    # Load resolver module
    spec = importlib.util.spec_from_file_location("resolver", str(root / "resolver.py"))
    assert spec and spec.loader
    resolver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(resolver)  # type: ignore

    # Load example DB (you can swap to your real DB)
    db_path = root / "db.example.json"
    if not db_path.exists():
        print("FAIL: missing db.example.json (copy from phase1 package or point to your DB)")
        return 2
    db = _load_json(db_path)

    codebook = _load_json(root / "CODEBOOK.json")
    fx = root / "FIXTURES" / "phase2" / "valid"

    for i in range(1, 4):
        facts = _load_json(fx / f"facts_sym_{i:02d}.json")["facts"]
        expected = _load_json(fx / f"expected_sym_{i:02d}.json")
        out = resolver.resolve(db, facts, codebook=codebook, with_trace=False)

        if i in (1, 2):
            if "must_select" in expected and not _contains_all(out["selected_ids"], expected["must_select"]):
                print(f"FAIL(must_select): {i}")
                print(out["selected_ids"])
                return 3
            if "must_emit" in expected and not _contains_all(out["emits"], expected["must_emit"]):
                print(f"FAIL(must_emit): {i}")
                print(out["emits"])
                return 4
        else:
            # expansion-only assertions
            if "must_have_expanded" in expected and not _contains_all(out.get("expanded_facts", []), expected["must_have_expanded"]):
                print("FAIL(expanded_facts)")
                print(out.get("expanded_facts"))
                return 5
            if out.get("unresolved_symbols") != expected.get("must_have_unresolved", []):
                print("FAIL(unresolved_symbols)")
                print(out.get("unresolved_symbols"))
                return 6

    print("OK: symbolic expansion + resolve expectations pass (3)")
    return 0


def test_symbols():
    """Pytest entry point."""
    assert main() == 0


if __name__ == "__main__":
    raise SystemExit(main())
