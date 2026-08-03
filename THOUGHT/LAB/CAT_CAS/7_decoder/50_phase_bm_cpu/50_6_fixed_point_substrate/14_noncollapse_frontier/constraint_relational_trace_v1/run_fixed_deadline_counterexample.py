from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    package_parent = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(package_parent))
    from constraint_relational_trace_v1.odd_parity_fixed_deadline_counterexample import (
        audit_odd_parity_fixed_deadline_counterexample,
    )
else:
    from .odd_parity_fixed_deadline_counterexample import (
        audit_odd_parity_fixed_deadline_counterexample,
    )


RESULT_PATH = Path(__file__).resolve().parent / "results" / "fixed_deadline_counterexample.json"


def build_fixed_deadline_counterexample_record() -> dict[str, object]:
    audit = audit_odd_parity_fixed_deadline_counterexample()
    record = asdict(audit)
    record["schema"] = "CONSTRAINT_RELATIONAL_TRACE_FIXED_DEADLINE_COUNTEREXAMPLE_V1"
    record["claim_boundary"] = {
        "fixed_deadline_three": "FALSIFIED",
        "all_formula_uniform_polynomial_deadlines": "UNRESOLVED",
        "public_seed_completeness": "NOT_ESTABLISHED",
        "p_equals_np": "NOT_PROVEN",
    }
    return record


def main() -> int:
    record = build_fixed_deadline_counterexample_record()
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(record["status"])
    return 0 if record["fixed_deadline_three_completeness_falsified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
