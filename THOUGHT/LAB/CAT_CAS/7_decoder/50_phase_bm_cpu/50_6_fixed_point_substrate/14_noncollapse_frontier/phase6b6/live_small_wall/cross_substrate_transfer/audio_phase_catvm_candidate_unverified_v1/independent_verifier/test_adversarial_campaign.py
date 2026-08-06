import json
from collections import Counter
from pathlib import Path

from adversarial_campaign import run


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE.parent


def test_predeclared_campaign_has_no_undetected_failures() -> None:
    result = run(
        CANDIDATE / "source_snapshot",
        CANDIDATE / "MUTATION_PLAN.json",
    )
    assert result["seed"] == 104729
    assert result["ledger"]["attempted"] > 1000
    assert result["ledger"]["failed"] == 0
    assert result["results"]["A"]["coefficient_cases"] == 256
    assert result["results"]["B"]["renumbered_cases"] == 32
    assert result["results"]["C"]["programs"] == 64
    assert result["results"]["D"]["homogeneous_cases"] == 224
    assert all(
        challenge["outside_homogeneous_law"]
        for challenge in result["results"]["D"]["mixed_layer_challenges"]
    )


def test_static_audit_summary_matches_finding_ledger() -> None:
    audit = json.loads(
        (CANDIDATE / "STATIC_ANTI_SMUGGLE_AUDIT.json").read_text(
            encoding="utf-8"
        )
    )
    severities = Counter(
        finding["severity"].lower() for finding in audit["findings"]
    )
    for severity in ("critical", "high", "medium", "low", "info"):
        assert audit["summary"][severity] == severities[severity]
    assert len(audit["findings"]) == 17
    assert audit["accepted_path_invalidated"] == {
        "A": True,
        "B": False,
        "C": False,
        "D": False,
    }
