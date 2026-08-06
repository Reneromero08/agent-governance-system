from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_REPORTS = [
    "SOURCE_RECEIPT.json",
    "TRANSFER_MANIFEST.json",
    "CLAIMS_UNDER_TEST.md",
    "INVESTMENT_GATE.json",
    "SOURCE_REPRODUCTION_REPORT.md",
    "STATIC_ANTI_SMUGGLE_AUDIT.json",
    "INDEPENDENT_RECONSTRUCTION_REPORT.md",
    "MUTATION_CAMPAIGN.json",
    "RESTORATION_REUSE_STRESS.json",
    "MACHINE_BOUNDARY_ATTACK_REPORT.md",
    "BASELINE_CHALLENGE_REPORT.md",
    "QUOTIENT_GENERALIZATION_REPORT.md",
    "TRANSFER_RELEVANCE_REPORT.md",
    "INDEPENDENT_REVIEW_LEDGER.json",
    "FINAL_PROVISIONAL_DECISION.md",
]


def read_json(name: str):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def test_required_reports_exist_and_are_nonempty():
    missing = [name for name in REQUIRED_REPORTS if not (ROOT / name).is_file()]
    assert missing == []
    empty = [name for name in REQUIRED_REPORTS if (ROOT / name).stat().st_size == 0]
    assert empty == []


def test_transaction_report_records_source_ordering_defect():
    report = read_json("RESTORATION_REUSE_STRESS.json")
    assert report["canonical"] is False
    assert report["small_wall_crossed"] is False
    assert (
        report["candidate_a"]["source_primary_transaction"][
            "final_boundary_before_restore"
        ]
        is True
    )
    assert (
        report["candidate_a"]["source_reuse_cycles"]["all_coefficients_match"]
        is True
    )
    assert report["branch_local_transfer_reference"]["all_cycles_match"] is True
    assert report["branch_local_transfer_reference"]["all_cycles_restored"] is True
    assert (
        report["branch_local_transfer_reference"]["negative_modes_fail_closed"]
        is True
    )


def test_final_decision_uses_required_classifications():
    text = (ROOT / "FINAL_PROVISIONAL_DECISION.md").read_text(encoding="utf-8")
    assert "`REJECTED_SOURCE_DEFECT`" in text
    assert "`REJECTED_FIXTURE_SPECIALIZATION`" in text
    assert "`INDEPENDENTLY_VERIFIED_TRANSFERABLE_OBSTRUCTION`" in text
    assert "`INDEPENDENTLY_VERIFIED_SOURCE_LOCAL`" in text
    assert "Small Wall not crossed" in text
    assert "Physical Family 10h evidence not modified" in text


def test_review_ledger_does_not_use_source_reviews_as_proof():
    ledger = read_json("INDEPENDENT_REVIEW_LEDGER.json")
    assert ledger["canonical"] is False
    assert ledger["small_wall_crossed"] is False
    assert ledger["source_review_json_used_as_proof"] is False
    assert len(ledger["reviewers"]) == 5
    assert all(item["decision"] for item in ledger["reviewers"])
