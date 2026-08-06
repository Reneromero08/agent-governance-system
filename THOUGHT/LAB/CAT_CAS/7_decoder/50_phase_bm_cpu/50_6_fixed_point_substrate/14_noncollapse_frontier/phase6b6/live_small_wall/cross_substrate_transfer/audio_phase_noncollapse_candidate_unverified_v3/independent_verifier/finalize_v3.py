#!/usr/bin/env python3
"""Fail-closed corrective V3 adjudication.

This finalizer intentionally supersedes the original V3 closure style.  It does
not contain a preset final classification table.  It loads the available
machine-readable evidence, checks the predicates that V3 actually established,
and downgrades claims whose independence was overstated.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_outputs"
SOURCE_HEAD = "7c79414f2beb34c29bf0d63783a6effea26c65ed"
MOVING_AUDIO_HEAD = "9165ef74d9750dec63f5ed2a3a1c69f156f23135"

ALLOWED_CORRECTIVE_CLASSIFICATIONS = {
    "I": {
        "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_QUOTIENT_OBSTRUCTION",
        "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST",
        "REJECTED_THEOREM_DEFECT",
    },
    "J": {
        "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_SOFTWARE_BISIMULATION_OBSTRUCTION",
        "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST",
        "REJECTED_BISIMULATION_DEFECT",
    },
    "K": {
        "SOURCE_REPRODUCED_SOURCE_LOCAL_MULTI_PORT_TT_OBSTRUCTION",
        "REJECTED_RANK_OR_BASELINE_DEFECT",
    },
    "L": {
        "SOURCE_REPRODUCED_TRANSFERABLE_BOUNDARY_HEIGHT_OBSTRUCTION_CANDIDATE",
        "SOURCE_QUALIFIER_FAIL_OPEN",
        "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST",
        "REJECTED_LOWER_BOUND_DEFECT",
    },
    "M": {
        "SOURCE_WILCZEK_ZEE_PACKAGE_REPRODUCED_STRICT_SCOPE",
        "REJECTED_HOLONOMY_OR_RESTORATION_DEFECT",
    },
    "N": {
        "REJECTED_SOURCE_DEFECT",
        "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST",
    },
    "O": {
        "SOURCE_REPRODUCED_FAMILY_SCOPED_TRANSFER_CLOSURE",
        "REJECTED_SOURCE_DEFECT",
    },
}


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / ".git").exists():
            return path
    raise RuntimeError(f"no git root above {start}")


REPO = find_repo_root(ROOT)


def git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def json_sha256(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    )


def read_json(rel: str) -> Any:
    path = ROOT / rel
    if not path.exists():
        raise RuntimeError(f"required evidence missing: {rel}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"malformed JSON evidence {rel}: {exc}") from exc


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def source_reproduced(source: dict[str, Any], candidate: str) -> bool:
    return source["classifications"].get(candidate) == "SOURCE_REPRODUCED"


def derive_classifications(evidence: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    source = evidence["source"]
    ij = evidence["ij"]
    kl = evidence["kl"]
    m = evidence["m"]
    n = evidence["n"]
    o = evidence["o"]
    obstruction = evidence["obstruction"]
    mechanism = evidence["mechanism"]

    require(source["scientific_source_commit"] == SOURCE_HEAD, "source reproduction bound to unexpected source head")
    require(source_reproduced(source, "I"), "Candidate I source reproduction not closed")
    require(source_reproduced(source, "J"), "Candidate J source reproduction not closed")
    require(source_reproduced(source, "K"), "Candidate K source reproduction not closed")
    require(source_reproduced(source, "L"), "Candidate L source reproduction not closed")
    require(source_reproduced(source, "M"), "Candidate M source reproduction not closed")
    require(source_reproduced(source, "O"), "Candidate O source reproduction not closed")

    l_marker = source.get("corrected_primary_result_controls", {}).get(
        "L_corrective_marker_result_full_json", {}
    )
    require(l_marker.get("passed") is True, "Candidate L corrective marker stale control did not pass")

    require(
        obstruction.get("classification") == "BRANCH_LOCAL_OBSTRUCTION_CHECKLIST_ESTABLISHED"
        and obstruction.get("executable_harness_complete") is False,
        "branch-local obstruction output is not the corrected checklist status",
    )
    require(
        mechanism.get("classification")
        == "BRANCH_LOCAL_EXACT_NONCOMMUTING_REVERSIBLE_FRAME_TRANSACTION_ESTABLISHED"
        and mechanism.get("full_two_port_machine_law_established") is False,
        "branch-local mechanism output is not the corrected narrow toy status",
    )

    classifications: dict[str, str] = {}
    reasons: dict[str, str] = {}

    i = ij["candidate_i"]
    if i.get("theorem_survives_declared_scope") is True:
        classifications["I"] = "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_QUOTIENT_OBSTRUCTION"
        reasons["I"] = "independent exact HT,e0 infinite-projective-orbit theorem survived; only fixed finite exact lossless quotients are rejected"
    else:
        classifications["I"] = "REJECTED_THEOREM_DEFECT"
        reasons["I"] = "theorem predicate failed"

    j = ij["candidate_j"]
    if j.get("obstruction_survives") is True:
        classifications["J"] = "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_SOFTWARE_BISIMULATION_OBSTRUCTION"
        reasons["J"] = "independent Q3 symbolic implementation exercised canonical, legal-placement, permutation, and inverse controls"
    else:
        classifications["J"] = "REJECTED_BISIMULATION_DEFECT"
        reasons["J"] = "bisimulation predicate failed"

    k = kl["candidate_k"]
    if k.get("strict_bounded_diagnostic_survives") is True:
        classifications["K"] = "SOURCE_REPRODUCED_SOURCE_LOCAL_MULTI_PORT_TT_OBSTRUCTION"
        reasons["K"] = "source/oracle rank pattern and storage arithmetic are consistent, but V3 did not independently reconstruct tensors/SVD/rank controls"
    else:
        classifications["K"] = "REJECTED_RANK_OR_BASELINE_DEFECT"
        reasons["K"] = "bounded TT diagnostic predicate failed"

    l = kl["candidate_l"]
    if l.get("strict_scope_survives") is True:
        classifications["L"] = "SOURCE_REPRODUCED_TRANSFERABLE_BOUNDARY_HEIGHT_OBSTRUCTION_CANDIDATE"
        reasons["L"] = "source package and marker stale control reproduced; V3 did not independently reconstruct recurrence certificate or cycle data"
    else:
        classifications["L"] = "REJECTED_LOWER_BOUND_DEFECT"
        reasons["L"] = "height/lower-bound scope predicate failed"

    if m.get("mechanism_survives") is True:
        classifications["M"] = "SOURCE_WILCZEK_ZEE_PACKAGE_REPRODUCED_STRICT_SCOPE"
        reasons["M"] = "source WZ package reproduced at matrix level; branch-local noncommuting toy is separate and does not verify WZ geometry"
    else:
        classifications["M"] = "REJECTED_HOLONOMY_OR_RESTORATION_DEFECT"
        reasons["M"] = "source WZ matrix/restoration predicate failed"

    n_failures = set(n.get("fail_open_cases", []))
    required_n_failures = {
        "oversize_initialize_plus_one",
        "oversize_initialize_plus_16",
        "concatenated_initialize_stop",
        "cross_record_splice_31_plus_1",
        "rejected_initialize_mutates_nonce_state",
    }
    if n.get("classification") == "REJECTED_SOURCE_DEFECT" and required_n_failures.issubset(n_failures):
        classifications["N"] = "REJECTED_SOURCE_DEFECT"
        reasons["N"] = "oversized, concatenated, cross-record-spliced packets accepted and rejected initialize mutates nonce state"
    else:
        classifications["N"] = "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST"
        reasons["N"] = "corrective N attack set incomplete"

    if o.get("exact_family_closure_survives") is True:
        classifications["O"] = "SOURCE_REPRODUCED_FAMILY_SCOPED_TRANSFER_CLOSURE"
        reasons["O"] = "source/oracle exact family closure reproduced, but V3 did not independently implement the exact transfer DP and two-message baseline dominates"
    else:
        classifications["O"] = "REJECTED_SOURCE_DEFECT"
        reasons["O"] = "family closure predicate failed"

    for candidate, classification in classifications.items():
        require(
            classification in ALLOWED_CORRECTIVE_CLASSIFICATIONS[candidate],
            f"candidate {candidate} produced disallowed classification {classification}",
        )
    require(set(classifications) == set("IJKLMNO"), "not all V3 candidates classified")
    return classifications, reasons


def review_ledger() -> dict[str, Any]:
    required = [
        "reviews/reviewer_i.json",
        "reviews/reviewer_j.json",
        "reviews/reviewer_k.json",
        "reviews/reviewer_l.json",
        "reviews/reviewer_m.json",
        "reviews/reviewer_n1.json",
        "reviews/reviewer_n2.json",
        "reviews/reviewer_n3.json",
        "reviews/reviewer_o.json",
        "reviews/reviewer_r.json",
        "reviews/reviewer_t.json",
    ]
    return {
        "schema_version": "audio_noncollapse_v3_corrective_review_ledger",
        "canonical": False,
        "small_wall_crossed": False,
        "original_static_ledger_retracted": True,
        "original_static_ledger_counted_as_evidence": False,
        "file_backed_independent_review_artifacts_available": False,
        "decision": "REVIEW_LEDGER_NOT_COUNTED_AS_INDEPENDENT_EVIDENCE",
        "missing_required_review_artifacts": required,
        "repair_status": "no synthetic reviewer decisions generated; future review pass must write separate reviewer artifacts before consolidation",
    }


def render_decision(
    branch: str, head: str, classifications: dict[str, str], reasons: dict[str, str]
) -> str:
    rows = "\n".join(
        f"| {candidate} | `{classifications[candidate]}` | {reasons[candidate]} |"
        for candidate in "IJKLMNO"
    )
    return f"""# Final Provisional Decision V3 — Corrective Audit

Frozen scientific source: `{SOURCE_HEAD}`
Moving audio branch observed: `{MOVING_AUDIO_HEAD}`
Corrective branch state when generated: `{branch}` @ `{head}`

This file supersedes the original V3 closure wording. It preserves the evidence, but narrows claims whose independence, review separation, transfer completion, or fail-closed adjudication was overstated.

None of these decisions is canonical. No Small Wall claim is promoted. No physical Family 10h evidence is modified.

| Candidate | Corrective provisional classification | Corrective reason |
|---|---|---|
{rows}

Branch-level corrective verdict:

- V3 is valuable but is not accepted as a completed fail-closed independent-verification closure.
- I and J remain strong independently verified obstruction results.
- N remains rejected, now with stronger framing/state evidence.
- K and O are conservative source-local/family-scoped warnings, not independently reimplemented scientific closures.
- L remains a strong transferable obstruction candidate, but V3 did not independently reconstruct the recurrence certificate.
- M is split: source WZ package reproduced in strict scope; branch-local noncommuting toy transaction demonstrated separately.
- The branch-local obstruction output is a checklist, not an executable harness.
- The branch-local mechanism output is a toy exact-rational reversible-frame transaction, not a full two-port CATVM machine law.

Small Wall position:

Unchanged. The strongest compact baselines remain identical or stronger for every positive semantic mechanism.
"""


def render_family_report() -> str:
    return """# Family 10h Relevance Report

Status: no physical Family 10h evidence was modified; no Small Wall position changed.

Unmodified Family 10h diagnostic relevance:

- I/J transfer as negative review gates against fixed finite exact quotients and finite software systems masquerading as new resources.
- L is a strong exact-boundary-height obstruction candidate, but it requires an independent recurrence/cycle reconstruction before being counted as fully independently verified.
- K/O warn against accepting expanded or factorized representations without the strongest compact baseline.
- M supplies a counterfactual noncommuting hidden-frame design pattern only after narrowing: the branch-local toy is not a Wilczek-Zee geometry proof and present Family 10h observations do not expose enough control to infer such a shared frame.
- N does not transfer: malformed seqpacket framing and rejected-initialize state mutation are decisive protocol defects.

Counterfactual Family 10h twin relevance:

A useful twin would still need a typed hidden carrier, exact owner/type/generation/lease custody at every consumer, public descriptors, noncommuting updates, reverse rematerialization, verified restoration before final-only response, packet/disconnect/replay controls, and a complete resource ledger. V3's branch-local mechanism demonstrates only an exact noncommuting reversible-frame toy transaction; it does not establish full two-port custody.

Investment justified next:

Continue investing in obstruction gates and counterfactual carrier laws. Do not invest in N until packet framing, rejected-state mutation, and path-depth defects are repaired and rerun. Do not treat any audio arithmetic as physical CPU evidence.
"""


def render_corrective_audit(classifications: dict[str, str]) -> str:
    return f"""# Corrective Audit Report V3

Status: corrective adjudication applied.

Corrected candidate classifications:

```json
{json.dumps(classifications, indent=2, sort_keys=True)}
```

Corrections applied:

1. Replaced constant final classifications with evidence-loaded fail-closed adjudication.
2. Retracted the synthetic independent-review ledger as evidentiary support.
3. Added Candidate L marker stale-control proof with pre/post hashes and marker disappearance.
4. Added Candidate N cross-record splicing and rejected-initialize nonce-mutation attacks.
5. Downgraded K/L/M/O labels where V3 did not independently rebuild the claimed scientific object.
6. Reclassified the branch-local obstruction output as a checklist, not an executable harness.
7. Narrowed the branch-local mechanism output to an exact-rational noncommuting reversible-frame toy transaction.
8. Added a hash-bound corrective closure manifest for reports, scripts, snapshots, logs, and raw outputs.

Remaining scientific gaps:

- K still needs independent tensor construction, high-precision SVD or exact/modular rank, alternate orderings/matricizations, tolerance sweep, and symmetry quotient search.
- L still needs independent public-operator compilation, characteristic/valuation derivation, normalized recurrence reconstruction, and distinct cycle detection.
- M still needs independent CP2/dark-frame/transport/gauge reconstruction to support Wilczek-Zee-specific transfer.
- O still needs an independent exact transfer dynamic program, streaming baseline implementation, perturbation controls, and block-power baseline.
- The obstruction checklist still needs executable candidate-input gates if it is to become a harness.
- The mechanism toy still needs full tuple validation at every consumer, protocol framing, disconnect cleanup, stage-cut enforcement, nonce/replay controls, and hidden-state nonserialization before it can support a two-port machine-law claim.
"""


def inventory_files() -> tuple[list[dict[str, Any]], str]:
    excluded = {
        "CORRECTIVE_CLOSURE_MANIFEST_V3.json",
        "VERIFICATION_CLOSURE_V3.json",
    }
    records: list[dict[str, Any]] = []
    for path in sorted(p for p in ROOT.rglob("*") if p.is_file()):
        rel = str(path.relative_to(ROOT))
        if rel in excluded:
            continue
        if rel.startswith(".git/"):
            continue
        records.append(
            {
                "path": rel,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    aggregate = json_sha256(records)
    return records, aggregate


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    evidence = {
        "source": read_json("SOURCE_REPRODUCTION_DATA_V3.json"),
        "ij": read_json("raw_outputs/independent_ij_v3/ij_independent_data.json"),
        "kl": read_json("raw_outputs/independent_kl_v3/kl_independent_data.json"),
        "m": read_json("raw_outputs/independent_m_v3/m_independent_data.json"),
        "n": read_json("TWO_PORT_CATVM_RUNTIME_ATTACK_REPORT.json"),
        "o": read_json("raw_outputs/independent_o_v3/o_independent_data.json"),
        "obstruction": read_json("raw_outputs/branch_local_transfer_v3/branch_local_obstruction_harness.json"),
        "mechanism": read_json("raw_outputs/branch_local_transfer_v3/branch_local_mechanism_transfer.json"),
    }
    classifications, reasons = derive_classifications(evidence)
    branch = git(["branch", "--show-current"])
    head = git(["rev-parse", "HEAD"])
    status = git(["status", "--short"])
    stash = git(["stash", "list"])

    write_json(ROOT / "INDEPENDENT_REVIEW_LEDGER_V3.json", review_ledger())
    (ROOT / "FINAL_PROVISIONAL_DECISION_V3.md").write_text(
        render_decision(branch, head, classifications, reasons), encoding="utf-8"
    )
    (ROOT / "FAMILY10H_RELEVANCE_REPORT.md").write_text(
        render_family_report(), encoding="utf-8"
    )
    (ROOT / "CORRECTIVE_AUDIT_REPORT_V3.md").write_text(
        render_corrective_audit(classifications), encoding="utf-8"
    )

    inventory, aggregate = inventory_files()
    manifest = {
        "schema_version": "audio_noncollapse_v3_corrective_closure_manifest",
        "canonical": False,
        "small_wall_crossed": False,
        "scientific_source_head": SOURCE_HEAD,
        "moving_audio_branch_head_recorded": MOVING_AUDIO_HEAD,
        "branch": branch,
        "generated_at_head": head,
        "inventory_count": len(inventory),
        "aggregate_sha256": aggregate,
        "inventory": inventory,
    }
    write_json(ROOT / "CORRECTIVE_CLOSURE_MANIFEST_V3.json", manifest)
    manifest_sha = file_sha256(ROOT / "CORRECTIVE_CLOSURE_MANIFEST_V3.json")

    closure = {
        "schema_version": "audio_noncollapse_v3_corrective_closure",
        "canonical": False,
        "small_wall_crossed": False,
        "scientific_source_head": SOURCE_HEAD,
        "moving_audio_branch_head_recorded": MOVING_AUDIO_HEAD,
        "branch": branch,
        "closure_generated_at_head": head,
        "evidence_root": str(ROOT.relative_to(REPO)),
        "classifications": classifications,
        "classification_reasons": reasons,
        "fail_closed_adjudication": True,
        "original_v3_completed_closure_accepted": False,
        "review_ledger_counted_as_independent_evidence": False,
        "source_reproduction": evidence["source"]["classifications"],
        "corrective_controls": {
            "candidate_l_marker_stale_control_passed": evidence["source"][
                "corrected_primary_result_controls"
            ]["L_corrective_marker_result_full_json"]["passed"],
            "candidate_n_fail_open_cases": evidence["n"]["fail_open_cases"],
            "obstruction_output_classification": evidence["obstruction"]["classification"],
            "mechanism_output_classification": evidence["mechanism"]["classification"],
        },
        "transfer_results": {
            "obstruction_harness_complete": False,
            "obstruction_checklist_established": True,
            "mechanism_transfer_complete": False,
            "noncommuting_reversible_frame_toy_established": True,
            "full_two_port_machine_law_established": False,
            "physical_family10h_claim_changed": False,
            "counterfactual_twin_component_identified": True,
        },
        "corrective_closure_manifest": "CORRECTIVE_CLOSURE_MANIFEST_V3.json",
        "corrective_closure_manifest_sha256": manifest_sha,
        "worktree_status_when_generated": status,
        "stash_state_when_generated": stash,
    }
    write_json(ROOT / "VERIFICATION_CLOSURE_V3.json", closure)
    print(json.dumps({"head": head, "classifications": classifications}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
