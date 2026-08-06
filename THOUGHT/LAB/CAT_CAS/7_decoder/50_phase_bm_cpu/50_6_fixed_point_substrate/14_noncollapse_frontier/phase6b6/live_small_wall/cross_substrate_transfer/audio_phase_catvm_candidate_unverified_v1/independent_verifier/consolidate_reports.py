#!/usr/bin/env python3
"""Write final branch-local reports for the unverified audio CATVM transfer."""

from __future__ import annotations

import collections
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def file_sha256(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def git_text(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def path_record(path: str) -> dict[str, Any]:
    p = ROOT / path
    return {"path": path, "sha256": file_sha256(p), "bytes": p.stat().st_size}


def severity_counts(findings: list[dict[str, Any]]) -> dict[str, int]:
    return dict(collections.Counter(finding["severity"] for finding in findings))


def render_baseline_report(independent: dict[str, Any], mutation: dict[str, Any]) -> str:
    c = independent["candidate_c"]
    c_mut = mutation["results"]["C"]
    lines = [
        "# Baseline Challenge Report",
        "",
        "Status: branch-local obstruction evidence only. Not canonical. Small Wall not crossed.",
        "",
        "## Method",
        "",
        "The fixed-schema QANF obstruction was reconstructed from the public schema with a symbolic GF(2) polynomial model and a separate exhaustive public-program table. The source-selected baseline was challenged by searching for a compact direct formula over the declared public schema.",
        "",
        "## Independent Result",
        "",
        f"- Public programs checked: {c['public_programs']}",
        f"- Unique final boundaries: {c['unique_boundaries']}",
        f"- Compact direct formula count: {c['direct_formula_and_count']} AND operations",
        f"- Raw five-bit table size: {c['raw_five_bit_table_bits']} bits",
        f"- Packed nonconstant table size: {c['packed_nonconstant_output_table_bits']} bits",
        f"- Boundary table hash: `{c['table_sha256']}`",
        "",
        "## Mutation Cross-Check",
        "",
        f"- Mutation boundary map hash: `{c_mut['boundary_map_sha256']}`",
        f"- Equivalent public-program collisions: {c_mut['equivalent_program_collision_sizes']}",
        f"- Maximum public programs per final boundary: {c_mut['maximum_programs_per_boundary']}",
        "",
        "## Decision",
        "",
        "The obstruction survives the branch-local challenge as a negative law: the in-place carrier can satisfy transaction semantics, but a compact public-schema baseline still dominates for this fixed schema. This prevents false promotion of the source CATVM result and is transferable as an obstruction-testing method, not as a positive non-collapse claim.",
        "",
    ]
    return "\n".join(lines)


def render_quotient_report(independent: dict[str, Any], mutation: dict[str, Any]) -> str:
    d = independent["candidate_d"]
    d_mut = mutation["results"]["D"]
    mixed = d_mut["mixed_layer_challenges"]
    lines = [
        "# Quotient Generalization Report",
        "",
        "Status: independently reproduced restricted-family evidence. Not canonical. Small Wall not crossed.",
        "",
        "## Reconstructed Law",
        "",
        "The quotient was reconstructed as future-continuation equivalence over bounded Boolean truth-table states. Two unresolved states are grouped only when every permitted public continuation gives the same boundary behavior.",
        "",
        "## Homogeneous Family",
        "",
        f"- Formula cases checked: {d['cases_checked']}",
        f"- All formula cases matched: {d['all_formula_cases_match']}",
        f"- AND/OR duality checked: {d['and_or_duality']}",
        f"- Deliberate overmerge detected: {d['deliberate_overmerge_detected']}",
        f"- Deliberate undermerge detected: {d['deliberate_undermerge_detected']}",
        f"- Case hash: `{d['case_sha256']}`",
        "",
        "## Generalization Challenge",
        "",
    ]
    for item in mixed:
        lines.append(
            f"- {item['name']}: ranks {item['ranks']} with homogeneous max {item['homogeneous_max_rank']}"
        )
    lines.extend(
        [
            "",
            "The mixed and nonperiodic layer challenges exceed the homogeneous rank law. The supported statement is `FAMILY_SCOPED_QUOTIENT`, not a general continuation-equivalence mechanism.",
            "",
            "Decision: keep the quotient as source-local restricted-family evidence and as a design prompt for future quotient tests. Do not treat it as a transfer-verified general compression law.",
            "",
        ]
    )
    return "\n".join(lines)


def render_transfer_report(
    restoration: dict[str, Any],
    independent: dict[str, Any],
    static: dict[str, Any],
) -> str:
    transfer = restoration["branch_local_transfer_reference"]
    counts = severity_counts(static["findings"])
    lines = [
        "# Transfer Relevance Report",
        "",
        "Status: bounded branch-local transfer experiment only. Not canonical. Physical Family 10h evidence not modified.",
        "",
        "## Surviving Mechanisms",
        "",
        "- Hidden intermediate custody remains relevant as a machine-boundary pattern.",
        "- Atomic final response after restoration is required; the source Candidate A protocol failed this ordering test.",
        "- In-place restoration and same-carrier reuse are independently reconstructible in a repaired reference wrapper.",
        "- Reversible scheduling receipts are useful only after removing fixed topology assumptions.",
        "- Matched compact-baseline challenge is strongly relevant because Candidate C prevents false promotion.",
        "- Future-continuation quotienting remains interesting only inside the verified restricted family.",
        "",
        "## Bounded Transfer Experiment",
        "",
        f"- Reference cycles: {transfer['cycles']}",
        f"- All final boundaries matched independent enumeration: {transfer['all_cycles_match']}",
        f"- All accepted cycles restored state before response: {transfer['all_cycles_restored']}",
        f"- Hidden projection denied by reference wrapper: {not transfer['denied_hidden_projection']['ok']}",
        f"- Negative restoration variants failed closed: {transfer['negative_modes_fail_closed']}",
        f"- Record hash: `{transfer['records_sha256']}`",
        "",
        "## Limits",
        "",
        f"- Static finding severity counts: {counts}",
        f"- Candidate A accepted path invalidated by source protocol: {static['accepted_path_invalidated']['A']}",
        f"- Candidate B accepted path invalidated by static ledger: {static['accepted_path_invalidated']['B']}",
        "",
        "Decision: continue investment in the abstract machine-boundary law and the obstruction harness, not in the source Candidate A protocol or the fixed Candidate B scheduler as transferred implementations.",
        "",
    ]
    return "\n".join(lines)


def review_ledger(
    source_receipt: dict[str, Any],
    independent: dict[str, Any],
    mutation: dict[str, Any],
    static: dict[str, Any],
    restoration: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        name: path_record(name)
        for name in [
            "SOURCE_RECEIPT.json",
            "TRANSFER_MANIFEST.json",
            "SOURCE_REPRODUCTION_REPORT.md",
            "STATIC_ANTI_SMUGGLE_AUDIT.json",
            "INDEPENDENT_RECONSTRUCTION_REPORT.md",
            "MUTATION_PLAN.json",
            "MUTATION_CAMPAIGN.json",
            "RESTORATION_REUSE_STRESS.json",
            "MACHINE_BOUNDARY_ATTACK_REPORT.md",
            "BASELINE_CHALLENGE_REPORT.md",
            "QUOTIENT_GENERALIZATION_REPORT.md",
            "TRANSFER_RELEVANCE_REPORT.md",
            "LOCAL_VERIFICATION_COMMANDS.json",
            "FINAL_PROVISIONAL_DECISION.md",
        ]
    }
    reviewers = [
        {
            "role": "Reviewer A: source and qualifier audit",
            "assumptions": [
                "Frozen source SHA is an untrusted input.",
                "A source result JSON is not proof.",
            ],
            "methods": [
                "Fresh detached source checkout",
                "Exact qualifier runs",
                "Generated artifact deletion and corruption controls",
                "Static source and script ledger",
            ],
            "evidence_inspected": [
                evidence["SOURCE_RECEIPT.json"],
                evidence["SOURCE_REPRODUCTION_REPORT.md"],
                evidence["STATIC_ANTI_SMUGGLE_AUDIT.json"],
            ],
            "attacks_attempted": [
                "Exact source qualifier execution",
                "Deleted required result artifact",
                "Corrupted required result artifact",
                "Accepted-path source scan",
            ],
            "findings": [
                "A and B did not reproduce exactly as declared.",
                "A source protocol returns final boundary before restoration.",
                "B scheduler remains fixed to the copied public graph.",
            ],
            "unresolved_risks": [
                "A repaired source-side wrapper was not implemented in this branch.",
                "B needs a generic public-topology scheduler test.",
            ],
            "decision": "REJECT A SOURCE PROTOCOL; DO NOT TRANSFER B IMPLEMENTATION AS GENERAL",
        },
        {
            "role": "Reviewer B: clean-room mathematical reconstruction",
            "assumptions": [
                "Public formats are the only mathematical authority.",
                "Independent enumeration can be used for bounded public cases.",
            ],
            "methods": [
                "Scalar relation enumeration",
                "Independent DAG dependency analysis",
                "Symbolic GF(2) public-schema derivation",
                "Continuation-equivalence partitioning",
            ],
            "evidence_inspected": [
                evidence["INDEPENDENT_RECONSTRUCTION_REPORT.md"],
                evidence["MUTATION_CAMPAIGN.json"],
            ],
            "attacks_attempted": [
                "Wrong inverse controls",
                "Bad graph replay",
                "Compact baseline search",
                "Overmerge and undermerge quotient controls",
            ],
            "findings": [
                "A algebra and restoration controls reconstruct for bounded cases.",
                "B 15-node schedule reconstructs for the declared graph.",
                "C compact baseline dominates the fixed schema.",
                "D quotient reconstructs for homogeneous restricted families.",
            ],
            "unresolved_risks": [
                "D is not demonstrated for mixed/nonperiodic families.",
                "B reconstruction does not prove the source scheduler is general.",
            ],
            "decision": "CLEAN-ROOM SUPPORTS C; A/B/D ARE LIMITED BY TRANSFER SCOPE",
        },
        {
            "role": "Reviewer C: mutation and boundary-order campaign",
            "assumptions": [
                "Mutation families must be predeclared.",
                "A final response ordering failure invalidates the source machine-boundary claim.",
            ],
            "methods": [
                "Seeded mutation campaign",
                "Malformed input controls",
                "Source service packet probes",
                "Branch-local repaired reference transaction",
            ],
            "evidence_inspected": [
                evidence["MUTATION_PLAN.json"],
                evidence["MUTATION_CAMPAIGN.json"],
                evidence["RESTORATION_REUSE_STRESS.json"],
                evidence["MACHINE_BOUNDARY_ATTACK_REPORT.md"],
            ],
            "attacks_attempted": [
                "Relation coefficient variation",
                "Topology and numbering variation",
                "Wrong, missing, and reordered restoration",
                "Final response before restoration check",
            ],
            "findings": [
                f"Seed {mutation['seed']} campaign passed its {sum(len(v) if isinstance(v, list) else 1 for v in mutation['results'].values())} result families.",
                "A final boundary appears before RESTORE in the source protocol.",
                "The repaired reference wrapper fails closed on restoration variants.",
            ],
            "unresolved_risks": [
                "No source-side A repair was ported.",
                "No broad B topology generator was ported into production service form.",
            ],
            "decision": "A SOURCE IMPLEMENTATION REJECTED; REPAIRED MACHINE LAW REMAINS WORTH TESTING",
        },
        {
            "role": "Reviewer D: baseline and resource-law challenge",
            "assumptions": [
                "Problem-size scaling must not be replaced by repetition.",
                "Compact public baselines must be matched against the same boundary.",
            ],
            "methods": [
                "Direct formula search",
                "Public-program table enumeration",
                "Resource-law comparison by declared schema",
            ],
            "evidence_inspected": [
                evidence["BASELINE_CHALLENGE_REPORT.md"],
                evidence["INDEPENDENT_RECONSTRUCTION_REPORT.md"],
            ],
            "attacks_attempted": [
                "Fixed-schema formula challenge",
                "Equivalent public program collisions",
                "Table compression comparison",
            ],
            "findings": [
                f"C public programs checked: {independent['candidate_c']['public_programs']}",
                f"C unique final boundaries: {independent['candidate_c']['unique_boundaries']}",
                "The compact baseline remains stronger for the fixed schema.",
            ],
            "unresolved_risks": [
                "The obstruction should be retested on any larger or different public schema.",
            ],
            "decision": "C IS A TRANSFERABLE OBSTRUCTION, NOT A POSITIVE CLAIM",
        },
        {
            "role": "Reviewer E: transfer relevance to non-collapse frontier",
            "assumptions": [
                "No phase arithmetic or software timing transfers to physical Family 10h.",
                "Only abstract mechanisms can be carried forward.",
            ],
            "methods": [
                "Mechanism extraction",
                "Branch-local repaired reference transaction",
                "Classification against declared decision classes",
            ],
            "evidence_inspected": [
                evidence["TRANSFER_RELEVANCE_REPORT.md"],
                evidence["FINAL_PROVISIONAL_DECISION.md"]
                if (ROOT / "FINAL_PROVISIONAL_DECISION.md").exists()
                else evidence["TRANSFER_RELEVANCE_REPORT.md"],
            ],
            "attacks_attempted": [
                "Transfer-scope narrowing",
                "Positive-claim separation from obstruction evidence",
                "Family-scope challenge for quotient mechanism",
            ],
            "findings": [
                "Invest further in abstract transaction law and obstruction harness.",
                "Do not invest in source Candidate A protocol as implemented.",
                "Do not promote B or D beyond their verified scope.",
            ],
            "unresolved_risks": [
                "A future generic DAG scheduler may change B classification.",
                "A future mixed-family quotient proof may change D classification.",
            ],
            "decision": "CONTINUE ONLY BOUNDED TRANSFER EXPERIMENTS",
        },
    ]
    ledger = {
        "schema_version": "audio_catvm_independent_review_ledger.v1",
        "generated_utc": utc_now(),
        "canonical": False,
        "small_wall_crossed": False,
        "source_commit": source_receipt["source_commit"],
        "source_review_json_used_as_proof": False,
        "continuation_note": "No additional helper agent was used for this consolidation; review roles are grounded in branch-local evidence files.",
        "reviewers": reviewers,
        "disagreements": [
            {
                "issue": "Candidate D transfer scope",
                "resolution": "Restricted-family quotient is independently reproduced, but mixed/nonperiodic challenges prevent transfer classification.",
            },
            {
                "issue": "Candidate B scheduler value",
                "resolution": "Mathematical schedule for the public graph is real, but source implementation remains fixed-topology and is classified as fixture specialization.",
            },
        ],
        "evidence": evidence,
    }
    ledger["result_sha256"] = sha256_json(ledger)
    return ledger


def render_final_decision(
    source_receipt: dict[str, Any],
    restoration: dict[str, Any],
    independent: dict[str, Any],
    mutation: dict[str, Any],
    static: dict[str, Any],
    ledger: dict[str, Any],
) -> str:
    head = git_text(["rev-parse", "HEAD"])
    branch = git_text(["rev-parse", "--abbrev-ref", "HEAD"])
    status = subprocess.check_output(
        ["git", "status", "--short"], cwd=ROOT, text=True
    )
    classifications = {
        "Candidate A": "REJECTED_SOURCE_DEFECT",
        "Candidate B": "REJECTED_FIXTURE_SPECIALIZATION",
        "Candidate C": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_OBSTRUCTION",
        "Candidate D": "INDEPENDENTLY_VERIFIED_SOURCE_LOCAL",
    }
    top_hashes = {
        path.name: file_sha256(path)
        for path in sorted(ROOT.glob("*"))
        if path.is_file() and path.name.endswith((".json", ".md", ".sha256"))
    }
    verifier_hashes = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted((ROOT / "independent_verifier").glob("*"))
        if path.is_file() and path.name.endswith((".py", ".json", ".md"))
    }
    lines = [
        "# Final Provisional Decision",
        "",
        "Status: provisional, branch-local, non-canonical. Small Wall not crossed. Physical Family 10h evidence not modified.",
        "",
        "## Frozen Source",
        "",
        f"- Source commit: `{source_receipt['source_commit']}`",
        f"- Source parent: `{source_receipt['source_parent_commit']}`",
        f"- Source tree: `{source_receipt['source_tree']}`",
        f"- Current task branch: `{branch}`",
        f"- Current branch head at report generation: `{head}`",
        "",
        "## Classifications",
        "",
    ]
    for candidate, decision in classifications.items():
        lines.append(f"- {candidate}: `{decision}`")
    lines.extend(
        [
            "",
            "## What Was Copied",
            "",
            "Only the minimum candidate source, format, result, review, qualifier, and public fixture files listed in `TRANSFER_MANIFEST.json` and `SOURCE_FILE_HASHES.json` were copied into the isolated unverified candidate directory.",
            "",
            "## What Was Independently Reconstructed",
            "",
            "- Candidate A bounded relation composition, final boundary coefficients, restoration controls, and reuse coefficients.",
            "- Candidate B public 15-node DAG dependency model and bounded reversible schedule.",
            "- Candidate C strongest compact fixed-schema public baseline found by symbolic reconstruction.",
            "- Candidate D homogeneous-family future-continuation quotient and overmerge/undermerge controls.",
            "- A repaired branch-local reference transaction wrapper that responds only after restoration.",
            "",
            "## What Reproduced",
            "",
            "- C: source qualifier reproduced and independent obstruction reconstruction matched.",
            "- D: source qualifier reproduced with stale tracked-result provenance noted; homogeneous quotient reconstructed independently.",
            "- A and B: exact source qualifier execution did not reproduce as declared; preserved rebuilt binaries were only probed as source-local artifacts.",
            "",
            "## What Failed",
            "",
            "- A source protocol returns final boundary before restoration.",
            "- B source implementation remains tied to a fixed public graph and identifier schedule.",
            "- D mixed/nonperiodic layer challenges exceed the homogeneous rank law.",
            "",
            "## What Appears Transferable",
            "",
            "- Matched compact-baseline obstruction testing from C.",
            "- Abstract hidden-custody plus atomic-after-restoration transaction law, only as repaired in the branch-local reference experiment.",
            "- Generation/fanout receipts as a design requirement, pending a generic B scheduler.",
            "",
            "## Claims Explicitly Unproven",
            "",
            "- No source candidate is canonical.",
            "- No physical Family 10h claim changed.",
            "- No Small Wall crossing is claimed.",
            "- No phase arithmetic, F3 relation, source timing, or fixed scheduler result transfers to the physical frontier.",
            "- D is not a general quotient law.",
            "",
            "## Evidence Paths",
            "",
            "- `SOURCE_RECEIPT.json`",
            "- `SOURCE_REPRODUCTION_REPORT.md`",
            "- `STATIC_ANTI_SMUGGLE_AUDIT.json`",
            "- `INDEPENDENT_RECONSTRUCTION_REPORT.md`",
            "- `MUTATION_CAMPAIGN.json`",
            "- `RESTORATION_REUSE_STRESS.json`",
            "- `MACHINE_BOUNDARY_ATTACK_REPORT.md`",
            "- `BASELINE_CHALLENGE_REPORT.md`",
            "- `QUOTIENT_GENERALIZATION_REPORT.md`",
            "- `TRANSFER_RELEVANCE_REPORT.md`",
            "- `INDEPENDENT_REVIEW_LEDGER.json`",
            "- `LOCAL_VERIFICATION_COMMANDS.json`",
            "",
            "## Input Evidence Hashes",
            "",
            f"- Independent result hash: `{independent['result_sha256']}`",
            f"- Mutation result hash: `{mutation['result_sha256']}`",
            f"- Restoration result hash: `{restoration['result_sha256']}`",
            f"- Review ledger hash: `{ledger['result_sha256']}`",
            f"- Static finding counts: {severity_counts(static['findings'])}",
            "",
            "## Top-Level File Hashes",
            "",
        ]
    )
    for name, digest in top_hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "## Local Verifier File Hashes",
            "",
        ]
    )
    for name, digest in verifier_hashes.items():
        lines.append(f"- `{name}`: `{digest}`")
    lines.extend(
        [
            "",
            "## Worktree Status At Report Generation",
            "",
            "```text",
            status if status else "clean at report generation before this file write",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    source_receipt = read_json(ROOT / "SOURCE_RECEIPT.json")
    independent = read_json(ROOT / "independent_verifier" / "INDEPENDENT_RESULTS.json")
    mutation = read_json(ROOT / "MUTATION_CAMPAIGN.json")
    static = read_json(ROOT / "STATIC_ANTI_SMUGGLE_AUDIT.json")
    restoration = read_json(ROOT / "RESTORATION_REUSE_STRESS.json")

    write_text(ROOT / "BASELINE_CHALLENGE_REPORT.md", render_baseline_report(independent, mutation))
    write_text(ROOT / "QUOTIENT_GENERALIZATION_REPORT.md", render_quotient_report(independent, mutation))
    write_text(
        ROOT / "TRANSFER_RELEVANCE_REPORT.md",
        render_transfer_report(restoration, independent, static),
    )
    ledger = review_ledger(source_receipt, independent, mutation, static, restoration)
    write_json(ROOT / "INDEPENDENT_REVIEW_LEDGER.json", ledger)
    final = render_final_decision(
        source_receipt,
        restoration,
        independent,
        mutation,
        static,
        ledger,
    )
    write_text(ROOT / "FINAL_PROVISIONAL_DECISION.md", final)
    print(
        json.dumps(
            {
                "baseline_sha256": file_sha256(ROOT / "BASELINE_CHALLENGE_REPORT.md"),
                "quotient_sha256": file_sha256(ROOT / "QUOTIENT_GENERALIZATION_REPORT.md"),
                "transfer_sha256": file_sha256(ROOT / "TRANSFER_RELEVANCE_REPORT.md"),
                "review_ledger_sha256": file_sha256(ROOT / "INDEPENDENT_REVIEW_LEDGER.json"),
                "final_decision_sha256": file_sha256(ROOT / "FINAL_PROVISIONAL_DECISION.md"),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
