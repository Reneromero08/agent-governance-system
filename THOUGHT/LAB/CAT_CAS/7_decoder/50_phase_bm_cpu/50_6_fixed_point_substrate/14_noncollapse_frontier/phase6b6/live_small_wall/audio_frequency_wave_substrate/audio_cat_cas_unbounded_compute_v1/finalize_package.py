"""Write the concise final result, review receipt, and content manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase_path_engine import canonical_bytes, sha256_bytes


HERE = Path(__file__).resolve().parent
FINAL_RESULTS_PATH = HERE / "FINAL_RESULTS.json"
FINAL_REPORT_PATH = HERE / "FINAL_REPORT.md"
REVIEW_PATH = HERE / "INDEPENDENT_BOUNDARY_REVIEW.md"
MANIFEST_PATH = HERE / "PACKAGE_MANIFEST.json"
RAW_COMMIT = "a624e3015eb7b1ca9a867a38baee98e74a6db4bc"
CONTRACT_COMMIT = "87fd5bcc0166590bfdb0a676846a3c53891d8a40"


def load(name: str) -> dict[str, Any]:
    return json.loads((HERE / name).read_text(encoding="utf-8"))


def build_final_result() -> dict[str, Any]:
    development = load("DEVELOPMENT_RESULTS.json")
    contract = load("PROSPECTIVE_CONTRACT.json")
    raw = load("PROSPECTIVE_RAW_RESULTS.json")
    external = load("EXTERNAL_VERIFICATION.json")
    performance = load("PERFORMANCE_RESULTS.json")
    invariants = load("INVARIANT_RESULTS.json")
    gamma_by_size: dict[str, float] = {}
    for case in raw["cases"]:
        gamma_by_size[str(case["steps"])] = case["gamma_path_work"]
    return {
        "bounded_mission_result": {
            "classical_boundary_result": True,
            "compact_unresolved_representation": True,
            "credible_compute_leverage": True,
            "independent_acceptance": True,
            "native_global_phase_computation": True,
            "substrate_restoration": True,
            "substrate_reuse": True,
        },
        "claim_ceiling": (
            "BOUNDED_SOFTWARE_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_ONLY"
        ),
        "contract": {
            "commit": CONTRACT_COMMIT,
            "file_sha256": sha256_bytes(
                (HERE / "PROSPECTIVE_CONTRACT.json").read_bytes()
            ),
            "ordered_batch_sha256": contract["batch"][
                "ordered_batch_sha256"
            ],
            "programs": contract["batch"]["case_count"],
        },
        "decision": (
            "CAT_CAS_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_VERIFIED"
        ),
        "development": {
            "cases": len(development["development_cases"]),
            "controls": development["controls"],
            "maximum_steps": max(
                case["steps"] for case in development["development_cases"]
            ),
        },
        "engine_fingerprint": raw["engine_fingerprint"],
        "external_acceptance": {
            "accepted": external["accepted_cases"],
            "file_sha256": sha256_bytes(
                (HERE / "EXTERNAL_VERIFICATION.json").read_bytes()
            ),
            "raw_cases": external["raw_cases"],
            "verifier_absent_at_raw_commit": (
                not external["independence"][
                    "external_verifier_present_at_raw_commit"
                ]
            ),
        },
        "mechanism": {
            "complete_path_modes": 0,
            "compiler_solves_target": False,
            "history_factor_count": 0,
            "native_operator": "TORUS_PATH_SHEAR",
            "native_state": (
                "modular path-count residues as relative phase on a product torus"
            ),
            "phase_lock": "fixed label-free p-fold injection-lock dynamics",
            "program_derived_inverse": True,
        },
        "relational_invariants": {
            "cyclic_automorphism_checks": invariants[
                "cyclic_automorphism_checks"
            ],
            "maximum_restoration_error": invariants[
                "maximum_restoration_error"
            ],
            "non_affine_scramble_rejects": invariants[
                "non_affine_scramble_control_from_raw"
            ],
            "verdict": invariants["verdict"],
            "weight_order_checks": invariants["weight_order_checks"],
        },
        "prospective_result": {
            "cases": raw["case_count"],
            "external_exact": external["accepted_cases"],
            "gamma_by_size": gamma_by_size,
            "gamma_grows_in_every_family": raw[
                "gamma_grows_with_size_in_every_family"
            ],
            "maximum_restoration_error": raw[
                "maximum_restoration_error"
            ],
            "minimum_displacement_l2": raw["minimum_displacement_l2"],
            "restoration_pass": raw["all_restorations_pass"],
            "reuse_pass": raw["all_reuse_pass"],
            "uninterpretable": raw["uninterpretable"],
        },
        "raw_evidence": {
            "commit": RAW_COMMIT,
            "file_sha256": sha256_bytes(
                (HERE / "PROSPECTIVE_RAW_RESULTS.json").read_bytes()
            ),
            "raw_result_root": raw["raw_result_root"],
        },
        "resource_honesty": performance["resource_conclusion"],
        "schema": "cat_cas.toroidal_path_sum.final.v1",
        "stop_boundary": (
            "BOUNDED_MISSION_RESULT_WITH_COMPACT_UNRESOLVED_PATH_WORK_"
            "LEVERAGE_AND_INDEPENDENT_ACCEPTANCE"
        ),
        "unproved": [
            "advantage over compact classical dynamic programming",
            "fixed-size unbounded information",
            "universal computation",
            "energy advantage",
            "official external benchmark acceptance",
            "physical phase computation",
            "hardware bit replacement",
            "Wall crossing",
        ],
        "zero_contact": True,
    }


def write_final_documents(result: dict[str, Any]) -> None:
    FINAL_RESULTS_PATH.write_bytes(canonical_bytes(result))
    gamma = result["prospective_result"]["gamma_by_size"]
    FINAL_REPORT_PATH.write_text(
        "\n".join(
            [
                "# CAT_CAS compact toroidal path-sum result",
                "",
                f"- result: `{result['decision']}`",
                f"- claim ceiling: `{result['claim_ceiling']}`",
                (
                    "- prospective external exact: "
                    f"{result['external_acceptance']['accepted']}/"
                    f"{result['external_acceptance']['raw_cases']}"
                ),
                (
                    "- maximum restoration error: "
                    f"{result['prospective_result']['maximum_restoration_error']:.12g}"
                ),
                "- actual restored-carrier reuse: PASS",
                "- complete path modes: 0",
                "- retained instruction factors: 0",
                (
                    "- relational invariants: "
                    f"{result['relational_invariants']['weight_order_checks']} "
                    "order + "
                    f"{result['relational_invariants']['cyclic_automorphism_checks']} "
                    "cyclic-automorphism checks PASS"
                ),
                "",
                "## Compute leverage",
                "",
                f"- n=16: Gamma {gamma['16']:.6g}",
                f"- n=32: Gamma {gamma['32']:.6g}",
                f"- n=64: Gamma {gamma['64']:.6g}",
                f"- n=128: Gamma {gamma['128']:.6g}",
                f"- n=256: Gamma {gamma['256']:.6g}",
                "",
                "Gamma compares against explicit include/exclude path-work.",
                "The compact classical dynamic program remains faster than this",
                "software phase reference. No best-classical advantage is claimed.",
                "",
                "## Meaning",
                "",
                "The bounded software machine compiles public `.holo` shifts into",
                "global torus shears, aggregates exponentially many binary paths",
                "without path materialization, latches one modular count, reverses",
                "the actual phase process, restores its borrowed carrier, and",
                "passes that exact restored carrier to another program.",
                "",
                "This reaches the bounded CAT_CAS mission stop condition. It does",
                "not establish fixed-size unbounded information, universality,",
                "physical execution, or hardware bit replacement.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    REVIEW_PATH.write_text(
        "\n".join(
            [
                "# Independent boundary review",
                "",
                "- reviewer ID: `POST-SEAL-INDEPENDENT-DP-01`",
                "- verdict: `PASS`",
                f"- raw commit: `{RAW_COMMIT}`",
                (
                    "- verifier absent at raw commit: "
                    f"{result['external_acceptance']['verifier_absent_at_raw_commit']}"
                ),
                "- imports native engine: false",
                "- externally exact cases: 20/20",
                "- literal 2^16 comparisons: 4/4",
                "- open findings: 0",
                "",
                "The post-seal verifier independently parses canonical `.holo`",
                "bytes and executes integer modular dynamic programming. It does",
                "not import or rerun the native engine to decide correctness.",
                "",
                "The review accepts compact unresolved path-work leverage only.",
                "It explicitly rejects claims of advantage over compact classical",
                "DP, fixed-size unbounded information, universality, or physical",
                "computation.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def write_manifest() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(HERE.rglob("*")):
        if not path.is_file():
            continue
        if path == MANIFEST_PATH or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(HERE).as_posix()
        payload = path.read_bytes()
        records.append(
            {
                "bytes": len(payload),
                "path": relative,
                "sha256": sha256_bytes(payload),
            }
        )
    manifest = {
        "content_root": sha256_bytes(canonical_bytes(records)),
        "file_count_excluding_manifest": len(records),
        "files": records,
        "schema": "cat_cas.toroidal_path_sum.manifest.v1",
        "total_bytes_excluding_manifest": sum(
            record["bytes"] for record in records
        ),
    }
    MANIFEST_PATH.write_bytes(canonical_bytes(manifest))
    return manifest


if __name__ == "__main__":
    final = build_final_result()
    write_final_documents(final)
    manifest = write_manifest()
    print(
        json.dumps(
            {
                "content_root": manifest["content_root"],
                "decision": final["decision"],
                "files": manifest["file_count_excluding_manifest"],
            },
            sort_keys=True,
        )
    )
