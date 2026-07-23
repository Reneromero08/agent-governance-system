"""Independent post-seal verifier.

This module intentionally does not import the native engine, compiler, builder,
or runner.  It independently parses the public .holo bytes, performs compact
integer dynamic programming, and compares only the already-sealed classical
boundary records.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "PROSPECTIVE_CONTRACT.json"
RAW_PATH = HERE / "PROSPECTIVE_RAW_RESULTS.json"
RESULT_PATH = HERE / "EXTERNAL_VERIFICATION.json"
REPORT_PATH = HERE / "EXTERNAL_VERIFICATION.md"
RAW_COMMIT = "a624e3015eb7b1ca9a867a38baee98e74a6db4bc"
RAW_FILE_SHA256 = (
    "ce3e656e5556a123babfd5012382ff737b35661ac2d94fd0f58ee10b808bb4e4"
)
REPO_RELATIVE_PACKAGE = (
    "THOUGHT/LAB/CAT_CAS/7_decoder/50_phase_bm_cpu/"
    "50_6_fixed_point_substrate/14_noncollapse_frontier/phase6b6/"
    "live_small_wall/audio_frequency_wave_substrate/"
    "audio_cat_cas_unbounded_compute_v1"
)


def canonical_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", *arguments],
        check=check,
        capture_output=True,
    )


def _strict_holo(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    document = json.loads(payload.decode("utf-8"))
    if payload != canonical_bytes(document):
        raise RuntimeError(f"noncanonical holo source: {path.name}")
    if set(document) != {
        "collapse_boundary",
        "geometry",
        "max_steps",
        "name",
        "process",
        "schema",
    }:
        raise RuntimeError("holo schema drift")
    if document["schema"] != "cat_cas.holo.path_sum.v1":
        raise RuntimeError("holo schema identity drift")
    return document


def _independent_program_sha256(document: dict[str, Any]) -> str:
    source_sha256 = sha256_bytes(canonical_bytes(document))
    instructions = [
        {"op": "TORUS_PATH_SHEAR", "shift": weight, "step": step}
        for step, weight in enumerate(
            document["process"]["binary_choice_weights"]
        )
    ]
    return sha256_bytes(
        canonical_bytes(
            {
                "instructions": instructions,
                "source_sha256": source_sha256,
            }
        )
    )


def _compact_dp(document: dict[str, Any]) -> int:
    residue_modulus = document["geometry"]["residue_modulus"]
    phase_product = math.prod(document["geometry"]["phase_moduli"])
    counts = np.zeros(residue_modulus, dtype=np.int64)
    counts[0] = 1
    for weight in document["process"]["binary_choice_weights"]:
        counts = (counts + np.roll(counts, weight)) % phase_product
    target = document["collapse_boundary"]["target_residue"]
    return int(counts[target])


def _explicit_paths(document: dict[str, Any]) -> int:
    residue_modulus = document["geometry"]["residue_modulus"]
    target = document["collapse_boundary"]["target_residue"]
    residues = [0]
    for weight in document["process"]["binary_choice_weights"]:
        residues.extend(
            (residue + weight) % residue_modulus
            for residue in tuple(residues)
        )
    return sum(residue == target for residue in residues) % math.prod(
        document["geometry"]["phase_moduli"]
    )


def verify() -> dict[str, Any]:
    contract_payload = CONTRACT_PATH.read_bytes()
    raw_payload = RAW_PATH.read_bytes()
    contract = json.loads(contract_payload.decode("utf-8"))
    raw = json.loads(raw_payload.decode("utf-8"))
    if sha256_bytes(raw_payload) != RAW_FILE_SHA256:
        raise RuntimeError("sealed raw file hash mismatch")

    committed_raw = _git(
        "show",
        f"{RAW_COMMIT}:{REPO_RELATIVE_PACKAGE}/PROSPECTIVE_RAW_RESULTS.json",
    ).stdout
    if committed_raw != raw_payload:
        raise RuntimeError("working raw bytes differ from raw commit")
    verifier_at_raw = _git(
        "cat-file",
        "-e",
        f"{RAW_COMMIT}:{REPO_RELATIVE_PACKAGE}/external_verifier.py",
        check=False,
    )
    if verifier_at_raw.returncode == 0:
        raise RuntimeError("external verifier existed at raw execution commit")

    root_document = dict(raw)
    root_document["raw_result_root"] = None
    if sha256_bytes(canonical_bytes(root_document)) != raw["raw_result_root"]:
        raise RuntimeError("raw result root mismatch")
    if raw["contract_sha256"] != sha256_bytes(contract_payload):
        raise RuntimeError("raw-to-contract binding mismatch")
    if raw["oracle_or_external_evaluator_loaded"]:
        raise RuntimeError("raw execution reports evaluator contact")

    raw_by_path = {case["path"]: case for case in raw["cases"]}
    adjudications: list[dict[str, Any]] = []
    explicit_matches = 0
    for entry in contract["batch"]["entries"]:
        path = HERE / entry["path"]
        payload = path.read_bytes()
        if sha256_bytes(payload) != entry["holo_sha256"]:
            raise RuntimeError("holo hash mismatch")
        document = _strict_holo(path)
        if _independent_program_sha256(document) != entry["program_sha256"]:
            raise RuntimeError("independent program identity mismatch")
        expected = _compact_dp(document)
        raw_case = raw_by_path[entry["path"]]
        observed = raw_case["boundary"]["count_mod_crt"]
        if expected != observed:
            raise RuntimeError(f"external mismatch: {entry['path']}")
        explicit: int | None = None
        if entry["steps"] == 16:
            explicit = _explicit_paths(document)
            if explicit != expected:
                raise RuntimeError("explicit path comparison mismatch")
            explicit_matches += 1
        adjudications.append(
            {
                "accepted": True,
                "external_count_mod_crt": expected,
                "explicit_path_count_mod_crt": explicit,
                "path": entry["path"],
                "raw_count_mod_crt": observed,
                "steps": entry["steps"],
            }
        )

    by_family: dict[str, list[tuple[int, float]]] = {
        family: [] for family in contract["batch"]["families"]
    }
    for entry in contract["batch"]["entries"]:
        case = raw_by_path[entry["path"]]
        by_family[entry["family"]].append(
            (entry["steps"], case["gamma_path_work"])
        )
    gamma_grows = all(
        all(
            right[1] > left[1]
            for left, right in zip(
                sorted(values), sorted(values)[1:]
            )
        )
        for values in by_family.values()
    )
    result = {
        "accepted_cases": len(adjudications),
        "adjudications": adjudications,
        "all_external_results_exact": True,
        "claim_boundary": {
            "advantage_over_compact_classical_dp": False,
            "bounded_software_reference": True,
            "fixed_size_unbounded_information": False,
            "physical_computation": False,
            "universal_computation": False,
        },
        "contract_sha256": sha256_bytes(contract_payload),
        "decision": "CAT_CAS_COMPACT_TOROIDAL_PATH_SUM_REFERENCE_VERIFIED",
        "explicit_path_matches": explicit_matches,
        "gamma_grows_with_size_in_every_family": gamma_grows,
        "independence": {
            "external_verifier_present_at_raw_commit": False,
            "imports_native_engine": False,
            "raw_commit": RAW_COMMIT,
            "raw_file_sha256": RAW_FILE_SHA256,
            "raw_result_root": raw["raw_result_root"],
        },
        "raw_cases": raw["case_count"],
        "restoration_and_reuse_from_sealed_raw": (
            raw["all_restorations_pass"] and raw["all_reuse_pass"]
        ),
        "schema": "cat_cas.toroidal_path_sum.external_verification.v1",
        "uninterpretable": 0,
    }
    if (
        result["accepted_cases"] != contract["batch"]["case_count"]
        or not result["gamma_grows_with_size_in_every_family"]
        or not result["restoration_and_reuse_from_sealed_raw"]
    ):
        raise RuntimeError("prospective promotion criterion failed")
    return result


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.write_bytes(canonical_bytes(result))
    REPORT_PATH.write_text(
        "\n".join(
            [
                "# Independent toroidal path-sum verification",
                "",
                f"- decision: `{result['decision']}`",
                (
                    "- external exact cases: "
                    f"{result['accepted_cases']}/{result['raw_cases']}"
                ),
                (
                    "- explicit 2^16 path comparisons: "
                    f"{result['explicit_path_matches']}"
                ),
                (
                    "- Gamma grows in every family: "
                    f"{result['gamma_grows_with_size_in_every_family']}"
                ),
                f"- raw commit: `{RAW_COMMIT}`",
                f"- raw file SHA-256: `{RAW_FILE_SHA256}`",
                "",
                "The verifier was absent at the raw commit and imports no",
                "native engine or compiler. It independently executes compact",
                "integer dynamic programming against the sealed boundary.",
                "",
                "This accepts a bounded compact phase path-sum reference. It",
                "does not establish advantage over compact classical DP.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


if __name__ == "__main__":
    verification = verify()
    write_result(verification)
    print(
        json.dumps(
            {
                "accepted": verification["accepted_cases"],
                "decision": verification["decision"],
                "raw_commit": RAW_COMMIT,
            },
            sort_keys=True,
        )
    )
