from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
RESULT_FILE = PACKAGE_DIR / "BATCH_RESULTS.json"
REVIEW_FILE = PACKAGE_DIR / "FINDINGS_NORMALIZED.json"
LANE_STATE_FILE = PACKAGE_DIR / "LANE_STATE.md"
MANIFEST_FILE = PACKAGE_DIR / "FINAL_MANIFEST.json"

DECISION = "CATALYTIC_WAVEFORM_ISING_V2_STABILITY_BATCH_NOT_ESTABLISHED"
CLAIM_CEILING = "BOUNDED_SOFTWARE_REJECT_ONLY_WAVEFORM_STABILITY_REFERENCE_ONLY"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def lane_state_bytes(results: dict[str, Any], review: dict[str, Any]) -> bytes:
    summary = results["summary"]
    nominal = summary["nominal_counts"]
    gated = summary["stability_counts"]
    lines = [
        "# V2 Stability-Gate Lane State",
        "",
        f"Decision: `{results['decision']}`",
        f"Review: `{review['reviewer_id']} {review['verdict']}`",
        f"Claim ceiling: `{results['claim_ceiling']}`",
        "",
        "```text",
        f"batch size                       {summary['batch_size']}",
        f"unique optima                    {summary['unique_count']}",
        f"non-unique optima                 {summary['non_unique_count']}",
        f"nominal accepted correct         {nominal['NOMINAL_ACCEPTED_CORRECT']}",
        f"nominal accepted incorrect        {nominal['NOMINAL_ACCEPTED_INCORRECT']}",
        f"nominal rejected incorrect        {nominal['NOMINAL_REJECTED_INCORRECT']}",
        f"stability accepted correct       {gated['STABILITY_ACCEPTED_CORRECT']}",
        f"stability accepted incorrect      {gated['STABILITY_ACCEPTED_INCORRECT']}",
        f"false-accept reduction             {summary['false_accept_reduction_count']}",
        f"correct-result retention          {summary['correct_result_retention_rate']}",
        "strict and stability controls     64 / 64",
        "restoration and reuse             64 / 64",
        "```",
        "",
        "The discriminator remained waveform-native, reject-only, restorable, and",
        "reusable, but it did not change any unseen acceptance decision. Five nominal",
        "accepted-incorrect results remained accepted, so the discriminator did not",
        "provide credible unseen false-accept reduction and is not promoted.",
        "",
        "The frozen V2 machine and all predecessor results remain unchanged. No hardware,",
        "playback, recording, procurement, fabrication, physical contact, reliability",
        "promotion, physical-computation claim, bit-replacement claim, advantage claim,",
        "or Wall claim is authorized.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def manifest_document() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for path in sorted(PACKAGE_DIR.iterdir(), key=lambda value: value.name):
        if not path.is_file() or path == MANIFEST_FILE:
            continue
        records.append(
            {
                "bytes": path.stat().st_size,
                "path": path.name,
                "sha256": sha256_file(path),
            }
        )
    return {
        "claim_ceiling": CLAIM_CEILING,
        "decision": DECISION,
        "file_count_excluding_manifest": len(records),
        "files": records,
        "manifest_exclusion": MANIFEST_FILE.name,
        "package_bytes_excluding_manifest": sum(record["bytes"] for record in records),
        "package_root_sha256": sha256_bytes(canonical_bytes(records)),
        "schema": "catalytic_waveform_ising_v2_stability_final_manifest_v1",
    }


def validated_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    results = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    if results["decision"] != DECISION or results["summary"]["decision"] != DECISION:
        raise ValueError("result decision differs from frozen final decision")
    if results["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("result claim ceiling changed")
    summary = results["summary"]
    if summary["batch_size"] != 64 or summary["uninterpretable_count"] != 0:
        raise ValueError("batch execution is incomplete or uninterpretable")
    if summary["meaningful_false_accept_reduction"]:
        raise ValueError("negative adjudication conflicts with reported discrimination")
    if summary["false_accept_reduction_count"] != 0:
        raise ValueError("negative adjudication requires zero false-accept reduction")
    if summary["stability_counts"]["STABILITY_ACCEPTED_INCORRECT"] != 5:
        raise ValueError("accepted-incorrect count drifted")
    if summary["promotion_pass"]:
        raise ValueError("negative adjudication conflicts with promotion pass")
    if review["verdict"] != "PASS" or review["findings"]:
        raise ValueError("focused independent review did not pass cleanly")
    if review["authorized_decision"] != DECISION:
        raise ValueError("review did not authorize the exact decision")
    if review["authorized_claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("review claim ceiling differs")
    if review["candidate"]["results_sha256"] != sha256_file(RESULT_FILE):
        raise ValueError("review is not bound to the exact result bytes")
    return results, review


def build() -> dict[str, Any]:
    results, review = validated_inputs()
    write_atomic(LANE_STATE_FILE, lane_state_bytes(results, review))
    manifest = manifest_document()
    write_atomic(MANIFEST_FILE, canonical_bytes(manifest))
    return manifest


def verify() -> dict[str, Any]:
    results, review = validated_inputs()
    if LANE_STATE_FILE.read_bytes() != lane_state_bytes(results, review):
        raise ValueError("lane state does not reproduce")
    manifest = manifest_document()
    if MANIFEST_FILE.read_bytes() != canonical_bytes(manifest):
        raise ValueError("final manifest does not reproduce")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    manifest = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "file_count": manifest["file_count_excluding_manifest"],
                "package_bytes": manifest["package_bytes_excluding_manifest"],
                "package_root_sha256": manifest["package_root_sha256"],
                "status": "FINAL_PACKAGE_MANIFEST_REPRODUCED",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
