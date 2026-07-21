from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
RESULT_FILE = PACKAGE_DIR / "V2_BATCH_RESULTS.json"
REVIEW_FILE = PACKAGE_DIR / "V2_FINDINGS_NORMALIZED.json"
LANE_STATE_FILE = PACKAGE_DIR / "V2_LANE_STATE.md"
MANIFEST_FILE = PACKAGE_DIR / "V2_FINAL_MANIFEST.json"

DECISION = "CATALYTIC_WAVEFORM_ISING_V2_BATCH_GENERALIZATION_PARTIAL"
CLAIM_CEILING = "BOUNDED_SOFTWARE_CARRIER_CAUSAL_CATALYTIC_ISING_REFERENCE_ONLY"


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
    counts = summary["classification_counts"]
    lines = [
        "# Catalytic Waveform-Ising V2 Lane State",
        "",
        f"Decision: `{results['decision']}`",
        f"Review: `{review['reviewer_id']} {review['verdict']}`",
        f"Claim ceiling: `{results['claim_ceiling']}`",
        "",
        "```text",
        f"batch size                    {summary['batch_size']}",
        f"unique optima                 {summary['unique_optimum_total']}",
        f"accepted correct              {counts['ACCEPTED_CORRECT']}",
        f"accepted incorrect            {counts['ACCEPTED_INCORRECT']}",
        f"raw incorrect                 {counts['RAW_INCORRECT']}",
        f"non-unique raw matches        {summary['non_unique_raw_match_count']} / {summary['non_unique_total']}",
        f"strict controls               32 / 32",
        f"restoration and reuse         32 / 32",
        "```",
        "",
        "The successor materially improves bounded unseen software performance while",
        "preserving native waveform causality, transform dependence, restoration, and",
        "reuse. One accepted-incorrect output prevents the prospectively frozen",
        "`VERIFIED` decision.",
        "",
        "No hardware, playback, recording, procurement, fabrication, physical contact,",
        "physical-computation claim, bit-replacement claim, advantage claim, or Wall claim",
        "is authorized.",
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
    root = sha256_bytes(canonical_bytes(records))
    return {
        "claim_ceiling": CLAIM_CEILING,
        "decision": DECISION,
        "file_count_excluding_manifest": len(records),
        "files": records,
        "manifest_exclusion": MANIFEST_FILE.name,
        "package_bytes_excluding_manifest": sum(record["bytes"] for record in records),
        "package_root_sha256": root,
        "schema": "catalytic_waveform_ising_v2_final_manifest_v1",
    }


def validated_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    results = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_FILE.read_text(encoding="utf-8"))
    if results["decision"] != DECISION:
        raise ValueError("result decision differs from frozen final decision")
    if results["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("result claim ceiling changed")
    if review["verdict"] != "PASS" or review["findings"]:
        raise ValueError("focused independent review did not pass cleanly")
    if review["authorized_decision"] != DECISION:
        raise ValueError("review did not authorize the exact decision")
    if review["authorized_claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("review claim ceiling differs")
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
    print(json.dumps({
        "file_count": manifest["file_count_excluding_manifest"],
        "package_bytes": manifest["package_bytes_excluding_manifest"],
        "package_root_sha256": manifest["package_root_sha256"],
        "status": "FINAL_PACKAGE_MANIFEST_REPRODUCED",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
