from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
FREEZE_FILE = PACKAGE_DIR / "SIX_SITE_FREEZE.json"
PREORACLE_SEAL_FILE = PACKAGE_DIR / "SIX_SITE_PREORACLE_SEAL.json"
RESULTS_FILE = PACKAGE_DIR / "SIX_SITE_BATCH_RESULTS.json"
INDEPENDENT_FILE = PACKAGE_DIR / "SIX_SITE_INDEPENDENT_VERIFICATION.json"
STATE_FILE = PACKAGE_DIR / "SIX_SITE_LANE_STATE.md"
MANIFEST_FILE = PACKAGE_DIR / "SIX_SITE_FINAL_MANIFEST.json"

DECISION = "CATALYTIC_WAVEFORM_ISING_V3_SIX_SITE_VERIFIED"
CLAIM_CEILING = "BOUNDED_SOFTWARE_RECURSIVE_SPECTRAL_PHASE_REFERENCE_ONLY"
FINAL_FILES = (
    "SIX_SITE_BATCH_REPORT.md",
    "SIX_SITE_BATCH_RESULTS.json",
    "SIX_SITE_INDEPENDENT_REVIEW.md",
    "SIX_SITE_INDEPENDENT_VERIFICATION.json",
    "SIX_SITE_ORACLE_TRACE.json",
    "finalize_package.py",
    "independent_verifier.py",
    "oracle_adjudicator.py",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def load_and_validate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    freeze = json.loads(FREEZE_FILE.read_text(encoding="utf-8"))
    preoracle = json.loads(PREORACLE_SEAL_FILE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    independent = json.loads(INDEPENDENT_FILE.read_text(encoding="utf-8"))
    if results["decision"] != DECISION or not results["summary"]["promotion_pass"]:
        raise ValueError("prospective V3 promotion did not pass")
    if results["claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("V3 claim ceiling drift")
    if independent["verdict"] != "PASS" or independent["findings"]:
        raise ValueError("independent review did not pass without findings")
    if independent["authorized_decision"] != DECISION:
        raise ValueError("independent review does not authorize the V3 decision")
    if independent["authorized_claim_ceiling"] != CLAIM_CEILING:
        raise ValueError("independent review claim ceiling drift")
    identities = {
        freeze["machine_fingerprint"],
        results["machine_fingerprint"],
    }
    if len(identities) != 1:
        raise ValueError("machine identity drift")
    if preoracle["oracle_call_count"] != 0 or preoracle["energy_call_count"] != 0:
        raise ValueError("oracle-order seal failed")
    return freeze, preoracle, results, independent


def state_bytes(
    freeze: dict[str, Any],
    preoracle: dict[str, Any],
    results: dict[str, Any],
    independent: dict[str, Any],
) -> bytes:
    summary = results["summary"]
    counts = summary["classification_counts"]
    text = f"""# Catalytic Waveform-Ising V3 Six-Site Lane State

Decision: `{DECISION}`
Claim ceiling: `{CLAIM_CEILING}`

## Frozen custody

```text
machine fingerprint        {freeze['machine_fingerprint']}
ordered batch SHA-256      {freeze['batch_ordered_sha256']}
pre-oracle evidence SHA-256 {preoracle['evidence_sha256']}
oracle calls before seal   {preoracle['oracle_call_count']}
energy calls before seal   {preoracle['energy_call_count']}
```

## Prospective result

```text
batch size                 {summary['batch_size']}
unique optima              {summary['unique_count']}
unique raw correct         {summary['unique_raw_correct']}
accepted incorrect         {counts['UNIQUE_ACCEPTED_INCORRECT']}
rejected unique correct    {counts['UNIQUE_REJECTED_CORRECT']}
non-unique rejected        {counts['NON_UNIQUE_REJECTED']}
promotion pass             {summary['promotion_pass']}
```

## Independent verification

Reviewer: `{independent['reviewer_id']}`
Verdict: `{independent['verdict']}`
Findings: `{len(independent['findings'])}`
States independently enumerated: `{independent['state_count_enumerated']}`

The bounded software mechanism uses a complete 64-mode recursive spectral phase
tree. It establishes correctness, reject-only handling of tied optima, exact
inverse restoration within the frozen tolerance, carrier reuse, and the frozen
anti-smuggling controls for the six-site reference domain.

It does not establish computational advantage, scalable complexity advantage,
physical waveform computation, hardware persistence, or bit replacement.
"""
    return text.encode("utf-8")


def manifest_document() -> dict[str, Any]:
    nested = sorted(
        path.relative_to(PACKAGE_DIR).as_posix()
        for path in PACKAGE_DIR.rglob("*")
        if path.is_file() and path.parent != PACKAGE_DIR
    )
    if nested:
        raise ValueError("final package contains unsealed nested inputs: " + ", ".join(nested))
    names = sorted(
        path.name
        for path in PACKAGE_DIR.iterdir()
        if path.is_file() and path.name != MANIFEST_FILE.name
    )
    records = [
        {
            "bytes": (PACKAGE_DIR / name).stat().st_size,
            "path": name,
            "sha256": sha256_file(PACKAGE_DIR / name),
        }
        for name in names
    ]
    return {
        "claim_ceiling": CLAIM_CEILING,
        "decision": DECISION,
        "file_count_excluding_manifest": len(records),
        "files": records,
        "package_bytes_excluding_manifest": sum(record["bytes"] for record in records),
        "schema": "catalytic_waveform_ising_v3_six_site_final_manifest_v1",
    }


def build() -> dict[str, Any]:
    freeze, preoracle, results, independent = load_and_validate()
    write_atomic(STATE_FILE, state_bytes(freeze, preoracle, results, independent))
    manifest = manifest_document()
    write_atomic(MANIFEST_FILE, canonical_bytes(manifest))
    return manifest


def verify() -> dict[str, Any]:
    freeze, preoracle, results, independent = load_and_validate()
    expected_state = state_bytes(freeze, preoracle, results, independent)
    if STATE_FILE.read_bytes() != expected_state:
        raise ValueError("V3 lane state does not reproduce")
    expected_manifest = manifest_document()
    if MANIFEST_FILE.read_bytes() != canonical_bytes(expected_manifest):
        raise ValueError("V3 final manifest does not reproduce")
    return expected_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    manifest = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "decision": manifest["decision"],
                "file_count_excluding_manifest": manifest[
                    "file_count_excluding_manifest"
                ],
                "manifest_sha256": sha256_file(MANIFEST_FILE),
                "package_bytes_excluding_manifest": manifest[
                    "package_bytes_excluding_manifest"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
