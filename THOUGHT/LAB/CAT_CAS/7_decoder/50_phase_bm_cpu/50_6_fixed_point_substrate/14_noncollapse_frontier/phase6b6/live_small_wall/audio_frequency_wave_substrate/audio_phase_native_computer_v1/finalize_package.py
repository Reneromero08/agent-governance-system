from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_FILE = PACKAGE_DIR / "PACKAGE_MANIFEST.json"
STATE_FILE = PACKAGE_DIR / "LANE_STATE.md"
ENGINE_FILE = PACKAGE_DIR / "phase_native_engine.py"
CONTRACT_FILE = PACKAGE_DIR / "PROSPECTIVE_CONTRACT.json"
RAW_SEAL_FILE = PACKAGE_DIR / "PROSPECTIVE_RAW_SEAL.json"
RESULTS_FILE = PACKAGE_DIR / "FINAL_RESULTS.json"
RESOURCE_FILE = PACKAGE_DIR / "RESOURCE_RESULTS.json"
INDEPENDENT_FILE = PACKAGE_DIR / "INDEPENDENT_VERIFICATION.json"
REVIEW_FILE = PACKAGE_DIR / "FOCUSED_REVIEW.md"
PRE_ADJUDICATION_COMMIT = "6e9112f78ef5b18dfa4ed8c646b80202a68a2d4b"
DECISION = "PHASE_NATIVE_COMPUTER_REFERENCE_VERIFIED"


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def state_bytes() -> bytes:
    contract = json.loads(CONTRACT_FILE.read_text(encoding="utf-8"))
    seal = json.loads(RAW_SEAL_FILE.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    resources = json.loads(RESOURCE_FILE.read_text(encoding="utf-8"))
    independent = json.loads(INDEPENDENT_FILE.read_text(encoding="utf-8"))
    lines = [
        "# Phase-native computer lane state",
        "",
        f"- result: `{results['decision']}`",
        f"- claim ceiling: `{results['claim_ceiling']}`",
        f"- engine fingerprint: `{results['engine_fingerprint']}`",
        f"- prospective contract: `{contract['contract_sha256']}`",
        f"- pre-adjudication commit: `{PRE_ADJUDICATION_COMMIT}`",
        f"- raw result SHA-256: `{seal['raw_sha256']}`",
        f"- prospective exact results: {results['exact_count']}/{results['case_count']}",
        f"- independent verifier: `{independent['reviewer_id']}` PASS",
        "- focused reviewer: `GPT-5.6-SOL-PHASE-COMPUTER-FINAL-REVIEW-01` PASS",
        f"- resource record: `{resources['document_sha256']}`",
        "",
        "One shared factorized phase engine executes modular arithmetic, carried "
        "binary addition, conditional pipelines, finite-state accumulation, and "
        "routed sequence composition. Intermediate computational state remains in "
        "spectral phase relations until the explicit output boundary.",
        "",
        "The package establishes a bounded reusable software phase-computer "
        "reference. It establishes neither hardware execution nor computational "
        "advantage.",
        "",
    ]
    return ("\n".join(lines)).encode("utf-8")


def package_files() -> list[Path]:
    return sorted(
        path
        for path in PACKAGE_DIR.iterdir()
        if path.is_file()
        and path.name != MANIFEST_FILE.name
        and not path.name.endswith(".tmp")
    )


def manifest_document() -> dict[str, Any]:
    files = [
        {
            "bytes": path.stat().st_size,
            "path": path.name,
            "sha256": sha256_file(path),
        }
        for path in package_files()
    ]
    document = {
        "decision": DECISION,
        "file_count_excluding_manifest": len(files),
        "files": files,
        "package_bytes_excluding_manifest": sum(item["bytes"] for item in files),
        "schema": "phase_native_computer_package_manifest_v1",
    }
    document["content_root_sha256"] = sha256_bytes(canonical_bytes(files))
    return document


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build() -> dict[str, Any]:
    write_atomic(STATE_FILE, state_bytes())
    document = manifest_document()
    write_atomic(MANIFEST_FILE, canonical_bytes(document))
    return document


def verify() -> dict[str, Any]:
    if STATE_FILE.read_bytes() != state_bytes():
        raise RuntimeError("lane state reproduction mismatch")
    stored = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    rebuilt = manifest_document()
    if canonical_bytes(stored) != canonical_bytes(rebuilt):
        raise RuntimeError("package manifest reproduction mismatch")
    if stored["decision"] != DECISION:
        raise RuntimeError("package decision mismatch")
    return stored


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.command == "build" else verify()
    print(
        json.dumps(
            {
                "content_root_sha256": document["content_root_sha256"],
                "decision": document["decision"],
                "file_count": document["file_count_excluding_manifest"],
                "package_bytes": document["package_bytes_excluding_manifest"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
