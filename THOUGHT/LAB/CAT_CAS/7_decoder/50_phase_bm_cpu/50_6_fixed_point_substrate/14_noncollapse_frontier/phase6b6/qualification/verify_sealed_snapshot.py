"""Software-only sealed snapshot verifier for future Phase 6B.6 qualification."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

try:
    from .qualification_contract import (
        AUTHORITY_FALSE_FIELDS,
        AUTHORITY_TRUE_FIELDS,
        EXPECTED_MERGED_MAIN_HEAD,
        PHASE6B6_ROOT,
        QUALIFIED_V2_SOURCE,
        canonical_json,
        digest,
        qualification_contract,
        validate_schema,
    )
except ImportError:  # pragma: no cover
    from qualification_contract import (  # type: ignore
        AUTHORITY_FALSE_FIELDS,
        AUTHORITY_TRUE_FIELDS,
        EXPECTED_MERGED_MAIN_HEAD,
        PHASE6B6_ROOT,
        QUALIFIED_V2_SOURCE,
        canonical_json,
        digest,
        qualification_contract,
        validate_schema,
    )


class SnapshotVerificationError(ValueError):
    """Raised when a sealed snapshot identity fails closed."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_inventory(snapshot_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for path in sorted(snapshot_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(snapshot_dir).as_posix()
        files[rel] = file_sha256(path)
    return files


def verify_snapshot_identity(snapshot_dir: Path, identity: dict[str, Any]) -> dict[str, Any]:
    validate_schema("snapshot_identity.schema.json", identity)
    contract = qualification_contract()
    expected_contract_digest = contract["qualification_contract_sha256"]

    checks: list[str] = []
    if identity["expected_commit"] != EXPECTED_MERGED_MAIN_HEAD:
        raise SnapshotVerificationError("wrong expected commit")
    checks.append("expected_commit")

    if identity["expected_tree"] != identity["observed_tree"]:
        raise SnapshotVerificationError("wrong expected tree")
    checks.append("expected_tree")

    if identity["qualification_contract_digest"] != expected_contract_digest:
        raise SnapshotVerificationError("qualification contract digest mismatch")
    checks.append("qualification_contract_digest")

    if identity["phase6b6_source_package_identity"]["root"] != PHASE6B6_ROOT.name:
        raise SnapshotVerificationError("Phase 6B.6 source package identity mismatch")
    checks.append("phase6b6_source_package_identity")

    if identity["v2_source_identity"]["sha256"] != QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]:
        raise SnapshotVerificationError("V2 source identity mismatch")
    checks.append("v2_source_identity")

    if identity["generated_final_campaign_sessions_present"] is not False:
        raise SnapshotVerificationError("generated final campaign sessions present")
    checks.append("no_generated_final_campaign_sessions")

    authority = identity["authority"]
    for field in AUTHORITY_TRUE_FIELDS:
        if authority.get(field) is not True:
            raise SnapshotVerificationError(f"qualification authority flag is false: {field}")
    for field in AUTHORITY_FALSE_FIELDS:
        if authority.get(field) is not False:
            raise SnapshotVerificationError(f"hardware or acquisition authority flag is true: {field}")
    checks.append("no_hardware_authority")

    observed_inventory = snapshot_inventory(snapshot_dir)
    expected_inventory = identity["file_sha256_inventory"]
    if observed_inventory != expected_inventory:
        missing = sorted(set(expected_inventory) - set(observed_inventory))
        extra = sorted(set(observed_inventory) - set(expected_inventory))
        changed = sorted(path for path in set(expected_inventory) & set(observed_inventory) if expected_inventory[path] != observed_inventory[path])
        raise SnapshotVerificationError(
            f"snapshot inventory mismatch missing={missing} extra={extra} changed={changed}"
        )
    checks.append("file_sha256_inventory")

    result = {
        "schema_id": "CAT_CAS_PHASE6B6_SNAPSHOT_VERIFICATION_RESULT_V1",
        "status": "PHASE6B6_SEALED_SNAPSHOT_VERIFICATION_PASS",
        "checked": checks,
        "snapshot_inventory_sha256": digest(observed_inventory),
        "qualification_contract_digest": expected_contract_digest,
    }
    result["snapshot_verification_sha256"] = digest(result)
    return result
