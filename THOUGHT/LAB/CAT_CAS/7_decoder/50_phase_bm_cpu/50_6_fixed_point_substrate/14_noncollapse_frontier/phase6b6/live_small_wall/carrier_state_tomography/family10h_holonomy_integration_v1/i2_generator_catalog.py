#!/usr/bin/env python3
"""Validate the receiver-side generator candidate catalog.

This audit classifies already committed runtime operations. It never executes the
Family 10h runtime, contacts the target, or treats operation labels as physical
transport evidence.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
CATALOG_PATH = PACKAGE_ROOT / "I2_GENERATOR_CATALOG.json"


def find_repo_root() -> Path:
    """Find the repository root from this package location."""

    for parent in Path(__file__).resolve().parents:
        if (parent / "THOUGHT").is_dir() and (parent / ".github").is_dir():
            return parent
    raise RuntimeError("repository root not found")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object root: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    """Return the canonical Git blob SHA-1 for ``path``."""

    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity, not cryptography
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def require_markers(path: Path, *markers: str) -> None:
    """Require literal source markers without interpreting unbound code."""

    text = path.read_text(encoding="utf-8")
    missing = [marker for marker in markers if marker not in text]
    if missing:
        raise AssertionError(f"missing markers in {path}: {missing}")


def validate_source_blobs(repo_root: Path, catalog: dict[str, Any]) -> dict[str, str]:
    """Bind the catalog to the exact audited runtime and checkpoint blobs."""

    verified: dict[str, str] = {}
    for source_id, source in catalog["source_blobs"].items():
        path = repo_root / source["path"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = git_blob_sha(path)
        if actual != source["blob_sha"]:
            raise AssertionError(
                f"source drift for {source_id}: {actual} != {source['blob_sha']}"
            )
        verified[source_id] = actual
    return verified


def validate_runtime_classification(repo_root: Path, catalog: dict[str, Any]) -> dict[str, bool]:
    """Check the code facts that separate preparation, readout, reset, and primitive."""

    sources = catalog["source_blobs"]
    path = lambda source_id: repo_root / sources[source_id]["path"]  # noqa: E731

    require_markers(
        path("base_runtime_h"),
        "int f10_carrier_prepare(F10CarrierPreparation prep, F10CarrierState *state);",
        "uint64_t f10_carrier_query_mapped(const F10CarrierState *state",
    )
    require_markers(
        path("base_runtime_c"),
        "if (!f10_carrier_prepare(prep, state))",
        "if (waitpid(child, &status, 0) != child)",
        "query = fields[13];",
        "(void)f10_carrier_query_mapped(state, query, map_variant);",
        "query_A_then_B",
        "query_B_then_A",
        "munmap(shared, sizeof(F10CarrierShared));",
        "source_alive_during_query\\\":false",
    )
    require_markers(
        path("relation_runtime_h"),
        "RELATION_SPATIAL_R0 = 0",
        "RELATION_SPATIAL_R1 = 1",
        "RELATION_SPATIAL_CONTROL_RESET_DOUBLE_FLUSH_PRIMARY",
        "RELATION_SPATIAL_CONTROL_RESTORE_OPPOSITE_PRIMARY",
        "RELATION_SPATIAL_CONTROL_COMPOSE_R0_R1_PRIMARY",
    )
    require_markers(
        path("relation_runtime_c"),
        "static int relation_spatial_prepare_composition(",
        "shared->preparation_ok = relation_spatial_prepare(prep, &shared->state);",
        "while (!shared->release_source)",
        "source_alive_at_pair_measurement = 0;",
        "flush_state_lines(&shared->state);",
        "relation_spatial_prefault(&shared->state);",
        "if (!relation_spatial_prepare(restore, &shared->state))",
        "source_alive_during_spatial_pair_probe",
        "source_dead_then_source_cpu_mutate_opposite_relation_before_spatial_pair_probe",
    )
    require_markers(
        path("coherence_worker_c"),
        "OP_REMOTE_READ_SHARED = 2",
        "OP_REMOTE_STORE_SAME_VALUE = 7",
        "enum coherence_operator",
    )

    checkpoint = load_json(path("coherence_checkpoint"))
    assert checkpoint["classification"] == "CONTROLLED_COHERENCE_STATE_FOUND"
    assert checkpoint["geometry"]["all_operator_windows_restored"] is True
    assert checkpoint["acceptance"]["store_same_value_change_to_dirty_moved"] is True
    assert checkpoint["acceptance"]["carrier_restored"] is True
    assert "coherence holonomy" in checkpoint["claim_exclusions"]
    assert "Small Wall crossing" in checkpoint["claim_exclusions"]
    assert checkpoint["contrast_counts"]["change_to_dirty"]["remote_store_same_value"] == 2104

    return {
        "base_prepare_is_source_child_operation": True,
        "base_queries_execute_after_waitpid": True,
        "base_queries_are_readout_api": True,
        "base_runtime_unmaps_state_after_one_query": True,
        "relation_composition_is_source_preparation": True,
        "relation_primary_probe_can_keep_source_alive": True,
        "post_death_reset_controls_are_destructive": True,
        "post_death_mutation_reprepares_full_state": True,
        "remote_store_same_value_is_confirmed_legacy_primitive": True,
        "remote_store_same_value_not_bound_to_successor_word_runtime": True,
    }


def validate_catalog_logic(catalog: dict[str, Any]) -> dict[str, Any]:
    """Enforce the fail-closed generator admission decision."""

    families = catalog["operation_families"]
    summary = catalog["catalog_summary"]
    decision = catalog["i2_decision"]

    if len(families) != summary["operation_family_count"]:
        raise AssertionError("operation family count mismatch")
    admitted = [family for family in families if family["h1_admissible"] is True]
    if admitted:
        raise AssertionError(f"unexpected H1-admissible families: {admitted}")
    if summary["h1_admissible_generator_count"] != 0:
        raise AssertionError("admissible generator count must remain zero")
    if summary["h1_admissible_generator_pair_count"] != 0:
        raise AssertionError("admissible generator pair count must remain zero")

    extraction = [family for family in families if family.get("extraction_candidate")]
    if [family["id"] for family in extraction] != ["remote_store_same_value"]:
        raise AssertionError("exactly remote_store_same_value may be an extraction candidate")
    if extraction[0]["inverse_candidate"] != "none qualified":
        raise AssertionError("the extraction primitive may not inherit an inverse")

    forbidden_admitted_statuses = {
        "SOURCE_ONLY_PREPARATION_NOT_GENERATOR",
        "SOURCE_AUTHORED_COMPOSITION_NOT_H3_COMPOSITION",
        "POST_SOURCE_CONTROL_NOT_INVERSE",
        "POST_SOURCE_DESTRUCTIVE_REPREPARATION_NOT_GENERATOR",
    }
    for family in families:
        if family["status"] in forbidden_admitted_statuses and family["h1_admissible"]:
            raise AssertionError(f"forbidden admission: {family['id']}")

    assert decision["result"] == (
        "I2_RECEIVER_SIDE_GENERATOR_CATALOG_COMPLETE__NO_H1_ADMISSIBLE_GENERATOR_PAIR"
    )
    assert decision["h1_passed"] is False
    assert decision["physical_transport_candidate_freeze_allowed"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2A_POST_SOURCE_OPERATOR_RUNTIME_EXTRACTION_DESIGN"

    return {
        "family_count": len(families),
        "admitted_generator_count": len(admitted),
        "admitted_pair_count": summary["h1_admissible_generator_pair_count"],
        "extraction_candidate": extraction[0]["id"],
        "next_gate": decision["next_gate"],
    }


def run_catalog_audit() -> dict[str, Any]:
    """Run the complete read-only I2 catalog validation."""

    repo_root = find_repo_root()
    catalog = load_json(CATALOG_PATH)
    return {
        "catalog_id": catalog["catalog_id"],
        "source_identity_verified": validate_source_blobs(repo_root, catalog),
        "runtime_classification_checks": validate_runtime_classification(repo_root, catalog),
        "catalog_checks": validate_catalog_logic(catalog),
        "decision": catalog["i2_decision"]["result"],
        "claim_ceiling": catalog["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_catalog_audit(), indent=2, sort_keys=True))
