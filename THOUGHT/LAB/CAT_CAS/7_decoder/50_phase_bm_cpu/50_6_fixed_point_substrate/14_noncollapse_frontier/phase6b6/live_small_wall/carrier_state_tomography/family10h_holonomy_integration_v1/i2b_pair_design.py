#!/usr/bin/env python3
"""Validate the I2B second-generator and inverse candidate design.

The validator binds the design to retained checkpoints, proves only synthetic line-set
geometry, and fails closed on every physical H1/H2 claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


PACKAGE_ROOT = Path(__file__).resolve().parent
DESIGN_PATH = PACKAGE_ROOT / "I2B_PAIR_DESIGN.json"


def find_repo_root() -> Path:
    """Locate the repository root in full or sparse Git checkouts."""

    for parent in Path(__file__).resolve().parents:
        has_repo_tree = (parent / "THOUGHT").is_dir()
        has_repo_marker = (parent / ".git").exists() or (parent / ".github").is_dir()
        if has_repo_tree and has_repo_marker:
            return parent
    raise RuntimeError("repository root not found")


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected object: {path}")
    return value


def git_blob_sha(path: Path) -> str:
    """Return the canonical Git blob SHA-1 for ``path``."""

    payload = path.read_bytes()
    return hashlib.sha1(  # noqa: S324 - Git object identity
        f"blob {len(payload)}\0".encode("ascii") + payload
    ).hexdigest()


def validate_source_blobs(repo_root: Path, design: dict[str, Any]) -> dict[str, str]:
    """Verify all retained checkpoint and interface identities."""

    verified: dict[str, str] = {}
    for source_id, source in design["source_blobs"].items():
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


def validate_retained_attacks(repo_root: Path, design: dict[str, Any]) -> dict[str, bool]:
    """Check retained results that kill the obvious pair constructions."""

    source = design["source_blobs"]
    read = lambda source_id: load_json(repo_root / source[source_id]["path"])  # noqa: E731

    coherence = read("coherence_operator_checkpoint")
    assert coherence["classification"] == "CONTROLLED_COHERENCE_STATE_FOUND"
    assert coherence["geometry"]["all_operator_windows_restored"] is True
    assert coherence["acceptance"]["store_same_value_change_to_dirty_moved"] is True
    assert "coherence holonomy" in coherence["claim_exclusions"]

    path_rw = read("path_rw_checkpoint")
    assert path_rw["classification"] == "PATH_RW_OBSERVE_NOT_ESTABLISHED"
    assert path_rw["acceptance"]["sign_reversal"] is False
    assert path_rw["acceptance"]["controls_small"] is False
    assert path_rw["acceptance"]["path_dependence_pilot"] is False
    assert path_rw["areas_cycles_normalized"]["forward"] > 0.0
    assert path_rw["areas_cycles_normalized"]["reverse"] > 0.0

    route = read("route_operator_checkpoint")
    assert route["classification"] == "ROUTE_STABLE_CONTROLLED_COHERENCE_OPERATOR_FOUND"
    assert route["acceptance"]["route45_controlled_state_found"] is True
    assert route["acceptance"]["route23_controlled_state_found"] is True
    assert "route-stable rather than route-selective" in route["interpretation"]["surviving_wall"]

    route_state = read("route_state_checkpoint")
    assert route_state["classification"] == "ROUTE_STATE_NOT_ESTABLISHED"
    assert route_state["acceptance"]["direct_route_moved"] is False
    assert route_state["acceptance"]["swapped_route_moved"] is True
    assert route_state["acceptance"]["route_state_response"] is False

    interface = read("i2a_runtime_interface")
    assert interface["live_backend_implemented"] is False
    assert interface["acceptance"]["h1_generator_established"] is False
    assert interface["acceptance"]["h1_generator_pair_established"] is False

    return {
        "coherence_primitive_retained": True,
        "read_store_rectangle_rejected": True,
        "route_instance_pair_rejected": True,
        "route_state_axis_rejected": True,
        "i2a_physical_backend_absent": True,
    }


def validate_partial_overlap_fixture(design: dict[str, Any]) -> dict[str, int]:
    """Validate the design-only partial-overlap line-set geometry."""

    fixture = design["design_only_candidate"]["line_set_law"]["synthetic_fixture"]
    line_a = set(fixture["line_set_A"])
    line_b = set(fixture["line_set_B"])
    intersection = line_a & line_b

    if len(line_a) != len(line_b):
        raise AssertionError("candidate line sets must have equal cardinality")
    if not intersection:
        raise AssertionError("candidate line sets may not be disjoint")
    if line_a == line_b:
        raise AssertionError("candidate line sets may not be identical")
    if len(intersection) != fixture["intersection_size"]:
        raise AssertionError("intersection-size receipt mismatch")
    if fixture["fixture_is_physical_schedule"] is not False:
        raise AssertionError("synthetic fixture may not be labeled physical")

    disjoint_control = set(range(100, 116))
    identical_control = set(line_a)
    assert not (line_a & disjoint_control)
    assert line_a == identical_control

    return {
        "line_set_size": len(line_a),
        "intersection_size": len(intersection),
        "union_size": len(line_a | line_b),
    }


def validate_claim_boundary(design: dict[str, Any]) -> dict[str, Any]:
    """Enforce zero physical pair/inverse promotion."""

    attacks = design["attacked_pair_families"]
    if len(attacks) != 7:
        raise AssertionError("unexpected attacked-family count")
    if any("REJECTED" not in attack["status"] for attack in attacks):
        raise AssertionError("every obvious pair family must remain rejected")

    candidate = design["design_only_candidate"]
    assert candidate["physical_support_status"] == "NOT_ESTABLISHED"
    assert candidate["topology_assignment_frozen"] is False
    assert candidate["operators"]["A"]["inverse_established"] is False
    assert candidate["operators"]["B"]["inverse_established"] is False

    admission = design["synthetic_admission"]
    assert admission["pair_geometry_fixture_valid"] is True
    for field in (
        "physical_generator_A_established",
        "physical_generator_B_established",
        "inverse_A_established",
        "inverse_B_established",
        "h1_pair_established",
        "h2_inverse_laws_established",
    ):
        if admission[field] is not False:
            raise AssertionError(f"forbidden positive admission field: {field}")

    decision = design["i2b_decision"]
    assert decision["result"] == (
        "I2B_SECOND_GENERATOR_AND_INVERSE_DESIGN_COMPLETE__NO_PHYSICALLY_SUPPORTED_PAIR"
    )
    assert decision["design_grammar_freeze_allowed"] is True
    assert decision["physical_pair_freeze_allowed"] is False
    assert decision["live_execution_authorized"] is False
    assert decision["next_gate"] == "I2C_SYNTHETIC_BIDIRECTIONAL_OWNERSHIP_MODEL"

    return {
        "attacked_family_count": len(attacks),
        "candidate_id": candidate["id"],
        "physical_pair_supported": False,
        "next_gate": decision["next_gate"],
    }


def run_pair_design_audit() -> dict[str, Any]:
    """Run the complete source-bound I2B audit."""

    repo_root = find_repo_root()
    design = load_json(DESIGN_PATH)
    return {
        "design_id": design["design_id"],
        "source_identity_verified": validate_source_blobs(repo_root, design),
        "retained_attack_checks": validate_retained_attacks(repo_root, design),
        "synthetic_geometry": validate_partial_overlap_fixture(design),
        "claim_boundary": validate_claim_boundary(design),
        "decision": design["i2b_decision"]["result"],
        "claim_ceiling": design["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_pair_design_audit(), indent=2, sort_keys=True))
