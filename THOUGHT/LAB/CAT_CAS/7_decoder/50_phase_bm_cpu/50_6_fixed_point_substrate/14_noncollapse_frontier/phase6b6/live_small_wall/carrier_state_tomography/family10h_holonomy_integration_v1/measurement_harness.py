#!/usr/bin/env python3
"""Fixture-only prospective measurement harness for the I2D gap contract.

The harness validates record structure and custody rules. It contains no Family 10h
backend, no numerical thresholds, and no path that can emit a physical promotion.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Iterable


PACKAGE_ROOT = Path(__file__).resolve().parent
CONTRACT_PATH = PACKAGE_ROOT / "I2E_MEASUREMENT_HARNESS_CONTRACT.json"


class HarnessValidationError(ValueError):
    """Fail-closed fixture validation error with a stable class token."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def load_contract() -> dict[str, Any]:
    """Load the frozen I2E contract."""

    value = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("I2E contract root must be an object")
    return value


def synthetic_digest(label: str) -> str:
    """Return a stable non-cryptographic fixture label in digest-shaped form."""

    return f"sha256:synthetic-{label}"


def build_state_observation(
    checkpoint: str,
    carrier_id: str,
    offset: float,
) -> dict[str, Any]:
    """Build one complete synthetic multivariate state record."""

    observers = {
        name: {
            "change_to_dirty": offset + float(index + 1),
            "probe_dirty": offset + float((index + 1) * 2),
        }
        for index, name in enumerate(
            ("home_core", "remote_core_A", "remote_core_B", "route_control")
        )
    }
    strata = {
        name: {
            "D_single": offset + float(index) / 10.0,
            "D_local": offset - float(index) / 20.0,
        }
        for index, name in enumerate(
            ("A_only", "A_intersection_B", "B_only", "outside_union")
        )
    }
    return {
        "carrier_id": carrier_id,
        "logical_byte_digest": synthetic_digest(f"bytes-{checkpoint}"),
        "D_single": offset + 1.0,
        "D_local": offset + 0.5,
        "multi_observer_coherence_vector": observers,
        "overlap_strata": strata,
        "timing_probe_vector": {
            "cycles": 1000.0 + offset,
            "probe_coordinate": 10.0 + offset,
            "disturbance_bound_frozen": False,
        },
        "measurement_receipt_id": f"synthetic-measurement-{checkpoint}",
        "synthetic_fixture": True,
    }


def build_valid_synthetic_transaction() -> dict[str, Any]:
    """Build a structurally valid but physically non-promotable fixture."""

    carrier_id = "synthetic-carrier-001"
    checkpoints = (
        "baseline_pre",
        "after_A",
        "after_B",
        "after_forward_commutator",
        "after_output_decoupled",
        "after_inverse_commutator",
    )
    observations = {
        checkpoint: build_state_observation(checkpoint, carrier_id, float(index))
        for index, checkpoint in enumerate(checkpoints)
    }

    operation_roles = (
        "A",
        "B",
        "A_INVERSE_CANDIDATE",
        "B_INVERSE_CANDIDATE",
        "B_INVERSE_CANDIDATE",
        "A_INVERSE_CANDIDATE",
        "B",
        "A",
    )
    receipts = []
    for index, role in enumerate(operation_roles):
        receipts.append(
            {
                "operation_index": index,
                "operator_role": role,
                "operator_instance_id": f"synthetic-{role}-{index}",
                "carrier_id": carrier_id,
                "acting_core": f"synthetic-core-{index % 3}",
                "route_id": f"synthetic-route-{index % 2}",
                "line_set_id": "A" if "A" in role else "B",
                "amplitude": 1,
                "operation_budget": 16,
                "backend_receipt_id": f"synthetic-backend-{index}",
                "byte_digest_before": synthetic_digest(f"op-{index}-before"),
                "byte_digest_after": synthetic_digest(f"op-{index}-before"),
                "byte_digest_preserved": True,
                "state_before_id": checkpoints[min(index, len(checkpoints) - 1)],
                "state_after_id": checkpoints[min(index + 1, len(checkpoints) - 1)],
                "synthetic_backend": True,
                "qualified_inverse": False,
                "destructive_reset": False,
            }
        )

    return {
        "identity": {
            "schema_version": "1.0.0",
            "transaction_id": "i2e-synthetic-fixture-valid",
            "mode": "synthetic_fixture",
            "physical_backend": False,
            "live_authority": False,
        },
        "carrier": {
            "carrier_id": carrier_id,
            "allocation_id": "synthetic-allocation-001",
            "page_identity_digest": synthetic_digest("pages"),
            "line_set_manifest_digest": synthetic_digest("line-sets"),
            "logical_byte_digest_initial": synthetic_digest("bytes-baseline"),
            "public_codewords": ["public-H-like", "public-A-like", "public-B-like"],
            "heldout_rank_claimed": None,
        },
        "source_death": {
            "source_pid": 101,
            "waitpid_exited_zero": True,
            "source_alive": False,
            "open_source_ipc": 0,
            "source_helper_count": 0,
            "preparation_nonce": "prep-001",
            "seal_nonce": "seal-001",
            "challenge_selected_before_source_death": False,
        },
        "challenge": {
            "selected_after_source_death": True,
            "challenge_nonce": "challenge-001",
            "forward_word": ["A", "B", "A_INVERSE_CANDIDATE", "B_INVERSE_CANDIDATE"],
            "inverse_word": ["B_INVERSE_CANDIDATE", "A_INVERSE_CANDIDATE", "B", "A"],
            "word_hash": synthetic_digest("word"),
        },
        "operator_specs": {
            "A": {"role": "A", "qualified_generator": False},
            "B": {"role": "B", "qualified_generator": False},
            "A_INVERSE_CANDIDATE": {
                "role": "A_INVERSE_CANDIDATE",
                "qualified_inverse": False,
            },
            "B_INVERSE_CANDIDATE": {
                "role": "B_INVERSE_CANDIDATE",
                "qualified_inverse": False,
            },
        },
        "operation_receipts": receipts,
        "state_observations": observations,
        "accumulator": {
            "implementation": "synthetic_carrier_coupled_fixture",
            "carrier_coupled": True,
            "reads_public_word": False,
            "carrier_off_output": 0,
            "word_only_replay_output": 0,
            "frozen_output": 8,
            "decoupled_before_inverse": True,
            "output_after_inverse": 8,
            "physical_accumulator": False,
        },
        "restoration": {
            "logical_byte_digest_equal": True,
            "synthetic_reference_state_equivalence": True,
            "physical_state_equivalence": False,
            "independent_output_retained_in_fixture": True,
            "qualified_inverse": False,
            "r2_restoration_claimed": False,
        },
        "environment": {
            field: {
                "classification": "synthetic_fixture_only",
                "measured_physically": False,
            }
            for field in (
                "cache_coherence",
                "page_identity",
                "frequency_policy",
                "temperature",
                "source_process",
                "receiver_process",
                "measurement_registers",
                "accumulator_state",
                "external_bath",
            )
        },
        "thresholds": {
            "frozen": False,
            "equivalence_margin": None,
            "use_floor": None,
            "inverse_deadline": None,
            "null_ceiling": None,
            "error_rate": None,
        },
        "claim_boundary": {
            "fixture_schema_valid": True,
            "physical_backend_implemented": False,
            "physical_generator_pair_established": False,
            "physical_inverse_established": False,
            "physical_accumulator_established": False,
            "r2_restoration_established": False,
            "family10h_holonomy_established": False,
            "small_wall_crossed": False,
            "live_execution_authorized": False,
        },
    }


def require_sections(record: dict[str, Any], sections: Iterable[str]) -> None:
    """Require all top-level transaction sections."""

    missing = [section for section in sections if section not in record]
    if missing:
        raise HarnessValidationError("HARNESS_SCHEMA_INVALID", f"missing sections: {missing}")


def validate_state_observations(record: dict[str, Any], contract: dict[str, Any]) -> None:
    """Validate every required state checkpoint and multivariate component."""

    observations = record["state_observations"]
    carrier_id = record["carrier"]["carrier_id"]
    for checkpoint in contract["required_state_checkpoints"]:
        if checkpoint not in observations:
            raise HarnessValidationError(
                "HARNESS_SCHEMA_INVALID", f"missing checkpoint: {checkpoint}"
            )
        observation = observations[checkpoint]
        if observation.get("readout_only_state_vector") is True:
            raise HarnessValidationError(
                "READOUT_ONLY_STATE_VECTOR", f"readout-only checkpoint: {checkpoint}"
            )
        for component in contract["required_state_components"]:
            if component not in observation:
                raise HarnessValidationError(
                    "HARNESS_SCHEMA_INVALID",
                    f"missing state component {component} at {checkpoint}",
                )
        if observation["carrier_id"] != carrier_id:
            raise HarnessValidationError(
                "CARRIER_ID_SUBSTITUTION", f"carrier changed at {checkpoint}"
            )
        if observation["synthetic_fixture"] is not True:
            raise HarnessValidationError(
                "HARNESS_SCHEMA_INVALID", f"checkpoint not synthetic: {checkpoint}"
            )
        observers = observation["multi_observer_coherence_vector"]
        missing_observers = [
            observer
            for observer in contract["required_observers"]
            if observer not in observers
        ]
        if missing_observers:
            raise HarnessValidationError(
                "MISSING_OBSERVER", f"{checkpoint}: {missing_observers}"
            )
        strata = observation["overlap_strata"]
        missing_strata = [
            stratum
            for stratum in contract["required_overlap_strata"]
            if stratum not in strata
        ]
        if missing_strata:
            raise HarnessValidationError(
                "MISSING_OVERLAP_STRATUM", f"{checkpoint}: {missing_strata}"
            )


def validate_transaction(record: dict[str, Any]) -> dict[str, Any]:
    """Validate one fixture-only transaction and return its nonphysical class."""

    contract = load_contract()
    require_sections(record, contract["required_transaction_sections"])

    identity = record["identity"]
    if identity.get("mode") != "synthetic_fixture":
        raise HarnessValidationError("HARNESS_SCHEMA_INVALID", "mode must be synthetic_fixture")
    if identity.get("physical_backend") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "physical backend may not be true"
        )
    if identity.get("live_authority") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "live authority may not be true"
        )

    death = record["source_death"]
    if death.get("source_alive") is not False or death.get("waitpid_exited_zero") is not True:
        raise HarnessValidationError("SOURCE_STILL_ALIVE", "source death seal failed")
    if death.get("open_source_ipc") != 0 or death.get("source_helper_count") != 0:
        raise HarnessValidationError("OPEN_SOURCE_IPC", "source capability remained")

    challenge = record["challenge"]
    if (
        death.get("challenge_selected_before_source_death") is not False
        or challenge.get("selected_after_source_death") is not True
    ):
        raise HarnessValidationError(
            "EARLY_CHALLENGE_SELECTION", "challenge was not selected after source death"
        )

    roles = set(record["operator_specs"])
    missing_roles = set(contract["required_operation_roles"]) - roles
    if missing_roles:
        raise HarnessValidationError(
            "HARNESS_SCHEMA_INVALID", f"missing operator roles: {sorted(missing_roles)}"
        )
    for role, spec in record["operator_specs"].items():
        if "INVERSE" in role and spec.get("qualified_inverse") is not False:
            raise HarnessValidationError(
                "PHYSICAL_CLAIM_WITHOUT_BACKEND", f"inverse qualified in fixture: {role}"
            )

    carrier_id = record["carrier"]["carrier_id"]
    for receipt in record["operation_receipts"]:
        if receipt.get("carrier_id") != carrier_id:
            raise HarnessValidationError(
                "CARRIER_ID_SUBSTITUTION", "operation receipt changed carrier"
            )
        if receipt.get("destructive_reset") is True or receipt.get("operator_role") == "DESTRUCTIVE_RESET":
            raise HarnessValidationError(
                "DESTRUCTIVE_RESET_AS_INVERSE", "destructive reset entered operator word"
            )
        if receipt.get("synthetic_backend") is not True:
            raise HarnessValidationError(
                "PHYSICAL_CLAIM_WITHOUT_BACKEND", "receipt not synthetic"
            )
        if receipt.get("qualified_inverse") is not False:
            raise HarnessValidationError(
                "PHYSICAL_CLAIM_WITHOUT_BACKEND", "receipt qualified an inverse"
            )

    validate_state_observations(record, contract)

    accumulator = record["accumulator"]
    if accumulator.get("reads_public_word") is not False:
        raise HarnessValidationError(
            "WORD_LABEL_ACCUMULATOR", "accumulator reads the public word"
        )
    if accumulator.get("carrier_coupled") is not True:
        raise HarnessValidationError(
            "WORD_LABEL_ACCUMULATOR", "accumulator is not carrier-coupled"
        )
    if accumulator.get("carrier_off_output") != 0:
        raise HarnessValidationError(
            "CARRIER_OFF_OUTPUT_NONZERO", "carrier-off output must be zero"
        )
    if accumulator.get("word_only_replay_output") != 0:
        raise HarnessValidationError(
            "WORD_LABEL_ACCUMULATOR", "word-only replay output must be zero"
        )
    if accumulator.get("physical_accumulator") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "physical accumulator may not be true"
        )

    thresholds = record["thresholds"]
    if thresholds.get("frozen") is not False:
        raise HarnessValidationError("POSTHOC_THRESHOLD", "fixture thresholds may not freeze")
    for key, value in thresholds.items():
        if key != "frozen" and value is not None:
            raise HarnessValidationError(
                "POSTHOC_THRESHOLD", f"numeric threshold present: {key}"
            )

    for field in contract["required_environment_fields"]:
        if field not in record["environment"]:
            raise HarnessValidationError(
                "HARNESS_SCHEMA_INVALID", f"missing environment field: {field}"
            )

    restoration = record["restoration"]
    if restoration.get("physical_state_equivalence") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "physical state equivalence claimed"
        )
    if restoration.get("qualified_inverse") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "qualified inverse claimed"
        )
    if restoration.get("r2_restoration_claimed") is not False:
        raise HarnessValidationError(
            "PHYSICAL_CLAIM_WITHOUT_BACKEND", "R2 claimed in fixture"
        )

    claims = record["claim_boundary"]
    for field, value in claims.items():
        if field == "fixture_schema_valid":
            if value is not True:
                raise HarnessValidationError(
                    "HARNESS_SCHEMA_INVALID", "fixture_schema_valid must be true"
                )
        elif value is not False:
            raise HarnessValidationError(
                "PHYSICAL_CLAIM_WITHOUT_BACKEND", f"positive claim field: {field}"
            )

    return {
        "classification": (
            "I2E_SYNTHETIC_FIXTURE_VALID__PHYSICAL_PACKAGE_NOT_FREEZE_READY"
        ),
        "transaction_id": identity["transaction_id"],
        "state_checkpoint_count": len(record["state_observations"]),
        "operation_receipt_count": len(record["operation_receipts"]),
        "negative_physical_claims_enforced": True,
        "physical_package_freeze_ready": False,
        "live_execution_authorized": False,
    }


def negative_fixtures() -> dict[str, dict[str, Any]]:
    """Return one mutation for every frozen negative fixture class."""

    base = build_valid_synthetic_transaction()
    fixtures: dict[str, dict[str, Any]] = {}

    fixtures["EARLY_CHALLENGE_SELECTION"] = deepcopy(base)
    fixtures["EARLY_CHALLENGE_SELECTION"]["source_death"][
        "challenge_selected_before_source_death"
    ] = True

    fixtures["SOURCE_STILL_ALIVE"] = deepcopy(base)
    fixtures["SOURCE_STILL_ALIVE"]["source_death"]["source_alive"] = True

    fixtures["OPEN_SOURCE_IPC"] = deepcopy(base)
    fixtures["OPEN_SOURCE_IPC"]["source_death"]["open_source_ipc"] = 1

    fixtures["CARRIER_ID_SUBSTITUTION"] = deepcopy(base)
    fixtures["CARRIER_ID_SUBSTITUTION"]["state_observations"]["after_B"][
        "carrier_id"
    ] = "replacement-carrier"

    fixtures["MISSING_OBSERVER"] = deepcopy(base)
    del fixtures["MISSING_OBSERVER"]["state_observations"]["after_A"][
        "multi_observer_coherence_vector"
    ]["remote_core_B"]

    fixtures["MISSING_OVERLAP_STRATUM"] = deepcopy(base)
    del fixtures["MISSING_OVERLAP_STRATUM"]["state_observations"]["after_A"][
        "overlap_strata"
    ]["A_intersection_B"]

    fixtures["READOUT_ONLY_STATE_VECTOR"] = deepcopy(base)
    fixtures["READOUT_ONLY_STATE_VECTOR"]["state_observations"]["after_A"][
        "readout_only_state_vector"
    ] = True

    fixtures["DESTRUCTIVE_RESET_AS_INVERSE"] = deepcopy(base)
    fixtures["DESTRUCTIVE_RESET_AS_INVERSE"]["operation_receipts"][2][
        "operator_role"
    ] = "DESTRUCTIVE_RESET"
    fixtures["DESTRUCTIVE_RESET_AS_INVERSE"]["operation_receipts"][2][
        "destructive_reset"
    ] = True

    fixtures["WORD_LABEL_ACCUMULATOR"] = deepcopy(base)
    fixtures["WORD_LABEL_ACCUMULATOR"]["accumulator"]["reads_public_word"] = True

    fixtures["CARRIER_OFF_OUTPUT_NONZERO"] = deepcopy(base)
    fixtures["CARRIER_OFF_OUTPUT_NONZERO"]["accumulator"]["carrier_off_output"] = 8

    fixtures["POSTHOC_THRESHOLD"] = deepcopy(base)
    fixtures["POSTHOC_THRESHOLD"]["thresholds"]["frozen"] = True
    fixtures["POSTHOC_THRESHOLD"]["thresholds"]["equivalence_margin"] = 0.1

    fixtures["PHYSICAL_CLAIM_WITHOUT_BACKEND"] = deepcopy(base)
    fixtures["PHYSICAL_CLAIM_WITHOUT_BACKEND"]["claim_boundary"][
        "family10h_holonomy_established"
    ] = True

    return fixtures


def validate_negative_fixtures() -> dict[str, str]:
    """Require every negative fixture to fail with its declared class."""

    contract = load_contract()
    fixtures = negative_fixtures()
    expected_classes = contract["negative_fixture_classes"]
    if list(fixtures) != expected_classes:
        raise AssertionError("negative fixture order or class set drifted")

    results: dict[str, str] = {}
    for expected_code, fixture in fixtures.items():
        try:
            validate_transaction(fixture)
        except HarnessValidationError as exc:
            if exc.code != expected_code:
                raise AssertionError(
                    f"fixture {expected_code} failed as {exc.code}: {exc.detail}"
                ) from exc
            results[expected_code] = exc.code
        else:
            raise AssertionError(f"negative fixture unexpectedly passed: {expected_code}")
    return results


def run_harness_self_test() -> dict[str, Any]:
    """Validate one good fixture and every declared negative fixture."""

    valid = validate_transaction(build_valid_synthetic_transaction())
    negative = validate_negative_fixtures()
    contract = load_contract()
    acceptance = contract["harness_acceptance"]

    assert acceptance["transaction_schema_complete"] is True
    assert acceptance["valid_synthetic_fixture_available"] is True
    assert acceptance["negative_fixture_count"] == len(negative)
    assert acceptance["all_negative_fixtures_rejected"] is True
    assert acceptance["physical_backend_present"] is False
    assert acceptance["numerical_thresholds_frozen"] is False
    assert acceptance["physical_package_freeze_ready"] is False

    decision = contract["i2e_decision"]
    assert decision["result"] == (
        "I2E_PROSPECTIVE_MEASUREMENT_HARNESS_SKELETON_COMPLETE__"
        "FIXTURE_ONLY__PHYSICAL_PACKAGE_NOT_FREEZE_READY"
    )
    assert decision["schema_freeze_allowed"] is True
    assert decision["physical_package_freeze_ready"] is False
    assert decision["live_execution_authorized"] is False

    return {
        "schema": "FAMILY10H_I2E_MEASUREMENT_HARNESS_SELF_TEST_V1",
        "valid_fixture": valid,
        "negative_fixture_results": negative,
        "decision": decision["result"],
        "next_gate": decision["next_gate"],
        "claim_ceiling": contract["claim_ceiling"],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(run_harness_self_test(), indent=2, sort_keys=True))
