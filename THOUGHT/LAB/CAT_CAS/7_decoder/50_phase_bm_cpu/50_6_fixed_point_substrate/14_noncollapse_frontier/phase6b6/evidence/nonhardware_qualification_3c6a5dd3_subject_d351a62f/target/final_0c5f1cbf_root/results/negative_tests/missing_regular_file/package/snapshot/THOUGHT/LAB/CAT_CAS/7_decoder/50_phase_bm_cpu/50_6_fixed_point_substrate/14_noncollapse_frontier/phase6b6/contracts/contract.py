"""Frozen Phase 6B.6 software-entry contract constants."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any


REBOOT_BLOCKS = tuple(f"b{i}" for i in range(6))
ROUTES = ("v4s5", "v2s3")
ROUTE_CORES = {"v4s5": (4, 5), "v2s3": (2, 3)}
TRAIN_BLOCKS = ("b0", "b1", "b2")
VAL_BLOCKS = ("b3",)
TEST_BLOCKS = ("b4", "b5")

READ_HZ = 8000
SLOT_S = 0.5
NOMINAL_SAMPLES_PER_SLOT = 4000
PIN_KHZ = 1600000
TEMPERATURE_VETO_C = 68.0
AUTOMATIC_RETRY = False

PREAMBLE_SLOTS = 96
PREPARED_ORDER_SLOTS = 360
TRAJECTORY_SLOTS = 384
TAIL_DRIFT_SLOTS = 24
SLOTS_PER_SESSION = 864
SESSION_COUNT = 12
TOTAL_SLOTS = SESSION_COUNT * SLOTS_PER_SESSION
NOMINAL_CAMPAIGN_SAMPLE_COUNT = TOTAL_SLOTS * NOMINAL_SAMPLES_PER_SLOT

TONE_COUNT = 12
AMPLITUDE_LEVELS = (0, 1, 2)
SIGNS = (-1, 1)
ORDER_FAMILIES = ("FWD", "REV", "RND1", "RND2", "ORDER_LABEL_SHAM")
ORDER_ARRAYS = {
    "FWD": tuple(range(12)),
    "REV": tuple(reversed(range(12))),
    "RND1": (3, 10, 1, 8, 5, 0, 11, 6, 2, 9, 4, 7),
    "RND2": (8, 2, 11, 4, 0, 7, 3, 10, 5, 1, 9, 6),
}
HORIZONS = (1, 2, 4, 8)
DELAY_CANDIDATES = (2, 4, 8, 16)
REGULARIZATION_LADDER = (0.0, 1e-6, 1e-4, 1e-2, 1.0)
O4_FIXED_LIFTS = (
    "z",
    "conjugate_z",
    "abs_z_squared",
    "z_tensor_u",
    "exp_i_executed_phase",
    "fixed_quadratic_cross_terms",
)

MEASURED_RESPONSE_FIELDS = ("lockin_I", "lockin_Q", "ring_osc_period")
EXECUTED_CONTROL_FIELDS = (
    "drive_on",
    "executed_mode",
    "amplitude_level",
    "phase_action",
    "physical_tone_index",
    "executed_order_family",
    "executed_order_position",
    "executed_codeword_signs",
    "codeword_source_index",
    "codeword_sign",
)
DECLARATION_FIELDS = (
    "declared_mode",
    "declared_amplitude_level",
    "declared_phase_action",
    "declared_physical_tone_index",
    "order_control_family",
    "declared_order_family",
    "declared_order_position",
)
CONTEXT_FIELDS = (
    "route",
    "sender_core",
    "receiver_core",
    "reboot_block",
    "session_chronology",
    "session_tsc_origin",
    "actual_start_tsc",
    "actual_end_tsc",
    "measured_tsc_hz",
    "empirical_sample_rate",
    "temperature",
    "P_state",
    "capture_quality",
)
SESSION_GAUGE_FIELDS = (
    "complex_anchor_alpha",
    "amplitude_floor",
    "preamble_drift_estimate",
    "local_idle_covariance",
)
PROHIBITED_MEASURED_STATE_FIELDS = (
    "session_id",
    "route",
    "target_label",
    "declared_mode",
    "order_control_family",
    "declared_order_family",
    "declared_order_position",
    "session_chronology",
    "future_value",
    "public_candidate_identity",
    "sender_core",
    "receiver_core",
)

AUTHORITY = {
    "schema_id": "CAT_CAS_PHASE6B6_SOFTWARE_ENTRY_RUNTIME_AUTHORITY_V1",
    "architecture_review": 4588082595,
    "software_entry_review": 4588098104,
    "project_owner_decision": "APPROVE_PHASE6B6_SOFTWARE_ENTRY_ONLY",
    "implementation_authorized": True,
    "software_qualification_authorized": True,
    "non_hardware_target_qualification_authorized": True,
    "hardware_ran": False,
    "authorization_artifact_created": False,
    "calibration_authorized": False,
    "scientific_acquisition_authorized": False,
    "restoration_authorized": False,
    "target_coupling_authorized": False,
    "small_wall_authorized": False,
    "automatic_retry": AUTOMATIC_RETRY,
}

CLAIM_CEILING = {
    "maximum_positive_claim": "EMPIRICAL_PREDICTIVE_OBSERVABILITY_OF_TESTED_MEASURED_EQUIVALENCE_CLASS",
    "forbidden_claims": (
        "complete physical observability",
        "physical HoloGeometry",
        "inverse physical dynamics",
        "physical restoration",
        "target coupling",
        "orientation recovery",
        "fold-odd invariant recovery",
        "Small Wall crossing",
    ),
}

CONTRACT: dict[str, Any] = {
    "schema_id": "CAT_CAS_PHASE6B6_SCIENTIFIC_CONTRACT_V1",
    "reboot_blocks": REBOOT_BLOCKS,
    "routes": ROUTES,
    "route_cores": ROUTE_CORES,
    "splits": {"train": TRAIN_BLOCKS, "validation": VAL_BLOCKS, "test": TEST_BLOCKS},
    "session_count": SESSION_COUNT,
    "slots_per_session": SLOTS_PER_SESSION,
    "total_slots": TOTAL_SLOTS,
    "read_hz": READ_HZ,
    "slot_s": SLOT_S,
    "nominal_samples_per_slot": NOMINAL_SAMPLES_PER_SLOT,
    "nominal_campaign_sample_count": NOMINAL_CAMPAIGN_SAMPLE_COUNT,
    "pin_khz": PIN_KHZ,
    "temperature_veto_c": TEMPERATURE_VETO_C,
    "automatic_retry": AUTOMATIC_RETRY,
    "order_arrays": ORDER_ARRAYS,
    "order_families": ORDER_FAMILIES,
    "horizons": HORIZONS,
    "delay_candidates": DELAY_CANDIDATES,
    "regularization_ladder": REGULARIZATION_LADDER,
    "o4_fixed_lifts": O4_FIXED_LIFTS,
    "measured_response_fields": MEASURED_RESPONSE_FIELDS,
    "executed_control_fields": EXECUTED_CONTROL_FIELDS,
    "declaration_fields": DECLARATION_FIELDS,
    "context_fields": CONTEXT_FIELDS,
    "session_gauge_fields": SESSION_GAUGE_FIELDS,
    "prohibited_measured_state_fields": PROHIBITED_MEASURED_STATE_FIELDS,
    "authority": AUTHORITY,
    "claim_ceiling": CLAIM_CEILING,
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


try:
    from .v2_interface import PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT, QUALIFIED_V2_SOURCE, TONE_CODEWORD_TABLE
except ImportError:  # pragma: no cover
    from v2_interface import PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT, QUALIFIED_V2_SOURCE, TONE_CODEWORD_TABLE  # type: ignore

CONTRACT["qualified_v2_source"] = QUALIFIED_V2_SOURCE
CONTRACT["imported_tone_codeword_table"] = TONE_CODEWORD_TABLE
CONTRACT["pre_acquisition_v2_equivalence_requirement"] = PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT


def contract_manifest() -> dict[str, Any]:
    manifest = deepcopy(CONTRACT)
    manifest["contract_sha256"] = digest(CONTRACT)
    return manifest


def split_for_block(reboot_block: str) -> str:
    if reboot_block in TRAIN_BLOCKS:
        return "train"
    if reboot_block in VAL_BLOCKS:
        return "validation"
    if reboot_block in TEST_BLOCKS:
        return "test"
    raise ValueError(f"unknown reboot block: {reboot_block}")


def route_order_for_block(block_index: int) -> tuple[str, str]:
    if block_index % 2 == 0:
        return ("v4s5", "v2s3")
    return ("v2s3", "v4s5")


def order_family_sequence(reboot_block: str, route: str) -> tuple[str, ...]:
    block_index = int(reboot_block[1:])
    route_index = ROUTES.index(route)
    rotation = (block_index + 2 * route_index) % len(ORDER_FAMILIES)
    sequence = ORDER_FAMILIES[rotation:] + ORDER_FAMILIES[:rotation]
    if reboot_block == "b5":
        sequence = tuple(reversed(sequence))
    return sequence


def declared_and_executed_order(family: str, reboot_block: str) -> tuple[str, tuple[int, ...], str, tuple[int, ...]]:
    if family != "ORDER_LABEL_SHAM":
        return family, ORDER_ARRAYS[family], family, ORDER_ARRAYS[family]
    block_index = int(reboot_block[1:])
    if block_index % 2 == 0:
        return "RND2", ORDER_ARRAYS["RND2"], "RND1", ORDER_ARRAYS["RND1"]
    return "RND1", ORDER_ARRAYS["RND1"], "RND2", ORDER_ARRAYS["RND2"]
