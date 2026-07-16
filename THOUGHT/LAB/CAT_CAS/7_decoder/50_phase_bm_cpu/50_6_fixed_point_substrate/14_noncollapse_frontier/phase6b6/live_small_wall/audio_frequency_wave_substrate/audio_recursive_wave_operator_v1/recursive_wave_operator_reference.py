#!/usr/bin/env python3
"""Deterministic R1 candidate for complete-tree temporal phase recurrence.

Each native step creates a new phase node whose children are the entire previous
recursive phase tree and an entire drive tree. The previous state is not decoded to a
spin, score, energy, FFT magnitude, or flat coefficient bank. It remains an executable
subtree of the next state.

This is an ordinary-software architecture candidate. It establishes no catalytic-loop,
Ising, physical-wave, optimization, restoration, or Wall claim.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import platform
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
R0_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_phase_tree_v1"
    / "recursive_phase_tree_reference.py"
)

_spec = importlib.util.spec_from_file_location("catcas_recursive_phase_tree_r0", R0_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load the established R0 recursive phase-tree reference")
r0 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r0
_spec.loader.exec_module(r0)

GENERATOR_ID = "recursive_wave_operator_reference_v1"
CLAIM_CEILING = "SOFTWARE_COMPLETE_TREE_TEMPORAL_RECURRENCE_REFERENCE_ONLY"
ESTABLISHED_TOKEN = "AUDIO_RECURSIVE_WAVE_OPERATOR_ESTABLISHED"
RESULT_SCHEMA = "recursive_wave_operator_candidate_result_v1"
STEP_SCHEMA = "recursive_wave_temporal_step_v1"
MAX_STEPS = 6
UNIT_MODULUS_TOL = 1e-12
NONIDENTITY_TOL = 1e-6
ORDER_RESPONSE_MAX = 0.99


@dataclass(frozen=True)
class TemporalStepSpec:
    """Prospective parameters for one complete-tree temporal lift."""

    step_index: int
    root_id: str
    carrier_frequency_hz: float
    root_phase_rad: float
    state_modulation_index: float
    drive_modulation_index: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.step_index, bool)
            or not isinstance(self.step_index, int)
            or not 1 <= self.step_index <= MAX_STEPS
        ):
            raise ValueError("step_index is outside the bounded R1 envelope")
        expected_prefix = f"time{self.step_index}."
        if not isinstance(self.root_id, str) or not self.root_id.startswith(
            expected_prefix
        ):
            raise ValueError("root_id must be scoped to its step index")

        numeric = {
            "carrier_frequency_hz": self.carrier_frequency_hz,
            "root_phase_rad": self.root_phase_rad,
            "state_modulation_index": self.state_modulation_index,
            "drive_modulation_index": self.drive_modulation_index,
        }
        for name, value in numeric.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")

        if not 0.0 < float(self.carrier_frequency_hz) < r0.SAMPLE_RATE_HZ / 2:
            raise ValueError("carrier_frequency_hz must be inside (0, Nyquist)")
        if abs(float(self.root_phase_rad)) > 2.0 * math.pi:
            raise ValueError("root_phase_rad exceeds the frozen envelope")
        if not 0.0 < float(self.state_modulation_index) <= 2.0:
            raise ValueError("state_modulation_index exceeds the frozen envelope")
        if not 0.0 < float(self.drive_modulation_index) <= 2.0:
            raise ValueError("drive_modulation_index exceeds the frozen envelope")

    def document(self) -> dict[str, Any]:
        return {"schema": STEP_SCHEMA, **asdict(self)}

    def digest(self) -> str:
        return hashlib.sha256(
            r0.canonical_json_bytes(self.document(), pretty=False)
        ).hexdigest()


@dataclass(frozen=True)
class TemporalStepReceipt:
    """Role and ancestry binding for one native step."""

    spec_digest: str
    result_digest: str
    state_digest: str
    state_root_id: str
    drive_digest: str
    drive_root_id: str

    def document(self) -> dict[str, str]:
        return asdict(self)


def node_ids(beam: Any) -> set[str]:
    return {node.node_id for node in beam.root.walk()}


def require_pre_ising_orientation(beam: Any, label: str) -> None:
    if beam.global_spin_phase_rad != 0.0:
        raise ValueError(
            f"{label} global orientation must remain zero before the Ising package"
        )


def temporal_step(
    state: Any, drive: Any, spec: TemporalStepSpec
) -> tuple[Any, TemporalStepReceipt]:
    """Embed the complete state and drive trees under one new phase root."""

    require_pre_ising_orientation(state, "state")
    require_pre_ising_orientation(drive, "drive")
    state_ids = node_ids(state)
    drive_ids = node_ids(drive)
    if state_ids & drive_ids:
        raise ValueError("state and drive node identities must be disjoint")
    if spec.root_id in state_ids | drive_ids:
        raise ValueError("step root identity collides with a child tree")

    root = r0.PhaseNode(
        spec.root_id,
        spec.carrier_frequency_hz,
        spec.root_phase_rad,
        (
            r0.PhaseEdge(spec.state_modulation_index, state.root),
            r0.PhaseEdge(spec.drive_modulation_index, drive.root),
        ),
    )
    result = r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)
    receipt = TemporalStepReceipt(
        spec_digest=spec.digest(),
        result_digest=result.digest(),
        state_digest=state.digest(),
        state_root_id=state.root.node_id,
        drive_digest=drive.digest(),
        drive_root_id=drive.root.node_id,
    )
    return result, receipt


def extract_predecessor(result: Any, receipt: TemporalStepReceipt) -> Any:
    """Recover the exact embedded predecessor identified by the frozen receipt."""

    if result.digest() != receipt.result_digest:
        raise ValueError("result digest does not match the temporal receipt")
    candidates = [
        edge.child
        for edge in result.root.children
        if edge.child.node_id == receipt.state_root_id
    ]
    if len(candidates) != 1:
        raise ValueError("receipt does not identify exactly one state subtree")
    predecessor = r0.RecursivePhaseBeam(
        root=candidates[0], global_spin_phase_rad=0.0
    )
    if predecessor.digest() != receipt.state_digest:
        raise ValueError("predecessor digest does not match the temporal receipt")
    return predecessor


def trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
) -> tuple[list[Any], list[TemporalStepReceipt]]:
    if not drives or len(drives) != len(specs):
        raise ValueError("trajectory requires equal nonempty drive and spec sequences")
    state = initial
    states = [state]
    receipts: list[TemporalStepReceipt] = []
    for drive, spec in zip(drives, specs, strict=True):
        state, receipt = temporal_step(state, drive, spec)
        states.append(state)
        receipts.append(receipt)
    return states, receipts


def drive_tree(step_index: int, variant: int = 0) -> Any:
    """Create a small, step-scoped complete drive tree."""

    base = f"drive{step_index}v{variant}"
    leaf = r0.PhaseNode(
        f"{base}.leaf",
        23.0 + 14.0 * step_index + 2.0 * variant,
        phase_rad=0.11 * (step_index + 1) + 0.03 * variant,
    )
    root = r0.PhaseNode(
        f"{base}.root",
        83.0 + 48.0 * step_index + 5.0 * variant,
        phase_rad=-0.07 * (step_index + 1) + 0.02 * variant,
        children=(r0.PhaseEdge(0.29 + 0.04 * step_index, leaf),),
    )
    return r0.RecursivePhaseBeam(root=root, global_spin_phase_rad=0.0)


def step_spec(step_index: int) -> TemporalStepSpec:
    return TemporalStepSpec(
        step_index=step_index,
        root_id=f"time{step_index}.root",
        carrier_frequency_hz=1601.0 + 137.0 * step_index,
        root_phase_rad=0.09 * step_index,
        state_modulation_index=0.63 + 0.03 * step_index,
        drive_modulation_index=0.41 + 0.02 * step_index,
    )


def collapsed_flat_trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
    t: np.ndarray,
) -> np.ndarray:
    """Declared baseline that retains only a rendered flat waveform between steps."""

    state_wave = r0.flat_multitone_replacement(initial, t)
    for drive, spec in zip(drives, specs, strict=True):
        drive_wave = r0.flat_multitone_replacement(drive, t)
        phase = (
            2.0 * math.pi * spec.carrier_frequency_hz * t
            + spec.root_phase_rad
            + spec.state_modulation_index * np.sin(np.angle(state_wave))
            + spec.drive_modulation_index * np.sin(np.angle(drive_wave))
        )
        state_wave = np.exp(1j * phase)
    return state_wave


def collapsed_spin_trajectory(
    initial: Any,
    drives: Sequence[Any],
    specs: Sequence[TemporalStepSpec],
    t: np.ndarray,
) -> np.ndarray:
    """Declared baseline that collapses each complete state to one global sign."""

    state_wave = initial.render(t)
    for drive, spec in zip(drives, specs, strict=True):
        spin = 1.0 if float(np.real(np.mean(state_wave))) >= 0.0 else -1.0
        drive_wave = drive.render(t)
        phase = (
            2.0 * math.pi * spec.carrier_frequency_hz * t
            + spec.root_phase_rad
            + (0.0 if spin > 0.0 else math.pi)
            + spec.drive_modulation_index * np.sin(np.angle(drive_wave))
        )
        state_wave = np.exp(1j * phase)
    return state_wave


def run_reference_tests() -> dict[str, Any]:
    t = r0.sample_times()
    initial = r0.hierarchy_a()
    drives = [drive_tree(index) for index in range(1, 4)]
    specs = [step_spec(index) for index in range(1, 4)]
    states, receipts = trajectory(initial, drives, specs)
    final = states[-1]
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append(
            {
                "id": test_id,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
        )

    depths = [state.root.max_depth() for state in states]
    counts = [len(node_ids(state)) for state in states]
    record("depth_grows_one_per_step", depths == [3, 4, 5, 6], depths)
    record("node_count_growth", counts == [3, 6, 9, 12], counts)

    recovered = final
    for receipt in reversed(receipts):
        recovered = extract_predecessor(recovered, receipt)
    record(
        "exact_ancestry_recovery",
        recovered.canonical_bytes() == initial.canonical_bytes(),
        recovered.digest(),
    )

    unit_error = max(
        float(np.max(np.abs(np.abs(state.render(t)) - 1.0))) for state in states
    )
    record("unit_modulus_trajectory", unit_error <= UNIT_MODULUS_TOL, unit_error)

    step_responses = [
        abs(r0.matched_response(states[index].render(t), states[index + 1].render(t)))
        for index in range(len(states) - 1)
    ]
    record(
        "each_step_changes_the_complete_state",
        all(response < 1.0 - NONIDENTITY_TOL for response in step_responses),
        step_responses,
    )

    repeated_states, repeated_receipts = trajectory(initial, drives, specs)
    deterministic = [state.digest() for state in states] == [
        state.digest() for state in repeated_states
    ] and [receipt.document() for receipt in receipts] == [
        receipt.document() for receipt in repeated_receipts
    ]
    record(
        "deterministic_trajectory_and_receipts",
        deterministic,
        [state.digest() for state in states],
    )

    reordered_drives = [drives[1], drives[0], drives[2]]
    reordered_states, _ = trajectory(initial, reordered_drives, specs)
    order_response = abs(
        r0.matched_response(final.render(t), reordered_states[-1].render(t))
    )
    record(
        "drive_order_is_relational",
        final.digest() != reordered_states[-1].digest()
        and order_response <= ORDER_RESPONSE_MAX,
        {
            "canonical_digest": final.digest(),
            "reordered_digest": reordered_states[-1].digest(),
            "matched_response": order_response,
        },
    )

    flat_wave = collapsed_flat_trajectory(initial, drives, specs, t)
    flat_response = abs(r0.matched_response(final.render(t), flat_wave))
    record(
        "flat_baseline_has_no_ancestry_receipts",
        not hasattr(flat_wave, "canonical_bytes"),
        {
            "matched_response_diagnostic": flat_response,
            "structural_ancestry": False,
        },
    )

    spin_wave = collapsed_spin_trajectory(initial, drives, specs, t)
    spin_response = abs(r0.matched_response(final.render(t), spin_wave))
    record(
        "decoded_spin_baseline_is_not_native_state",
        float(np.max(np.abs(final.render(t) - spin_wave))) > NONIDENTITY_TOL,
        {
            "matched_response_diagnostic": spin_response,
            "structural_ancestry": False,
        },
    )

    native_source = inspect.getsource(temporal_step)
    forbidden_tokens = [
        token
        for token in ("matched_response", "np.sign", "energy", "argmin", "argmax")
        if token in native_source
    ]
    record(
        "native_step_has_no_scalar_feedback",
        not forbidden_tokens,
        forbidden_tokens,
    )

    overlap_rejected = False
    try:
        temporal_step(initial, initial, specs[0])
    except ValueError:
        overlap_rejected = True
    record("identity_overlap_rejected", overlap_rejected, overlap_rejected)

    wrong_receipt = TemporalStepReceipt(
        spec_digest=receipts[-1].spec_digest,
        result_digest=receipts[-1].result_digest,
        state_digest="0" * 64,
        state_root_id=receipts[-1].state_root_id,
        drive_digest=receipts[-1].drive_digest,
        drive_root_id=receipts[-1].drive_root_id,
    )
    wrong_receipt_rejected = False
    try:
        extract_predecessor(final, wrong_receipt)
    except ValueError:
        wrong_receipt_rejected = True
    record(
        "wrong_ancestry_receipt_rejected",
        wrong_receipt_rejected,
        wrong_receipt_rejected,
    )

    record(
        "step_spec_identity_is_deterministic",
        specs[0].digest() == step_spec(1).digest(),
        specs[0].digest(),
    )

    passed = sum(test["status"] == "PASS" for test in tests)
    result = {
        "claim_ceiling": CLAIM_CEILING,
        "execution_environment": {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "python": ".".join(str(part) for part in sys.version_info[:3]),
        },
        "generator": GENERATOR_ID,
        "measurements": {
            "decoded_spin_response_diagnostic": float(spin_response),
            "final_depth": final.root.max_depth(),
            "final_node_count": len(node_ids(final)),
            "flat_response_diagnostic": float(flat_response),
            "order_response": float(order_response),
            "trajectory_digests": [state.digest() for state in states],
        },
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "schema": RESULT_SCHEMA,
        "summary": {
            "failed": len(tests) - passed,
            "passed": passed,
            "test_count": len(tests),
        },
        "tests": tests,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("self-test",), nargs="?", default="self-test"
    )
    parser.parse_args()
    result = run_reference_tests()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
