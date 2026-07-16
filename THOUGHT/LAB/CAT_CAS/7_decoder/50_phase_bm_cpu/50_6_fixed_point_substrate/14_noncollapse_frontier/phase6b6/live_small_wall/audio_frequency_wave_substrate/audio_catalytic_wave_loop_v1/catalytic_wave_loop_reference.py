#!/usr/bin/env python3
"""Deterministic R2S candidate for a software catalytic recursive-wave loop.

The loop borrows a deterministic complex carrier, drives it through the established R1
complete-tree trajectory with reversible phase-and-transport operators, extracts and
latches one predeclared complex relational observable, restores the carrier, and unwinds
the exact R1 ancestry while the latch remains outside the reversed history.

This ordinary-software candidate establishes no physical carrier, physical restoration,
Ising computation, optimization advantage, hardware bit replacement, or Wall crossing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
R1_SOURCE = (
    PACKAGE_DIR.parent
    / "audio_recursive_wave_operator_v1"
    / "recursive_wave_operator_reference.py"
)

_spec = importlib.util.spec_from_file_location("catcas_recursive_wave_operator_r1", R1_SOURCE)
if _spec is None or _spec.loader is None:
    raise RuntimeError("unable to load the established R1 recursive-wave reference")
r1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = r1
_spec.loader.exec_module(r1)
r0 = r1.r0

GENERATOR_ID = "catalytic_wave_loop_reference_v1"
CLAIM_CEILING = "SOFTWARE_CATALYTIC_WAVE_LOOP_REFERENCE_ONLY"
ESTABLISHED_TOKEN = "AUDIO_SOFTWARE_CATALYTIC_WAVE_LOOP_ESTABLISHED"
LATCH_SCHEMA = "catalytic_wave_relational_latch_v1"
SHIFT_SCHEDULE = (17, -29, 43)
CARRIER_RESTORE_TOL = 1e-12
MIN_FORWARD_DISPLACEMENT_L2 = 1.0
MIN_WRONG_RESTORE_ERROR = 0.05
MIN_QUERY_CHANGE = 1e-6


def _metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def _complex_bytes(values: np.ndarray) -> bytes:
    array = np.asarray(values, dtype=np.complex128)
    if array.ndim != 1 or array.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier must be one complex vector with the R0 sample count")
    if not np.all(np.isfinite(array)):
        raise ValueError("carrier contains non-finite values")
    return np.asarray(array, dtype="<c16", order="C").tobytes(order="C")


def carrier_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(_complex_bytes(values)).hexdigest()


@dataclass(frozen=True)
class RelationalLatch:
    """Immutable external record copied before carrier and ancestry restoration."""

    query_tree_digest: str
    final_tree_digest: str
    carrier_before_sha256: str
    carrier_displaced_sha256: str
    response_real: float
    response_imag: float

    def __post_init__(self) -> None:
        for name in (
            "query_tree_digest",
            "final_tree_digest",
            "carrier_before_sha256",
            "carrier_displaced_sha256",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        for name in ("response_real", "response_imag"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"{name} must be a finite number")

    def response(self) -> complex:
        return complex(self.response_real, self.response_imag)

    def document(self) -> dict[str, Any]:
        return {"schema": LATCH_SCHEMA, **asdict(self)}

    def canonical_bytes(self) -> bytes:
        return r0.canonical_json_bytes(self.document())

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()


@dataclass(frozen=True)
class CatalyticClosure:
    """Closed result that intentionally retains no temporal history objects."""

    latch: RelationalLatch
    carrier_before_sha256: str
    carrier_restored_sha256: str
    carrier_byte_exact: bool
    recovered_initial_tree_digest: str
    forward_displacement_l2: float
    restore_max_error: float

    def __post_init__(self) -> None:
        for name in (
            "carrier_before_sha256",
            "carrier_restored_sha256",
            "recovered_initial_tree_digest",
        ):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(ch not in "0123456789abcdef" for ch in value)
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if not isinstance(self.carrier_byte_exact, bool):
            raise ValueError("carrier_byte_exact must be a boolean")
        if self.forward_displacement_l2 <= 0.0:
            raise ValueError("forward carrier displacement must be nonzero")
        if self.restore_max_error < 0.0:
            raise ValueError("restore error must be nonnegative")


def forward_carrier_step(
    carrier: np.ndarray, state: Any, shift_samples: int
) -> np.ndarray:
    """Reversible phase multiplication followed by reversible circular transport."""

    r1.require_native_tree(state, "carrier state")
    if isinstance(shift_samples, bool) or not isinstance(shift_samples, int):
        raise ValueError("shift_samples must be an integer")
    carrier_array = np.asarray(carrier, dtype=np.complex128)
    if carrier_array.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier shape does not match the frozen sample count")
    beam = state.render(r0.sample_times())
    return np.roll(r0.apply_phase_operator(carrier_array, beam), shift_samples)


def inverse_carrier_step(
    carrier: np.ndarray, state: Any, shift_samples: int
) -> np.ndarray:
    """Exact inverse of forward_carrier_step."""

    r1.require_native_tree(state, "inverse carrier state")
    if isinstance(shift_samples, bool) or not isinstance(shift_samples, int):
        raise ValueError("shift_samples must be an integer")
    carrier_array = np.asarray(carrier, dtype=np.complex128)
    if carrier_array.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier shape does not match the frozen sample count")
    unshifted = np.roll(carrier_array, -shift_samples)
    return r0.uncompute_phase_operator(
        unshifted, state.render(r0.sample_times())
    )


def forward_carrier(
    carrier: np.ndarray,
    trajectory_states: Sequence[Any],
    shifts: Sequence[int] = SHIFT_SCHEDULE,
) -> np.ndarray:
    if not trajectory_states or len(trajectory_states) != len(shifts):
        raise ValueError("forward carrier requires equal nonempty state and shift sequences")
    current = np.asarray(carrier, dtype=np.complex128).copy()
    for state, shift in zip(trajectory_states, shifts, strict=True):
        current = forward_carrier_step(current, state, shift)
    return current


def restore_carrier(
    displaced: np.ndarray,
    trajectory_states: Sequence[Any],
    shifts: Sequence[int] = SHIFT_SCHEDULE,
) -> np.ndarray:
    if not trajectory_states or len(trajectory_states) != len(shifts):
        raise ValueError("restore carrier requires equal nonempty state and shift sequences")
    current = np.asarray(displaced, dtype=np.complex128).copy()
    for state, shift in reversed(list(zip(trajectory_states, shifts, strict=True))):
        current = inverse_carrier_step(current, state, shift)
    return current


def carrier_query_response(
    carrier_before: np.ndarray,
    carrier_displaced: np.ndarray,
    query_tree: Any,
    shifts: Sequence[int] = SHIFT_SCHEDULE,
) -> complex:
    """Read a matched relational observable from the actually displaced carrier.

    The public query tree is applied to the same borrowed carrier and transported through
    the same circular-shift schedule, but it does not receive the hidden trajectory phase
    operators. This creates a matched carrier query without reconstructing the final tree
    or feeding the scalar response back into evolution.
    """

    r1.require_native_tree(query_tree, "query tree")
    before = np.asarray(carrier_before, dtype=np.complex128)
    displaced = np.asarray(carrier_displaced, dtype=np.complex128)
    if before.shape != displaced.shape or before.shape != (r0.SAMPLE_COUNT,):
        raise ValueError("carrier query inputs have incompatible shapes")
    if len(shifts) != len(SHIFT_SCHEDULE):
        raise ValueError("query transport must use the complete frozen shift schedule")
    query_carrier = r0.apply_phase_operator(
        before, query_tree.render(r0.sample_times())
    )
    for shift in shifts:
        if isinstance(shift, bool) or not isinstance(shift, int):
            raise ValueError("query shifts must be integers")
        query_carrier = np.roll(query_carrier, shift)
    return r0.matched_response(displaced, query_carrier)


def latch_relational_observable(
    carrier_before: np.ndarray,
    carrier_displaced: np.ndarray,
    final_tree: Any,
    query_tree: Any,
) -> RelationalLatch:
    response = carrier_query_response(carrier_before, carrier_displaced, query_tree)
    return RelationalLatch(
        query_tree_digest=query_tree.digest(),
        final_tree_digest=final_tree.digest(),
        carrier_before_sha256=carrier_sha256(carrier_before),
        carrier_displaced_sha256=carrier_sha256(carrier_displaced),
        response_real=_metric(response.real),
        response_imag=_metric(response.imag),
    )


def reference_trajectory() -> tuple[list[Any], list[Any], list[Any], list[Any]]:
    initial = r0.hierarchy_a()
    drives = [r1.drive_tree(index) for index in range(1, 4)]
    specs = [r1.step_spec(index) for index in range(1, 4)]
    states, receipts = r1.trajectory(initial, drives, specs)
    r1.validate_trajectory(states, drives, specs, receipts)
    return states, drives, specs, receipts


def close_reference_loop(query_tree: Any | None = None) -> CatalyticClosure:
    """Execute forward displacement, latch, restore, and exact ancestry unwind."""

    states, drives, specs, receipts = reference_trajectory()
    query = r0.hierarchy_a() if query_tree is None else query_tree
    carrier_before = r0.borrowed_tape(r0.sample_times())
    carrier_displaced = forward_carrier(carrier_before, states[1:])
    forward_displacement = float(np.linalg.norm(carrier_displaced - carrier_before))
    if forward_displacement < MIN_FORWARD_DISPLACEMENT_L2:
        raise ValueError("forward carrier displacement is below the frozen minimum")

    latch = latch_relational_observable(
        carrier_before, carrier_displaced, states[-1], query
    )
    latch_before_restore = latch.canonical_bytes()

    restored = restore_carrier(carrier_displaced, states[1:])
    restore_error = float(np.max(np.abs(restored - carrier_before)))
    if restore_error > CARRIER_RESTORE_TOL:
        raise ValueError("correct reverse schedule did not restore the carrier")

    recovered = r1.validate_trajectory(states, drives, specs, receipts)
    if recovered.canonical_bytes() != states[0].canonical_bytes():
        raise ValueError("exact ancestry unwind did not recover the initial tree")
    if latch.canonical_bytes() != latch_before_restore:
        raise ValueError("latched observable changed during restoration")

    return CatalyticClosure(
        latch=latch,
        carrier_before_sha256=carrier_sha256(carrier_before),
        carrier_restored_sha256=carrier_sha256(restored),
        carrier_byte_exact=carrier_sha256(restored) == carrier_sha256(carrier_before),
        recovered_initial_tree_digest=recovered.digest(),
        forward_displacement_l2=_metric(forward_displacement),
        restore_max_error=_metric(restore_error),
    )


def run_reference_tests() -> dict[str, Any]:
    states, _, _, _ = reference_trajectory()
    carrier_before = r0.borrowed_tape(r0.sample_times())
    displaced = forward_carrier(carrier_before, states[1:])
    closure = close_reference_loop()
    tests: list[dict[str, Any]] = []

    def record(test_id: str, passed: bool, observed: Any) -> None:
        tests.append(
            {
                "id": test_id,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
            }
        )

    displacement = float(np.linalg.norm(displaced - carrier_before))
    record(
        "borrowed_carrier_is_actually_displaced",
        displacement >= MIN_FORWARD_DISPLACEMENT_L2,
        _metric(displacement),
    )
    record(
        "correct_reverse_restores_carrier_equivalence",
        closure.restore_max_error <= CARRIER_RESTORE_TOL,
        {
            "before_sha256": closure.carrier_before_sha256,
            "byte_exact": closure.carrier_byte_exact,
            "max_error": closure.restore_max_error,
            "restored_sha256": closure.carrier_restored_sha256,
        },
    )
    record(
        "exact_ancestry_is_recovered",
        closure.recovered_initial_tree_digest == states[0].digest(),
        closure.recovered_initial_tree_digest,
    )
    latch_bytes = closure.latch.canonical_bytes()
    record(
        "latch_is_external_to_restored_history",
        closure.latch.canonical_bytes() == latch_bytes
        and not hasattr(closure, "states")
        and not hasattr(closure, "receipts"),
        closure.latch.digest(),
    )

    wrong_order = np.asarray(displaced, dtype=np.complex128).copy()
    for state, shift in zip(states[1:], SHIFT_SCHEDULE, strict=True):
        wrong_order = inverse_carrier_step(wrong_order, state, shift)
    wrong_order_error = float(np.max(np.abs(wrong_order - carrier_before)))
    record(
        "forward_order_inverse_fails",
        wrong_order_error >= MIN_WRONG_RESTORE_ERROR,
        _metric(wrong_order_error),
    )

    reordered_states, _ = r1.trajectory(
        states[0],
        [r1.drive_tree(2), r1.drive_tree(1), r1.drive_tree(3)],
        [r1.step_spec(index) for index in range(1, 4)],
    )
    wrong_beam = restore_carrier(displaced, reordered_states[1:])
    wrong_beam_error = float(np.max(np.abs(wrong_beam - carrier_before)))
    record(
        "wrong_trajectory_inverse_fails",
        wrong_beam_error >= MIN_WRONG_RESTORE_ERROR,
        _metric(wrong_beam_error),
    )

    omitted = np.asarray(displaced, dtype=np.complex128).copy()
    for state, shift in reversed(
        list(zip(states[2:], SHIFT_SCHEDULE[1:], strict=True))
    ):
        omitted = inverse_carrier_step(omitted, state, shift)
    omitted_error = float(np.max(np.abs(omitted - carrier_before)))
    record(
        "omitted_inverse_step_fails",
        omitted_error >= MIN_WRONG_RESTORE_ERROR,
        _metric(omitted_error),
    )
    record(
        "no_restore_control_remains_displaced",
        float(np.max(np.abs(displaced - carrier_before))) >= MIN_WRONG_RESTORE_ERROR,
        _metric(float(np.max(np.abs(displaced - carrier_before)))),
    )

    exact_latch = closure.latch
    wrong_latch = latch_relational_observable(
        carrier_before, displaced, states[-1], r0.hierarchy_b()
    )
    query_change = abs(exact_latch.response() - wrong_latch.response())
    record(
        "query_changes_latched_relation",
        query_change >= MIN_QUERY_CHANGE
        and exact_latch.query_tree_digest != wrong_latch.query_tree_digest,
        _metric(query_change),
    )
    record(
        "latch_is_not_a_spin_or_energy",
        not any(
            key in exact_latch.document()
            for key in ("spin", "energy", "winner", "candidate", "score")
        ),
        sorted(exact_latch.document()),
    )
    record(
        "result_object_excludes_history",
        set(asdict(closure)) == {
            "latch",
            "carrier_before_sha256",
            "carrier_restored_sha256",
            "carrier_byte_exact",
            "recovered_initial_tree_digest",
            "forward_displacement_l2",
            "restore_max_error",
        },
        sorted(asdict(closure)),
    )

    passed = sum(test["status"] == "PASS" for test in tests)
    failed = len(tests) - passed
    result = {
        "claim_ceiling": CLAIM_CEILING,
        "contact_counts": {
            "audio_playback": 0,
            "audio_recording": 0,
            "hardware": 0,
            "target": 0,
        },
        "established_token_candidate": ESTABLISHED_TOKEN,
        "generator": GENERATOR_ID,
        "measurements": {
            "forward_displacement_l2": _metric(displacement),
            "latch_digest": exact_latch.digest(),
            "latch_response_imag": exact_latch.response_imag,
            "latch_response_real": exact_latch.response_real,
            "restore_max_error": closure.restore_max_error,
            "wrong_order_restore_error": _metric(wrong_order_error),
            "wrong_trajectory_restore_error": _metric(wrong_beam_error),
        },
        "ordinary_software_only": True,
        "physical_claims_established": [],
        "restoration_semantics": {
            "byte_exact_required": False,
            "equivalence_metric": "max_abs_complex_error",
            "tolerance": CARRIER_RESTORE_TOL,
        },
        "summary": {"failed": failed, "passed": passed, "test_count": len(tests)},
        "tests": tests,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test",))
    args = parser.parse_args()
    if args.command == "self-test":
        result = run_reference_tests()
        print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
        return 0 if result["summary"]["failed"] == 0 else 1
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
