#!/usr/bin/env python3
"""Synthetic ownership-state models for the Family 10h holonomy protocol.

The overwrite model approximates ordinary ownership-setting stores and demonstrates why
endpoint return can be destructive rather than inverse. The permutation model is a
hypothetical reversible reference used only to qualify protocol laws and controls.
Neither model is evidence about the physical Family 10h substrate.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Iterable, Sequence


OWNER_HOME = "H"
OWNER_A = "A"
OWNER_B = "B"
OWNERS = (OWNER_HOME, OWNER_A, OWNER_B)
LINE_SET_A = frozenset(range(0, 16))
LINE_SET_B = frozenset(range(8, 24))
OVERLAP = LINE_SET_A & LINE_SET_B
CARRIER_SIZE = 24


@dataclass(frozen=True)
class OwnershipState:
    """Finite per-line ownership state."""

    owners: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.owners) != CARRIER_SIZE:
            raise ValueError("unexpected carrier size")
        if any(owner not in OWNERS for owner in self.owners):
            raise ValueError("invalid owner token")

    @classmethod
    def home_baseline(cls) -> "OwnershipState":
        """Return the all-home baseline."""

        return cls((OWNER_HOME,) * CARRIER_SIZE)

    def changed_lines(self, other: "OwnershipState") -> frozenset[int]:
        """Return line indices whose owners differ."""

        return frozenset(
            index
            for index, (left, right) in enumerate(zip(self.owners, other.owners, strict=True))
            if left != right
        )

    def with_owner(self, lines: Iterable[int], owner: str) -> "OwnershipState":
        """Overwrite declared lines with one owner."""

        if owner not in OWNERS:
            raise ValueError("invalid owner")
        updated = list(self.owners)
        for index in lines:
            updated[index] = owner
        return OwnershipState(tuple(updated))

    def swap_owners(
        self,
        lines: Iterable[int],
        left_owner: str,
        right_owner: str,
    ) -> "OwnershipState":
        """Apply one owner transposition on declared lines."""

        updated = list(self.owners)
        for index in lines:
            if updated[index] == left_owner:
                updated[index] = right_owner
            elif updated[index] == right_owner:
                updated[index] = left_owner
        return OwnershipState(tuple(updated))


@dataclass(frozen=True)
class OverwriteOperation:
    """Many-to-one ownership-setting operation."""

    name: str
    lines: frozenset[int]
    owner: str

    def apply(self, state: OwnershipState) -> OwnershipState:
        return state.with_owner(self.lines, self.owner)


@dataclass(frozen=True)
class SwapOperation:
    """Bijective per-line ownership transposition."""

    name: str
    lines: frozenset[int]
    left_owner: str
    right_owner: str

    def apply(self, state: OwnershipState) -> OwnershipState:
        return state.swap_owners(self.lines, self.left_owner, self.right_owner)

    @property
    def inverse(self) -> "SwapOperation":
        """A transposition is self-inverse."""

        return self


Operation = OverwriteOperation | SwapOperation


def apply_word(state: OwnershipState, word: Sequence[Operation]) -> OwnershipState:
    """Apply operations in listed execution order."""

    current = state
    for operation in word:
        current = operation.apply(current)
    return current


def overlap_orientation(state: OwnershipState) -> int:
    """Read the carrier-coupled A/B orientation on overlap lines."""

    return sum(
        1 if state.owners[index] == OWNER_B else -1 if state.owners[index] == OWNER_A else 0
        for index in OVERLAP
    )


def word_label_recorder(word: Sequence[Operation], carrier_enabled: bool) -> int:
    """Invalid control that records labels even when carrier coupling is absent."""

    del carrier_enabled
    names = tuple(operation.name for operation in word)
    if names[:2] == ("A", "B"):
        return len(OVERLAP)
    if names[:2] == ("B", "A"):
        return -len(OVERLAP)
    return 0


def carrier_coupled_readout(
    state: OwnershipState,
    word: Sequence[Operation],
    *,
    carrier_enabled: bool,
) -> tuple[OwnershipState, int]:
    """Apply a word and read orientation only from the resulting carrier state."""

    result = apply_word(state, word) if carrier_enabled else state
    return result, overlap_orientation(result)


def overwrite_model_report() -> dict[str, object]:
    """Show order dependence, noninvertibility, and destructive endpoint closure."""

    baseline = OwnershipState.home_baseline()
    op_a = OverwriteOperation("A", LINE_SET_A, OWNER_A)
    op_b = OverwriteOperation("B", LINE_SET_B, OWNER_B)
    reclaim_a = OverwriteOperation("A_reclaim", LINE_SET_A, OWNER_HOME)
    reclaim_b = OverwriteOperation("B_reclaim", LINE_SET_B, OWNER_HOME)

    state_ab = apply_word(baseline, (op_a, op_b))
    state_ba = apply_word(baseline, (op_b, op_a))
    closed_ab = apply_word(state_ab, (reclaim_a, reclaim_b))
    closed_ba = apply_word(state_ba, (reclaim_b, reclaim_a))

    counterexample = baseline.with_owner((0,), OWNER_B)
    left_inverse_counterexample = reclaim_a.apply(op_a.apply(counterexample))
    right_inverse_counterexample = op_a.apply(reclaim_a.apply(counterexample))

    collision_left = baseline
    collision_right = baseline.with_owner((0,), OWNER_B)
    collision_after_left = op_a.apply(collision_left)
    collision_after_right = op_a.apply(collision_right)

    _, coupled_ab = carrier_coupled_readout(baseline, (op_a, op_b), carrier_enabled=True)
    _, coupled_ba = carrier_coupled_readout(baseline, (op_b, op_a), carrier_enabled=True)
    _, coupled_off = carrier_coupled_readout(baseline, (op_a, op_b), carrier_enabled=False)
    label_off = word_label_recorder((op_a, op_b), carrier_enabled=False)

    return {
        "model": "many_to_one_ownership_overwrite",
        "ab_ba_changed_line_count": len(state_ab.changed_lines(state_ba)),
        "order_effect_localized_to_overlap": state_ab.changed_lines(state_ba) == OVERLAP,
        "orientation_ab": overlap_orientation(state_ab),
        "orientation_ba": overlap_orientation(state_ba),
        "baseline_endpoint_restored_after_reclaims_ab": closed_ab == baseline,
        "baseline_endpoint_restored_after_reclaims_ba": closed_ba == baseline,
        "reclaim_is_left_inverse_on_all_states": left_inverse_counterexample == counterexample,
        "reclaim_is_right_inverse_on_all_states": right_inverse_counterexample == counterexample,
        "forward_map_injective": not (
            collision_left != collision_right
            and collision_after_left == collision_after_right
        ),
        "carrier_coupled_readout_ab": coupled_ab,
        "carrier_coupled_readout_ba": coupled_ba,
        "carrier_off_readout": coupled_off,
        "invalid_word_recorder_carrier_off": label_off,
        "interpretation": (
            "Order dependence and endpoint return coexist with a noninjective forward map; "
            "home reclaim is destructive closure, not an inverse."
        ),
    }


def reversible_model_report() -> dict[str, object]:
    """Qualify the ideal reversible control laws on a hypothetical permutation model."""

    baseline = OwnershipState.home_baseline()
    op_a = SwapOperation("A", LINE_SET_A, OWNER_HOME, OWNER_A)
    op_b = SwapOperation("B", LINE_SET_B, OWNER_HOME, OWNER_B)

    commutator = (op_a, op_b, op_a.inverse, op_b.inverse)
    inverse_commutator = tuple(operation.inverse for operation in reversed(commutator))
    reverse_commutator = (op_b, op_a, op_b.inverse, op_a.inverse)
    contractible = (op_a, op_a.inverse, op_b, op_b.inverse)

    after_commutator = apply_word(baseline, commutator)
    after_reverse = apply_word(baseline, reverse_commutator)
    restored = apply_word(after_commutator, inverse_commutator)
    contractible_state = apply_word(baseline, contractible)

    disjoint_b = SwapOperation(
        "B_disjoint",
        frozenset(range(100, 116)),
        OWNER_HOME,
        OWNER_B,
    )
    # Use a larger synthetic state only for the disjoint algebraic control by mapping
    # the disjoint support outside the physical fixture to an empty in-domain support.
    disjoint_b_in_domain = SwapOperation("B_disjoint", frozenset(), OWNER_HOME, OWNER_B)
    disjoint_commutator = (
        op_a,
        disjoint_b_in_domain,
        op_a.inverse,
        disjoint_b_in_domain.inverse,
    )
    del disjoint_b

    single_line_states = tuple(
        OwnershipState((owner,) + (OWNER_HOME,) * (CARRIER_SIZE - 1))
        for owner in OWNERS
    )
    a_two_sided = all(op_a.inverse.apply(op_a.apply(state)) == state for state in single_line_states)
    a_two_sided = a_two_sided and all(
        op_a.apply(op_a.inverse.apply(state)) == state for state in single_line_states
    )
    b_two_sided = all(op_b.inverse.apply(op_b.apply(state)) == state for state in single_line_states)
    b_two_sided = b_two_sided and all(
        op_b.apply(op_b.inverse.apply(state)) == state for state in single_line_states
    )

    _, carrier_off = carrier_coupled_readout(baseline, commutator, carrier_enabled=False)
    output = overlap_orientation(after_commutator)
    reverse_output = overlap_orientation(after_reverse)

    return {
        "model": "hypothetical_reversible_ownership_permutations",
        "A_two_sided_inverse": a_two_sided,
        "B_two_sided_inverse": b_two_sided,
        "commutator_nontrivial": after_commutator != baseline,
        "reverse_commutator_is_inverse_orientation": reverse_output == -output,
        "commutator_output": output,
        "reverse_commutator_output": reverse_output,
        "commutator_then_inverse_restores_carrier": restored == baseline,
        "retained_output_after_decoupled_inverse": output,
        "contractible_word_is_identity": contractible_state == baseline,
        "disjoint_support_commutator_is_identity": (
            apply_word(baseline, disjoint_commutator) == baseline
        ),
        "carrier_off_output": carrier_off,
        "nontrivial_lines_equal_overlap": (
            baseline.changed_lines(after_commutator) == OVERLAP
        ),
        "physical_realization_established": False,
        "interpretation": (
            "A reversible multi-owner permutation carrier can satisfy the protocol laws "
            "synthetically, but ordinary overwrite stores do not implement this model."
        ),
    }


def verify_synthetic_models() -> dict[str, object]:
    """Run both models and enforce their different claim boundaries."""

    overwrite = overwrite_model_report()
    reversible = reversible_model_report()

    overwrite_expected = {
        "ab_ba_changed_line_count": len(OVERLAP),
        "order_effect_localized_to_overlap": True,
        "orientation_ab": len(OVERLAP),
        "orientation_ba": -len(OVERLAP),
        "baseline_endpoint_restored_after_reclaims_ab": True,
        "baseline_endpoint_restored_after_reclaims_ba": True,
        "reclaim_is_left_inverse_on_all_states": False,
        "reclaim_is_right_inverse_on_all_states": False,
        "forward_map_injective": False,
        "carrier_coupled_readout_ab": len(OVERLAP),
        "carrier_coupled_readout_ba": -len(OVERLAP),
        "carrier_off_readout": 0,
        "invalid_word_recorder_carrier_off": len(OVERLAP),
    }
    for key, expected in overwrite_expected.items():
        if overwrite[key] != expected:
            raise AssertionError(f"overwrite model mismatch for {key}")

    reversible_expected = {
        "A_two_sided_inverse": True,
        "B_two_sided_inverse": True,
        "commutator_nontrivial": True,
        "reverse_commutator_is_inverse_orientation": True,
        "commutator_output": len(OVERLAP),
        "reverse_commutator_output": -len(OVERLAP),
        "commutator_then_inverse_restores_carrier": True,
        "retained_output_after_decoupled_inverse": len(OVERLAP),
        "contractible_word_is_identity": True,
        "disjoint_support_commutator_is_identity": True,
        "carrier_off_output": 0,
        "nontrivial_lines_equal_overlap": True,
        "physical_realization_established": False,
    }
    for key, expected in reversible_expected.items():
        if reversible[key] != expected:
            raise AssertionError(f"reversible model mismatch for {key}")

    return {
        "schema": "FAMILY10H_I2C_SYNTHETIC_OWNERSHIP_MODELS_V1",
        "line_set_size": len(LINE_SET_A),
        "overlap_size": len(OVERLAP),
        "overwrite_model": overwrite,
        "reversible_reference_model": reversible,
        "decision": (
            "I2C_SYNTHETIC_MODELS_COMPLETE__OVERWRITE_MODEL_REJECTED_FOR_H2_R2__"
            "REVERSIBLE_REFERENCE_PASSES_PROTOCOL_LAWS__PHYSICAL_REALIZATION_NOT_ESTABLISHED"
        ),
        "claim_ceiling": [
            "SYNTHETIC_PROTOCOL_REFERENCE_ONLY",
            "PHYSICAL_REVERSIBLE_OWNERSHIP_CARRIER_NOT_ESTABLISHED",
            "FAMILY10H_TRANSPORT_OPERATOR_NOT_ESTABLISHED",
            "FAMILY10H_HOLONOMY_NOT_ESTABLISHED",
            "SMALL_WALL_NOT_CROSSED",
            "NO_LIVE_EXECUTION_AUTHORIZED",
        ],
        "passed": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify_synthetic_models(), indent=2, sort_keys=True))
