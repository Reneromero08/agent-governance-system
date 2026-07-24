from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping

from .catalytic_existential_trace import CLAIM_CEILING
from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError, Literal

REFERENCE_WIDTH_VARIABLE_LIMIT = 16


@dataclass(frozen=True)
class RelationalWidthAudit:
    variable_order: tuple[str, ...]
    cut_widths: tuple[int, ...]
    maximum_width: int
    maximum_cut: int
    exact_reference: bool
    status: str
    claim_ceiling: str = CLAIM_CEILING


def _assignments(variables: tuple[str, ...]) -> Iterable[dict[str, bool]]:
    for values in product((False, True), repeat=len(variables)):
        yield dict(zip(variables, values, strict=True))


def _residual_signature(
    holo: ConstraintHolo,
    prefix_assignment: Mapping[str, bool],
    suffix_variables: tuple[str, ...],
) -> bytes:
    bits = bytearray()
    current = 0
    used = 0
    for suffix_assignment in _assignments(suffix_variables):
        accepted = holo.accepts({**prefix_assignment, **suffix_assignment})
        current |= int(accepted) << used
        used += 1
        if used == 8:
            bits.append(current)
            current = 0
            used = 0
    if used:
        bits.append(current)
    return bytes(bits)


def audit_residual_relation_width(
    holo: ConstraintHolo,
    variable_order: Iterable[str],
) -> RelationalWidthAudit:
    order = tuple(variable_order)
    if len(holo.variables) > REFERENCE_WIDTH_VARIABLE_LIMIT:
        raise ConstraintHoloError("residual-width audit exceeds the reference variable limit")
    if len(order) != len(holo.variables) or set(order) != set(holo.variables):
        raise ConstraintHoloError("variable order must be a permutation of the public boundary")

    cut_widths: list[int] = []
    for cut in range(len(order) + 1):
        prefix = order[:cut]
        suffix = order[cut:]
        signatures = {
            _residual_signature(holo, assignment, suffix)
            for assignment in _assignments(prefix)
        }
        cut_widths.append(len(signatures))

    maximum_width = max(cut_widths)
    maximum_cut = cut_widths.index(maximum_width)
    return RelationalWidthAudit(
        variable_order=order,
        cut_widths=tuple(cut_widths),
        maximum_width=maximum_width,
        maximum_cut=maximum_cut,
        exact_reference=True,
        status="EXACT_RESIDUAL_OPEN_RELATION_WIDTH_MEASURED",
    )


def equality_relation_holo(pair_count: int) -> ConstraintHolo:
    """Build x_i iff y_i using exact padded three-literal clauses."""

    if pair_count < 1:
        raise ConstraintHoloError("pair count must be positive")
    variables: list[str] = []
    clauses: list[ClauseRelation] = []
    for index in range(1, pair_count + 1):
        left = f"x{index}"
        right = f"y{index}"
        variables.extend((left, right))
        clauses.append(
            ClauseRelation(
                (
                    Literal(left, False),
                    Literal(right, True),
                    Literal(right, True),
                )
            )
        )
        clauses.append(
            ClauseRelation(
                (
                    Literal(left, True),
                    Literal(right, False),
                    Literal(right, False),
                )
            )
        )
    return ConstraintHolo.build(variables, clauses)
