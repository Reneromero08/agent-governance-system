from __future__ import annotations

from itertools import combinations

from .constraint_holo import ClauseRelation, ConstraintHolo, ConstraintHoloError, Literal


def _clause(*literals: Literal) -> ClauseRelation:
    if len(literals) == 2:
        literals = (literals[0], literals[1], literals[1])
    if len(literals) != 3:
        raise ConstraintHoloError("structured family clauses must have arity two or three")
    return ClauseRelation(tuple(literals))  # type: ignore[arg-type]


def unit_clause(variable: str, value: bool) -> ClauseRelation:
    literal = Literal(variable, value)
    return _clause(literal, literal, literal)


def pad_binary_clause_exact_three(
    first: Literal,
    second: Literal,
    auxiliary: str,
) -> tuple[ClauseRelation, ClauseRelation]:
    """Replace `(first OR second)` by two distinct-terminal 3-clauses."""

    if auxiliary in (first.variable, second.variable):
        raise ConstraintHoloError("binary-clause auxiliary must be a fresh public variable")
    return (
        _clause(first, second, Literal(auxiliary)),
        _clause(first, second, Literal(auxiliary, False)),
    )


def pad_unit_clause_exact_three(
    literal: Literal,
    first_auxiliary: str,
    second_auxiliary: str,
) -> tuple[ClauseRelation, ClauseRelation, ClauseRelation, ClauseRelation]:
    """Replace one unit relation by four distinct-terminal 3-clauses."""

    if len({literal.variable, first_auxiliary, second_auxiliary}) != 3:
        raise ConstraintHoloError("unit-clause auxiliaries must be fresh and distinct")
    return (
        _clause(literal, Literal(first_auxiliary), Literal(second_auxiliary)),
        _clause(literal, Literal(first_auxiliary), Literal(second_auxiliary, False)),
        _clause(literal, Literal(first_auxiliary, False), Literal(second_auxiliary)),
        _clause(literal, Literal(first_auxiliary, False), Literal(second_auxiliary, False)),
    )


def xor_edge_clauses(left: str, right: str, parity: int) -> tuple[ClauseRelation, ClauseRelation]:
    """Encode left XOR right = parity using exact 3-CNF with duplicate literals."""

    if parity not in (0, 1):
        raise ConstraintHoloError("XOR edge parity must be zero or one")
    if parity == 0:
        return (
            _clause(Literal(left), Literal(right, False)),
            _clause(Literal(left, False), Literal(right)),
        )
    return (
        _clause(Literal(left), Literal(right)),
        _clause(Literal(left, False), Literal(right, False)),
    )


def parity_cycle_holo(variable_count: int, total_charge: int = 0) -> ConstraintHolo:
    """Build a cycle whose XOR edge charges sum to total_charge.

    A cycle is globally satisfiable exactly when the XOR of all edge parities is zero.
    The final edge carries the requested total charge, making the odd-charge instance a
    locally satisfiable but globally inconsistent control.
    """

    if variable_count < 3:
        raise ConstraintHoloError("parity cycle requires at least three variables")
    if total_charge not in (0, 1):
        raise ConstraintHoloError("total parity charge must be zero or one")

    variables = tuple(f"x{index}" for index in range(variable_count))
    clauses: list[ClauseRelation] = []
    for index, left in enumerate(variables):
        right = variables[(index + 1) % variable_count]
        parity = total_charge if index == variable_count - 1 else 0
        clauses.extend(xor_edge_clauses(left, right, parity))
    return ConstraintHolo.build(variables, clauses)


def exact_three_parity_cycle_holo(
    variable_count: int,
    total_charge: int = 0,
) -> ConstraintHolo:
    """Build the parity-cycle control using only distinct-variable 3-clauses."""

    if variable_count < 3:
        raise ConstraintHoloError("parity cycle requires at least three variables")
    if total_charge not in (0, 1):
        raise ConstraintHoloError("total parity charge must be zero or one")

    core_variables = tuple(f"x{index}" for index in range(variable_count))
    auxiliaries: list[str] = []
    clauses: list[ClauseRelation] = []
    for edge_index, left in enumerate(core_variables):
        right = core_variables[(edge_index + 1) % variable_count]
        parity = total_charge if edge_index == variable_count - 1 else 0
        binary = xor_edge_clauses(left, right, parity)
        for relation_index, relation in enumerate(binary):
            auxiliary = f"a_e{edge_index}_r{relation_index}"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(
                    relation.literals[0],
                    relation.literals[1],
                    auxiliary,
                )
            )
    return ConstraintHolo.build(core_variables + tuple(auxiliaries), clauses)


def exact_three_unique_solution_holo(variable_count: int) -> ConstraintHolo:
    if variable_count < 1:
        raise ConstraintHoloError("unique-solution family requires at least one variable")
    core_variables = tuple(f"x{index}" for index in range(variable_count))
    auxiliaries: list[str] = []
    clauses: list[ClauseRelation] = []
    for index, variable in enumerate(core_variables):
        first_auxiliary = f"u{index}_a"
        second_auxiliary = f"u{index}_b"
        auxiliaries.extend((first_auxiliary, second_auxiliary))
        clauses.extend(
            pad_unit_clause_exact_three(
                Literal(variable),
                first_auxiliary,
                second_auxiliary,
            )
        )
    return ConstraintHolo.build(core_variables + tuple(auxiliaries), clauses)


def pigeonhole_holo(pigeons: int, holes: int) -> ConstraintHolo:
    if pigeons < 1 or holes not in (2, 3):
        raise ConstraintHoloError("reference pigeonhole family supports two or three holes")

    def variable(pigeon: int, hole: int) -> str:
        return f"p{pigeon}_h{hole}"

    variables = tuple(variable(pigeon, hole) for pigeon in range(pigeons) for hole in range(holes))
    clauses: list[ClauseRelation] = []

    for pigeon in range(pigeons):
        literals = tuple(Literal(variable(pigeon, hole)) for hole in range(holes))
        clauses.append(_clause(*literals))
        for first, second in combinations(range(holes), 2):
            clauses.append(
                _clause(
                    Literal(variable(pigeon, first), False),
                    Literal(variable(pigeon, second), False),
                )
            )

    for hole in range(holes):
        for first, second in combinations(range(pigeons), 2):
            clauses.append(
                _clause(
                    Literal(variable(first, hole), False),
                    Literal(variable(second, hole), False),
                )
            )

    return ConstraintHolo.build(variables, clauses)


def exact_three_pigeonhole_holo(pigeons: int, holes: int) -> ConstraintHolo:
    if pigeons < 1 or holes not in (2, 3):
        raise ConstraintHoloError("reference pigeonhole family supports two or three holes")

    def variable(pigeon: int, hole: int) -> str:
        return f"p{pigeon}_h{hole}"

    core_variables = tuple(
        variable(pigeon, hole)
        for pigeon in range(pigeons)
        for hole in range(holes)
    )
    auxiliaries: list[str] = []
    clauses: list[ClauseRelation] = []

    for pigeon in range(pigeons):
        literals = tuple(Literal(variable(pigeon, hole)) for hole in range(holes))
        if holes == 3:
            clauses.append(_clause(*literals))
        else:
            auxiliary = f"a_p{pigeon}_atleast"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(literals[0], literals[1], auxiliary)
            )
        for first, second in combinations(range(holes), 2):
            auxiliary = f"a_p{pigeon}_pair{first}_{second}"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(
                    Literal(variable(pigeon, first), False),
                    Literal(variable(pigeon, second), False),
                    auxiliary,
                )
            )

    for hole in range(holes):
        for first, second in combinations(range(pigeons), 2):
            auxiliary = f"a_h{hole}_p{first}_{second}"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(
                    Literal(variable(first, hole), False),
                    Literal(variable(second, hole), False),
                    auxiliary,
                )
            )

    return ConstraintHolo.build(core_variables + tuple(auxiliaries), clauses)


def graph_three_coloring_holo(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> ConstraintHolo:
    if vertex_count < 1:
        raise ConstraintHoloError("graph coloring requires at least one vertex")
    normalized_edges = tuple(sorted({tuple(sorted(edge)) for edge in edges}))
    if any(left == right or left < 0 or right >= vertex_count for left, right in normalized_edges):
        raise ConstraintHoloError("graph coloring edge is outside the public graph")

    def variable(vertex: int, color: int) -> str:
        return f"v{vertex}_c{color}"

    variables = tuple(variable(vertex, color) for vertex in range(vertex_count) for color in range(3))
    clauses: list[ClauseRelation] = []

    for vertex in range(vertex_count):
        clauses.append(
            _clause(*(Literal(variable(vertex, color)) for color in range(3)))
        )
        for first, second in combinations(range(3), 2):
            clauses.append(
                _clause(
                    Literal(variable(vertex, first), False),
                    Literal(variable(vertex, second), False),
                )
            )

    for left, right in normalized_edges:
        for color in range(3):
            clauses.append(
                _clause(
                    Literal(variable(left, color), False),
                    Literal(variable(right, color), False),
                )
            )

    return ConstraintHolo.build(variables, clauses)


def exact_three_graph_coloring_holo(
    vertex_count: int,
    edges: tuple[tuple[int, int], ...],
) -> ConstraintHolo:
    if vertex_count < 1:
        raise ConstraintHoloError("graph coloring requires at least one vertex")
    normalized_edges = tuple(sorted({tuple(sorted(edge)) for edge in edges}))
    if any(left == right or left < 0 or right >= vertex_count for left, right in normalized_edges):
        raise ConstraintHoloError("graph coloring edge is outside the public graph")

    def variable(vertex: int, color: int) -> str:
        return f"v{vertex}_c{color}"

    core_variables = tuple(
        variable(vertex, color)
        for vertex in range(vertex_count)
        for color in range(3)
    )
    auxiliaries: list[str] = []
    clauses: list[ClauseRelation] = []

    for vertex in range(vertex_count):
        clauses.append(
            _clause(*(Literal(variable(vertex, color)) for color in range(3)))
        )
        for first, second in combinations(range(3), 2):
            auxiliary = f"a_v{vertex}_pair{first}_{second}"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(
                    Literal(variable(vertex, first), False),
                    Literal(variable(vertex, second), False),
                    auxiliary,
                )
            )

    for edge_index, (left, right) in enumerate(normalized_edges):
        for color in range(3):
            auxiliary = f"a_e{edge_index}_c{color}"
            auxiliaries.append(auxiliary)
            clauses.extend(
                pad_binary_clause_exact_three(
                    Literal(variable(left, color), False),
                    Literal(variable(right, color), False),
                    auxiliary,
                )
            )

    return ConstraintHolo.build(core_variables + tuple(auxiliaries), clauses)


def cycle_graph_edges(vertex_count: int) -> tuple[tuple[int, int], ...]:
    if vertex_count < 3:
        raise ConstraintHoloError("cycle graph requires at least three vertices")
    return tuple((index, (index + 1) % vertex_count) for index in range(vertex_count))


def complete_graph_edges(vertex_count: int) -> tuple[tuple[int, int], ...]:
    return tuple(combinations(range(vertex_count), 2))
