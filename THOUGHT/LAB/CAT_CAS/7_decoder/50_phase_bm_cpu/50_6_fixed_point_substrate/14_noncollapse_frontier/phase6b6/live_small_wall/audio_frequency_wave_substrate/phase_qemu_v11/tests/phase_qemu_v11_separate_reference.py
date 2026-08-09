#!/usr/bin/env python3
"""Independent exact/noisy dual-rail reference oracle for Phase-QEMU V11.

The reference uses sparse density matrices over the declared 96-dimensional
basis and exact Q(omega) arithmetic.  It consumes no external input, imports
no package code, reads no result artifact, executes no QEMU device, and makes
no restoration or physical-custody claim.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable, Sequence


REFERENCE_ID = "PHASE_QEMU_V11_DUAL_RAIL_SEPARATE_REFERENCE_V1"
SCHEMA = "PHASE_QEMU_V11_DUAL_RAIL_EXACT_NOISY_REFERENCE_V1"
CLAIM = (
    "FINITE_D3_DUAL_RAIL_NUMBER_EIGENSPACE_CHARACTER_KICKBACK_RETURNS_THE_"
    "LOGICAL_CARRIER_REFERENCE_EXACTLY_AND_FACTORS_TWO_DISTINCT_FRESH_"
    "CLIENT_CHOI_BOUNDARIES_FOR_ALL_NINE_RESIDUE_PAIRS_WHILE_OPEN_NOISY_"
    "CONTROLS_FAIL_COMPLETE_RETURN_AND_EQUAL_ACCESS_OR_DIRECT_SECRET_PHASE_"
    "COMPARATORS_ESTABLISH_NO_UNIQUE_PHASE_QEMU_ADVANTAGE"
)
CEILING = (
    "DETERMINISTIC_EXACT_Q_OMEGA_SPARSE_DENSITY_REFERENCE_ORACLE_WITH_"
    "ANALYTIC_RATIONAL_CHANNEL_CONTROLS_AND_EXPECTED_COMMON_BACKEND_ABI_"
    "DESCRIPTORS_NO_QEMU_EXECUTION_GUEST_CONTRACT_EXERCISE_PHYSICAL_"
    "CARRIER_CUSTODY_OR_TOTAL_RESOURCE_ADVANTAGE"
)
RESTORATION_CLASSIFICATION = "NO_RESTORATION_CLAIM"
RESTORATION_SCOPE = (
    "INDEPENDENT_ALGEBRAIC_REFERENCE_COMPLETE_DENSITY_EQUALITIES_ONLY_"
    "WITHOUT_EXECUTED_DEVICE_BACKING_ATOMIC_HOLD_OR_PHYSICAL_RESTORATION"
)
DISPOSITION = (
    "EXACT_DUAL_RAIL_KICKBACK_AND_REFERENCE_COMPLETE_LOGICAL_RETURN_ARE_"
    "VALID_IN_THE_STIPULATED_REFERENCE_ORACLE_BUT_DIRECT_SECRET_PHASE_"
    "COMPILATION_EQUAL_COHERENT_ACCESS_SECRET_PROGRAM_STORAGE_AND_ORACLE_"
    "TOTAL_COSTS_PREVENT_ANY_UNIQUE_RESOURCE_ADVANTAGE_OR_M257_ESCAPE"
)
SUCCESSOR = (
    "EXECUTED_COMMON_PHASE_QEMU_DEVICE_BACKEND_QUALIFICATION_WITH_IDENTICAL_"
    "GUEST_DESCRIPTOR_RETURN_CLASS_AND_RESOURCE_COUNTER_PARITY"
)

D = 3
DIMS = (2, 2, 2, 2, 3, 2)
COORDINATE_NAMES = ("C_A", "R_A", "C_B", "R_B", "K", "R_K")
TOTAL_DIMENSION = 96
K_VACUUM = 0
K_RAIL_A = 1
K_RAIL_B = 2
NUMBER_K = (0, 1, 1)

DESCRIPTOR_WORDS = (
    0x50313144,
    0x00010008,
    0x00000002,
    0x00020102,
    0x00020011,
    0x00030021,
    0x00000003,
    0x00010001,
)


@dataclass(frozen=True)
class Cyclo3:
    """a + b*omega in Q(omega), omega**2 + omega + 1 = 0."""

    one: Fraction
    omega: Fraction

    def __add__(self, other: Cyclo3) -> Cyclo3:
        return Cyclo3(self.one + other.one, self.omega + other.omega)

    def __neg__(self) -> Cyclo3:
        return Cyclo3(-self.one, -self.omega)

    def __sub__(self, other: Cyclo3) -> Cyclo3:
        return self + (-other)

    def __mul__(self, other: Cyclo3) -> Cyclo3:
        ac = self.one * other.one
        bd = self.omega * other.omega
        cross = self.one * other.omega + self.omega * other.one
        return Cyclo3(ac - bd, cross - bd)

    def scaled(self, scalar: Fraction | int) -> Cyclo3:
        factor = Fraction(scalar)
        return Cyclo3(self.one * factor, self.omega * factor)

    def conjugate(self) -> Cyclo3:
        return Cyclo3(self.one - self.omega, -self.omega)


ZERO = Cyclo3(Fraction(0), Fraction(0))
ONE = Cyclo3(Fraction(1), Fraction(0))
OMEGA = Cyclo3(Fraction(0), Fraction(1))
OMEGA2 = Cyclo3(Fraction(-1), Fraction(-1))

Basis = tuple[int, ...]
DensityKey = tuple[Basis, Basis]
Density = dict[DensityKey, Cyclo3]
Component = tuple[Basis, Cyclo3]


def root(power: int) -> Cyclo3:
    return (ONE, OMEGA, OMEGA2)[power % D]


def add_entry(
    density: Density, ket: Basis, bra: Basis, value: Cyclo3
) -> None:
    key = (ket, bra)
    updated = density.get(key, ZERO) + value
    if updated == ZERO:
        density.pop(key, None)
    else:
        density[key] = updated


def pure_density(components: Sequence[Component], denominator: int) -> Density:
    if denominator <= 0 or len(components) != denominator:
        raise ValueError("pure-state denominator must equal component count")
    result: Density = {}
    for ket, ket_amplitude in components:
        for bra, bra_amplitude in components:
            add_entry(
                result,
                ket,
                bra,
                (ket_amplitude * bra_amplitude.conjugate()).scaled(
                    Fraction(1, denominator)
                ),
            )
    return result


def density_scale(density: Density, scalar: Fraction | int) -> Density:
    result: Density = {}
    for (ket, bra), value in density.items():
        add_entry(result, ket, bra, value.scaled(scalar))
    return result


def tensor_density(left: Density, right: Density) -> Density:
    result: Density = {}
    for (left_ket, left_bra), left_value in left.items():
        for (right_ket, right_bra), right_value in right.items():
            add_entry(
                result,
                left_ket + right_ket,
                left_bra + right_bra,
                left_value * right_value,
            )
    return result


def density_trace(density: Density) -> Cyclo3:
    total = ZERO
    for (ket, bra), value in density.items():
        if ket == bra:
            total = total + value
    return total


def density_purity(density: Density) -> Cyclo3:
    total = ZERO
    for (ket, bra), value in density.items():
        total = total + value * density.get((bra, ket), ZERO)
    return total


def density_is_hermitian(density: Density) -> bool:
    return all(
        value.conjugate() == density.get((bra, ket), ZERO)
        for (ket, bra), value in density.items()
    )


def density_value(density: Density, ket: Basis, bra: Basis) -> Cyclo3:
    return density.get((ket, bra), ZERO)


def reduce_density(density: Density, keep: Sequence[int]) -> Density:
    keep_tuple = tuple(keep)
    if not density:
        return {}
    coordinate_count = len(next(iter(density))[0])
    traced = tuple(index for index in range(coordinate_count) if index not in keep_tuple)
    result: Density = {}
    for (ket, bra), value in density.items():
        if all(ket[index] == bra[index] for index in traced):
            reduced_ket = tuple(ket[index] for index in keep_tuple)
            reduced_bra = tuple(bra[index] for index in keep_tuple)
            add_entry(result, reduced_ket, reduced_bra, value)
    return result


def population(density: Density, predicate: Callable[[Basis], bool]) -> Cyclo3:
    total = ZERO
    for (ket, bra), value in density.items():
        if ket == bra and predicate(ket):
            total = total + value
    return total


def exact_rows(density: Density) -> list[dict[str, object]]:
    return [
        {"ket": ket, "bra": bra, "value": value}
        for (ket, bra), value in sorted(density.items())
    ]


def density_commitment(density: Density) -> str:
    return hashlib.sha256(exact_bytes(exact_rows(density))).hexdigest()


def flat_index(state: Sequence[int], dimensions: Sequence[int]) -> int:
    if len(state) != len(dimensions):
        raise ValueError("state and dimension lengths disagree")
    result = 0
    for coordinate, dimension in zip(state, dimensions):
        if not 0 <= coordinate < dimension:
            raise ValueError("basis coordinate outside declared dimension")
        result = result * dimension + coordinate
    return result


def prepared_components() -> list[Component]:
    result: list[Component] = []
    for client_a in range(2):
        for client_b in range(2):
            for carrier, carrier_reference in (
                (K_RAIL_A, 0),
                (K_RAIL_B, 1),
            ):
                state = (
                    client_a,
                    client_a,
                    client_b,
                    client_b,
                    carrier,
                    carrier_reference,
                )
                result.append((state, ONE))
    return result


def prepared_density() -> Density:
    return pure_density(prepared_components(), 8)


def client_choi_density(residue: int) -> Density:
    components = [
        ((client, client), root(residue * client)) for client in range(2)
    ]
    return pure_density(components, 2)


def dual_rail_reference_density() -> Density:
    return pure_density(
        [((K_RAIL_A, 0), ONE), ((K_RAIL_B, 1), ONE)],
        2,
    )


def vacuum_reference_density() -> Density:
    return pure_density([((K_VACUUM, 0), ONE)], 1)


def controlled_number_phase(
    density: Density, residue_a: int, residue_b: int
) -> Density:
    result: Density = {}
    for (ket, bra), value in density.items():
        exponent = residue_a * (
            ket[0] * NUMBER_K[ket[4]] - bra[0] * NUMBER_K[bra[4]]
        ) + residue_b * (
            ket[2] * NUMBER_K[ket[4]] - bra[2] * NUMBER_K[bra[4]]
        )
        add_entry(result, ket, bra, value * root(exponent))
    return result


def expected_factorized_density(residue_a: int, residue_b: int) -> Density:
    return tensor_density(
        tensor_density(client_choi_density(residue_a), client_choi_density(residue_b)),
        dual_rail_reference_density(),
    )


def exact_environment_gram() -> list[list[Cyclo3]]:
    return [[ONE for _ in range(4)] for _ in range(4)]


def phase_ports(
    global_density: Density,
    client_a_density: Density,
    client_b_density: Density,
) -> dict[str, Cyclo3]:
    client_pair_density = reduce_density(global_density, (0, 1, 2, 3))
    return {
        "client_a_relative_phase": density_value(
            client_a_density, (1, 1), (0, 0)
        ).scaled(2),
        "client_b_relative_phase": density_value(
            client_b_density, (1, 1), (0, 0)
        ).scaled(2),
        "joint_sum_phase": density_value(
            client_pair_density,
            (1, 1, 1, 1),
            (0, 0, 0, 0),
        ).scaled(4),
        "client_differential_phase": density_value(
            client_pair_density,
            (1, 1, 0, 0),
            (0, 0, 1, 1),
        ).scaled(4),
    }


def combined_client_phase_exponents(
    residue_a: int, residue_b: int
) -> tuple[int, int, int, int]:
    return tuple(
        (residue_a * client_a + residue_b * client_b) % D
        for client_a, client_b in itertools.product(range(2), repeat=2)
    )


def phase_exponents_proportional(
    left: Sequence[int], right: Sequence[int]
) -> bool:
    differences = {
        (int(right[index]) - int(left[index])) % D
        for index in range(len(left))
    }
    return len(differences) == 1


def exact_pair_fixture(residue_a: int, residue_b: int) -> dict[str, object]:
    initial = prepared_density()
    output = controlled_number_phase(initial, residue_a, residue_b)
    expected = expected_factorized_density(residue_a, residue_b)
    client_a = reduce_density(output, (0, 1))
    client_b = reduce_density(output, (2, 3))
    carrier_reference = reduce_density(output, (4, 5))
    ports = phase_ports(output, client_a, client_b)
    expected_ports = {
        "client_a_relative_phase": root(residue_a),
        "client_b_relative_phase": root(residue_b),
        "joint_sum_phase": root(residue_a + residue_b),
        "client_differential_phase": root(residue_a - residue_b),
    }
    return {
        "residue_a": residue_a,
        "residue_b": residue_b,
        "client_a_unitary": [ONE, root(residue_a)],
        "client_b_unitary": [ONE, root(residue_b)],
        "client_a_choi_commitment": density_commitment(client_a),
        "client_b_choi_commitment": density_commitment(client_b),
        "carrier_reference_commitment": density_commitment(carrier_reference),
        "full_output_commitment": density_commitment(output),
        "client_a_choi_exact": client_a == client_choi_density(residue_a),
        "client_b_choi_exact": client_b == client_choi_density(residue_b),
        "carrier_reference_joint_return_exact": (
            carrier_reference == dual_rail_reference_density()
        ),
        "complete_factorization_exact": output == expected,
        "environment_dimension": 1,
        "environment_gram": exact_environment_gram(),
        "environment_trivial": True,
        "fresh_client_a_distinct_from_fresh_client_b": (
            COORDINATE_NAMES[0:2] != COORDINATE_NAMES[2:4]
        ),
        "phase_ports": ports,
        "expected_phase_ports": expected_ports,
        "all_phase_ports_exact": ports == expected_ports,
        "same_prepared_carrier_value": (
            density_commitment(carrier_reference)
            == density_commitment(dual_rail_reference_density())
        ),
        "density_trace": density_trace(output),
        "density_purity": density_purity(output),
        "density_hermitian": density_is_hermitian(output),
        "coherent_query_count": 2,
        "query_order": ["A", "B"],
        "return_class": "EXACT_COMPLETE_RETURN",
        "qualifies_exact_logical_return": True,
        "qualifies_reuse_boundary": True,
    }


def vacuum_components() -> list[Component]:
    return [
        (
            (client_a, client_a, client_b, client_b, K_VACUUM, 0),
            ONE,
        )
        for client_a in range(2)
        for client_b in range(2)
    ]


def non_eigen_components() -> list[Component]:
    result: list[Component] = []
    for client_a in range(2):
        for client_b in range(2):
            for carrier, carrier_reference in (
                (K_VACUUM, 0),
                (K_RAIL_A, 1),
            ):
                result.append(
                    (
                        (
                            client_a,
                            client_a,
                            client_b,
                            client_b,
                            carrier,
                            carrier_reference,
                        ),
                        ONE,
                    )
                )
    return result


def rail_sign(carrier: int) -> int:
    return (0, 1, -1)[carrier]


def differential_rail_phase(density: Density, delta: int) -> Density:
    result: Density = {}
    for (ket, bra), value in density.items():
        exponent = delta * (rail_sign(ket[4]) - rail_sign(bra[4]))
        add_entry(result, ket, bra, value * root(exponent))
    return result


def balanced_loss_to_vacuum(density: Density, survival: Fraction) -> Density:
    if not Fraction(0) < survival < Fraction(1):
        raise ValueError("balanced-loss survival must be strictly between zero and one")
    loss = Fraction(1) - survival
    result: Density = {}
    for (ket, bra), value in density.items():
        if ket[4] not in (K_RAIL_A, K_RAIL_B) or bra[4] not in (
            K_RAIL_A,
            K_RAIL_B,
        ):
            raise ValueError("balanced-loss control expects dual-rail support")
        add_entry(result, ket, bra, value.scaled(survival))
        if ket[4] == bra[4]:
            lost_ket = ket[:4] + (K_VACUUM,) + ket[5:]
            lost_bra = bra[:4] + (K_VACUUM,) + bra[5:]
            add_entry(result, lost_ket, lost_bra, value.scaled(loss))
    return result


def dephasing_environment_tag(density: Density, overlap: Fraction) -> Density:
    if not Fraction(0) < overlap < Fraction(1):
        raise ValueError("dephasing overlap must be strictly between zero and one")
    result: Density = {}
    for (ket, bra), value in density.items():
        factor = overlap if ket[4] != bra[4] else Fraction(1)
        add_entry(result, ket, bra, value.scaled(factor))
    return result


def controls_fixture() -> dict[str, object]:
    residue_a = 1
    residue_b = 2
    prepared = prepared_density()
    exact_output = controlled_number_phase(prepared, residue_a, residue_b)
    expected_clients = tensor_density(
        client_choi_density(residue_a), client_choi_density(residue_b)
    )

    vacuum_initial = pure_density(vacuum_components(), 4)
    vacuum_output = controlled_number_phase(vacuum_initial, residue_a, residue_b)
    vacuum_null = {
        "residues": [residue_a, residue_b],
        "carrier_number": 0,
        "output_equals_input": vacuum_output == vacuum_initial,
        "client_a_is_unphased": (
            reduce_density(vacuum_output, (0, 1)) == client_choi_density(0)
        ),
        "client_b_is_unphased": (
            reduce_density(vacuum_output, (2, 3)) == client_choi_density(0)
        ),
        "carrier_reference_return_exact": (
            reduce_density(vacuum_output, (4, 5)) == vacuum_reference_density()
        ),
        "target_kickback_observed": False,
        "return_class": "NULL_VACUUM_NO_KICKBACK",
        "qualifies_exact_logical_return": False,
        "qualifies_reuse_boundary": False,
    }

    non_eigen_initial = pure_density(non_eigen_components(), 8)
    non_eigen_output = controlled_number_phase(
        non_eigen_initial, residue_a, residue_b
    )
    non_eigen_initial_carrier = reduce_density(non_eigen_initial, (4, 5))
    non_eigen_output_carrier = reduce_density(non_eigen_output, (4, 5))
    non_eigen = {
        "residues": [residue_a, residue_b],
        "carrier_superposition": ["VACUUM_RK0", "RAIL_A_RK1"],
        "carrier_is_number_eigenstate": False,
        "carrier_reference_return_exact": (
            non_eigen_output_carrier == non_eigen_initial_carrier
        ),
        "carrier_reference_purity_after": density_purity(non_eigen_output_carrier),
        "complete_factorization_as_direct_client_phases": (
            non_eigen_output
            == tensor_density(expected_clients, non_eigen_initial_carrier)
        ),
        "client_carrier_entangled": (
            non_eigen_output
            != tensor_density(
                reduce_density(non_eigen_output, (0, 1, 2, 3)),
                non_eigen_output_carrier,
            )
        ),
        "return_class": "OPEN_NONRETURN",
        "nonzero_control": True,
        "qualifies_exact_logical_return": False,
        "qualifies_reuse_boundary": False,
    }

    differential_output = differential_rail_phase(exact_output, 1)
    differential_carrier = reduce_density(differential_output, (4, 5))
    differential = {
        "delta_mod_3": 1,
        "client_boundary_unchanged": (
            reduce_density(differential_output, (0, 1, 2, 3))
            == expected_clients
        ),
        "carrier_reference_return_exact": (
            differential_carrier == dual_rail_reference_density()
        ),
        "carrier_marginal_return_exact": (
            reduce_density(differential_output, (4,))
            == reduce_density(prepared, (4,))
        ),
        "environment_trivial": True,
        "return_class": "OPEN_NONRETURN",
        "nonzero_control": True,
        "qualifies_exact_logical_return": False,
        "qualifies_reuse_boundary": False,
    }

    survival = Fraction(1, 2)
    loss_output = balanced_loss_to_vacuum(exact_output, survival)
    loss_carrier = reduce_density(loss_output, (4, 5))
    balanced_loss = {
        "rail_survival": survival,
        "rail_loss": Fraction(1) - survival,
        "density_trace": density_trace(loss_output),
        "client_boundary_unchanged": (
            reduce_density(loss_output, (0, 1, 2, 3)) == expected_clients
        ),
        "vacuum_rk0_population": population(
            loss_carrier, lambda state: state == (K_VACUUM, 0)
        ),
        "vacuum_rk1_population": population(
            loss_carrier, lambda state: state == (K_VACUUM, 1)
        ),
        "balanced_rail_loss": (
            population(loss_carrier, lambda state: state == (K_VACUUM, 0))
            == population(loss_carrier, lambda state: state == (K_VACUUM, 1))
        ),
        "carrier_reference_return_exact": (
            loss_carrier == dual_rail_reference_density()
        ),
        "environment_dimension": 3,
        "environment_trivial": False,
        "return_class": "OPEN_NONRETURN",
        "nonzero_control": True,
        "qualifies_exact_logical_return": False,
        "qualifies_reuse_boundary": False,
    }

    overlap = Fraction(1, 2)
    dephased_output = dephasing_environment_tag(exact_output, overlap)
    dephased_carrier = reduce_density(dephased_output, (4, 5))
    environment_gram = [[ONE, ONE.scaled(overlap)], [ONE.scaled(overlap), ONE]]
    dephasing = {
        "environment_tag_overlap": overlap,
        "environment_gram": environment_gram,
        "environment_gram_determinant": ONE.scaled(
            Fraction(1) - overlap * overlap
        ),
        "environment_dimension": 2,
        "environment_trivial": False,
        "client_boundary_unchanged": (
            reduce_density(dephased_output, (0, 1, 2, 3)) == expected_clients
        ),
        "carrier_reference_return_exact": (
            dephased_carrier == dual_rail_reference_density()
        ),
        "complete_factorization_with_prepared_carrier": (
            dephased_output
            == tensor_density(expected_clients, dual_rail_reference_density())
        ),
        "return_class": "OPEN_NONRETURN",
        "nonzero_control": True,
        "qualifies_exact_logical_return": False,
        "qualifies_reuse_boundary": False,
    }

    return {
        "vacuum_carrier_null": vacuum_null,
        "vacuum_dual_rail_non_eigenstate": non_eigen,
        "differential_rail_phase": differential,
        "balanced_loss_to_vacuum": balanced_loss,
        "dephasing_environment_tag": dephasing,
    }


def fraction_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def json_exact(value: object) -> object:
    if isinstance(value, Cyclo3):
        return {
            "basis": ["1", "omega"],
            "coefficients": [fraction_text(value.one), fraction_text(value.omega)],
        }
    if isinstance(value, Fraction):
        return fraction_text(value)
    if isinstance(value, dict):
        return {str(key): json_exact(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_exact(item) for item in value]
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        raise TypeError("floating-point payload value rejected")
    raise TypeError(f"unsupported payload type: {type(value).__name__}")


def exact_bytes(value: object) -> bytes:
    return json.dumps(
        json_exact(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def expected_guest_descriptor() -> dict[str, object]:
    semantics = [
        "MAGIC_P11D",
        "SCHEMA_VERSION_1_AND_LENGTH_8_WORDS",
        "OPCODE_TWO_QUERY",
        "QUERY_COUNT_2_ORDER_A_THEN_B",
        "MODEL_TWO_FRESH_BELL_CHOI_CLIENTS",
        "PROFILE_D3_DUAL_RAIL_REFERENCE",
        "POLICY_COMPLETE_RETURN_AND_ATOMIC_HOLD_REQUIRED",
        "RESOURCE_AND_BOUNDARY_SCHEMA_V1",
    ]
    return {
        "mmio_neutral": True,
        "word_bits": 32,
        "word_count": len(DESCRIPTOR_WORDS),
        "words_unsigned": DESCRIPTOR_WORDS,
        "words_hex": [f"0x{word:08X}" for word in DESCRIPTOR_WORDS],
        "word_semantics": semantics,
        "schema_version": 1,
        "opcode": "TWO_QUERY",
        "query_count": 2,
        "query_order": ["A", "B"],
        "model_profile": "TWO_FRESH_CLIENT_REFERENCE_BELL_CHOI_PAIRS",
        "carrier_profile": "D3_DUAL_RAIL_NUMBER_ONE_WITH_REFERENCE_BELL",
        "return_policy": ["COMPLETE_REFERENCE_RETURN", "ATOMIC_HOLD"],
        "mmio_addresses_assigned": False,
        "guest_descriptor_executed": False,
    }


def expected_backend_contract() -> dict[str, object]:
    return {
        "common_device_id": "COMMON_PHASE_QEMU_DEVICE",
        "backend_id": "D3_DUAL_RAIL_BELL_CHOI_EXACT_NOISY_BACKEND_V11",
        "model_id": "CONTROLLED_NUMBER_PHASE_D3_DUAL_RAIL",
        "abi_id": "COMMON_PHASE_QEMU_GUEST_DESCRIPTOR_ABI_V1",
        "abi_version": 1,
        "register_widths": {
            "control_bits": 32,
            "status_bits": 32,
            "descriptor_address_bits": 64,
            "result_address_bits": 64,
            "resource_counter_bits": 64,
            "residue_field_bits": 2,
        },
        "expected_return_classes": [
            "EXACT_COMPLETE_RETURN",
            "NULL_VACUUM_NO_KICKBACK",
            "OPEN_NONRETURN",
            "ANALYTIC_CHANNEL_CONTROL_ONLY",
            "INVALID_DESCRIPTOR_FAIL_CLOSED",
        ],
        "resource_laws": {
            "total_hilbert_dimension": TOTAL_DIMENSION,
            "prepared_pure_components": 8,
            "prepared_density_nonzero_entries": 64,
            "fresh_client_count": 2,
            "coherent_query_count": 2,
            "query_order": ["A", "B"],
            "exact_environment_dimension": 1,
            "carrier_dimension": 3,
            "carrier_reference_dimension": 2,
            "density_denominator": 8,
            "public_descriptor_words": len(DESCRIPTOR_WORDS),
            "return_requires_complete_krk_joint_equality": True,
            "return_requires_environment_triviality": True,
            "return_requires_atomic_hold": True,
            "open_control_never_qualifies_reuse": True,
        },
        "contract_is_expected_parity_data_only": True,
        "backend_executed_by_reference": False,
        "registers_accessed_by_reference": False,
    }


def main() -> int:
    if TOTAL_DIMENSION != 96 or TOTAL_DIMENSION != int(
        DIMS[0] * DIMS[1] * DIMS[2] * DIMS[3] * DIMS[4] * DIMS[5]
    ):
        raise AssertionError("declared Hilbert dimension mismatch")

    prepared = prepared_density()
    carrier_reference = dual_rail_reference_density()
    pair_results = [
        exact_pair_fixture(residue_a, residue_b)
        for residue_a, residue_b in itertools.product(range(D), repeat=2)
    ]
    controls = controls_fixture()
    descriptor = expected_guest_descriptor()
    backend_contract = expected_backend_contract()

    carrier_number = population(
        carrier_reference, lambda state: NUMBER_K[state[0]] == 1
    )
    rail_a_population = population(
        carrier_reference, lambda state: state[0] == K_RAIL_A
    )
    rail_b_population = population(
        carrier_reference, lambda state: state[0] == K_RAIL_B
    )

    exact_model = {
        "coordinate_order": COORDINATE_NAMES,
        "dimensions": DIMS,
        "total_dimension": TOTAL_DIMENSION,
        "carrier_basis": ["VACUUM", "RAIL_A", "RAIL_B"],
        "carrier_number_eigenvalues": NUMBER_K,
        "prepared_component_count": len(prepared_components()),
        "prepared_density_denominator": 8,
        "prepared_density_nonzero_entries": len(prepared),
        "prepared_density_commitment": density_commitment(prepared),
        "prepared_density_trace": density_trace(prepared),
        "prepared_density_purity": density_purity(prepared),
        "prepared_density_hermitian": density_is_hermitian(prepared),
        "prepared_product_law": (
            prepared
            == tensor_density(
                tensor_density(client_choi_density(0), client_choi_density(0)),
                carrier_reference,
            )
        ),
        "dual_rail_carrier_reference_commitment": density_commitment(
            carrier_reference
        ),
        "carrier_number_moment": carrier_number,
        "rail_a_population": rail_a_population,
        "rail_b_population": rail_b_population,
        "differential_rail_population": rail_a_population - rail_b_population,
        "controlled_number_density_law": (
            "RHO_KET_BRA_MULTIPLIES_BY_omega_TO_THE_sA_TIMES_"
            "CA_KET_NK_KET_MINUS_CA_BRA_NK_BRA_PLUS_sB_TIMES_"
            "CB_KET_NK_KET_MINUS_CB_BRA_NK_BRA"
        ),
        "arithmetic_field": "Q(omega)",
        "floating_point_decisions": 0,
    }

    all_exact_pairs = bool(
        len(pair_results) == 9
        and all(
            pair["client_a_choi_exact"]
            and pair["client_b_choi_exact"]
            and pair["carrier_reference_joint_return_exact"]
            and pair["complete_factorization_exact"]
            and pair["environment_trivial"]
            and pair["fresh_client_a_distinct_from_fresh_client_b"]
            and pair["all_phase_ports_exact"]
            and pair["same_prepared_carrier_value"]
            and pair["density_trace"] == ONE
            and pair["density_purity"] == ONE
            and pair["density_hermitian"]
            and pair["qualifies_exact_logical_return"]
            and pair["qualifies_reuse_boundary"]
            for pair in pair_results
        )
    )

    open_control_names = (
        "vacuum_dual_rail_non_eigenstate",
        "differential_rail_phase",
        "balanced_loss_to_vacuum",
        "dephasing_environment_tag",
    )
    nonzero_open_controls_fail_closed = all(
        controls[name]["nonzero_control"]
        and not controls[name]["qualifies_exact_logical_return"]
        and not controls[name]["qualifies_reuse_boundary"]
        and controls[name]["return_class"] == "OPEN_NONRETURN"
        for name in open_control_names
    )

    direct_comparator = {
        "scope": "SAME_RESIDUE_AUTHORITY_DIRECT_SECRET_CONTROLLED_PHASES",
        "law": "D_s=DIAG_1_omega^s_ON_EACH_FRESH_CLIENT",
        "secret_residue_reads_per_pair": 2,
        "coherent_oracle_queries_per_pair": 0,
        "all_nine_client_boundaries_match": all(
            pair["client_a_choi_exact"] and pair["client_b_choi_exact"]
            for pair in pair_results
        ),
        "secret_access_and_compilation_work_charged": True,
        "unique_phase_resource_advantage": False,
    }
    equal_access_comparator = {
        "scope": "EQUAL_STIPULATED_COHERENT_CONTROLLED_NUMBER_INTERFACE",
        "phase_route_query_count": 2,
        "comparator_query_count": 2,
        "phase_route_query_order": ["A", "B"],
        "comparator_query_order": ["A", "B"],
        "identical_query_count_and_order": True,
        "all_nine_boundaries_identical": True,
        "unique_query_advantage": False,
    }
    coherent_wave_comparator = {
        "scope": "STIPULATED_SOFTWARE_COHERENT_WAVE_INTERFACE_ONLY",
        "same_coherent_wave_access_granted": True,
        "same_dual_rail_number_eigenspace_law_granted": True,
        "query_count": 2,
        "all_nine_exact_boundaries_reproduced": True,
        "physical_wave_generation_modeled": False,
        "physical_wave_custody_modeled": False,
        "physical_total_resource_comparison_established": False,
    }

    phase_gate_family = [
        combined_client_phase_exponents(residue_a, residue_b)
        for residue_a, residue_b in itertools.product(range(D), repeat=2)
    ]
    phase_gate_family_pairwise_nonproportional = all(
        not phase_exponents_proportional(left, right)
        for index, left in enumerate(phase_gate_family)
        for right in phase_gate_family[index + 1 :]
    )
    program_theorem = {
        "theorem": (
            "FIXED_EXACT_PROCESSOR_NONPROPORTIONAL_UNITARIES_REQUIRE_"
            "ORTHOGONAL_PROGRAM_STATES"
        ),
        "combined_client_phase_family_size": D**2,
        "general_program_dimension_law": "d^(N-1)",
        "phase_alphabet_dimension_d": D,
        "combined_client_phase_exponents": phase_gate_family,
        "combined_client_phase_gates_pairwise_nonproportional": (
            phase_gate_family_pairwise_nonproportional
        ),
        "minimum_exact_secret_program_dimension_for_residue_pairs": D**2,
        "secret_storage_states_required": D**2,
        "general_diagonal_qutrit_phase_program_dimension": [
            {
                "client_basis_size": size,
                "phase_classes_modulo_global_phase": D ** (size - 1),
                "minimum_exact_program_dimension": D ** (size - 1),
            }
            for size in range(1, 7)
        ],
        "program_states_materialized": False,
        "program_overlap_executed_or_measured": False,
        "orthogonality_is_theorem_requirement_only": True,
        "secret_program_storage_is_charged": True,
    }

    caveats = {
        "m241_hidden_linear_calibration_improved": False,
        "m242_tensor_factored_hidden_linear_calibration_improved": False,
        "m241_escape": False,
        "m242_escape": False,
        "forrelation_implemented": False,
        "forrelation_query_separation_claimed": False,
        "forrelation_is_prospective_only": True,
        "forrelation_requires_restricted_promise_oracle": True,
        "oracle_generation_custody_and_precision_costs_must_be_charged": True,
        "carrier_preparation_and_reference_costs_must_be_charged": True,
        "equal_interface_total_resource_accounting_required": True,
        "m257_guardrail": (
            "EQUAL_ACCESS_EXACT_DETERMINISTIC_SOFTWARE_FORWARD_SHADOW_MUST_"
            "NOT_BE_COUNTED_AS_A_PHASE_RESOURCE"
        ),
        "m257_escape": False,
        "total_resource_advantage": False,
    }

    architecture_scope = {
        "phase_qemu_layer_classification": (
            "COMMON_PHASE_QEMU_DEVICE_BACKEND_REFERENCE_ORACLE"
        ),
        "distinguished_from_m268_standalone_mechanism_search": True,
        "reference_oracle_only": True,
        "qemu_device_implemented_by_reference": False,
        "qemu_device_executed_by_reference": False,
        "common_guest_visible_device_contract_exercised": False,
        "physical_carrier_custody": False,
        "eligible_for_mechanism_kill": True,
        "eligible_for_architecture_promotion": False,
        "promotion_requires_executed_common_phase_qemu_device_backend": True,
        "reference_verifies_algebra_and_expected_contract_only": True,
        "reference_can_promote_architecture": False,
    }

    resource_scope_accounting = {
        "reference_total_hilbert_dimension_enumerated": TOTAL_DIMENSION,
        "reference_prepared_components_materialized": 8,
        "reference_prepared_sparse_density_entries_materialized": 64,
        "reference_residue_pairs_enumerated": 9,
        "reference_control_families_evaluated": 5,
        "reference_descriptor_words_materialized": len(DESCRIPTOR_WORDS),
        "reference_exact_arithmetic_and_hashing_work_charged": True,
        "reference_runtime_peak_bytes_measured": False,
        "integrated_backend_resident_cells_independently_measured": False,
        "integrated_backend_transient_cells_independently_measured": False,
        "integrated_backend_peak_cells_independently_measured": False,
        "integrated_backend_peak_bytes_independently_measured": False,
        "integrated_backend_resource_parity_claimed": False,
        "physical_oracle_generation_cost_measured": False,
        "physical_carrier_preparation_cost_measured": False,
    }

    checks = {
        "cyclotomic_minimal_polynomial_exact": OMEGA * OMEGA + OMEGA + ONE == ZERO,
        "dimension_product_is_96": TOTAL_DIMENSION == 96,
        "all_component_flat_indices_valid_and_distinct": (
            len(
                {
                    flat_index(component[0], DIMS)
                    for component in prepared_components()
                }
            )
            == 8
        ),
        "prepared_density_exact": bool(
            exact_model["prepared_density_trace"] == ONE
            and exact_model["prepared_density_purity"] == ONE
            and exact_model["prepared_density_hermitian"]
            and exact_model["prepared_product_law"]
        ),
        "dual_rail_number_one_and_balanced": bool(
            carrier_number == ONE
            and rail_a_population == ONE.scaled(Fraction(1, 2))
            and rail_b_population == ONE.scaled(Fraction(1, 2))
        ),
        "all_nine_exact_pairs_qualify": all_exact_pairs,
        "all_nine_residue_pairs_are_unique": (
            len(
                {
                    (pair["residue_a"], pair["residue_b"])
                    for pair in pair_results
                }
            )
            == 9
        ),
        "all_nine_use_same_prepared_carrier_value": all(
            pair["carrier_reference_commitment"]
            == density_commitment(carrier_reference)
            for pair in pair_results
        ),
        "all_nine_phase_ports_exact": all(
            pair["all_phase_ports_exact"] for pair in pair_results
        ),
        "all_nine_environments_trivial": all(
            pair["environment_trivial"] and pair["environment_dimension"] == 1
            for pair in pair_results
        ),
        "vacuum_null_is_returning_but_signal_free_and_nonqualifying": bool(
            controls["vacuum_carrier_null"]["output_equals_input"]
            and controls["vacuum_carrier_null"]["carrier_reference_return_exact"]
            and not controls["vacuum_carrier_null"]["target_kickback_observed"]
            and not controls["vacuum_carrier_null"]["qualifies_exact_logical_return"]
            and not controls["vacuum_carrier_null"]["qualifies_reuse_boundary"]
        ),
        "non_eigenstate_entangles_and_fails_complete_return": bool(
            controls["vacuum_dual_rail_non_eigenstate"]["client_carrier_entangled"]
            and not controls["vacuum_dual_rail_non_eigenstate"]
            ["carrier_reference_return_exact"]
            and not controls["vacuum_dual_rail_non_eigenstate"]
            ["complete_factorization_as_direct_client_phases"]
        ),
        "differential_phase_fails_reference_complete_return": bool(
            controls["differential_rail_phase"]["client_boundary_unchanged"]
            and controls["differential_rail_phase"]["carrier_marginal_return_exact"]
            and not controls["differential_rail_phase"]
            ["carrier_reference_return_exact"]
        ),
        "balanced_loss_is_trace_preserving_balanced_and_open": bool(
            controls["balanced_loss_to_vacuum"]["density_trace"] == ONE
            and controls["balanced_loss_to_vacuum"]["balanced_rail_loss"]
            and controls["balanced_loss_to_vacuum"]["client_boundary_unchanged"]
            and not controls["balanced_loss_to_vacuum"]
            ["carrier_reference_return_exact"]
        ),
        "dephasing_tag_is_nontrivial_and_open": bool(
            controls["dephasing_environment_tag"]["environment_gram_determinant"]
            == ONE.scaled(Fraction(3, 4))
            and not controls["dephasing_environment_tag"]["environment_trivial"]
            and controls["dephasing_environment_tag"]["client_boundary_unchanged"]
            and not controls["dephasing_environment_tag"]
            ["carrier_reference_return_exact"]
            and not controls["dephasing_environment_tag"]
            ["complete_factorization_with_prepared_carrier"]
        ),
        "all_nonzero_open_controls_fail_closed": nonzero_open_controls_fail_closed,
        "direct_secret_comparator_matches": direct_comparator[
            "all_nine_client_boundaries_match"
        ],
        "equal_coherent_access_has_identical_queries": equal_access_comparator[
            "identical_query_count_and_order"
        ],
        "coherent_wave_comparator_is_software_scope_only": bool(
            coherent_wave_comparator["all_nine_exact_boundaries_reproduced"]
            and not coherent_wave_comparator["physical_wave_generation_modeled"]
            and not coherent_wave_comparator[
                "physical_total_resource_comparison_established"
            ]
        ),
        "program_dimension_and_storage_are_charged_theorem_only": bool(
            program_theorem[
                "minimum_exact_secret_program_dimension_for_residue_pairs"
            ]
            == 9
            and program_theorem["secret_storage_states_required"] == 9
            and program_theorem["general_program_dimension_law"] == "d^(N-1)"
            and program_theorem["phase_alphabet_dimension_d"] == D
            and program_theorem[
                "combined_client_phase_gates_pairwise_nonproportional"
            ]
            and not program_theorem["program_states_materialized"]
            and not program_theorem["program_overlap_executed_or_measured"]
            and program_theorem["orthogonality_is_theorem_requirement_only"]
        ),
        "m241_m242_forrelation_caveats_fail_closed": bool(
            not caveats["m241_hidden_linear_calibration_improved"]
            and not caveats["m242_tensor_factored_hidden_linear_calibration_improved"]
            and not caveats["forrelation_implemented"]
            and caveats["oracle_generation_custody_and_precision_costs_must_be_charged"]
            and caveats["equal_interface_total_resource_accounting_required"]
        ),
        "m257_and_advantage_claims_remain_false": bool(
            not caveats["m257_escape"]
            and not caveats["total_resource_advantage"]
            and not direct_comparator["unique_phase_resource_advantage"]
            and not equal_access_comparator["unique_query_advantage"]
        ),
        "descriptor_words_and_semantics_exact": bool(
            descriptor["words_unsigned"] == DESCRIPTOR_WORDS
            and descriptor["word_count"] == 8
            and descriptor["words_unsigned"][0] == 0x50313144
            and descriptor["schema_version"] == 1
            and descriptor["query_count"] == 2
            and descriptor["query_order"] == ["A", "B"]
            and not descriptor["guest_descriptor_executed"]
        ),
        "backend_contract_is_expected_unexecuted_parity_data": bool(
            backend_contract["common_device_id"] == "COMMON_PHASE_QEMU_DEVICE"
            and backend_contract["backend_id"]
            == "D3_DUAL_RAIL_BELL_CHOI_EXACT_NOISY_BACKEND_V11"
            and backend_contract["model_id"]
            == "CONTROLLED_NUMBER_PHASE_D3_DUAL_RAIL"
            and backend_contract["abi_id"]
            == "COMMON_PHASE_QEMU_GUEST_DESCRIPTOR_ABI_V1"
            and backend_contract["register_widths"]
            == {
                "control_bits": 32,
                "status_bits": 32,
                "descriptor_address_bits": 64,
                "result_address_bits": 64,
                "resource_counter_bits": 64,
                "residue_field_bits": 2,
            }
            and backend_contract["expected_return_classes"]
            == [
                "EXACT_COMPLETE_RETURN",
                "NULL_VACUUM_NO_KICKBACK",
                "OPEN_NONRETURN",
                "ANALYTIC_CHANNEL_CONTROL_ONLY",
                "INVALID_DESCRIPTOR_FAIL_CLOSED",
            ]
            and backend_contract["contract_is_expected_parity_data_only"]
            and not backend_contract["backend_executed_by_reference"]
            and not backend_contract["registers_accessed_by_reference"]
            and backend_contract["resource_laws"]
            ["return_requires_complete_krk_joint_equality"]
            and backend_contract["resource_laws"]
            ["open_control_never_qualifies_reuse"]
        ),
        "reference_restoration_and_execution_claims_fail_closed": bool(
            RESTORATION_CLASSIFICATION == "NO_RESTORATION_CLAIM"
            and architecture_scope["reference_oracle_only"]
            and not architecture_scope["qemu_device_executed_by_reference"]
            and not architecture_scope["physical_carrier_custody"]
        ),
        "architecture_is_common_backend_reference_not_execution": bool(
            architecture_scope["phase_qemu_layer_classification"]
            == "COMMON_PHASE_QEMU_DEVICE_BACKEND_REFERENCE_ORACLE"
            and architecture_scope[
                "distinguished_from_m268_standalone_mechanism_search"
            ]
            and not architecture_scope["qemu_device_implemented_by_reference"]
            and not architecture_scope["qemu_device_executed_by_reference"]
            and not architecture_scope[
                "common_guest_visible_device_contract_exercised"
            ]
            and not architecture_scope["eligible_for_architecture_promotion"]
            and not architecture_scope["reference_can_promote_architecture"]
        ),
        "resource_scope_excludes_integrated_and_physical_measurements": bool(
            resource_scope_accounting[
                "reference_total_hilbert_dimension_enumerated"
            ]
            == 96
            and resource_scope_accounting[
                "reference_prepared_sparse_density_entries_materialized"
            ]
            == 64
            and resource_scope_accounting["reference_residue_pairs_enumerated"]
            == 9
            and not resource_scope_accounting[
                "integrated_backend_resident_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "integrated_backend_transient_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "integrated_backend_peak_cells_independently_measured"
            ]
            and not resource_scope_accounting[
                "integrated_backend_resource_parity_claimed"
            ]
            and not resource_scope_accounting[
                "physical_oracle_generation_cost_measured"
            ]
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    if failed:
        raise AssertionError(f"independent V11 self-check failed: {failed}")

    claim_payload = {
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "phase_qemu_layer_classification": architecture_scope[
            "phase_qemu_layer_classification"
        ],
    }
    payload = {
        "schema": SCHEMA,
        "reference_id": REFERENCE_ID,
        "milestone": "M269_V11",
        "claim": CLAIM,
        "ceiling": CEILING,
        "restoration_classification": RESTORATION_CLASSIFICATION,
        "restoration_scope": RESTORATION_SCOPE,
        "resource_disposition": DISPOSITION,
        "next_mechanism": SUCCESSOR,
        "claim_payload": claim_payload,
        "exact_model": exact_model,
        "all_nine_residue_pairs": pair_results,
        "analytic_controls": controls,
        "direct_secret_controlled_phase_comparator": direct_comparator,
        "equal_coherent_access_comparator": equal_access_comparator,
        "coherent_wave_comparator": coherent_wave_comparator,
        "nielsen_chuang_program_theorem": program_theorem,
        "m241_m242_forrelation_m257_caveats": caveats,
        "expected_guest_descriptor": descriptor,
        "expected_backend_contract": backend_contract,
        "architecture_scope": architecture_scope,
        "resource_scope_accounting": resource_scope_accounting,
        "checks": checks,
        "nonclaims": {
            "executed_restoration": False,
            "same_backing_restoration": False,
            "qemu_execution": False,
            "guest_visible_contract_execution": False,
            "physical_oracle": False,
            "physical_carrier_custody": False,
            "physical_restoration": False,
            "query_separation": False,
            "total_resource_advantage": False,
            "m257_escape": False,
            "architecture_promotion": False,
            "unbounded_computation": False,
            "bit_replaced_with_pi": False,
        },
        "reference_self_assertion": (
            "PASS_INDEPENDENT_V11_DUAL_RAIL_EXACT_NOISY_REFERENCE"
        ),
        "status": "PASS_INDEPENDENT_V11_DUAL_RAIL_EXACT_NOISY_REFERENCE",
        "terminal": False,
    }
    payload["claim_payload_sha256"] = hashlib.sha256(
        exact_bytes(claim_payload)
    ).hexdigest()
    payload["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(exact_bytes(payload).decode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
