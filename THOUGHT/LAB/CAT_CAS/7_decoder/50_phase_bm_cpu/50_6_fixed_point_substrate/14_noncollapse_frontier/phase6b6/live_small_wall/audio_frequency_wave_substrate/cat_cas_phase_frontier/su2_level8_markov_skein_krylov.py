#!/usr/bin/env python3
"""Package the exact M218 link-pattern skein/Markov-boundary diagnostic.

This is a representation and boundary-functional change from M217.  The
carrier is the noncrossing link-pattern module, each braid generator is
applied by the native Kauffman skein law ``A I + A^-1 e_i``, and the only
projection is the normalized public cup/cap Markov closure.  At at most
sixteen strands the level-eight Jones--Wenzl relation has not yet removed a
link pattern, so the package explicitly measures the pre-truncation ceiling.

The result is a negative diagnostic if the Markov scalar Krylov degree still
equals the full link-pattern dimension.  It does not claim phase/classical
separation, CATVM custody, a Small Wall crossing, or a physical computation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import su2_level8_fusion_path_braid_phase_relation as braid


PRIMARY_STRANDS = 16
PRIMARY_ROUNDS = 10
PRIMARY_FAMILY = 0
REUSE_ROUNDS = 7
REUSE_FAMILY = 1
SPLIT_PRIMES = (241, 401)
EXPECTED_DIMENSIONS = (2, 5, 14, 42, 132, 429, 1430)
DELTA = braid.QUANTUM_DIMENSIONS[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pairing_token(pairing: tuple[int, ...]) -> str:
    return ",".join(map(str, pairing))


def path_to_pairing(path: tuple[int, ...]) -> tuple[int, ...]:
    pairing = [-1] * (len(path) - 1)
    stack: list[int] = []
    for position, (left, right) in enumerate(zip(path, path[1:])):
        if right == left + 1:
            stack.append(position)
        elif right == left - 1 and stack:
            peer = stack.pop()
            pairing[position] = peer
            pairing[peer] = position
        else:
            raise ValueError("fusion path is not a Dyck path")
    if stack or any(peer < 0 for peer in pairing):
        raise ValueError("Dyck path did not close to a pairing")
    return tuple(pairing)


@dataclass(frozen=True)
class DiagramTopology:
    strands: int
    pairings: tuple[tuple[int, ...], ...]
    index: dict[tuple[int, ...], int]
    cup_index: int
    e_targets: tuple[tuple[int, ...], ...]
    e_delta_flags: tuple[tuple[bool, ...], ...]
    commitment: str

    @staticmethod
    def compile(strands: int) -> "DiagramTopology":
        paths = braid.FusionPathTopology.compile(strands)
        path_work = braid.Work()
        pairings = tuple(
            path_to_pairing(paths.unrank(index, path_work))
            for index in range(paths.dimension)
        )
        index = {pairing: position for position, pairing in enumerate(pairings)}
        if len(index) != len(pairings):
            raise RuntimeError("link-pattern topology contains duplicates")
        cups = tuple(position ^ 1 for position in range(strands))
        cup_index = index[cups]
        targets: list[tuple[int, ...]] = [tuple()]
        flags: list[tuple[bool, ...]] = [tuple()]
        for generator in range(1, strands):
            left, right = generator - 1, generator
            generator_targets = []
            generator_flags = []
            for pairing in pairings:
                if pairing[left] == right:
                    generator_targets.append(index[pairing])
                    generator_flags.append(True)
                    continue
                transformed = list(pairing)
                left_peer, right_peer = transformed[left], transformed[right]
                transformed[left], transformed[right] = right, left
                transformed[left_peer], transformed[right_peer] = (
                    right_peer,
                    left_peer,
                )
                transformed_tuple = tuple(transformed)
                if transformed_tuple not in index:
                    raise RuntimeError("skein reconnection left noncrossing basis")
                generator_targets.append(index[transformed_tuple])
                generator_flags.append(False)
            targets.append(tuple(generator_targets))
            flags.append(tuple(generator_flags))
        token = f"strands:{strands}|" + "|".join(map(pairing_token, pairings))
        return DiagramTopology(
            strands,
            pairings,
            index,
            cup_index,
            tuple(targets),
            tuple(flags),
            hashlib.sha256(token.encode("ascii")).hexdigest(),
        )

    @property
    def dimension(self) -> int:
        return len(self.pairings)

    @property
    def retained_pairing_integer_cells(self) -> int:
        return self.dimension * self.strands

    @property
    def retained_action_records(self) -> int:
        return (self.strands - 1) * self.dimension

    @property
    def retained_action_integer_cells(self) -> int:
        return 2 * self.retained_action_records


def union_loop_count(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    seen: set[int] = set()
    loops = 0
    for start in range(len(left)):
        if start in seen:
            continue
        loops += 1
        pending = [start]
        while pending:
            item = pending.pop()
            if item in seen:
                continue
            seen.add(item)
            pending.extend((left[item], right[item]))
    return loops


@dataclass
class Work:
    field_additions: int = 0
    field_multiplications: int = 0
    gate_applications: int = 0
    skein_column_updates: int = 0
    closure_pairings_streamed: int = 0
    closure_loop_graph_integer_visits: int = 0
    forward_operations: int = 0
    inverse_operations: int = 0
    port_leases: int = 0
    port_releases: int = 0
    public_descriptor_hashes: int = 0
    public_descriptor_integers_hashed: int = 0
    state_commitment_hashes: int = 0
    state_commitment_field_cells_hashed: int = 0
    boundary_commitment_hashes: int = 0
    peak_carrier_field_cells: int = 0
    peak_scratch_field_cells: int = 0
    peak_total_live_field_cells: int = 0
    peak_carrier_payload_bits: int = 0
    peak_scratch_payload_bits: int = 0

    def add(self, left: braid.K, right: braid.K) -> braid.K:
        self.field_additions += 1
        return left + right

    def multiply(self, left: braid.K, right: braid.K) -> braid.K:
        self.field_multiplications += 1
        return left * right

    def observe(self, state: list[braid.K], scratch: list[braid.K]) -> None:
        state_bits = braid.field_payload_bits(state)
        scratch_bits = braid.field_payload_bits(scratch)
        self.peak_carrier_field_cells = max(self.peak_carrier_field_cells, len(state))
        self.peak_scratch_field_cells = max(self.peak_scratch_field_cells, len(scratch))
        self.peak_total_live_field_cells = max(
            self.peak_total_live_field_cells, len(state) + len(scratch)
        )
        self.peak_carrier_payload_bits = max(self.peak_carrier_payload_bits, state_bits)
        self.peak_scratch_payload_bits = max(self.peak_scratch_payload_bits, scratch_bits)

    def as_dict(self) -> dict[str, int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def state_commitment(values: list[braid.K]) -> str:
    return hashlib.sha256(
        "|".join(value.token() for value in values).encode("ascii")
    ).hexdigest()


def program_commitment(program: braid.BraidProgram) -> str:
    return hashlib.sha256(program.token().encode("ascii")).hexdigest()


def apply_gate(
    coefficients: list[braid.K],
    scratch: list[braid.K],
    topology: DiagramTopology,
    operation: braid.BraidOperation,
    work: Work,
) -> None:
    alpha, beta = braid.local_braid_scalars(operation.exponent)
    scratch[:] = [braid.ZERO] * topology.dimension
    targets = topology.e_targets[operation.generator]
    delta_flags = topology.e_delta_flags[operation.generator]
    for column, value in enumerate(coefficients):
        scratch[column] = work.add(scratch[column], work.multiply(alpha, value))
        e_value = work.multiply(DELTA, value) if delta_flags[column] else value
        row = targets[column]
        scratch[row] = work.add(scratch[row], work.multiply(beta, e_value))
        work.skein_column_updates += 1
    coefficients[:] = scratch
    work.gate_applications += 1
    work.observe(coefficients, scratch)


def normalized_markov_boundary(
    coefficients: list[braid.K], topology: DiagramTopology, work: Work
) -> braid.K:
    cups = topology.pairings[topology.cup_index]
    normalization = braid.ONE
    for _ in range(topology.strands // 2):
        normalization = work.multiply(normalization, DELTA)
    inverse_normalization = normalization.inverse()
    result = braid.ZERO
    for value, pairing in zip(coefficients, topology.pairings, strict=True):
        loops = union_loop_count(cups, pairing)
        weight = braid.ONE
        for _ in range(loops):
            weight = work.multiply(weight, DELTA)
        result = work.add(result, work.multiply(value, weight))
        work.closure_pairings_streamed += 1
        work.closure_loop_graph_integer_visits += 2 * topology.strands
    return work.multiply(result, inverse_normalization)


@dataclass
class OpenDiagramPort:
    topology: DiagramTopology
    coefficients: list[braid.K]
    scratch: list[braid.K]
    live: bool = False
    owner: int = 0
    lease_generation: int = 0
    cursor: int = 0
    expected_steps: int = 0
    public_program_commitment: str = ""
    topology_commitment: str = ""

    def lease(
        self, owner: int, generation: int, program: braid.BraidProgram, work: Work
    ) -> None:
        if self.live:
            raise RuntimeError("diagram port already live")
        if len(self.coefficients) != self.topology.dimension or len(self.scratch) != self.topology.dimension:
            raise ValueError("null or wrong-width diagram carrier")
        if program.strands != self.topology.strands or owner <= 0 or generation <= 0:
            raise ValueError("invalid diagram lease descriptor")
        self.live = True
        self.owner = owner
        self.lease_generation = generation
        self.cursor = 0
        self.expected_steps = program.steps
        self.public_program_commitment = program_commitment(program)
        self.topology_commitment = self.topology.commitment
        work.public_descriptor_hashes += 2
        work.public_descriptor_integers_hashed += 3 + self.topology.retained_pairing_integer_cells
        work.port_leases += 1
        work.observe(self.coefficients, self.scratch)

    def require(self, owner: int, program: braid.BraidProgram, work: Work) -> None:
        if not self.live:
            raise RuntimeError("diagram port is not live")
        if owner != self.owner:
            raise PermissionError("diagram owner mismatch")
        if program_commitment(program) != self.public_program_commitment:
            raise ValueError("public braid program mismatch")
        if self.topology.commitment != self.topology_commitment:
            raise ValueError("public diagram topology mismatch")
        work.public_descriptor_hashes += 2
        work.public_descriptor_integers_hashed += 3 + self.topology.retained_pairing_integer_cells

    def forward(
        self, owner: int, program: braid.BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor or index >= self.expected_steps:
            raise ValueError("forward diagram cursor mismatch")
        apply_gate(self.coefficients, self.scratch, self.topology, program.operation(index), work)
        self.cursor += 1
        work.forward_operations += 1

    def inverse(
        self, owner: int, program: braid.BraidProgram, index: int, work: Work
    ) -> None:
        self.require(owner, program, work)
        if index != self.cursor - 1:
            raise ValueError("inverse diagram cursor mismatch")
        operation = program.operation(index)
        apply_gate(
            self.coefficients,
            self.scratch,
            self.topology,
            braid.BraidOperation(operation.generator, -operation.exponent),
            work,
        )
        self.cursor -= 1
        work.inverse_operations += 1

    def project_final(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> braid.K:
        self.require(owner, program, work)
        if self.cursor != self.expected_steps:
            raise PermissionError("nonfinal diagram projection rejected")
        return normalized_markov_boundary(self.coefficients, self.topology, work)

    def release(
        self, owner: int, program: braid.BraidProgram, work: Work
    ) -> int:
        self.require(owner, program, work)
        if self.cursor:
            raise RuntimeError("diagram port released before inverse")
        generation = self.lease_generation
        self.live = False
        self.owner = 0
        self.lease_generation = 0
        self.expected_steps = 0
        self.public_program_commitment = ""
        self.topology_commitment = ""
        self.scratch[:] = [braid.ZERO] * self.topology.dimension
        work.port_releases += 1
        return generation


@dataclass
class Carrier:
    port: OpenDiagramPort
    restoration_generation: int = 0


def source_state(topology: DiagramTopology) -> list[braid.K]:
    result = [braid.ZERO] * topology.dimension
    result[topology.cup_index] = braid.ONE
    return result


def canonical_restoration(
    carrier: Carrier, source: list[braid.K], generation: int
) -> bool:
    port = carrier.port
    return (
        port.coefficients == source
        and all(value == braid.ZERO for value in port.scratch)
        and not port.live
        and port.owner == 0
        and port.lease_generation == 0
        and port.cursor == 0
        and port.expected_steps == 0
        and port.public_program_commitment == ""
        and port.topology_commitment == ""
        and carrier.restoration_generation == generation
    )


def transaction(
    carrier: Carrier, source: list[braid.K], program: braid.BraidProgram
) -> tuple[dict[str, Any], Work, braid.K]:
    coefficient_backing = id(carrier.port.coefficients)
    scratch_backing = id(carrier.port.scratch)
    generation = carrier.restoration_generation + 1
    owner = 218000 + generation
    work = Work()
    carrier.port.lease(owner, generation, program, work)
    for index in range(program.steps):
        carrier.port.forward(owner, program, index, work)
    boundary = carrier.port.project_final(owner, program, work)
    forward_commitment = state_commitment(carrier.port.coefficients)
    work.state_commitment_hashes += 1
    work.state_commitment_field_cells_hashed += len(carrier.port.coefficients)
    boundary_commitment = braid.boundary_commitment(boundary)
    work.boundary_commitment_hashes += 1
    forward_payload = braid.field_payload_bits(carrier.port.coefficients)
    forward_nonzero = sum(value != braid.ZERO for value in carrier.port.coefficients)
    missing_inverse_error = sum(
        left != right
        for left, right in zip(carrier.port.coefficients, source, strict=True)
    ) + carrier.port.cursor
    for index in range(program.steps - 1, -1, -1):
        carrier.port.inverse(owner, program, index, work)
    carrier.restoration_generation = carrier.port.release(owner, program, work)
    return (
        {
            "boundary_commitment": boundary_commitment,
            "boundary_payload_bits": braid.field_payload_bits([boundary]),
            "forward_state_commitment": forward_commitment,
            "forward_field_cells": len(carrier.port.coefficients),
            "forward_nonzero_field_cells": forward_nonzero,
            "forward_payload_bits": forward_payload,
            "missing_inverse_error_cells_and_cursor": missing_inverse_error,
            "restoration_error_field_cells": sum(
                left != right
                for left, right in zip(carrier.port.coefficients, source, strict=True)
            ),
            "same_coefficient_backing": id(carrier.port.coefficients) == coefficient_backing,
            "same_scratch_backing": id(carrier.port.scratch) == scratch_backing,
            "canonical_post_restoration_state_exact": canonical_restoration(
                carrier, source, generation
            ),
            "restoration_generation": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        work,
        boundary,
    )


def primitive_root(prime: int) -> int:
    residual = prime - 1
    factors = []
    candidate = 2
    while candidate * candidate <= residual:
        if residual % candidate == 0:
            factors.append(candidate)
            while residual % candidate == 0:
                residual //= candidate
        candidate += 1
    if residual > 1:
        factors.append(residual)
    return next(
        value
        for value in range(2, prime)
        if all(pow(value, (prime - 1) // factor, prime) != 1 for factor in factors)
    )


def root40(prime: int) -> int:
    return pow(primitive_root(prime), (prime - 1) // 40, prime)


def evaluate_exact(value: braid.K, prime: int) -> int:
    root = root40(prime)
    return sum(
        coefficient.numerator
        * pow(coefficient.denominator, -1, prime)
        * pow(root, exponent, prime)
        for exponent, coefficient in enumerate(value.coefficients)
    ) % prime


def load_core(helper: Path) -> dict[str, Any]:
    resolved = helper.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("compiled M218 core is missing or not executable")
    if str(resolved).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M218 helper is forbidden")
    completed = subprocess.run(
        [str(resolved)], check=True, capture_output=True, text=True
    )
    if completed.stderr:
        raise RuntimeError("M218 core emitted unexpected stderr")
    result = json.loads(completed.stdout)
    if result.get("schema") != "cat_cas.su2_level8_markov_skein_krylov_core.v1":
        raise RuntimeError("M218 core schema changed")
    return result


def load_reference(helper: Path) -> dict[str, Any]:
    resolved = helper.resolve()
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise ValueError("compiled M218 separate reference is missing or not executable")
    if str(resolved).startswith(("/dev/shm/", "/run/shm/")):
        raise ValueError("RAM-backed M218 reference is forbidden")
    completed = subprocess.run(
        [str(resolved)], check=True, capture_output=True, text=True
    )
    if completed.stderr:
        raise RuntimeError("M218 separate reference emitted unexpected stderr")
    result = json.loads(completed.stdout)
    if result.get("schema") != "cat_cas.su2_level8_markov_skein_krylov_separate_reference.v1":
        raise RuntimeError("M218 separate-reference schema changed")
    return result


def exact_transactions() -> tuple[dict[str, Any], braid.K]:
    topology = DiagramTopology.compile(PRIMARY_STRANDS)
    source = source_state(topology)
    carrier = Carrier(OpenDiagramPort(topology, source.copy(), [braid.ZERO] * topology.dimension))
    primary_program = braid.BraidProgram(PRIMARY_STRANDS, PRIMARY_ROUNDS, PRIMARY_FAMILY)
    reuse_program = braid.BraidProgram(PRIMARY_STRANDS, REUSE_ROUNDS, REUSE_FAMILY)
    primary, primary_work, boundary = transaction(carrier, source, primary_program)
    reuse, reuse_work, _ = transaction(carrier, source, reuse_program)
    fresh = Carrier(OpenDiagramPort(topology, source.copy(), [braid.ZERO] * topology.dimension))
    fresh_reuse, fresh_work, _ = transaction(fresh, source, reuse_program)
    if reuse["boundary_commitment"] != fresh_reuse["boundary_commitment"]:
        raise RuntimeError("restored diagram reuse boundary differs from fresh")
    if reuse["forward_state_commitment"] != fresh_reuse["forward_state_commitment"]:
        raise RuntimeError("restored diagram reuse state differs from fresh")
    return (
        {
            "topology": {
                "link_pattern_cells": topology.dimension,
                "retained_pairing_integer_cells": topology.retained_pairing_integer_cells,
                "retained_skein_action_records": topology.retained_action_records,
                "retained_skein_action_integer_cells": topology.retained_action_integer_cells,
                "jones_wenzl_truncation_active": False,
                "first_possible_truncation_strands": 18,
            },
            "primary_program": {
                "strands": PRIMARY_STRANDS,
                "rounds": PRIMARY_ROUNDS,
                "family": PRIMARY_FAMILY,
                "steps": primary_program.steps,
            },
            "primary": primary,
            "primary_full_lifecycle_work": primary_work.as_dict(),
            "reuse_program": {
                "strands": PRIMARY_STRANDS,
                "rounds": REUSE_ROUNDS,
                "family": REUSE_FAMILY,
                "steps": reuse_program.steps,
            },
            "reuse": reuse,
            "reuse_full_lifecycle_work": reuse_work.as_dict(),
            "fresh_reuse": fresh_reuse,
            "fresh_reuse_full_lifecycle_work": fresh_work.as_dict(),
            "fresh_restored_reuse_boundary_agreement": True,
            "fresh_restored_reuse_state_agreement": True,
            "restoration_generation_after_reuse": carrier.restoration_generation,
            "baseline_reload_used": False,
        },
        boundary,
    )


def controls() -> dict[str, bool]:
    topology = DiagramTopology.compile(6)
    source = source_state(topology)
    accepted = braid.BraidProgram(6, 2, 0)
    wrong = braid.BraidProgram(6, 2, 1)
    port = OpenDiagramPort(topology, source.copy(), [braid.ZERO] * topology.dimension)
    work = Work()
    port.lease(218900, 1, accepted, work)
    wrong_owner = wrong_type = premature = wrong_program = reordered = False
    try:
        port.forward(218901, accepted, 0, work)
    except PermissionError:
        wrong_owner = True
    try:
        port.forward(218900, object(), 0, work)  # type: ignore[arg-type]
    except (AttributeError, TypeError, ValueError):
        wrong_type = True
    try:
        port.project_final(218900, accepted, work)
    except PermissionError:
        premature = True
    for index in range(accepted.steps):
        port.forward(218900, accepted, index, work)
    try:
        port.inverse(218900, wrong, accepted.steps - 1, work)
    except ValueError:
        wrong_program = True
    try:
        port.inverse(218900, accepted, accepted.steps - 2, work)
    except ValueError:
        reordered = True
    missing = port.coefficients != source and port.cursor == accepted.steps
    for index in range(accepted.steps - 1, -1, -1):
        port.inverse(218900, accepted, index, work)
    port.release(218900, accepted, work)
    null_rejected = False
    try:
        OpenDiagramPort(topology, [], []).lease(1, 1, accepted, Work())
    except ValueError:
        null_rejected = True
    alternate_program_parity = True
    for strands, rounds, family in ((4, 1, 1), (6, 3, 0), (8, 4, 1), (10, 2, 0)):
        alternate_topology = DiagramTopology.compile(strands)
        alternate_source = source_state(alternate_topology)
        alternate_program = braid.BraidProgram(strands, rounds, family)
        alternate_carrier = Carrier(
            OpenDiagramPort(
                alternate_topology,
                alternate_source.copy(),
                [braid.ZERO] * alternate_topology.dimension,
            )
        )
        alternate_result, _, alternate_boundary = transaction(
            alternate_carrier, alternate_source, alternate_program
        )
        path_topology, path_state, _ = braid.execute_forward(alternate_program)
        path_boundary = path_state[
            path_topology.rank(braid.vacuum_path(strands))
        ]
        alternate_program_parity = alternate_program_parity and (
            alternate_boundary == path_boundary
            and alternate_result["canonical_post_restoration_state_exact"]
        )
    return {
        "wrong_owner_rejected": wrong_owner,
        "wrong_operation_type_rejected": wrong_type,
        "premature_projection_rejected": premature,
        "wrong_public_program_inverse_rejected": wrong_program,
        "reordered_inverse_rejected": reordered,
        "missing_inverse_detected": missing,
        "null_carrier_rejected": null_rejected,
        "alternate_public_program_family_boundary_parity": alternate_program_parity,
        "intermediate_link_pattern_state_projected": False,
        "snapshot_command_available": hasattr(port, "snapshot"),
    }


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: su2_level8_markov_skein_krylov.py COMPILED_CORE COMPILED_REFERENCE"
        )
    here = Path(__file__).resolve().parent
    core = load_core(Path(sys.argv[1]))
    reference = load_reference(Path(sys.argv[2]))
    cases = core["cases"]
    dimensions = [case["link_pattern_cells"] for case in cases if case["family"] == 0]
    if tuple(dimensions) != EXPECTED_DIMENSIONS:
        raise RuntimeError("declared link-pattern dimension law changed")
    if not all(
        item["markov_scalar_recurrence_degree"] == case["link_pattern_cells"]
        and item["holdout_violations"] == 0
        and item["semantic_perturbation_changes_prefix"]
        for case in cases
        for item in case["prime_results"]
    ):
        raise RuntimeError("Markov-boundary full-degree diagnostic failed")
    if (
        reference.get("reference_imports_m218_production") is not False
        or not reference.get("all_cases_full_at_both_distinct_split_primes")
        or [case["link_pattern_cells"] for case in reference["cases"] if case["family"] == 0]
        != list(EXPECTED_DIMENSIONS)
        or not all(
            item["markov_scalar_recurrence_degree"] == case["link_pattern_cells"]
            and item["holdout_violations"] == 0
            for case in reference["cases"]
            for item in case["prime_results"]
        )
    ):
        raise RuntimeError("separate Markov-skein reference failed")
    m217 = json.loads(
        (here / "SU2_LEVEL8_PERIOD10_MONODROMY_KRYLOV_RESULTS.json").read_text()
    )
    sequences_identical_to_m217 = all(
        left["strands"] == right["strands"]
        and left["family"] == right["family"]
        and all(
            lp["first_terms"] == rp["first_terms"]
            and lp["sequence_digest_fnv1a64"] == rp["sequence_digest_fnv1a64"]
            for lp, rp in zip(left["prime_results"], right["prime_results"], strict=True)
        )
        for left, right in zip(cases, m217["core_diagnostic"]["cases"], strict=True)
    )
    if not sequences_identical_to_m217:
        raise RuntimeError("diagram Markov and M217 vacuum sequences diverged")
    transaction_result, exact_boundary = exact_transactions()
    primary_core = next(
        case for case in cases
        if case["strands"] == PRIMARY_STRANDS and case["family"] == PRIMARY_FAMILY
    )
    exact_modular = {str(prime): evaluate_exact(exact_boundary, prime) for prime in SPLIT_PRIMES}
    core_modular = {
        str(item["prime"]): item["first_terms"][1]
        for item in primary_core["prime_results"]
    }
    if exact_modular != core_modular:
        raise RuntimeError("exact and modular Markov boundaries disagree")
    primary_prime = primary_core["prime_results"][0]
    result = {
        "schema": "cat_cas.su2_level8_markov_skein_krylov.v1",
        "result": "PASS_EXACT_SU2_LEVEL8_MARKOV_SKEIN_FULL_BOUNDARY_DEGREE_OBSTRUCTION",
        "claim": "EXACT_PRETRUNCATION_KAUFFMAN_TEMPERLEY_LIEB_LINK_PATTERN_SKEIN_AND_FUSION_PATH_VACUUM_BOUNDARY_SEQUENCES_ARE_IDENTICAL_FOR_THE_PUBLIC_PERIOD10_SU2_LEVEL8_FAMILIES_AND_HAVE_FULL_MINIMAL_DEGREES2_5_14_42_132_429_1430_ACROSS_STRANDS4TO16_WHILE_THE_SKEIN_PATH_REQUIRES1430_CARRIER_PLUS1430_SCRATCH_CELLS_WITH_FINAL_ONLY_BOUNDARY_EXACT_RESTORATION_REUSE_AND_THE_IDENTICAL_CLASSICAL_SKEIN_RECURRENCE",
        "classification": "INDEPENDENTLY_VERIFIED_STRICT_SCOPE",
        "verification_level": "SEPARATE_REFERENCE_PARITY",
        "restoration_classification": "EXACT_ALGEBRAIC_RESTORATION",
        "claim_ceiling": "FORMAL_PRETRUNCATION_SU2_LEVEL8_QZETA40_NONCROSSING_LINK_PATTERN_MODULE_PUBLIC_PERIOD10_FAMILIES0_1_EVEN_STRANDS4_TO16_NORMALIZED_CUP_CAP_MARKOV_BOUNDARY_PRIMARY_SPLIT_PRIMES241_401_REFERENCE_SPLIT_PRIMES641_881_PRIMARY16_FAMILY0_REUSE16_ROUNDS7_FAMILY1_DIRECT_PROCESS_ONLY",
        "degree_law": {
            "strands": [4, 6, 8, 10, 12, 14, 16],
            "link_pattern_dimensions": list(EXPECTED_DIMENSIONS),
            "both_family_markov_minimal_degrees": list(EXPECTED_DIMENSIONS),
            "all_cases_full_at_both_primes": True,
            "jones_wenzl_truncation_active": False,
            "first_possible_truncation_strands": 18,
            "uniform_fixed_degree_markov_recurrence_rejected_for_declared_family": True,
            "arbitrary_braid_program_lower_bound": False,
            "all_primary_markov_sequences_identical_to_m217_vacuum_sequences": sequences_identical_to_m217,
        },
        "core_diagnostic": core,
        "separate_reference": reference,
        "representation_parity": {
            "all_primary_markov_sequences_identical_to_m217_vacuum_sequences": sequences_identical_to_m217,
            "exact_primary_boundary_identical_to_m217": transaction_result["primary"]["boundary_commitment"]
            == m217["transaction"]["primary"]["boundary_commitment"],
            "interpretation": "THE_LINK_PATTERN_SKEIN_IS_A_BASIS_LEVEL_REEXPRESSION_OF_THE_SAME_PRETRUNCATION_TEMPERLEY_LIEB_MODULE_NOT_A_NEW_COMPACT_CARRIER",
        },
        "exact_boundary_modular_parity": {
            "exact_boundary_commitment": transaction_result["primary"]["boundary_commitment"],
            "exact_modular_boundaries": exact_modular,
            "core_first_period_boundaries": core_modular,
            "agreement": True,
        },
        "transaction": transaction_result,
        "controls": {
            **controls(),
            "compiler_uses_only_public_topology_and_word": True,
            "compiler_reads_final_answer": False,
            "both_split_prime_degrees_full": True,
            "both_distinct_reference_split_prime_degrees_full": True,
            "exact_primary_boundary_reduces_to_both_modular_boundaries": True,
        },
        "resource_law": {
            "primary_exact_carrier_field_cells": transaction_result["primary"]["forward_field_cells"],
            "primary_exact_scratch_field_cells": primary_core["link_pattern_cells"],
            "primary_exact_peak_carrier_plus_scratch_field_cells": 2 * primary_core["link_pattern_cells"],
            "primary_retained_pairing_integer_cells": primary_core["retained_pairing_integer_cells"],
            "primary_retained_skein_action_records": primary_core["retained_skein_action_records"],
            "primary_retained_skein_action_integer_cells": primary_core["retained_skein_action_integer_cells"],
            "primary_scalar_sequence_field_cells": 2 * primary_core["link_pattern_cells"] + 64,
            "primary_public_recurrence_coefficient_slots": primary_prime["markov_scalar_recurrence_degree"] + 1,
            "primary_public_recurrence_nonzero_coefficients": primary_prime["nonzero_recurrence_coefficients"],
            "primary_period_applications_per_prime": 2 * primary_core["link_pattern_cells"] + 63,
            "primary_local_gate_applications_per_prime": (2 * primary_core["link_pattern_cells"] + 63) * 10 * (PRIMARY_STRANDS - 1),
            "accepted_transaction_retained_inverse_history": 0,
            "controller_backend_traffic_bytes": 0,
            "snapshot_traffic_bytes": 0,
            "baseline_reload_bytes": 0,
            "excluded_not_zero": "PYTHON_AND_CPP_CONTAINER_CAPACITY_HASH_TABLE_ALLOCATOR_PROCESS_IMAGE_JSON_SERIALIZATION_TIMING_AND_WHOLE_PROCESS_PEAKS",
        },
        "matched_classical_baselines": {
            "strongest_compact": "IDENTICAL_NONCROSSING_LINK_PATTERN_SKEIN_TRANSFER_AND_MARKOV_CLOSURE_RECURRENCE",
            "same_state_scratch_and_full_degree_law": True,
            "phase_specific_reduction": False,
            "computational_advantage": False,
        },
        "claim_limits": {
            "jones_wenzl_reduced_diagram_carrier_established": False,
            "first_truncated_18_strand_case_executed": False,
            "compact_markov_boundary_recurrence_established": False,
            "nonlinear_diagram_quotient_lower_bound": False,
            "catvm_custody": False,
            "distinct_phase_resource_established": False,
            "computational_advantage": False,
            "small_wall_crossed": False,
            "physical_waveform_execution": False,
            "physical_bit_replacement": False,
            "catalytic_inference_established": False,
            "unbounded_computation_established": False,
        },
        "source_dependencies": {
            "m214_exact_field_and_public_word_sha256": sha256_file(here / "su2_level8_fusion_path_braid_phase_relation.py"),
            "m217_modular_primitives_sha256": sha256_file(here / "su2_level8_period10_monodromy_krylov_core.cpp"),
            "m218_core_sha256": sha256_file(here / "su2_level8_markov_skein_krylov_core.cpp"),
            "m217_separate_reference_primitives_sha256": sha256_file(here / "su2_level8_period10_monodromy_krylov_separate_reference.cpp"),
            "m218_separate_reference_sha256": sha256_file(here / "su2_level8_markov_skein_krylov_separate_reference.cpp"),
            "m218_wrapper_sha256": sha256_file(Path(__file__).resolve()),
        },
        "terminal": False,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
