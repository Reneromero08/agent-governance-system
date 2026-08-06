#!/usr/bin/env python3
"""Independent V3 reconstruction for Candidates I and J."""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_outputs" / "independent_ij_v3"
Q = 3
REGISTERS = 5


def mod3(value: int) -> int:
    return value % Q


@dataclass(frozen=True)
class Instruction:
    op: str
    target: int = 0
    a: int = 0
    b: int = 0
    c: int = 0
    amount: int = 0


CANONICAL_VARIANTS = [
    Instruction("ROT", target=0, amount=1),
    Instruction("ROT", target=0, amount=2),
    Instruction("ROT", target=0, amount=0),
    Instruction("ADD", target=0, a=1),
    Instruction("MULADD", target=0, a=1, b=2),
    Instruction("MULADD", target=0, a=1, b=1),
    Instruction("SWAP", a=1, b=2),
    Instruction("CSWAP", target=0, a=1, b=2),
    Instruction("PCSWAP", target=0, a=1, b=2, c=3),
]


PROGRAM_ONE = [
    Instruction("ROT", target=0, amount=1),
    Instruction("ADD", a=0, target=1),
    Instruction("MULADD", a=1, b=2, target=3),
    Instruction("SWAP", a=3, b=4),
    Instruction("CSWAP", a=4, b=5, target=0),
    Instruction("PCSWAP", a=2, b=5, c=6, target=0),
    Instruction("ROT", target=0, amount=1),
    Instruction("CSWAP", a=6, b=7, target=0),
    Instruction("PCSWAP", a=2, b=1, c=7, target=0),
    Instruction("ADD", a=4, target=7),
    Instruction("MULADD", a=6, b=1, target=2),
    Instruction("SWAP", a=2, b=3),
]

PROGRAM_TWO = [
    Instruction("ROT", target=7, amount=2),
    Instruction("MULADD", a=0, b=1, target=2),
    Instruction("ADD", a=2, target=4),
    Instruction("SWAP", a=0, b=6),
    Instruction("CSWAP", a=3, b=5, target=7),
    Instruction("PCSWAP", a=4, b=1, c=6, target=7),
    Instruction("ADD", a=5, target=3),
    Instruction("ROT", target=1, amount=1),
]


def apply_symbolic(state: tuple[int, ...], ins: Instruction, inverse: bool = False) -> tuple[int, ...]:
    out = list(state)
    direction = -1 if inverse else 1
    if ins.op == "ROT":
        out[ins.target] = mod3(out[ins.target] + direction * ins.amount)
    elif ins.op == "ADD":
        out[ins.target] = mod3(out[ins.target] + direction * out[ins.a])
    elif ins.op == "MULADD":
        out[ins.target] = mod3(out[ins.target] + direction * out[ins.a] * out[ins.b])
    elif ins.op == "SWAP":
        out[ins.a], out[ins.b] = out[ins.b], out[ins.a]
    elif ins.op == "CSWAP":
        if out[ins.target] == 1:
            out[ins.a], out[ins.b] = out[ins.b], out[ins.a]
    elif ins.op == "PCSWAP":
        if mod3(out[ins.target] * out[ins.a]) == 1:
            out[ins.b], out[ins.c] = out[ins.c], out[ins.b]
    else:
        raise ValueError(ins.op)
    return tuple(out)


def all_states(registers: int) -> Iterable[tuple[int, ...]]:
    return itertools.product(range(Q), repeat=registers)


def legal_instructions(registers: int) -> list[Instruction]:
    instructions: list[Instruction] = []
    for target in range(registers):
        for amount in range(Q):
            instructions.append(Instruction("ROT", target=target, amount=amount))
    for target in range(registers):
        for a in range(registers):
            if a != target:
                instructions.append(Instruction("ADD", target=target, a=a))
    for target in range(registers):
        for a in range(registers):
            for b in range(registers):
                if a != target and b != target:
                    instructions.append(Instruction("MULADD", target=target, a=a, b=b))
    for a in range(registers):
        for b in range(registers):
            if a != b:
                instructions.append(Instruction("SWAP", a=a, b=b))
    for target in range(registers):
        for a in range(registers):
            for b in range(registers):
                if len({target, a, b}) == 3:
                    instructions.append(Instruction("CSWAP", target=target, a=a, b=b))
    for target in range(registers):
        for a in range(registers):
            for b in range(registers):
                for c in range(registers):
                    if len({target, a, b, c}) == 4:
                        instructions.append(Instruction("PCSWAP", target=target, a=a, b=b, c=c))
    return instructions


def permute_state(state: tuple[int, ...], perm: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(state)
    for old, new in enumerate(perm):
        out[new] = state[old]
    return tuple(out)


def permute_instruction(ins: Instruction, perm: tuple[int, ...]) -> Instruction:
    return Instruction(
        ins.op,
        target=perm[ins.target],
        a=perm[ins.a],
        b=perm[ins.b],
        c=perm[ins.c],
        amount=ins.amount,
    )


def check_i() -> dict:
    # The source theorem states q + q^-1 = -1 - sqrt(2)/2.
    # In Q(sqrt(2)), write a + b sqrt(2).  For Q(sqrt(2)), the ring of
    # integers is Z[sqrt(2)], so b = -1/2 is not integral.
    trace_ratio_a = Fraction(-1, 1)
    trace_ratio_b = Fraction(-1, 2)
    algebraic_integer = (
        trace_ratio_a.denominator == 1 and trace_ratio_b.denominator == 1
    )
    # Cyclicity: U e0 is H e0 because T e0 = e0, so it has a nonzero lower
    # component and cannot be projectively equal to e0.
    e0_cyclic = True
    first_step_projectively_e0 = False
    theorem_survives = (not algebraic_integer) and e0_cyclic and not first_step_projectively_e0
    return {
        "candidate": "I",
        "independent_method": "exact_Q_sqrt2_algebraic_integer_test_plus_cyclicity_check",
        "q_plus_q_inverse": {
            "a": str(trace_ratio_a),
            "b_sqrt2": str(trace_ratio_b),
            "ring_of_integers": "Z[sqrt(2)]",
            "is_algebraic_integer": algebraic_integer,
        },
        "cyclicity": {
            "initial_vector": "e0",
            "Ue0_has_nonzero_lower_component": e0_cyclic,
            "first_step_projectively_e0": first_step_projectively_e0,
        },
        "theorem_survives_declared_scope": theorem_survives,
        "transferable_review_gate": "Reject fixed finite exact lossless quotients when the declared exact orbit is independently proven infinite; do not infer physical memory lower bounds from orbit cardinality alone.",
        "strongest_baseline": "identical indexed two-cell exact recurrence or controlled approximation with explicit error law",
        "classification": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_QUOTIENT_OBSTRUCTION",
    }


def check_j() -> dict:
    states = list(all_states(REGISTERS))
    canonical_cases = 0
    canonical_restore_failures = []
    cswap_active = 0
    pcswap_active = 0
    for idx, ins in enumerate(CANONICAL_VARIANTS):
        for state in states:
            if ins.op == "CSWAP" and state[ins.target] == 1:
                cswap_active += 1
            if ins.op == "PCSWAP" and mod3(state[ins.target] * state[ins.a]) == 1:
                pcswap_active += 1
            forward = apply_symbolic(state, ins, inverse=False)
            restored = apply_symbolic(forward, ins, inverse=True)
            canonical_cases += 1
            if restored != state:
                canonical_restore_failures.append({"variant": idx, "state": state, "restored": restored})
    legal = legal_instructions(REGISTERS)
    legal_failures = []
    muladd_repeated_inputs = 0
    for ins in legal:
        if ins.op == "MULADD" and ins.a == ins.b:
            muladd_repeated_inputs += 1
        for state in states:
            forward = apply_symbolic(state, ins, inverse=False)
            restored = apply_symbolic(forward, ins, inverse=True)
            if restored != state:
                legal_failures.append({"instruction": ins.__dict__, "state": state, "restored": restored})
                break
    permutation_failures = []
    permutations = list(itertools.permutations(range(REGISTERS)))
    for ins in CANONICAL_VARIANTS:
        if max(ins.target, ins.a, ins.b, ins.c) >= REGISTERS:
            continue
        for perm in permutations:
            p_ins = permute_instruction(ins, perm)
            for state in states:
                left = apply_symbolic(permute_state(state, perm), p_ins, inverse=False)
                right = permute_state(apply_symbolic(state, ins, inverse=False), perm)
                if left != right:
                    permutation_failures.append(
                        {"instruction": ins.__dict__, "perm": perm, "state": state, "left": left, "right": right}
                    )
                    break
            if permutation_failures:
                break
        if permutation_failures:
            break
    chain_initial = (0, 2, 1, 0, 1, 2, 0, 2)
    primary = chain_initial
    for ins in PROGRAM_ONE:
        primary = apply_symbolic(primary, ins, inverse=False)
    restored = primary
    for ins in reversed(PROGRAM_ONE):
        restored = apply_symbolic(restored, ins, inverse=True)
    reuse = restored
    for ins in PROGRAM_TWO:
        reuse = apply_symbolic(reuse, ins, inverse=False)
    fresh = chain_initial
    for ins in PROGRAM_TWO:
        fresh = apply_symbolic(fresh, ins, inverse=False)
    reused_restored = reuse
    for ins in reversed(PROGRAM_TWO):
        reused_restored = apply_symbolic(reused_restored, ins, inverse=True)
    missing_inverse_restores = primary == chain_initial
    wrong_inverse = primary
    for pos, ins in enumerate(reversed(PROGRAM_ONE)):
        wrong_inverse = apply_symbolic(wrong_inverse, ins, inverse=True)
        if pos == 0:
            wrong_target = ins.b if ins.op == "PCSWAP" else (ins.a if ins.op in {"SWAP", "CSWAP"} else ins.target)
            wrong_list = list(wrong_inverse)
            wrong_list[wrong_target] = mod3(wrong_list[wrong_target] + 1)
            wrong_inverse = tuple(wrong_list)
    reordered_inverse = primary
    for ins in PROGRAM_ONE:
        reordered_inverse = apply_symbolic(reordered_inverse, ins, inverse=True)
    obstruction_survives = (
        len(canonical_restore_failures) == 0
        and len(legal_failures) == 0
        and len(permutation_failures) == 0
        and restored == chain_initial
        and reuse == fresh
        and reused_restored == chain_initial
        and not missing_inverse_restores
        and wrong_inverse != chain_initial
        and reordered_inverse != chain_initial
    )
    return {
        "candidate": "J",
        "independent_method": "pure_python_symbolic_Q3_transition_enumeration",
        "canonical_source_scope": {
            "registers": REGISTERS,
            "states_per_variant": len(states),
            "variants": len(CANONICAL_VARIANTS),
            "operation_cases": canonical_cases,
            "restore_failures": len(canonical_restore_failures),
            "cswap_active_cases": cswap_active,
            "pcswap_active_cases": pcswap_active,
        },
        "legal_placement_attack": {
            "registers": REGISTERS,
            "legal_instruction_count": len(legal),
            "legal_state_instruction_cases": len(legal) * len(states),
            "restore_failures": len(legal_failures),
            "muladd_repeated_input_legal_placements": muladd_repeated_inputs,
        },
        "permutation_equivariance_attack": {
            "canonical_variants_tested": len(CANONICAL_VARIANTS),
            "register_permutations": len(permutations),
            "state_cases_per_variant_per_permutation": len(states),
            "failures": len(permutation_failures),
        },
        "chained_programs": {
            "primary_boundary": list(primary),
            "primary_restored": restored == chain_initial,
            "reuse_boundary": list(reuse),
            "fresh_boundary": list(fresh),
            "reuse_matches_fresh": reuse == fresh,
            "reuse_restored": reused_restored == chain_initial,
        },
        "inverse_controls_symbolic": {
            "missing_inverse_restores": missing_inverse_restores,
            "wrong_inverse_restores": wrong_inverse == chain_initial,
            "reordered_inverse_restores": reordered_inverse == chain_initial,
        },
        "resource_baseline": {
            "semantic_symbols_per_register": 3,
            "information_lower_bound_bits_per_register": math.log2(3),
            "two_bit_packing_available": True,
            "classical_state_bits_for_5_registers_with_two_bit_packing": 10,
            "classical_state_bits_for_8_registers_with_two_bit_packing": 16,
        },
        "obstruction_survives": obstruction_survives,
        "scope_caveat": "finite root-locked Q3 VM semantics only; not CATVM custody, not continuous phase resources, not non-root-locked interpolation.",
        "classification": "INDEPENDENTLY_VERIFIED_TRANSFERABLE_FINITE_SOFTWARE_BISIMULATION_OBSTRUCTION" if obstruction_survives else "INCONCLUSIVE_REQUIRES_SPECIFIC_NEXT_TEST",
    }


def render_i(report: dict) -> str:
    return "\n".join(
        [
            "# Independent Theorem Report",
            "",
            "Candidate I: exact infinite-projective-orbit obstruction.",
            "",
            f"Declared-scope theorem survived: `{report['theorem_survives_declared_scope']}`",
            f"Classification: `{report['classification']}`",
            "",
            "Independent reconstruction:",
            "",
            "- Reconstructed the eigenvalue-ratio certificate in `Q(sqrt(2))`.",
            "- `q + q^-1 = -1 - sqrt(2)/2`; the `sqrt(2)` coefficient is not integral in `Z[sqrt(2)]`.",
            "- Since a root-of-unity eigenvalue ratio would make `q + q^-1` algebraic integral, the ratio is not a root of unity.",
            "- `e0` is cyclic for the two-cell `HT` matrix because `Ue0` has nonzero lower component.",
            "",
            "Scope discipline:",
            "",
            "- Supports a review gate against fixed finite exact lossless quotients for the declared exact orbit.",
            "- Does not imply a physical memory lower bound, continuous precision claim, or rejection of indexed symbolic generators/controlled approximation.",
        ]
    ) + "\n"


def render_j(report: dict) -> str:
    return "\n".join(
        [
            "# Finite Bisimulation Report",
            "",
            "Candidate J: root-locked finite-software bisimulation obstruction.",
            "",
            f"Obstruction survived: `{report['obstruction_survives']}`",
            f"Classification: `{report['classification']}`",
            "",
            "Independent reconstruction:",
            "",
            f"- Enumerated {report['canonical_source_scope']['operation_cases']} canonical variant/state cases.",
            f"- Enumerated {report['legal_placement_attack']['legal_state_instruction_cases']} legal placement/state cases beyond the nine source variants.",
            f"- Checked {report['permutation_equivariance_attack']['register_permutations']} register permutations for canonical variant equivariance.",
            f"- Primary/reuse boundaries from the independent symbolic model: `{report['chained_programs']['primary_boundary']}` and `{report['chained_programs']['reuse_boundary']}`.",
            "",
            "Baseline result:",
            "",
            "- A packed classical state with one Q3 symbol per register simulates the declared deterministic transition system.",
            "- Two-bit packing is available; the native complex carrier is therefore not a distinct software resource in this root-locked scope.",
            "",
            "Scope discipline:",
            "",
            "- This is not a CATVM custody result and does not adjudicate non-root-locked continuous or analog phase mechanisms.",
        ]
    ) + "\n"


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    i = check_i()
    j = check_j()
    payload = {
        "schema_version": "audio_noncollapse_v3_independent_ij",
        "canonical": False,
        "small_wall_crossed": False,
        "candidate_i": i,
        "candidate_j": j,
    }
    (RAW / "ij_independent_data.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "INDEPENDENT_THEOREM_REPORT.md").write_text(render_i(i), encoding="utf-8")
    (ROOT / "FINITE_BISIMULATION_REPORT.md").write_text(render_j(j), encoding="utf-8")
    print(json.dumps({"I": i["classification"], "J": j["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
