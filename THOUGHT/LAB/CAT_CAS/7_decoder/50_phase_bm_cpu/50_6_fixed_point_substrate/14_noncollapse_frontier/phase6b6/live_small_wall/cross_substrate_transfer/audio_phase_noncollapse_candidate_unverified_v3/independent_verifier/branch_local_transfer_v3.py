#!/usr/bin/env python3
"""Branch-local obstruction and mechanism transfer harness for V3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw_outputs" / "branch_local_transfer_v3"


def obstruction_harness() -> dict:
    finite_quotient_gate = {
        "name": "finite_exact_quotient_vs_infinite_orbit",
        "status": "CHECKLIST_ONLY",
        "law": "If exact projective orbit is infinite, fixed finite injective lossless quotient is impossible.",
        "scope": "cardinality/exact-boundary review gate only",
    }
    finite_machine_gate = {
        "name": "finite_deterministic_machine_vs_symbolic_bisimulation",
        "status": "CHECKLIST_ONLY",
        "law": "A deterministic finite transition system is simulated by storing its semantic finite state.",
        "example": "Q3 register can be packed into two bits; native representation is not automatically distinct.",
    }
    rank_gate = {
        "name": "factorized_representation_vs_exact_rank",
        "status": "CHECKLIST_ONLY",
        "law": "A proposed factorization must beat direct dense/baseline storage after rank and scratch are counted.",
        "example": "binary-width canonical rank makes TT cells exceed dense cells in K-like shape.",
    }
    precision_gate = {
        "name": "fixed_cells_vs_precision_growth",
        "status": "CHECKLIST_ONLY",
        "law": "Fixed logical cell count does not imply fixed material payload.",
        "toy": [
            {"n": n, "value": f"1/2^{n}", "denominator_bits": n + 1}
            for n in [1, 2, 4, 8, 16, 32]
        ],
    }
    boundary_gate = {
        "name": "boundary_alphabet_growth_vs_compact_index",
        "status": "CHECKLIST_ONLY",
        "law": "Unbounded exact boundary valuations reject fixed finite lossless alphabets, but an indexed ledger may remain compact.",
        "height_formula": "ceil((272*N+16)/3)",
        "horizon_code_width_bits_samples": {
            str(n): math.ceil(math.log2(((272 * n + 18) // 3) + 1))
            for n in [4, 16, 64, 256, 1024]
        },
    }
    gates = [finite_quotient_gate, finite_machine_gate, rank_gate, precision_gate, boundary_gate]
    return {
        "schema_version": "branch_local_obstruction_harness_v3",
        "canonical": False,
        "small_wall_crossed": False,
        "audio_arithmetic_used": False,
        "gates": gates,
        "classification": "BRANCH_LOCAL_OBSTRUCTION_CHECKLIST_ESTABLISHED",
        "all_gates_operational": False,
        "executable_harness_complete": False,
        "reason_not_executable": "records obstruction laws and sample calculations but does not accept candidate representations or produce pass/fail outcomes",
        "supported_transfer": [
            "finite exact quotient obstruction gate",
            "finite software bisimulation obstruction gate",
            "rank/baseline anti-compaction gate",
            "fixed logical slot versus material precision gate",
            "exact boundary alphabet growth gate with indexed-ledger caveat",
        ],
    }


Matrix = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
IDENTITY: Matrix = ((Fraction(1), Fraction(0)), (Fraction(0), Fraction(1)))
A: Matrix = ((Fraction(1), Fraction(1)), (Fraction(0), Fraction(1)))
A_INV: Matrix = ((Fraction(1), Fraction(-1)), (Fraction(0), Fraction(1)))
B: Matrix = ((Fraction(1), Fraction(0)), (Fraction(1), Fraction(1)))
B_INV: Matrix = ((Fraction(1), Fraction(0)), (Fraction(-1), Fraction(1)))


def mmul(left: Matrix, right: Matrix) -> Matrix:
    return (
        (
            left[0][0] * right[0][0] + left[0][1] * right[1][0],
            left[0][0] * right[0][1] + left[0][1] * right[1][1],
        ),
        (
            left[1][0] * right[0][0] + left[1][1] * right[1][0],
            left[1][0] * right[0][1] + left[1][1] * right[1][1],
        ),
    )


def mlist(value: Matrix) -> list[list[str]]:
    return [[str(cell) for cell in row] for row in value]


def inverse_for(op: str) -> Matrix:
    if op == "A":
        return A_INV
    if op == "B":
        return B_INV
    raise ValueError(op)


def matrix_for(op: str) -> Matrix:
    if op == "A":
        return A
    if op == "B":
        return B
    raise ValueError(op)


@dataclass
class TupleCustody:
    port_id: str
    port_type: str
    owner: int
    generation: int
    lease: int


@dataclass
class Carrier:
    frame: Matrix = IDENTITY
    generation: int = 0
    lease: int = 0xC0D3
    port_a: TupleCustody | None = None
    port_b: TupleCustody | None = None


def owner_for(program: list[str]) -> int:
    value = 2166136261
    for op in program:
        for byte in op.encode():
            value ^= byte
            value = (value * 16777619) & 0xFFFFFFFF
    return value


def compile_program(program: list[str], carrier: Carrier) -> tuple[TupleCustody, TupleCustody]:
    if not program or any(op not in {"A", "B"} for op in program):
        raise ValueError("malformed descriptor")
    if "A" not in program or "B" not in program:
        raise ValueError("both typed ports must be consumed")
    owner = owner_for(program)
    return (
        TupleCustody("port_a", "left_shear", owner, carrier.generation, carrier.lease),
        TupleCustody("port_b", "lower_shear", owner, carrier.generation, carrier.lease),
    )


def execute_transaction(carrier: Carrier, program: list[str], mutate_inverse: str | None = None) -> dict:
    before = carrier.frame
    port_a, port_b = compile_program(program, carrier)
    carrier.port_a = port_a
    carrier.port_b = port_b
    custody_checks = []
    for custody in [carrier.port_a, carrier.port_b]:
        checks = {
            "port_id": custody.port_id,
            "port_type": custody.port_type,
            "owner_checked": custody.owner == owner_for(program),
            "generation_checked": custody.generation == carrier.generation,
            "lease_checked": custody.lease == carrier.lease,
            "port_id_checked_at_consumer": False,
            "port_type_checked_at_consumer": False,
        }
        custody_checks.append(checks)
        if not (checks["owner_checked"] and checks["generation_checked"] and checks["lease_checked"]):
            raise RuntimeError("custody tuple mismatch")
    for op in program:
        carrier.frame = mmul(matrix_for(op), carrier.frame)
    hidden_after_forward = carrier.frame
    boundary = carrier.frame
    inverse_program = list(reversed(program))
    if mutate_inverse == "missing":
        inverse_program = inverse_program[:-1]
    for index, op in enumerate(inverse_program):
        inv = inverse_for(op)
        if mutate_inverse == "wrong" and index == 0:
            inv = matrix_for(op)
        carrier.frame = mmul(inv, carrier.frame)
    restored = carrier.frame == before
    accepted = restored and mutate_inverse is None
    if accepted:
        carrier.generation += 1
        response = {
            "status": "OK",
            "boundary": mlist(boundary),
            "restored_before_response": True,
            "generation": carrier.generation,
        }
    else:
        response = {
            "status": "ERROR",
            "boundary": None,
            "restored_before_response": restored,
            "generation": carrier.generation,
        }
    return {
        "program": program,
        "hidden_after_forward": mlist(hidden_after_forward),
        "response": response,
        "restored": restored,
        "same_carrier_object": True,
        "custody_checks": custody_checks,
        "consumer_check_count": 1,
    }


def mechanism_harness() -> dict:
    carrier = Carrier()
    primary = execute_transaction(carrier, ["A", "B"])
    reuse = execute_transaction(carrier, ["B", "A"])
    ab = mmul(B, A)
    ba = mmul(A, B)
    wrong = execute_transaction(Carrier(), ["A", "B"], mutate_inverse="wrong")
    missing = execute_transaction(Carrier(), ["A", "B"], mutate_inverse="missing")
    malformed_denied = False
    try:
        compile_program(["A"], Carrier())
    except ValueError:
        malformed_denied = True
    survived = (
        primary["response"]["status"] == "OK"
        and reuse["response"]["status"] == "OK"
        and ab != ba
        and wrong["response"]["status"] == "ERROR"
        and wrong["response"]["boundary"] is None
        and missing["response"]["status"] == "ERROR"
        and missing["response"]["boundary"] is None
        and malformed_denied
        and carrier.frame == IDENTITY
        and carrier.generation == 2
    )
    return {
        "schema_version": "branch_local_mechanism_transfer_v3",
        "canonical": False,
        "small_wall_crossed": False,
        "audio_arithmetic_used": False,
        "carrier": "exact rational 2x2 hidden noncommuting frame with two typed ports",
        "classification": "BRANCH_LOCAL_EXACT_NONCOMMUTING_REVERSIBLE_FRAME_TRANSACTION_ESTABLISHED",
        "primary": primary,
        "reuse": reuse,
        "order_sensitive": {
            "AB": mlist(ab),
            "BA": mlist(ba),
            "different": ab != ba,
        },
        "controls": {
            "wrong_inverse_releases_boundary": wrong["response"]["boundary"] is not None,
            "missing_inverse_releases_boundary": missing["response"]["boundary"] is not None,
            "malformed_single_port_descriptor_denied": malformed_denied,
        },
        "resource_ledger": {
            "hidden_frame_rational_cells": 4,
            "typed_hidden_ports": 2,
            "tuple_fields_per_port": 5,
            "retained_inverse_history": 0,
            "public_descriptor_ops_primary": 2,
            "public_descriptor_ops_reuse": 2,
            "strongest_compact_baseline": "same exact 2x2 matrix recurrence",
        },
        "toy_transaction_survived": survived,
        "mechanism_transfer_survived": False,
        "full_two_port_machine_law_established": False,
        "custody_scope": {
            "validated_at_consumer": ["owner", "generation", "lease"],
            "not_validated_at_consumer": ["port_id", "port_type"],
            "checked_once_before_operations": True,
            "checked_at_every_consumer": False,
            "both_ports_same_owner": True,
            "packet_protocol_present": False,
            "disconnect_cleanup_exercised": False,
            "stage_cut_enforcement_exercised": False,
            "nonce_replay_control_present": False,
            "reordered_inverse_control_present": False,
            "diagnostic_serializes_hidden_after_forward": True,
            "port_custody_records_cleared_after_restoration": False,
        },
        "scope": "software reference/counterfactual toy carrier only; no physical Family 10h claim and no full two-port CATVM custody claim",
    }


def render_obstruction(report: dict) -> str:
    gates = "\n".join(f"- {gate['name']}: `{gate['status']}` — {gate['law']}" for gate in report["gates"])
    return f"""# Branch-Local Obstruction Harness Report

Classification: `{report['classification']}`

This harness is independent of audio arithmetic and physical Family 10h evidence.

{gates}

Conclusion:

This file is a policy checklist with sample calculations, not an executable obstruction transfer harness. It does not accept candidate representations and cannot produce both passing and failing outcomes. The obstruction laws transfer as review gates only and prevent false promotion by forcing exact quotient, finite-state, rank/baseline, precision-payload, and boundary-alphabet claims to state their decoder/resource assumptions.
"""


def render_mechanism(report: dict) -> str:
    return f"""# Branch-Local Mechanism Transfer Report

Classification: `{report['classification']}`

Toy transaction survived: `{report['toy_transaction_survived']}`

Full two-port machine law established: `{report['full_two_port_machine_law_established']}`

Carrier: `{report['carrier']}`

What this actually demonstrates:

- hidden noncommuting shared frame;
- two typed hidden ports in the data model;
- public descriptor compilation;
- reverse rematerialization without retained inverse history;
- atomic final-only response after verified restoration;
- same-carrier unrelated reuse.

What it does not demonstrate:

- full owner/type/generation/lease custody at every consumer;
- port ID or port type validation at consumer sites;
- packet protocol framing;
- disconnect cleanup;
- stage-cut enforcement;
- nonce/replay handling;
- reordered inverse control;
- absence of diagnostic hidden-state serialization.

Controls:

- wrong inverse released boundary: `{report['controls']['wrong_inverse_releases_boundary']}`;
- missing inverse released boundary: `{report['controls']['missing_inverse_releases_boundary']}`;
- malformed single-port descriptor denied: `{report['controls']['malformed_single_port_descriptor_denied']}`.

Baseline:

The strongest compact baseline remains the same exact 2×2 matrix recurrence. This is useful as a counterfactual noncommuting reversible-frame toy law, not a transferable two-port CATVM machine law, resource separation, or physical Family 10h claim.
"""


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    obstruction = obstruction_harness()
    mechanism = mechanism_harness()
    (RAW / "branch_local_obstruction_harness.json").write_text(
        json.dumps(obstruction, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (RAW / "branch_local_mechanism_transfer.json").write_text(
        json.dumps(mechanism, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (ROOT / "BRANCH_LOCAL_OBSTRUCTION_HARNESS_REPORT.md").write_text(
        render_obstruction(obstruction), encoding="utf-8"
    )
    (ROOT / "BRANCH_LOCAL_MECHANISM_TRANSFER_REPORT.md").write_text(
        render_mechanism(mechanism), encoding="utf-8"
    )
    print(json.dumps({"obstruction": obstruction["classification"], "mechanism": mechanism["classification"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
