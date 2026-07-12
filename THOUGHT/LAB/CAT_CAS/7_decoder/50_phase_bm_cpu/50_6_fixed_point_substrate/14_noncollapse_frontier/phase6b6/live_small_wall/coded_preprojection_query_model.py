#!/usr/bin/env python3
"""Non-driving coded pre-projection query boundary model.

This script does not contact the lab device and does not execute a physical
stimulus.  It freezes the next access-model discriminator:

* the old passive projection is fold-even;
* a declared public quadrature query can preserve the conjugate coordinate only
  when it acts before projection;
* post-projection, query-off, source-off, declaration-sham, and scrambled-query
  controls remain fold-odd null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA_ID = "CAT_CAS_CODED_PREPROJECTION_QUERY_MODEL_V1"
N = 256
PUBLIC_FOLD_DISTANCE = 23
PHASES = (0.0, math.pi / 2.0, math.pi, 3.0 * math.pi / 2.0)
TOLERANCE = 1.0e-12


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def reconstruct_complex(responses: list[float], phases: tuple[float, ...] = PHASES) -> complex:
    if len(responses) != len(phases):
        raise ValueError("response and phase lengths differ")
    total = 0.0 + 0.0j
    for response, phase in zip(responses, phases):
        total += response * complex(math.cos(phase), math.sin(phase))
    return (2.0 / float(len(phases))) * total


def response_summary(name: str, responses: list[float]) -> dict[str, Any]:
    z = reconstruct_complex(responses)
    return {
        "name": name,
        "responses": responses,
        "reconstructed_complex": {
            "real": z.real,
            "imag": z.imag,
            "abs": abs(z),
        },
        "fold_even_coordinate": z.real,
        "fold_odd_coordinate": z.imag,
    }


def pre_projection_responses(private_fold_sign: int, phi: float) -> list[float]:
    return [math.cos(private_fold_sign * phi - phase) for phase in PHASES]


def post_projection_responses(phi: float) -> list[float]:
    folded_scalar = math.cos(phi)
    return [folded_scalar * math.cos(phase) for phase in PHASES]


def declaration_sham_responses(phi: float) -> list[float]:
    folded_scalar = math.cos(phi)
    return [folded_scalar for _phase in PHASES]


def scrambled_query_responses(phi: float) -> list[float]:
    folded_scalar = math.cos(phi)
    # Public non-quadrature phase pattern 0, pi, 0, pi.  Decoding it with the
    # frozen quadrature weights must destroy the first-harmonic fold-odd
    # coordinate.
    return [folded_scalar, -folded_scalar, folded_scalar, -folded_scalar]


def near_zero(value: float) -> bool:
    return abs(value) <= TOLERANCE


def build_model() -> dict[str, Any]:
    phi = 2.0 * math.pi * PUBLIC_FOLD_DISTANCE / N
    plus_branch = PUBLIC_FOLD_DISTANCE
    minus_branch = (N - PUBLIC_FOLD_DISTANCE) % N
    cos_plus = math.cos(phi)
    cos_minus = math.cos(-phi)
    sin_plus = math.sin(phi)
    sin_minus = math.sin(-phi)

    query_law = {
        "schema_id": "CAT_CAS_PUBLIC_CODED_QUERY_LAW_V1",
        "N": N,
        "public_fold_distance": PUBLIC_FOLD_DISTANCE,
        "phase_radians": list(PHASES),
        "phase_names": ["0", "pi/2", "pi", "3pi/2"],
        "decoder": "z = (2/K) * sum_k response_k * exp(i * theta_k)",
        "acts_before_projection": True,
        "forbidden_receiver_inputs": [
            "private_branch_label",
            "orientation_label",
            "target_identity",
            "session_chronology",
            "future_value",
        ],
    }

    pre_plus = response_summary("pre_projection_private_fold_plus", pre_projection_responses(+1, phi))
    pre_minus = response_summary("pre_projection_private_fold_minus", pre_projection_responses(-1, phi))
    post = response_summary("post_projection_control", post_projection_responses(phi))
    source_off = response_summary("source_off_control", [0.0 for _phase in PHASES])
    query_off = response_summary("query_off_control", [0.0 for _phase in PHASES])
    declaration_sham = response_summary("declaration_sham_control", declaration_sham_responses(phi))
    query_scramble = response_summary("query_scramble_control", scrambled_query_responses(phi))

    control_odd_abs = [
        abs(post["fold_odd_coordinate"]),
        abs(source_off["fold_odd_coordinate"]),
        abs(query_off["fold_odd_coordinate"]),
        abs(declaration_sham["fold_odd_coordinate"]),
        abs(query_scramble["fold_odd_coordinate"]),
    ]
    pre_odd_min_abs = min(
        abs(pre_plus["fold_odd_coordinate"]),
        abs(pre_minus["fold_odd_coordinate"]),
    )
    forbidden_terms = ("private_branch", "orientation_label", "target_identity")
    receiver_query_payload = {k: v for k, v in query_law.items() if k != "forbidden_receiver_inputs"}
    receiver_query_text = json.dumps(receiver_query_payload, sort_keys=True)
    no_branch_route = all(term not in receiver_query_text for term in forbidden_terms)
    private_fold_sign_reversal = (
        pre_plus["fold_odd_coordinate"] * pre_minus["fold_odd_coordinate"] < 0.0
        and abs(abs(pre_plus["fold_odd_coordinate"]) - abs(pre_minus["fold_odd_coordinate"])) <= TOLERANCE
    )

    acceptance = {
        "old_boundary_preserved": abs(cos_plus - cos_minus) <= TOLERANCE,
        "old_passive_sine_was_excluded": abs(sin_plus + sin_minus) <= TOLERANCE,
        "new_access_model_declared": True,
        "query_law_has_no_branch_route": no_branch_route,
        "pre_projection_fold_odd_sign_reversal": private_fold_sign_reversal,
        "post_projection_fold_odd_null": near_zero(post["fold_odd_coordinate"]),
        "source_off_fold_odd_null": near_zero(source_off["fold_odd_coordinate"]),
        "query_off_fold_odd_null": near_zero(query_off["fold_odd_coordinate"]),
        "declaration_sham_fold_odd_null": near_zero(declaration_sham["fold_odd_coordinate"]),
        "query_scramble_fold_odd_null": near_zero(query_scramble["fold_odd_coordinate"]),
        "pre_projection_exceeds_controls": pre_odd_min_abs > 16.0 * max(control_odd_abs),
    }
    acceptance["model_boundary_discriminator_passed"] = all(acceptance.values())

    result: dict[str, Any] = {
        "schema_id": SCHEMA_ID,
        "claim_ceiling": (
            "Access-model discriminator only: a public coded query can preserve a "
            "fold-odd coordinate in the model if and only if it acts before the "
            "fold-even projection. No physical coupling, restoration, OrbitState "
            "carrier coupling, or Small Wall crossing claim."
        ),
        "live_hardware_operations": 0,
        "lab_device_contact": False,
        "old_passive_boundary": {
            "N": N,
            "branch_pair": [plus_branch, minus_branch],
            "public_projection": "cos(2*pi*d/N)",
            "cos_plus": cos_plus,
            "cos_minus": cos_minus,
            "fold_even_difference": cos_plus - cos_minus,
            "hidden_sine_plus": sin_plus,
            "hidden_sine_minus": sin_minus,
            "excluded_coordinate": "sin(2*pi*d/N)",
        },
        "declared_new_access_model": {
            "query_law": query_law,
            "query_law_sha256": digest(query_law),
            "receiver_query_payload_sha256": digest(receiver_query_payload),
            "projection_boundary": "pre_projection responses are measured before scalar cos projection; post_projection controls apply the same decoder after scalar loss",
        },
        "responses": {
            "pre_projection_private_fold_plus": pre_plus,
            "pre_projection_private_fold_minus": pre_minus,
            "post_projection_control": post,
            "source_off_control": source_off,
            "query_off_control": query_off,
            "declaration_sham_control": declaration_sham,
            "query_scramble_control": query_scramble,
        },
        "controls": {
            "post_projection": "fold-odd null after scalar projection",
            "source_off": "zero response when source coupling is absent",
            "query_off": "zero response when coded query is absent",
            "declaration_sham": "declared phases present, source emits unphased folded scalar",
            "query_scramble": "non-quadrature public phase schedule decoded by frozen quadrature weights",
            "private_fold": "internal private fold changes sign of the pre-projection odd coordinate without changing query law",
        },
        "acceptance": acceptance,
        "next_experiment_contract": {
            "name": "coded_preprojection_loop_0",
            "status": "MODEL_READY_LIVE_MAPPING_PENDING",
            "carrier": "public coded query mapped onto CAT_CAS-owned timing or ownership-intent carrier",
            "operator": "four public phases 0, pi/2, pi, 3pi/2 before projection",
            "primary_observable": "quadrature reconstructed fold-odd coordinate from fixed response weights",
            "killing_controls": [
                "post_projection",
                "query_scramble",
                "query_off",
                "source_off",
                "declaration_sham",
                "private_fold",
            ],
            "restoration_class": "buffer digest closure plus neutral pre/post physical-state probe; exact live tolerance not yet claimed",
        },
    }
    result["result_sha256"] = digest({k: v for k, v in result.items() if k != "result_sha256"})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = build_model()
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.write_text(payload, encoding="utf-8")
    return 0 if result["acceptance"]["model_boundary_discriminator_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
