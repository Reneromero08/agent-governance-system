#!/usr/bin/env python3
"""High-precision independent oracle for Pancharatnam gauge transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mpmath as mp


mp.mp.dps = 80

PI = mp.pi
THETA = mp.mpf("1.1")
DEPTHS = (4, 8, 16, 32, 64, 128, 256, 512)
PRIMARY_SEGMENTS = 512
REUSE_REPETITIONS = 37
TOLERANCE = mp.mpf("2e-13")
REUSE_PATH = (
    (THETA, mp.mpf("0")),
    (mp.mpf("0.55"), mp.mpf("0")),
    (mp.mpf("0.55"), mp.mpf("1.25")),
    (THETA, mp.mpf("1.25")),
    (THETA, mp.mpf("0")),
)

Spinor = tuple[mp.mpc, mp.mpc]


def base(theta: mp.mpf, phi: mp.mpf) -> Spinor:
    return (
        mp.mpc(mp.cos(theta / 2), 0),
        mp.exp(mp.j * phi) * mp.sin(theta / 2),
    )


def overlap(left: Spinor, right: Spinor) -> mp.mpc:
    return mp.conj(left[0]) * right[0] + mp.conj(left[1]) * right[1]


def spinor_error(left: Spinor, right: Spinor) -> mp.mpf:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def transport_to(carrier: Spinor, target: Spinor) -> Spinor:
    value = overlap(carrier, target)
    if abs(value) <= mp.mpf("1e-60"):
        raise RuntimeError("independent transport reached orthogonal target")
    factor = mp.conj(value) / abs(value)
    return factor * target[0], factor * target[1]


def forward_latitude(
    carrier: Spinor,
    theta: mp.mpf,
    segments: int,
    orientation: int,
) -> Spinor:
    result = carrier
    for step in range(1, segments + 1):
        phi = orientation * 2 * PI * step / segments
        result = transport_to(result, base(theta, phi))
    return result


def inverse_latitude(
    carrier: Spinor,
    theta: mp.mpf,
    segments: int,
    orientation: int,
) -> Spinor:
    result = carrier
    for step in reversed(range(segments)):
        phi = orientation * 2 * PI * step / segments
        result = transport_to(result, base(theta, phi))
    return result


def forward_path(carrier: Spinor, repetitions: int) -> Spinor:
    result = carrier
    for _ in range(repetitions):
        for theta, phi in REUSE_PATH[1:]:
            result = transport_to(result, base(theta, phi))
    return result


def inverse_path(carrier: Spinor, repetitions: int) -> Spinor:
    result = carrier
    for _ in range(repetitions):
        for theta, phi in reversed(REUSE_PATH[:-1]):
            result = transport_to(result, base(theta, phi))
    return result


def latitude_formula(
    theta: mp.mpf,
    segments: int,
    orientation: int,
) -> mp.mpc:
    c = mp.cos(theta / 2)
    s = mp.sin(theta / 2)
    delta = orientation * 2 * PI / segments
    edge = c * c + s * s * mp.exp(mp.j * delta)
    return (mp.conj(edge) / abs(edge)) ** segments


def continuous_limit(theta: mp.mpf, orientation: int) -> mp.mpc:
    phase = -orientation * PI * (1 - mp.cos(theta))
    return mp.exp(mp.j * phase)


def scalar_baseline_path(repetitions: int) -> mp.mpc:
    phase = mp.mpf("0")
    for _ in range(repetitions):
        for left, right in zip(
            REUSE_PATH[:-1], REUSE_PATH[1:], strict=True
        ):
            phase -= mp.arg(overlap(base(*left), base(*right)))
    return mp.exp(mp.j * phase)


def decode_complex(value: object) -> mp.mpc:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("production complex schema mismatch")
    return mp.mpc(str(value[0]), str(value[1]))


def verify_production(
    production: dict[str, object],
) -> list[dict[str, object]]:
    runs = production.get("segment_runs")
    if not isinstance(runs, list) or len(runs) != len(DEPTHS):
        raise RuntimeError("production segment runs missing")
    initial = base(THETA, mp.mpf("0"))
    verified = []
    previous_limit_error = mp.inf
    for segments, production_run in zip(DEPTHS, runs, strict=True):
        if not isinstance(production_run, dict):
            raise RuntimeError("production segment run is not an object")
        carrier = forward_latitude(initial, THETA, segments, 1)
        holonomy = overlap(initial, carrier)
        formula = latitude_formula(THETA, segments, 1)
        limit = continuous_limit(THETA, 1)
        restored = inverse_latitude(carrier, THETA, segments, 1)
        production_holonomy = decode_complex(
            production_run.get("holonomy")
        )
        production_formula = decode_complex(
            production_run.get("analytic_discrete_holonomy")
        )
        production_limit = decode_complex(
            production_run.get("continuous_limit_holonomy")
        )
        limit_error = abs(holonomy - limit)
        if (
            production_run.get("segments") != segments
            or abs(production_holonomy - holonomy) > TOLERANCE
            or abs(production_formula - formula) > TOLERANCE
            or abs(production_limit - limit) > TOLERANCE
            or abs(holonomy - formula) > mp.mpf("1e-70")
            or spinor_error(restored, initial) > mp.mpf("1e-70")
            or abs(overlap(carrier, carrier) - 1) > mp.mpf("1e-70")
            or not limit_error < previous_limit_error
        ):
            raise RuntimeError(
                f"production segment {segments} failed independent check"
            )
        previous_limit_error = limit_error
        verified.append(
            {
                "segments": segments,
                "holonomy_matches": True,
                "analytic_discrete_formula_matches": True,
                "restored": True,
                "norm_one": True,
                "continuous_limit_error": float(limit_error),
            }
        )
    return verified


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(
            "usage: pancharatnam_gauge_transport_oracle.py "
            "PRODUCTION_RESULT"
        )
    production_value = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    if not isinstance(production_value, dict):
        raise RuntimeError("production result is not an object")
    verified_runs = verify_production(production_value)

    initial = base(THETA, mp.mpf("0"))
    primary = forward_latitude(initial, THETA, PRIMARY_SEGMENTS, 1)
    primary_holonomy = overlap(initial, primary)
    primary_restored = inverse_latitude(
        primary, THETA, PRIMARY_SEGMENTS, 1
    )
    reuse = forward_path(initial, REUSE_REPETITIONS)
    reuse_holonomy = overlap(initial, reuse)
    reuse_restored = inverse_path(reuse, REUSE_REPETITIONS)
    if (
        abs(
            decode_complex(production_value["primary"]["holonomy"])
            - primary_holonomy
        ) > TOLERANCE
        or abs(
            decode_complex(production_value["reuse"]["holonomy"])
            - reuse_holonomy
        ) > TOLERANCE
        or spinor_error(primary_restored, initial) > mp.mpf("1e-70")
        or spinor_error(reuse_restored, initial) > mp.mpf("1e-70")
        or abs(
            reuse_holonomy - scalar_baseline_path(REUSE_REPETITIONS)
        ) > mp.mpf("1e-70")
    ):
        raise RuntimeError("accepted path failed independent check")

    reverse = forward_latitude(initial, THETA, PRIMARY_SEGMENTS, -1)
    area_initial = base(mp.mpf("0.95"), mp.mpf("0"))
    area = forward_latitude(area_initial, mp.mpf("0.95"), 512, 1)
    canonicalized = base(THETA, 2 * PI)
    if (
        abs(overlap(initial, reverse) - mp.conj(primary_holonomy))
        > mp.mpf("1e-70")
        or abs(overlap(area_initial, area) - primary_holonomy)
        < mp.mpf("0.1")
        or abs(overlap(initial, canonicalized) - 1) > mp.mpf("1e-70")
    ):
        raise RuntimeError("semantic control failed independent check")

    output = {
        "result": "PASS",
        "oracle": (
            "INDEPENDENT_MPMATH80_PANCHARATNAM_TRANSPORT_"
            "AND_ANALYTIC_LATITUDE_FORMULA"
        ),
        "production_backend_imported": False,
        "precision_decimal_digits": 80,
        "tested_segments": list(DEPTHS),
        "verified_segment_runs": verified_runs,
        "all_discrete_holonomies_match": True,
        "all_analytic_formulas_match": True,
        "all_norms_one": True,
        "all_restorations_below_1e_70_at_oracle_precision": True,
        "continuous_limit_errors_strictly_decrease": True,
        "primary_boundary_matches": True,
        "reuse_boundary_matches": True,
        "fresh_restored_reuse_boundary_equal": True,
        "reverse_orientation_conjugates_holonomy": True,
        "area_perturbation_changes_holonomy": True,
        "premature_phase_canonicalization_erases_holonomy": True,
        "endpoint_only_state_distinguishes_closed_paths": False,
        "matched_compact_classical_scalar_recurrence_identical": True,
        "closed_form_public_path_product_available": True,
        "distinct_resource_unavailable_to_compact_classical": False,
        "terminal": False,
    }
    json.dump(output, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
