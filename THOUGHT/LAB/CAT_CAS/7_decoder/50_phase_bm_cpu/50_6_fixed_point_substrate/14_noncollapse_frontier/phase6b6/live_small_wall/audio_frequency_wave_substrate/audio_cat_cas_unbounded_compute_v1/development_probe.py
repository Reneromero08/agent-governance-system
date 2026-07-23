"""Development and scaling probe for the toroidal path-sum engine."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

from phase_path_engine import (
    DEFAULT_PHASE_MODULI,
    RESTORATION_MAX,
    HoloSource,
    borrowed_carrier,
    canonical_bytes,
    classical_path_work,
    collapse_boundary,
    compile_holo,
    compute_leverage,
    engine_contract,
    engine_fingerprint,
    execute_catalytic,
    maximum_abs_error,
    source_no_smuggle,
)


HERE = Path(__file__).resolve().parent
RESULT_PATH = HERE / "DEVELOPMENT_RESULTS.json"
REPORT_PATH = HERE / "DEVELOPMENT_REPORT.md"
RESIDUE_MODULUS = 31
PHASE_MODULI = DEFAULT_PHASE_MODULI
PHASE_PRODUCT = int(np.prod(PHASE_MODULI))
SIZES = (4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 256)


def deterministic_values(label: str, count: int, modulus: int) -> tuple[int, ...]:
    output: list[int] = []
    counter = 0
    while len(output) < count:
        digest = hashlib.sha256(f"{label}:{counter}".encode("utf-8")).digest()
        output.extend(1 + byte % (modulus - 1) for byte in digest)
        counter += 1
    return tuple(output[:count])


def make_source(label: str, steps: int, max_steps: int | None = None) -> HoloSource:
    target = hashlib.sha256(f"{label}:target".encode("utf-8")).digest()[0]
    return HoloSource(
        name=label,
        residue_modulus=RESIDUE_MODULUS,
        phase_moduli=PHASE_MODULI,
        weights=deterministic_values(label, steps, RESIDUE_MODULUS),
        target_residue=target % RESIDUE_MODULUS,
        max_steps=steps if max_steps is None else max_steps,
    )


def compact_dp(source: HoloSource) -> int:
    counts = np.zeros(source.residue_modulus, dtype=np.int64)
    counts[0] = 1
    modulus = int(np.prod(source.phase_moduli))
    for weight in source.weights:
        counts = (counts + np.roll(counts, weight)) % modulus
    return int(counts[source.target_residue])


def explicit_path_materialization(source: HoloSource) -> int:
    residues = [0]
    for weight in source.weights:
        residues.extend(
            (residue + weight) % source.residue_modulus
            for residue in tuple(residues)
        )
    return sum(
        residue == source.target_residue for residue in residues
    ) % int(np.prod(source.phase_moduli))


def median_ns(action: Any, repeats: int = 5) -> int:
    samples: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        action()
        samples.append(time.perf_counter_ns() - start)
    return int(statistics.median(samples))


def run() -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    gamma_values: list[float] = []
    max_restore = 0.0
    max_root_distance = 0.0
    explicit_matches = 0
    for steps in SIZES:
        source = make_source(f"development_path_sum_{steps}", steps)
        compile_start = time.perf_counter_ns()
        compiled = compile_holo(source)
        compile_ns = time.perf_counter_ns() - compile_start
        carrier = borrowed_carrier(compiled, identity=steps)
        execution = execute_catalytic(compiled, carrier)
        boundary = collapse_boundary(compiled, execution.result_latch)
        expected = compact_dp(source)
        if not boundary.valid or boundary.count_mod_crt != expected:
            raise RuntimeError(f"phase result mismatch at size {steps}")
        if execution.restoration_max_abs > RESTORATION_MAX:
            raise RuntimeError(f"restoration failed at size {steps}")
        gamma = compute_leverage(execution)
        gamma_values.append(gamma)
        max_restore = max(max_restore, execution.restoration_max_abs)
        max_root_distance = max(
            max_root_distance, max(boundary.root_distances)
        )
        dp_ns = median_ns(lambda: compact_dp(source))
        explicit_ns: int | None = None
        explicit_result: int | None = None
        if steps <= 20:
            explicit_start = time.perf_counter_ns()
            explicit_result = explicit_path_materialization(source)
            explicit_ns = time.perf_counter_ns() - explicit_start
            if explicit_result != expected:
                raise RuntimeError("explicit path baseline disagrees")
            explicit_matches += 1
        cases.append(
            {
                "boundary": boundary.document(),
                "carrier_complex_values": int(carrier.size),
                "compile_ns": compile_ns,
                "compact_dp_ns_median": dp_ns,
                "count_mod_crt": boundary.count_mod_crt,
                "declared_phase_work": execution.total_declared_work,
                "displacement_l2": execution.displacement_l2,
                "explicit_path_count": 1 << steps,
                "explicit_path_materialization_ns": explicit_ns,
                "explicit_result": explicit_result,
                "gamma_path_work": gamma,
                "max_steps": source.max_steps,
                "native_forward_ns": execution.forward_ns,
                "program_sha256": compiled.program_sha256,
                "restoration_max_abs": execution.restoration_max_abs,
                "restoration_ns": execution.restoration_ns,
                "steps": steps,
            }
        )

    # Same physical-size carrier, materially different program, direct reuse.
    first_source = make_source("development_reuse_a", 32, max_steps=64)
    second_source = make_source("development_reuse_b", 48, max_steps=64)
    first_compiled = compile_holo(first_source)
    second_compiled = compile_holo(second_source)
    borrowed = borrowed_carrier(first_compiled, identity=777)
    first = execute_catalytic(first_compiled, borrowed)
    first_boundary = collapse_boundary(first_compiled, first.result_latch)
    second = execute_catalytic(second_compiled, first.restored_carrier)
    second_boundary = collapse_boundary(second_compiled, second.result_latch)
    if first_boundary.count_mod_crt != compact_dp(first_source):
        raise RuntimeError("first reuse computation failed")
    if second_boundary.count_mod_crt != compact_dp(second_source):
        raise RuntimeError("cross-program reuse computation failed")
    reuse_delta = maximum_abs_error(first.restored_carrier, borrowed)
    reuse_restore = maximum_abs_error(second.restored_carrier, borrowed)

    control_source = make_source("development_controls", 64)
    control_compiled = compile_holo(control_source)
    control_carrier = borrowed_carrier(control_compiled, identity=991)
    nominal = execute_catalytic(control_compiled, control_carrier)
    nominal_boundary = collapse_boundary(
        control_compiled, nominal.result_latch
    )
    expected_control = compact_dp(control_source)
    if nominal_boundary.count_mod_crt != expected_control:
        raise RuntimeError("control nominal failed")
    controls: dict[str, bool] = {}
    for fault in (
        "remove_last",
        "scramble_geometry",
        "disable_phase_lock",
    ):
        altered = execute_catalytic(
            control_compiled, control_carrier, forward_fault=fault
        )
        latch_delta = maximum_abs_error(
            altered.result_latch, nominal.result_latch
        )
        controls[fault] = latch_delta > 1.0e-6
    for mode in ("wrong_program", "wrong_order", "omitted"):
        altered = execute_catalytic(
            control_compiled, control_carrier, inverse_mode=mode
        )
        controls[mode] = altered.restoration_max_abs > 1.0e-4
    controls["borrowed_carrier_required"] = False
    try:
        execute_catalytic(
            control_compiled, np.zeros_like(control_carrier)
        )
    except ValueError:
        controls["borrowed_carrier_required"] = True
    controls["result_survives_reverse"] = (
        collapse_boundary(
            control_compiled, nominal.result_latch
        ).count_mod_crt
        == expected_control
        and nominal.restoration_max_abs <= RESTORATION_MAX
    )
    controls["actual_restored_reuse"] = (
        reuse_delta <= RESTORATION_MAX
        and reuse_restore <= RESTORATION_MAX
    )
    controls["no_smuggle"] = source_no_smuggle()["passed"]
    if not all(controls.values()):
        raise RuntimeError(
            "development control failure: "
            + repr([key for key, value in controls.items() if not value])
        )

    result = {
        "claim_ceiling": engine_contract()["claim_ceiling"],
        "controls": controls,
        "development_cases": cases,
        "engine_contract": engine_contract(),
        "engine_fingerprint": engine_fingerprint(),
        "explicit_baseline_matches": explicit_matches,
        "gamma_grows_monotonically": all(
            right > left
            for left, right in zip(gamma_values, gamma_values[1:])
        ),
        "gamma_max": max(gamma_values),
        "gamma_min": min(gamma_values),
        "max_restoration_error": max_restore,
        "max_root_distance": max_root_distance,
        "mechanism": (
            "binary path counts modulo CRT are carried as relative phases; "
            "each choice is one global triangular torus shear"
        ),
        "native_complete_path_modes": 0,
        "phase_product_modulus": PHASE_PRODUCT,
        "reuse": {
            "first_program_result": first_boundary.count_mod_crt,
            "first_restoration_error": first.restoration_max_abs,
            "restored_carrier_directly_passed": True,
            "reuse_input_delta_from_original": reuse_delta,
            "second_program_result": second_boundary.count_mod_crt,
            "second_restoration_error": second.restoration_max_abs,
            "second_restored_delta_from_original": reuse_restore,
        },
        "source_no_smuggle": source_no_smuggle(),
        "status": "DEVELOPMENT_MECHANISM_PASSED",
    }
    return result


def write_result(result: dict[str, Any]) -> None:
    RESULT_PATH.write_bytes(canonical_bytes(result))
    lines = [
        "# Toroidal path-sum development result",
        "",
        f"- status: `{result['status']}`",
        f"- engine fingerprint: `{result['engine_fingerprint']}`",
        f"- cases: {len(result['development_cases'])}",
        (
            "- path-work leverage range: "
            f"{result['gamma_min']:.6g} to {result['gamma_max']:.6g}"
        ),
        (
            "- gamma monotonically increasing: "
            f"{result['gamma_grows_monotonically']}"
        ),
        (
            "- maximum restoration error: "
            f"{result['max_restoration_error']:.12g}"
        ),
        "- controls: all PASS",
        "",
        "This development result compares the phase process with both explicit",
        "binary-path materialization and a compact conventional dynamic program.",
        "Growing Gamma is against explicit matched path-work. No advantage over",
        "the compact dynamic program is claimed.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    development = run()
    write_result(development)
    print(
        json.dumps(
            {
                "cases": len(development["development_cases"]),
                "engine_fingerprint": development["engine_fingerprint"],
                "gamma_max": development["gamma_max"],
                "status": development["status"],
            },
            sort_keys=True,
        )
    )
