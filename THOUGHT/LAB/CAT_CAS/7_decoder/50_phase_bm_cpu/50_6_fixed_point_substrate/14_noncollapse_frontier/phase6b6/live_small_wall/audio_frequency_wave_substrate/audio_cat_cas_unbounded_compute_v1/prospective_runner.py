"""Execute sealed .holo programs without loading a classical result evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from phase_path_engine import (
    RESTORATION_MAX,
    borrowed_carrier,
    canonical_bytes,
    collapse_boundary,
    compile_holo,
    compute_leverage,
    engine_fingerprint,
    execute_catalytic,
    load_holo,
    maximum_abs_error,
    sha256_bytes,
    source_no_smuggle,
)


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "PROSPECTIVE_CONTRACT.json"
RAW_PATH = HERE / "PROSPECTIVE_RAW_RESULTS.json"


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value, dtype=np.complex128)
    return sha256_bytes(array.tobytes(order="C"))


def _verify_contract(contract: dict[str, Any]) -> None:
    if contract["engine_fingerprint"] != engine_fingerprint():
        raise RuntimeError("engine fingerprint drift")
    for relative, expected in contract["source_hashes"].items():
        actual = sha256_bytes((HERE / relative).read_bytes())
        if actual != expected:
            raise RuntimeError(f"source drift: {relative}")
    entries = contract["batch"]["entries"]
    if sha256_bytes(canonical_bytes(entries)) != contract["batch"][
        "ordered_batch_sha256"
    ]:
        raise RuntimeError("ordered prospective batch drift")
    for entry in entries:
        payload = (HERE / entry["path"]).read_bytes()
        if sha256_bytes(payload) != entry["holo_sha256"]:
            raise RuntimeError(f"holo source drift: {entry['path']}")


def _latch_document(value: np.ndarray) -> list[dict[str, float]]:
    return [
        {"imag": float(item.imag), "real": float(item.real)}
        for item in np.asarray(value, dtype=np.complex128)
    ]


def run() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    _verify_contract(contract)
    carrier_pool: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cases: list[dict[str, Any]] = []
    maximum_restoration = 0.0
    minimum_displacement = float("inf")
    for entry in contract["batch"]["entries"]:
        source = load_holo(HERE / entry["path"])
        compiled = compile_holo(source)
        if compiled.program_sha256 != entry["program_sha256"]:
            raise RuntimeError("compiled program drift")
        if source.max_steps not in carrier_pool:
            original = borrowed_carrier(compiled, identity=source.max_steps)
            carrier = original
            reused = False
        else:
            original, carrier = carrier_pool[source.max_steps]
            reused = True
        input_delta = maximum_abs_error(carrier, original)
        execution = execute_catalytic(compiled, carrier)
        boundary = collapse_boundary(compiled, execution.result_latch)
        if not boundary.valid:
            raise RuntimeError(f"invalid raw boundary: {source.name}")
        if execution.restoration_max_abs > RESTORATION_MAX:
            raise RuntimeError(f"restoration failure: {source.name}")
        restored_delta = maximum_abs_error(
            execution.restored_carrier, original
        )
        carrier_pool[source.max_steps] = (
            original,
            execution.restored_carrier,
        )
        maximum_restoration = max(
            maximum_restoration, execution.restoration_max_abs
        )
        minimum_displacement = min(
            minimum_displacement, execution.displacement_l2
        )
        cases.append(
            {
                "boundary": boundary.document(),
                "declared_phase_work": execution.total_declared_work,
                "displacement_l2": execution.displacement_l2,
                "gamma_path_work": compute_leverage(execution),
                "history_factor_count": execution.history_factor_count,
                "holo_sha256": entry["holo_sha256"],
                "input_carrier_sha256": array_sha256(carrier),
                "input_delta_from_original": input_delta,
                "path": entry["path"],
                "program_sha256": compiled.program_sha256,
                "restoration_max_abs": execution.restoration_max_abs,
                "restored_carrier_sha256": array_sha256(
                    execution.restored_carrier
                ),
                "restored_delta_from_original": restored_delta,
                "result_latch": _latch_document(execution.result_latch),
                "reused_actual_restored_carrier": reused,
                "steps": len(compiled.instructions),
            }
        )

    # Prospectively named mechanism controls use the largest alpha program.
    control_entry = next(
        entry
        for entry in contract["batch"]["entries"]
        if entry["family"] == "phase_path_alpha" and entry["steps"] == 256
    )
    control_source = load_holo(HERE / control_entry["path"])
    control_compiled = compile_holo(control_source)
    control_carrier = borrowed_carrier(control_compiled, identity=909)
    nominal = execute_catalytic(control_compiled, control_carrier)
    nominal_boundary = collapse_boundary(
        control_compiled, nominal.result_latch
    )
    control_results: dict[str, dict[str, Any]] = {}
    for fault in (
        "remove_last",
        "remove_phase_interaction",
        "scramble_geometry",
        "disable_phase_lock",
    ):
        altered = execute_catalytic(
            control_compiled, control_carrier, forward_fault=fault
        )
        altered_boundary = collapse_boundary(
            control_compiled, altered.result_latch
        )
        latch_delta = maximum_abs_error(
            nominal.result_latch, altered.result_latch
        )
        control_results[fault] = {
            "altered_boundary_valid": altered_boundary.valid,
            "latch_delta": latch_delta,
            "passed": (
                (not altered_boundary.valid)
                or latch_delta > 1.0e-6
            ),
        }
    for mode in ("wrong_program", "wrong_order", "omitted"):
        altered = execute_catalytic(
            control_compiled, control_carrier, inverse_mode=mode
        )
        control_results[mode] = {
            "restoration_max_abs": altered.restoration_max_abs,
            "passed": altered.restoration_max_abs > 1.0e-4,
        }
    control_results["no_smuggle"] = {
        **source_no_smuggle(),
        "passed": source_no_smuggle()["passed"],
    }
    control_results["result_survives_uncompute"] = {
        "boundary_unchanged": (
            collapse_boundary(
                control_compiled, nominal.result_latch
            ).document()
            == nominal_boundary.document()
        ),
        "restoration_max_abs": nominal.restoration_max_abs,
        "passed": nominal.restoration_max_abs <= RESTORATION_MAX,
    }
    failed_controls = [
        name
        for name, outcome in control_results.items()
        if not outcome["passed"]
    ]
    if failed_controls:
        raise RuntimeError(f"prospective controls failed: {failed_controls}")

    family_gamma: dict[str, list[float]] = {
        family: [] for family in contract["batch"]["families"]
    }
    for entry, case in zip(contract["batch"]["entries"], cases):
        family_gamma[entry["family"]].append(case["gamma_path_work"])
    gamma_monotonic = all(
        all(right > left for left, right in zip(values, values[1:]))
        for values in family_gamma.values()
    )
    if not gamma_monotonic:
        raise RuntimeError("prospective Gamma failed to grow")
    raw = {
        "all_boundaries_valid": all(
            case["boundary"]["valid"] for case in cases
        ),
        "all_controls_pass": True,
        "all_restorations_pass": all(
            case["restoration_max_abs"] <= RESTORATION_MAX for case in cases
        ),
        "all_reuse_pass": all(
            case["restored_delta_from_original"] <= RESTORATION_MAX
            for case in cases
        ),
        "case_count": len(cases),
        "cases": cases,
        "contract_sha256": sha256_bytes(CONTRACT_PATH.read_bytes()),
        "controls": control_results,
        "engine_fingerprint": engine_fingerprint(),
        "gamma_grows_with_size_in_every_family": gamma_monotonic,
        "maximum_restoration_error": maximum_restoration,
        "minimum_displacement_l2": minimum_displacement,
        "ordered_batch_sha256": contract["batch"][
            "ordered_batch_sha256"
        ],
        "oracle_or_external_evaluator_loaded": False,
        "raw_result_root": "",
        "schema": "cat_cas.toroidal_path_sum.raw.v1",
        "uninterpretable": 0,
    }
    root_document = dict(raw)
    root_document["raw_result_root"] = None
    raw["raw_result_root"] = sha256_bytes(canonical_bytes(root_document))
    return raw


def write_raw(raw: dict[str, Any]) -> None:
    RAW_PATH.write_bytes(canonical_bytes(raw))


if __name__ == "__main__":
    result = run()
    write_raw(result)
    print(
        json.dumps(
            {
                "cases": result["case_count"],
                "raw_result_root": result["raw_result_root"],
                "status": "PROSPECTIVE_RAW_EXECUTION_PASSED",
            },
            sort_keys=True,
        )
    )
