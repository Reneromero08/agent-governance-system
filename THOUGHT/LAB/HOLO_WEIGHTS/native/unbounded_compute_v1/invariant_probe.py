"""Relational invariance controls for the sealed path-sum programs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from phase_path_engine import (
    RESTORATION_MAX,
    HoloSource,
    borrowed_carrier,
    canonical_bytes,
    collapse_boundary,
    compile_holo,
    execute_catalytic,
    load_holo,
)


HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "PROSPECTIVE_CONTRACT.json"
RAW_PATH = HERE / "PROSPECTIVE_RAW_RESULTS.json"
RESULT_PATH = HERE / "INVARIANT_RESULTS.json"


def transformed_source(
    source: HoloSource, *, reverse: bool = False, unit: int = 1
) -> HoloSource:
    weights = tuple(reversed(source.weights)) if reverse else source.weights
    modulus = source.residue_modulus
    return HoloSource(
        name=(
            f"{source.name}__reverse"
            if reverse
            else f"{source.name}__unit_{unit}"
        ),
        residue_modulus=modulus,
        phase_moduli=source.phase_moduli,
        weights=tuple((unit * weight) % modulus for weight in weights),
        target_residue=(unit * source.target_residue) % modulus,
        max_steps=source.max_steps,
    )


def execute_count(source: HoloSource, identity: int) -> tuple[int, float]:
    compiled = compile_holo(source)
    carrier = borrowed_carrier(compiled, identity=identity)
    execution = execute_catalytic(compiled, carrier)
    boundary = collapse_boundary(compiled, execution.result_latch)
    if not boundary.valid or boundary.count_mod_crt is None:
        raise RuntimeError("invariance boundary invalid")
    if execution.restoration_max_abs > RESTORATION_MAX:
        raise RuntimeError("invariance restoration failed")
    return boundary.count_mod_crt, execution.restoration_max_abs


def run() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    raw = json.loads(RAW_PATH.read_text(encoding="utf-8"))
    raw_by_path = {case["path"]: case for case in raw["cases"]}
    records: list[dict[str, Any]] = []
    maximum_restoration = 0.0
    for case_index, entry in enumerate(contract["batch"]["entries"]):
        source = load_holo(HERE / entry["path"])
        nominal = raw_by_path[entry["path"]]["boundary"]["count_mod_crt"]
        reverse_count, reverse_restore = execute_count(
            transformed_source(source, reverse=True),
            identity=1100 + case_index,
        )
        if reverse_count != nominal:
            raise RuntimeError("weight-order invariant failed")
        unit_results: dict[str, int] = {}
        for unit in (3, 7):
            count, restore = execute_count(
                transformed_source(source, unit=unit),
                identity=2100 + 10 * case_index + unit,
            )
            if count != nominal:
                raise RuntimeError("cyclic automorphism invariant failed")
            unit_results[str(unit)] = count
            maximum_restoration = max(maximum_restoration, restore)
        maximum_restoration = max(maximum_restoration, reverse_restore)
        records.append(
            {
                "cyclic_unit_results": unit_results,
                "nominal_count_mod_crt": nominal,
                "path": entry["path"],
                "reversed_order_count_mod_crt": reverse_count,
                "steps": entry["steps"],
            }
        )
    return {
        "cases": len(records),
        "cyclic_automorphism_checks": 2 * len(records),
        "maximum_restoration_error": maximum_restoration,
        "non_affine_scramble_control_from_raw": raw["controls"][
            "scramble_geometry"
        ]["passed"],
        "records": records,
        "schema": "cat_cas.toroidal_path_sum.invariants.v1",
        "verdict": "PASS",
        "weight_order_checks": len(records),
    }


if __name__ == "__main__":
    result = run()
    RESULT_PATH.write_bytes(canonical_bytes(result))
    print(
        json.dumps(
            {
                "automorphisms": result["cyclic_automorphism_checks"],
                "cases": result["cases"],
                "order_checks": result["weight_order_checks"],
                "verdict": result["verdict"],
            },
            sort_keys=True,
        )
    )
