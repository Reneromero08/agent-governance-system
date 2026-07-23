from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time
import tracemalloc
import types
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np


sys.dont_write_bytecode = True


PACKAGE_DIR = Path(__file__).resolve().parent
SUBSTRATE_DIR = PACKAGE_DIR.parent
FIVE_DIR = SUBSTRATE_DIR / "audio_catalytic_waveform_ising_v3"
FIVE_SOURCE = FIVE_DIR / "v3_machine.py"
SIX_SOURCE = PACKAGE_DIR / "dimension_general_machine.py"
FIVE_RESULTS = FIVE_DIR / "DEVELOPMENT_RESULTS.json"
SIX_RESULTS = PACKAGE_DIR / "DEVELOPMENT_RESULTS.json"
SIX_CONTROLS = PACKAGE_DIR / "control_qualifier.py"
RESULT_FILE = PACKAGE_DIR / "SIX_SITE_RESOURCE_ACCOUNTING.json"
SAMPLE_CASE_COUNT = 32
WARMUP_COUNT = 4


def load_module(path: Path, name: str) -> Any:
    source = path.read_bytes()
    code = compile(source, str(path), "exec", dont_inherit=True, optimize=0)
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


five = load_module(FIVE_SOURCE, "catcas_v3_resource_five")
six = load_module(SIX_SOURCE, "catcas_v3_resource_six")
controls = load_module(SIX_CONTROLS, "catcas_v3_resource_controls")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def problem_rows(path: Path) -> list[tuple[np.ndarray, np.ndarray]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for record in document["records"][:SAMPLE_CASE_COUNT]:
        rows.append(
            (
                np.asarray(record["coupling_matrix_J"], dtype=np.float64),
                np.asarray(record["field_vector_h"], dtype=np.float64),
            )
        )
    if len(rows) != SAMPLE_CASE_COUNT:
        raise RuntimeError("resource sample corpus is incomplete")
    return rows


def median_elapsed_ns(callables: Sequence[Callable[[], Any]]) -> int:
    for callback in callables[:WARMUP_COUNT]:
        callback()
    samples: list[int] = []
    for callback in callables:
        started = time.perf_counter_ns()
        callback()
        samples.append(time.perf_counter_ns() - started)
    return int(statistics.median(samples))


def peak_python_bytes(callback: Callable[[], Any]) -> int:
    tracemalloc.start()
    callback()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return int(peak)


def machine_measurement(module: Any, rows: list[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    borrowed = module.borrowed_carrier()

    def cycle(coupling: np.ndarray, field: np.ndarray) -> Callable[[], Any]:
        def execute() -> Any:
            native = module.execute_native_cycle(borrowed, coupling, field)
            return module.project_boundary(native, "resource_probe")

        return execute

    cycle_calls = [cycle(coupling, field) for coupling, field in rows]
    first_coupling, first_field = rows[0]
    native = module.execute_native_cycle(borrowed, first_coupling, first_field)
    expected = module.as_carrier_bank(borrowed)

    def restore() -> Any:
        restored = module.restore_carrier(native)
        if module.maximum_abs_error(restored, expected) > module.RESTORATION_MAX:
            raise RuntimeError("resource probe restoration failed")
        return restored

    return {
        "active_spectral_bins": int(len(module.ACTIVE_BINS)),
        "median_native_boundary_runtime_ns": median_elapsed_ns(cycle_calls),
        "median_restoration_runtime_ns": median_elapsed_ns(
            [restore for _ in range(SAMPLE_CASE_COUNT)]
        ),
        "mode_count": int(module.MODE_COUNT),
        "native_carrier_state_bytes": int(native.displaced.nbytes),
        "peak_python_tracemalloc_bytes": peak_python_bytes(cycle_calls[0]),
        "sample_count": int(module.SAMPLE_COUNT),
        "site_count": int(module.SITE_COUNT),
    }


def package_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.iterdir()
        if candidate.is_file()
    )


def build_document() -> dict[str, Any]:
    five_measurement = machine_measurement(five, problem_rows(FIVE_RESULTS))
    six_rows = problem_rows(SIX_RESULTS)
    six_measurement = machine_measurement(six, six_rows)
    borrowed = six.borrowed_carrier()
    control_calls = [
        (
            lambda coupling=coupling, field=field: controls.execute_controls(
                {
                    "coupling": coupling,
                    "field": field,
                    "label": "resource_control",
                },
                borrowed,
            )
        )
        for coupling, field in six_rows
    ]
    control_runtime = median_elapsed_ns(control_calls)
    ratios = {
        "active_spectral_bins": metric(
            six_measurement["active_spectral_bins"]
            / five_measurement["active_spectral_bins"]
        ),
        "median_native_boundary_runtime": metric(
            six_measurement["median_native_boundary_runtime_ns"]
            / five_measurement["median_native_boundary_runtime_ns"]
        ),
        "median_restoration_runtime": metric(
            six_measurement["median_restoration_runtime_ns"]
            / five_measurement["median_restoration_runtime_ns"]
        ),
        "mode_count": metric(
            six_measurement["mode_count"] / five_measurement["mode_count"]
        ),
        "native_carrier_state_bytes": metric(
            six_measurement["native_carrier_state_bytes"]
            / five_measurement["native_carrier_state_bytes"]
        ),
        "peak_python_tracemalloc": metric(
            six_measurement["peak_python_tracemalloc_bytes"]
            / five_measurement["peak_python_tracemalloc_bytes"]
        ),
        "sample_count": metric(
            six_measurement["sample_count"] / five_measurement["sample_count"]
        ),
    }
    return {
        "control_cost": {
            "control_count_per_instance": 8,
            "median_six_site_control_runtime_ns": control_runtime,
            "sample_case_count": SAMPLE_CASE_COUNT,
        },
        "five_site": five_measurement,
        "measurement_law": {
            "clock": "time.perf_counter_ns",
            "peak_memory": "Python tracemalloc peak plus exact complex array bytes",
            "sample_case_count": SAMPLE_CASE_COUNT,
            "warmup_count": WARMUP_COUNT,
        },
        "package_bytes_at_measurement": {
            "five_site": package_bytes(FIVE_DIR),
            "six_site_before_resource_output": package_bytes(PACKAGE_DIR),
        },
        "ratios_six_over_five": ratios,
        "schema": "catalytic_waveform_ising_v3_six_site_resource_accounting_v1",
        "six_site": six_measurement,
        "source_sha256": {
            "five_site_machine": sha256_file(FIVE_SOURCE),
            "six_site_machine": sha256_file(SIX_SOURCE),
        },
    }


def validate_document(document: dict[str, Any]) -> None:
    if document["five_site"]["site_count"] != 5:
        raise ValueError("five-site resource identity failed")
    if document["six_site"]["site_count"] != 6:
        raise ValueError("six-site resource identity failed")
    expected = {
        "mode_count": 2.0,
        "active_spectral_bins": 2.0,
        "sample_count": 2.0,
        "native_carrier_state_bytes": 4.0,
    }
    for key, value in expected.items():
        if abs(float(document["ratios_six_over_five"][key]) - value) > 1.0e-12:
            raise ValueError(f"structural resource ratio drift: {key}")
    if document["source_sha256"]["five_site_machine"] != sha256_file(FIVE_SOURCE):
        raise ValueError("five-site source identity drift")
    if document["source_sha256"]["six_site_machine"] != sha256_file(SIX_SOURCE):
        raise ValueError("six-site source identity drift")
    for site in ("five_site", "six_site"):
        for key in (
            "median_native_boundary_runtime_ns",
            "median_restoration_runtime_ns",
            "peak_python_tracemalloc_bytes",
        ):
            if int(document[site][key]) <= 0:
                raise ValueError(f"invalid resource measurement: {site}.{key}")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    validate_document(document)
    write_atomic(RESULT_FILE, canonical_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    validate_document(document)
    return document


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    document = build() if args.mode == "build" else verify()
    print(
        json.dumps(
            {
                "five_site": document["five_site"],
                "ratios_six_over_five": document["ratios_six_over_five"],
                "six_site": document["six_site"],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
