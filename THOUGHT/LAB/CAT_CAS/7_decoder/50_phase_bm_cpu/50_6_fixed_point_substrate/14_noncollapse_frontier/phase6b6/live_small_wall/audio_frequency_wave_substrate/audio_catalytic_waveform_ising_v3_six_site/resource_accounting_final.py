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
CONTROL_SOURCE = PACKAGE_DIR / "control_qualifier.py"
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


five = load_module(FIVE_SOURCE, "catcas_v3_final_resource_five")
six = load_module(SIX_SOURCE, "catcas_v3_final_resource_six")
controls = load_module(CONTROL_SOURCE, "catcas_v3_final_resource_controls")


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def problem_rows(path: Path) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for record in document["records"][:SAMPLE_CASE_COUNT]:
        rows.append(
            {
                "coupling": np.asarray(
                    record["coupling_matrix_J"], dtype=np.float64
                ),
                "field": np.asarray(record["field_vector_h"], dtype=np.float64),
                "label": str(record["label"]),
                "problem_sha256": str(record["problem_sha256"]),
                "source_group": str(record.get("source_group", "six_site_development")),
            }
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


def native_array_bytes(execution: Any) -> int:
    total = 0
    for name in execution.__dataclass_fields__:
        value = getattr(execution, name)
        if isinstance(value, np.ndarray):
            total += int(value.nbytes)
    return total


def machine_measurement(module: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    borrowed = module.borrowed_carrier()

    def cycle(record: dict[str, Any]) -> Callable[[], Any]:
        def execute() -> Any:
            native = module.execute_native_cycle(
                borrowed, record["coupling"], record["field"]
            )
            return module.project_boundary(native, "resource_probe")

        return execute

    cycle_calls = [cycle(record) for record in rows]
    first = rows[0]
    native = module.execute_native_cycle(
        borrowed, first["coupling"], first["field"]
    )
    expected = module.as_carrier_bank(borrowed)

    def restore() -> Any:
        restored = module.restore_carrier(native)
        if module.maximum_abs_error(restored, expected) > module.RESTORATION_MAX:
            raise RuntimeError("resource probe restoration failed")
        return restored

    complex_bytes = np.dtype(np.complex128).itemsize
    return {
        "active_spectral_bins": int(len(module.ACTIVE_BINS)),
        "derived_dense_mode_sample_bytes": int(
            module.MODE_COUNT * module.SAMPLE_COUNT * complex_bytes
        ),
        "derived_dense_mode_sample_definition": (
            "mode_count * sample_count * sizeof(complex128); not an instantiated "
            "NativeExecution carrier array"
        ),
        "executed_native_arrays_total_bytes": native_array_bytes(native),
        "median_native_boundary_runtime_ns": median_elapsed_ns(cycle_calls),
        "median_restoration_runtime_ns": median_elapsed_ns(
            [restore for _ in range(SAMPLE_CASE_COUNT)]
        ),
        "mode_count": int(module.MODE_COUNT),
        "native_carrier_state_bytes": int(native.displaced.nbytes),
        "native_carrier_state_definition": (
            "NativeExecution.displaced site-by-sample complex128 array"
        ),
        "peak_python_tracemalloc_bytes": peak_python_bytes(cycle_calls[0]),
        "sample_count": int(module.SAMPLE_COUNT),
        "site_count": int(module.SITE_COUNT),
    }


def package_bytes(path: Path) -> int:
    return sum(
        candidate.stat().st_size
        for candidate in path.iterdir()
        if candidate.is_file() and candidate != RESULT_FILE
    )


def build_document() -> dict[str, Any]:
    five_rows = problem_rows(FIVE_RESULTS)
    six_rows = problem_rows(SIX_RESULTS)
    five_measurement = machine_measurement(five, five_rows)
    six_measurement = machine_measurement(six, six_rows)
    borrowed = six.borrowed_carrier()
    control_calls = [
        (
            lambda record=record: controls.execute_controls(record, borrowed)
        )
        for record in six_rows
    ]
    control_runtime = median_elapsed_ns(control_calls)
    ratios = {
        "active_spectral_bins": metric(
            six_measurement["active_spectral_bins"]
            / five_measurement["active_spectral_bins"]
        ),
        "derived_dense_mode_sample_bytes": metric(
            six_measurement["derived_dense_mode_sample_bytes"]
            / five_measurement["derived_dense_mode_sample_bytes"]
        ),
        "executed_native_arrays_total_bytes": metric(
            six_measurement["executed_native_arrays_total_bytes"]
            / five_measurement["executed_native_arrays_total_bytes"]
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
        "evidence_disposition": {
            "accounting_only": True,
            "frozen_machine_or_preoracle_source_changed": False,
            "supersedes": "uncommitted resource-accounting v1 output",
        },
        "five_site": five_measurement,
        "measurement_law": {
            "clock": "time.perf_counter_ns",
            "exact_array_bytes": (
                "NumPy ndarray.nbytes; native carrier is NativeExecution.displaced"
            ),
            "peak_memory": "Python tracemalloc peak only",
            "sample_case_count": SAMPLE_CASE_COUNT,
            "timing_reproduction": (
                "timings are observed measurements; verify checks schema, identities, "
                "structural values, and positivity rather than byte equality"
            ),
            "warmup_count": WARMUP_COUNT,
        },
        "package_bytes_at_measurement": {
            "five_site": package_bytes(FIVE_DIR),
            "six_site_excluding_resource_output": package_bytes(PACKAGE_DIR),
        },
        "ratios_six_over_five": ratios,
        "schema": "catalytic_waveform_ising_v3_six_site_resource_accounting_v2",
        "six_site": six_measurement,
        "source_sha256": {
            "control_qualifier": sha256_file(CONTROL_SOURCE),
            "five_site_machine": sha256_file(FIVE_SOURCE),
            "six_site_machine": sha256_file(SIX_SOURCE),
        },
    }


def validate_document(document: dict[str, Any]) -> None:
    if document["schema"] != (
        "catalytic_waveform_ising_v3_six_site_resource_accounting_v2"
    ):
        raise ValueError("resource schema drift")
    expected_disposition = {
        "accounting_only": True,
        "frozen_machine_or_preoracle_source_changed": False,
        "supersedes": "uncommitted resource-accounting v1 output",
    }
    if document["evidence_disposition"] != expected_disposition:
        raise ValueError("resource evidence disposition drift")
    expected_law = {
        "clock": "time.perf_counter_ns",
        "exact_array_bytes": (
            "NumPy ndarray.nbytes; native carrier is NativeExecution.displaced"
        ),
        "peak_memory": "Python tracemalloc peak only",
        "sample_case_count": SAMPLE_CASE_COUNT,
        "timing_reproduction": (
            "timings are observed measurements; verify checks schema, identities, "
            "structural values, and positivity rather than byte equality"
        ),
        "warmup_count": WARMUP_COUNT,
    }
    if document["measurement_law"] != expected_law:
        raise ValueError("resource measurement law drift")
    modules = {
        "five_site": (five, FIVE_RESULTS),
        "six_site": (six, SIX_RESULTS),
    }
    native_definition = (
        "NativeExecution.displaced site-by-sample complex128 array"
    )
    dense_definition = (
        "mode_count * sample_count * sizeof(complex128); not an instantiated "
        "NativeExecution carrier array"
    )
    complex_bytes = np.dtype(np.complex128).itemsize
    for name, (module, results_path) in modules.items():
        published = document[name]
        record = problem_rows(results_path)[0]
        execution = module.execute_native_cycle(
            module.borrowed_carrier(), record["coupling"], record["field"]
        )
        expected_structural = {
            "active_spectral_bins": int(len(module.ACTIVE_BINS)),
            "derived_dense_mode_sample_bytes": int(
                module.MODE_COUNT * module.SAMPLE_COUNT * complex_bytes
            ),
            "executed_native_arrays_total_bytes": native_array_bytes(execution),
            "mode_count": int(module.MODE_COUNT),
            "native_carrier_state_bytes": int(execution.displaced.nbytes),
            "sample_count": int(module.SAMPLE_COUNT),
            "site_count": int(module.SITE_COUNT),
        }
        for key, value in expected_structural.items():
            if published[key] != value:
                raise ValueError(f"resource structural value drift: {name}.{key}")
        if published["native_carrier_state_definition"] != native_definition:
            raise ValueError(f"native carrier definition drift: {name}")
        if published["derived_dense_mode_sample_definition"] != dense_definition:
            raise ValueError(f"dense reference definition drift: {name}")
        for key in (
            "median_native_boundary_runtime_ns",
            "median_restoration_runtime_ns",
            "peak_python_tracemalloc_bytes",
        ):
            if int(published[key]) <= 0:
                raise ValueError(f"invalid resource measurement: {name}.{key}")
    ratio_fields = {
        "active_spectral_bins": "active_spectral_bins",
        "derived_dense_mode_sample_bytes": "derived_dense_mode_sample_bytes",
        "executed_native_arrays_total_bytes": "executed_native_arrays_total_bytes",
        "median_native_boundary_runtime": "median_native_boundary_runtime_ns",
        "median_restoration_runtime": "median_restoration_runtime_ns",
        "mode_count": "mode_count",
        "native_carrier_state_bytes": "native_carrier_state_bytes",
        "peak_python_tracemalloc": "peak_python_tracemalloc_bytes",
        "sample_count": "sample_count",
    }
    if set(document["ratios_six_over_five"]) != set(ratio_fields):
        raise ValueError("resource ratio field set drift")
    for ratio_name, field_name in ratio_fields.items():
        expected_ratio = metric(
            document["six_site"][field_name]
            / document["five_site"][field_name]
        )
        if abs(
            float(document["ratios_six_over_five"][ratio_name])
            - expected_ratio
        ) > 1.0e-12:
            raise ValueError(f"resource ratio drift: {ratio_name}")
    if document["source_sha256"] != {
        "control_qualifier": sha256_file(CONTROL_SOURCE),
        "five_site_machine": sha256_file(FIVE_SOURCE),
        "six_site_machine": sha256_file(SIX_SOURCE),
    }:
        raise ValueError("resource source identity drift")
    if document["control_cost"]["control_count_per_instance"] != 8:
        raise ValueError("resource control-count drift")
    if document["control_cost"]["sample_case_count"] != SAMPLE_CASE_COUNT:
        raise ValueError("resource control sample-count drift")
    if int(document["control_cost"]["median_six_site_control_runtime_ns"]) <= 0:
        raise ValueError("invalid control runtime")
    if any(
        int(value) <= 0
        for value in document["package_bytes_at_measurement"].values()
    ):
        raise ValueError("invalid package byte measurement")

def functional_verify() -> None:
    record = problem_rows(SIX_RESULTS)[0]
    control = controls.execute_controls(record, six.borrowed_carrier())
    if not control["all_pass"]:
        raise ValueError("resource control probe failed")


def write_atomic(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def build() -> dict[str, Any]:
    document = build_document()
    validate_document(document)
    functional_verify()
    write_atomic(RESULT_FILE, canonical_bytes(document))
    return document


def verify() -> dict[str, Any]:
    document = json.loads(RESULT_FILE.read_text(encoding="utf-8"))
    validate_document(document)
    functional_verify()
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
                "schema": document["schema"],
                "six_site": document["six_site"],
                "status": "PASS",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
