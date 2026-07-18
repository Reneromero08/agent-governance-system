#!/usr/bin/env python3
"""Deterministic, offline P0 raw-record analyzer and evidence qualifier.

This program never opens an instrument or network connection.  Its only inputs are
strict JSON and little-endian signed-int16 files supplied on disk.  Synthetic evidence
can qualify the implementation but can never mint a physical claim.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timedelta
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import shutil
import struct
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

NUMERICAL_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
for _thread_key, _thread_value in NUMERICAL_THREAD_ENV.items():
    os.environ[_thread_key] = _thread_value

import numpy as np

SCHEMA = "p0.raw-bundle.v1"
RESULT_SCHEMA = "p0.scientific-result.v1"
FIXTURE_SCHEMA = "p0.scientific-fixtures.v1"
FS = 1_000_000
CHANNELS = 4
SAMPLES = 3_101_000
PAYLOAD_BYTES = SAMPLES * CHANNELS * 2
N_CMD = 1_101_000
WINDOW = 2_048
HOP = 256
NW_LAG = 7
BLOCK = 8
F_REF = 32_768.0
HEX64 = set("0123456789abcdef")
ROLE_ORDER = ("arm_0", "arm_pi", "zero_drive", "resonator_removed", "dummy_c0")
POSITIVE_CASES = (
    "ideal_source_off_ringdown",
    "ideal_matched_0_pi_antipode",
    "nonzero_noise_within_threshold",
    "frequency_and_decay_match_at_equality",
    "phase_relation_at_equality",
    "minimum_256_cycle_usable_window",
    "valid_ch2_transition_and_guard",
    "valid_environment_at_boundaries",
)
SCIENTIFIC_NEGATIVES = {
    "zero_drive": "ZERO_DRIVE_CONTROL",
    "resonator_removed": "RESONATOR_REMOVED_CONTROL",
    "dummy_c0_feedthrough": "DUMMY_C0_FEEDTHROUGH",
    "source_left_on": "SOURCE_OFF_NOT_WITNESSED",
    "off_resonance_response": "OFF_RESONANCE_RESPONSE",
    "detector_impulse_memory": "DETECTOR_MEMORY_OVER_10US",
    "controller_buffer_replay": "POST_BARRIER_BUFFER_FORBIDDEN",
    "analog_switch_charge_transient": "TRANSIENT_AFTER_ADMIT",
    "relay_bounce_transient": "CONTACT_BOUNCE_AFTER_ADMIT",
    "source_leakage_after_guard": "SOURCE_FEEDTHROUGH",
    "reference_leakage": "REFERENCE_FEEDTHROUGH",
    "amplitude_mismatch": "AMPLITUDE_MISMATCH",
    "frequency_mismatch": "FREQUENCY_MISMATCH",
    "decay_mismatch": "DECAY_MISMATCH",
    "pi_2_phase": "HALF_TURN_PHASE_MISMATCH",
    "fixed_random_phases": "FIXED_PHASE_CONTROL",
    "timing_mismatch": "TIMING_MISMATCH_CONTROL",
    "wrong_termination": "TERMINATION_IDENTITY_MISMATCH",
    "wrong_guard_interval": "GUARD_INTERVAL_SHORT",
    "ch2_illegal_code": "CH2_ILLEGAL_STATE",
    "ch2_nearest_code_ambiguity": "CH2_AMBIGUOUS_LEVEL",
    "post_off_state_reentry": "CH2_POST_OFF_REENTRY",
    "missing_witness": "CH2_WITNESS_MISSING",
    "channel_swap": "CHANNEL_ROLE_MISMATCH",
    "timebase_drift": "TIMEBASE_MISMATCH",
    "channel_skew_violation": "CHANNEL_SKEW_OVER_LIMIT",
    "clipping": "CLIPPING",
    "adc_saturation": "ADC_SATURATION",
    "environment_cadence_failure": "ENVIRONMENT_CADENCE",
    "temperature_mismatch": "TEMPERATURE_MISMATCH",
    "humidity_mismatch": "HUMIDITY_MISMATCH",
    "vibration_mismatch": "VIBRATION_MISMATCH",
}
MALFORMED_NEGATIVES = {
    "truncated_binary": "PAYLOAD_SIZE",
    "extra_samples": "PAYLOAD_SIZE",
    "wrong_channel_count": "CHANNEL_COUNT",
    "wrong_dtype": "DTYPE",
    "wrong_endian": "ENDIAN",
    "nonfinite_scale_metadata": "NONFINITE_JSON",
    "missing_calibration": "MISSING_FIELD",
    "malformed_json": "MALFORMED_JSON",
    "duplicate_json_key": "DUPLICATE_KEY",
    "unknown_field": "UNKNOWN_FIELD",
    "path_traversal": "UNSAFE_PATH",
    "hash_mutation": "HASH_MISMATCH",
    "manifest_role_substitution": "ROLE_SUBSTITUTION",
    "assignment_reveal_before_custody_closure": "EARLY_ASSIGNMENT_REVEAL",
    "threshold_mutation_after_primary_data": "THRESHOLD_CUSTODY",
}


class Reject(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dependency_identity() -> dict[str, Any]:
    distribution = importlib.metadata.distribution("numpy")
    record_text = distribution.read_text("RECORD")
    core_binary = Path(np.core._multiarray_umath.__file__).resolve()
    executable = Path(sys.executable).resolve()
    if record_text is None or not core_binary.is_file() or not executable.is_file():
        raise Reject("DEPENDENCY_CUSTODY")
    return {
        "byteorder": sys.byteorder,
        "machine": platform.machine(),
        "numpy_config": np.__config__.CONFIG,
        "numpy_core_binary_sha256": sha256_file(core_binary),
        "numpy_distribution_record_sha256": sha256_bytes(record_text.encode("utf-8")),
        "numpy_version": np.__version__,
        "platform": platform.platform(),
        "python_cache_tag": sys.implementation.cache_tag,
        "python_executable_sha256": sha256_file(executable),
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "thread_environment": dict(NUMERICAL_THREAD_ENV),
    }


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise Reject("DUPLICATE_KEY", key)
        out[key] = value
    return out


def load_json_strict(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Reject("UTF8", str(exc)) from exc
    try:
        value = json.loads(text, object_pairs_hook=_pairs, parse_constant=lambda x: (_ for _ in ()).throw(Reject("NONFINITE_JSON", x)))
    except Reject:
        raise
    except json.JSONDecodeError as exc:
        raise Reject("MALFORMED_JSON", str(exc)) from exc
    if not isinstance(value, dict):
        raise Reject("JSON_ROOT", "object required")
    if raw != canonical_bytes(value):
        raise Reject("NONCANONICAL_JSON", str(path))
    return value


def exact_fields(obj: dict[str, Any], required: set[str], where: str) -> None:
    missing = sorted(required - set(obj))
    unknown = sorted(set(obj) - required)
    if missing:
        raise Reject("MISSING_FIELD", f"{where}:{','.join(missing)}")
    if unknown:
        raise Reject("UNKNOWN_FIELD", f"{where}:{','.join(unknown)}")


def lower_hash(value: Any, where: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX64 for c in value):
        raise Reject("HASH_FORMAT", where)
    return value


DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:e-?[1-9][0-9]*)?$")
ENV_DECIMAL_RE = re.compile(r"^-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$")


def decimal(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str):
        raise Reject("NUMBER_ENCODING", where)
    if not DECIMAL_RE.fullmatch(value):
        raise Reject("NUMBER_ENCODING", where)
    try:
        parsed = float(value)
    except ValueError as exc:
        raise Reject("NUMBER_ENCODING", where) from exc
    if not math.isfinite(parsed):
        raise Reject("NONFINITE_JSON", where)
    if parsed == 0.0 and value.startswith("-"):
        raise Reject("NUMBER_ENCODING", where)
    return parsed


def environment_decimal(value: Any, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, str) or not ENV_DECIMAL_RE.fullmatch(value):
        raise Reject("NUMBER_ENCODING", where)
    parsed = float(value)
    if not math.isfinite(parsed) or (parsed == 0.0 and value.startswith("-")):
        raise Reject("NUMBER_ENCODING", where)
    return parsed


def nonnegative_decimal(value: Any, where: str) -> float:
    parsed = decimal(value, where)
    if parsed < 0:
        raise Reject("NUMBER_DOMAIN", where)
    return parsed


def nonnegative_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Reject("NUMBER_DOMAIN", where)
    return value


def safe_relative(base: Path, value: Any, where: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise Reject("UNSAFE_PATH", where)
    rel = Path(value)
    if rel.is_absolute() or any(part in ("", ".", "..") for part in rel.parts):
        raise Reject("UNSAFE_PATH", where)
    target = (base / rel).resolve()
    if base.resolve() not in target.parents:
        raise Reject("UNSAFE_PATH", where)
    return target


ENVIRONMENT_HEADER = (
    "nearest_raw_sample_index,monotonic_ns,utc_timestamp,sensor_serial_hex,"
    "command_hex,temperature_ticks_hex,temperature_crc8_hex,rh_ticks_hex,"
    "rh_crc8_hex,temperature_C,rh_percent"
)
UTC_RE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
HEX8_RE = re.compile(r"^[0-9a-f]{8}$")
HEX4_RE = re.compile(r"^[0-9a-f]{4}$")
HEX2_RE = re.compile(r"^[0-9a-f]{2}$")


def sht4x_crc8(word: bytes) -> int:
    value = 0xFF
    for byte in word:
        value ^= byte
        for _ in range(8):
            value = ((value << 1) ^ 0x31) & 0xFF if value & 0x80 else (value << 1) & 0xFF
    return value


def sht4x_temperature(ticks: int) -> float:
    return -45.0 + 175.0 * ticks / 65535.0


def sht4x_humidity(ticks: int) -> float:
    return -6.0 + 125.0 * ticks / 65535.0


def parse_environment_csv(
    path: Path,
    expected_bytes: int,
    expected_hash: str,
    expected_count: int,
    expected_sensor_serial: str,
    expected_command: str,
) -> list[tuple[int, float, float]]:
    raw = path.read_bytes()
    if len(raw) != expected_bytes or sha256_bytes(raw) != expected_hash:
        raise Reject("HASH_MISMATCH", "environment")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Reject("UTF8", "environment") from exc
    if not text.endswith("\n") or "\r" in text or '"' in text or text.startswith("\ufeff"):
        raise Reject("ENVIRONMENT_FORMAT")
    lines = text[:-1].split("\n")
    if not lines or lines[0] != ENVIRONMENT_HEADER or len(lines) - 1 != expected_count:
        raise Reject("ENVIRONMENT_CADENCE")
    records: list[tuple[int, float, float]] = []
    previous = -100_000
    previous_monotonic_ns = -100_000_000
    previous_time: datetime | None = None
    for line in lines[1:]:
        fields = line.split(",")
        if (
            len(fields) != 11
            or not fields[0].isdigit()
            or not fields[1].isdigit()
            or not UTC_RE.fullmatch(fields[2])
            or not HEX8_RE.fullmatch(fields[3])
            or not HEX2_RE.fullmatch(fields[4])
            or not HEX4_RE.fullmatch(fields[5])
            or not HEX2_RE.fullmatch(fields[6])
            or not HEX4_RE.fullmatch(fields[7])
            or not HEX2_RE.fullmatch(fields[8])
        ):
            raise Reject("ENVIRONMENT_FORMAT")
        index = int(fields[0])
        monotonic_ns = int(fields[1])
        if fields[0] != str(index) or fields[1] != str(monotonic_ns):
            raise Reject("ENVIRONMENT_FORMAT")
        try:
            timestamp = datetime.strptime(fields[2], "%Y-%m-%dT%H:%M:%S.%fZ")
        except ValueError as exc:
            raise Reject("ENVIRONMENT_FORMAT") from exc
        if fields[3] != expected_sensor_serial or fields[4] != expected_command:
            raise Reject("ENVIRONMENT_SENSOR_IDENTITY")
        temperature_ticks = int(fields[5], 16)
        humidity_ticks = int(fields[7], 16)
        if sht4x_crc8(temperature_ticks.to_bytes(2, "big")) != int(fields[6], 16) or sht4x_crc8(humidity_ticks.to_bytes(2, "big")) != int(fields[8], 16):
            raise Reject("ENVIRONMENT_CRC")
        temperature = environment_decimal(fields[9], "environment.temperature_C")
        humidity = environment_decimal(fields[10], "environment.rh_percent")
        if abs(temperature - sht4x_temperature(temperature_ticks)) > 5e-12 or abs(humidity - sht4x_humidity(humidity_ticks)) > 5e-12:
            raise Reject("ENVIRONMENT_CONVERSION")
        if index - previous != 100_000 or index < 0 or index >= SAMPLES:
            raise Reject("ENVIRONMENT_CADENCE")
        if monotonic_ns - previous_monotonic_ns != 100_000_000:
            raise Reject("ENVIRONMENT_CADENCE")
        if previous_time is not None and timestamp - previous_time != timedelta(microseconds=100_000):
            raise Reject("ENVIRONMENT_CADENCE")
        if not 20.0 <= temperature <= 30.0:
            raise Reject("TEMPERATURE_MISMATCH")
        if not 20.0 <= humidity <= 60.0:
            raise Reject("HUMIDITY_MISMATCH")
        records.append((index, temperature, humidity))
        previous = index
        previous_monotonic_ns = monotonic_ns
        previous_time = timestamp
    return records


def validate_metadata(meta: dict[str, Any], base: Path) -> tuple[Path, dict[str, Any]]:
    exact_fields(meta, {"schema", "run_id", "evidence_class", "role", "instrument", "export", "payload", "clock", "source", "witness", "environment", "custody", "thresholds"}, "root")
    if meta["schema"] != SCHEMA or meta["evidence_class"] not in ("SYNTHETIC", "PHYSICAL") or meta["role"] not in ROLE_ORDER:
        raise Reject("ENUM", "root")
    if not isinstance(meta["run_id"], str) or not meta["run_id"]:
        raise Reject("TYPE", "run_id")
    instrument = meta["instrument"]
    exact_fields(instrument, {"manufacturer", "model", "serial", "firmware", "driver", "configuration_queryback_sha256"}, "instrument")
    lower_hash(instrument["configuration_queryback_sha256"], "instrument.configuration_queryback_sha256")
    export = meta["export"]
    exact_fields(export, {"adapter_id", "adapter_sha256", "native_file_sha256", "native_file_bytes", "lossless_assertions"}, "export")
    lower_hash(export["adapter_sha256"], "export.adapter_sha256")
    lower_hash(export["native_file_sha256"], "export.native_file_sha256")
    if isinstance(export["native_file_bytes"], bool) or not isinstance(export["native_file_bytes"], int) or export["native_file_bytes"] <= 0:
        raise Reject("TYPE", "native_file_bytes")
    assertions = export["lossless_assertions"]
    exact_fields(assertions, {"sample_loss", "reordering", "averaging", "filtering", "resampling", "clipping_concealment", "unit_ambiguity"}, "lossless_assertions")
    if any(value is not False for value in assertions.values()):
        raise Reject("ADAPTER_TRANSFORM", "lossless assertions")
    payload = meta["payload"]
    exact_fields(payload, {"path", "sha256", "bytes", "dtype", "endian", "layout", "channels", "samples_per_channel", "sample_rate_hz", "scale_per_code", "offset"}, "payload")
    path = safe_relative(base, payload["path"], "payload.path")
    lower_hash(payload["sha256"], "payload.sha256")
    if payload["bytes"] != PAYLOAD_BYTES:
        raise Reject("PAYLOAD_SIZE", "metadata")
    if payload["dtype"] != "int16" or payload["endian"] != "little" or payload["layout"] != "sample-major-interleaved":
        raise Reject("DTYPE" if payload["dtype"] != "int16" else "ENDIAN" if payload["endian"] != "little" else "LAYOUT")
    if payload["channels"] != ["CH0_SOURCE", "CH1_CARRIER", "CH2_WITNESS", "CH3_VIBRATION"]:
        raise Reject("CHANNEL_COUNT" if len(payload["channels"]) != 4 else "CHANNEL_ROLE_MISMATCH")
    if payload["samples_per_channel"] != SAMPLES or payload["sample_rate_hz"] != FS:
        raise Reject("PAYLOAD_SIZE" if payload["samples_per_channel"] != SAMPLES else "TIMEBASE_MISMATCH")
    if not isinstance(payload["scale_per_code"], list) or len(payload["scale_per_code"]) != 4:
        raise Reject("CHANNEL_COUNT", "scale")
    scales = [decimal(x, "payload.scale_per_code") for x in payload["scale_per_code"]]
    if any(value <= 0 for value in scales):
        raise Reject("SCALE_MISMATCH")
    offsets = payload["offset"]
    if not isinstance(offsets, list) or len(offsets) != 4:
        raise Reject("CHANNEL_COUNT", "offset")
    parsed_offsets = [decimal(x, "payload.offset") for x in offsets]
    if any(value != 0 for value in parsed_offsets):
        raise Reject("OFFSET_MISMATCH")
    clock = meta["clock"]
    exact_fields(clock, {"identity", "frequency_hz", "sample_rate_hz", "channel_skew_seconds", "record_start_mode", "external_trigger_connected", "phase_gauge", "alignment"}, "clock")
    if decimal(clock["frequency_hz"], "clock.frequency_hz") <= 0 or decimal(clock["sample_rate_hz"], "clock.sample_rate_hz") != FS:
        raise Reject("TIMEBASE_MISMATCH")
    if decimal(clock["channel_skew_seconds"], "clock.channel_skew_seconds") > 0.1e-6:
        raise Reject("CHANNEL_SKEW_OVER_LIMIT")
    if clock["record_start_mode"] != "SOFTWARE_PREARM_FREE_RUN" or clock["external_trigger_connected"] is not False or clock["phase_gauge"] != "CH0_SOURCE" or clock["alignment"] != "CH2_WITNESS":
        raise Reject("ACQUISITION_ALIGNMENT")
    source = meta["source"]
    exact_fields(source, {"model", "phase_command_rad", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "frequency_hz", "amplitude_vpp", "offset_v", "reference_frequency_hz", "reference_amplitude_vpp", "reference_offset_v", "reference_phase_command_rad", "dual_channel_phase_locked", "monitor_network", "output_mode", "load_mode", "qualified_preparation_cycles", "source_remains_on_through_record", "output_ohms", "setup_queryback_sha256"}, "source")
    phase = decimal(source["phase_command_rad"], "source.phase_command_rad")
    if phase not in (0.0, math.pi):
        raise Reject("SOURCE_PHASE")
    f_ref = decimal(clock["frequency_hz"], "clock.frequency_hz")
    if (
        f_ref != F_REF
        or decimal(source["frequency_hz"], "source.frequency_hz") != f_ref
        or decimal(source["amplitude_vpp"], "source.amplitude_vpp") != 0.400
        or decimal(source["offset_v"], "source.offset_v") != 0.0
        or decimal(source["reference_frequency_hz"], "source.reference_frequency_hz") != 2.0 * f_ref
        or decimal(source["reference_amplitude_vpp"], "source.reference_amplitude_vpp") != 0.100
        or decimal(source["reference_offset_v"], "source.reference_offset_v") != 0.0
        or decimal(source["reference_phase_command_rad"], "source.reference_phase_command_rad") != 0.0
        or source["dual_channel_phase_locked"] is not True
        or source["monitor_network"] != "PASSIVE_100K_PLUS_100K_DUAL_TONE_SUM"
        or source["output_mode"] != "CONTINUOUS_SINE"
        or source["load_mode"] != "HIGH_Z"
        or source["qualified_preparation_cycles"] != 32768
        or source["source_remains_on_through_record"] is not True
        or decimal(source["output_ohms"], "source.output_ohms") != 50.0
    ):
        raise Reject("SOURCE_SETUP")
    nonnegative_decimal(source["phase_skew_standard_uncertainty_rad"], "source.phase_skew_standard_uncertainty_rad")
    nonnegative_decimal(source["phase_drive_cal_standard_uncertainty_rad"], "source.phase_drive_cal_standard_uncertainty_rad")
    lower_hash(source["setup_queryback_sha256"], "source.setup_queryback_sha256")
    witness = meta["witness"]
    exact_fields(witness, {"centroids_code", "sigma_code", "gate_search_start", "gate_search_stop", "stable_off_samples", "guard_samples", "max_transition_samples", "calibration_sha256"}, "witness")
    witness_body = {key: witness[key] for key in ("centroids_code", "sigma_code", "gate_search_start", "gate_search_stop", "stable_off_samples", "guard_samples", "max_transition_samples")}
    centroids = witness["centroids_code"]
    sigma_code = decimal(witness["sigma_code"], "witness.sigma_code")
    if not isinstance(centroids, list) or len(centroids) != 16 or any(isinstance(value, bool) or not isinstance(value, int) for value in centroids) or sigma_code <= 0 or witness["gate_search_start"] != 1_000_000 or witness["gate_search_stop"] != 1_200_000 or witness["stable_off_samples"] != 1000 or witness["guard_samples"] != 10000 or witness["max_transition_samples"] != 14500:
        raise Reject("WITNESS_CALIBRATION")
    if any(centroids[index + 1] - centroids[index] < 10 * sigma_code for index in range(15)):
        raise Reject("WITNESS_CALIBRATION")
    lower_hash(witness["calibration_sha256"], "witness.calibration_sha256")
    if witness["calibration_sha256"] != sha256_bytes(canonical_bytes(witness_body)):
        raise Reject("HASH_MISMATCH", "witness calibration")
    environment = meta["environment"]
    exact_fields(environment, {"cadence_hz", "temperature_c", "humidity_rh", "vibration_rms_m_s2", "vibration_peak_m_s2", "sensor_serial_hex", "measurement_command_hex", "clock_mapping_sha256", "calibration_sha256", "record_path", "record_sha256", "record_bytes", "record_count"}, "environment")
    if environment_decimal(environment["cadence_hz"], "environment.cadence_hz") != 10.0:
        raise Reject("ENVIRONMENT_CADENCE")
    temperature = environment_decimal(environment["temperature_c"], "environment.temperature_c")
    humidity = environment_decimal(environment["humidity_rh"], "environment.humidity_rh")
    vibration_rms = environment_decimal(environment["vibration_rms_m_s2"], "environment.vibration_rms_m_s2")
    vibration_peak = environment_decimal(environment["vibration_peak_m_s2"], "environment.vibration_peak_m_s2")
    if not 20 <= temperature <= 30:
        raise Reject("TEMPERATURE_MISMATCH")
    if not 20 <= humidity <= 60:
        raise Reject("HUMIDITY_MISMATCH")
    if vibration_rms > 0.05 or vibration_peak > 0.5:
        raise Reject("VIBRATION_MISMATCH")
    lower_hash(environment["calibration_sha256"], "environment.calibration_sha256")
    lower_hash(environment["clock_mapping_sha256"], "environment.clock_mapping_sha256")
    if not isinstance(environment["sensor_serial_hex"], str) or not HEX8_RE.fullmatch(environment["sensor_serial_hex"]) or environment["measurement_command_hex"] != "fd":
        raise Reject("ENVIRONMENT_SENSOR_IDENTITY")
    lower_hash(environment["record_sha256"], "environment.record_sha256")
    env_path = safe_relative(base, environment["record_path"], "environment.record_path")
    if isinstance(environment["record_bytes"], bool) or not isinstance(environment["record_bytes"], int) or isinstance(environment["record_count"], bool) or not isinstance(environment["record_count"], int):
        raise Reject("TYPE", "environment record")
    records = parse_environment_csv(env_path, environment["record_bytes"], environment["record_sha256"], environment["record_count"], environment["sensor_serial_hex"], environment["measurement_command_hex"])
    mean_temperature = sum(value[1] for value in records) / len(records)
    mean_humidity = sum(value[2] for value in records) / len(records)
    if abs(mean_temperature - temperature) > 1e-12:
        raise Reject("TEMPERATURE_MISMATCH")
    if abs(mean_humidity - humidity) > 1e-12:
        raise Reject("HUMIDITY_MISMATCH")
    custody = meta["custody"]
    exact_fields(custody, {"calibration_sha256", "assignment_commitment_sha256", "assignment_revealed", "thresholds_frozen_before_primary", "primary_observed"}, "custody")
    lower_hash(custody["calibration_sha256"], "custody.calibration_sha256")
    lower_hash(custody["assignment_commitment_sha256"], "custody.assignment_commitment_sha256")
    if custody["assignment_revealed"] and not custody["primary_observed"]:
        raise Reject("EARLY_ASSIGNMENT_REVEAL")
    if custody["assignment_revealed"] is not True or custody["primary_observed"] is not True:
        raise Reject("ASSIGNMENT_NOT_REVEALED")
    if not custody["thresholds_frozen_before_primary"]:
        raise Reject("THRESHOLD_CUSTODY")
    thresholds = meta["thresholds"]
    exact_fields(thresholds, {"neg", "amplitude", "frequency", "decay", "phase", "feedthrough", "calibration_u95", "sha256"}, "thresholds")
    limits = {"neg": 0.100, "amplitude": 0.050, "frequency": 5e-6, "decay": 0.050, "phase": 0.050, "feedthrough": 0.100}
    for key, cap in limits.items():
        value = decimal(thresholds[key], f"thresholds.{key}")
        if value < 0 or value > cap:
            raise Reject("THRESHOLD_CAP", key)
    calibration_u95 = thresholds["calibration_u95"]
    exact_fields(calibration_u95, set(limits), "thresholds.calibration_u95")
    for key in limits:
        value = decimal(calibration_u95[key], f"thresholds.calibration_u95.{key}")
        if value < 0 or value > decimal(thresholds[key], f"thresholds.{key}"):
            raise Reject("CALIBRATION_UNCERTAINTY", key)
    lower_hash(thresholds["sha256"], "thresholds.sha256")
    threshold_body = {key: thresholds[key] for key in (*limits, "calibration_u95")}
    if thresholds["sha256"] != sha256_bytes(canonical_bytes(threshold_body)):
        raise Reject("HASH_MISMATCH", "thresholds")
    calibration_body = {"clock_identity": clock["identity"], "environment_cadence_hz": environment["cadence_hz"], "environment_sensor_serial_hex": environment["sensor_serial_hex"], "environment_measurement_command_hex": environment["measurement_command_hex"], "clock_mapping_sha256": environment["clock_mapping_sha256"], "witness_calibration_sha256": witness["calibration_sha256"]}
    if environment["calibration_sha256"] != sha256_bytes(canonical_bytes(calibration_body)) or custody["calibration_sha256"] != environment["calibration_sha256"]:
        raise Reject("HASH_MISMATCH", "calibration")
    if not path.is_file() or path.stat().st_size != PAYLOAD_BYTES:
        raise Reject("PAYLOAD_SIZE", str(path))
    if sha256_file(path) != payload["sha256"]:
        raise Reject("HASH_MISMATCH", "payload")
    if export["native_file_sha256"] != payload["sha256"] or export["native_file_bytes"] != payload["bytes"]:
        raise Reject("HASH_MISMATCH", "native export")
    payload["_scales"] = scales
    meta["_f_ref"] = f_ref
    environment["_records"] = records
    return path, meta


def wrap(value: float | np.ndarray) -> Any:
    return (value + math.pi) % (2 * math.pi) - math.pi


def ascending_sum(values: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.cumsum(values, axis=axis, dtype=np.float64).take(-1, axis=axis)


def decode_witness(codes: np.ndarray, witness: dict[str, Any]) -> tuple[int, int, int, int, int]:
    centroids = np.asarray(witness["centroids_code"], dtype=np.int32)
    radius = 3.0 * decimal(witness["sigma_code"], "witness.sigma_code")
    distances = np.abs(codes.astype(np.int32)[:, None] - centroids[None, :])
    accepted = distances <= radius
    counts = accepted.sum(axis=1)
    if np.any(counts > 1):
        raise Reject("CH2_AMBIGUOUS_LEVEL")
    if np.any(counts == 0):
        raise Reject("CH2_WITNESS_MISSING")
    decoded = accepted.argmax(axis=1).astype(np.int8)
    gate_search_start = witness["gate_search_start"]
    gate_search_stop = witness["gate_search_stop"]
    if np.any(decoded[:gate_search_start] != 7):
        raise Reject("CH2_ILLEGAL_STATE")
    tail = decoded[gate_search_start:gate_search_stop]
    not_drive = np.flatnonzero(tail != 7)
    if not len(not_drive):
        raise Reject("SOURCE_OFF_NOT_WITNESSED")
    n_gate = gate_search_start + int(not_drive[0])
    if n_gate + 250 > len(decoded) or np.any(decoded[n_gate : n_gate + 250] != 6):
        raise Reject("GATE_FIRST_SEQUENCE")
    series_open = (decoded[n_gate + 250 :] == 0).astype(np.int32)
    series_runs = np.convolve(series_open, np.ones(1000, dtype=np.int32), mode="valid")
    series_stable = np.flatnonzero(series_runs == 1000)
    if not len(series_stable):
        raise Reject("SERIES_OPEN_SEQUENCE")
    series_start = n_gate + 250 + int(series_stable[0])
    n_series_open = series_start + 999
    if np.any(~np.isin(decoded[n_gate + 250 : series_start], np.asarray([0, 2, 4, 6], dtype=np.int8))):
        raise Reject("SERIES_OPEN_SEQUENCE")
    if np.any(decoded[series_start : n_series_open + 1] != 0):
        raise Reject("SERIES_OPEN_SEQUENCE")
    guard = (decoded[n_series_open + 1 :] == 8).astype(np.int32)
    guard_runs = np.convolve(guard, np.ones(1000, dtype=np.int32), mode="valid")
    stable = np.flatnonzero(guard_runs == 1000)
    if not len(stable):
        raise Reject("CH2_WITNESS_MISSING")
    off_start = n_series_open + 1 + int(stable[0])
    n_contact = off_start + 999
    if np.any(~np.isin(decoded[n_series_open + 1 : off_start], np.asarray([0, 8], dtype=np.int8))):
        raise Reject("GUARD_ORDER")
    if n_contact - n_gate > witness["max_transition_samples"]:
        raise Reject("CONTACT_TIMING")
    n_iso = max(n_gate + 1, n_contact)
    n_admit = n_iso + 10000
    if np.any(decoded[n_contact + 1 :] != 8):
        raise Reject("CH2_POST_OFF_REENTRY")
    return n_gate, n_series_open, n_contact, n_iso, n_admit


def cholesky_solve(g: np.ndarray, h: np.ndarray) -> np.ndarray:
    if not np.all(np.isfinite(g)) or not np.all(np.isfinite(h)):
        raise Reject("NONFINITE_MATH")
    if np.linalg.cond(g, 2) > 1e8:
        raise Reject("WLS_CONDITION")
    try:
        lower = np.linalg.cholesky(g)
        return np.linalg.solve(lower.T, np.linalg.solve(lower, h))
    except np.linalg.LinAlgError as exc:
        raise Reject("WLS_RANK") from exc


def hac_covariance(x: np.ndarray, residual: np.ndarray) -> np.ndarray:
    g = np.array([[float(ascending_sum(x[:, a] * x[:, b])) for b in range(x.shape[1])] for a in range(x.shape[1])], dtype=np.float64)
    inv = np.linalg.inv(g)
    meat = np.zeros_like(g)
    for lag in range(NW_LAG + 1):
        weight = 1.0 if lag == 0 else 1.0 - lag / 8.0
        gamma = np.zeros_like(g)
        for a in range(x.shape[1]):
            for b in range(x.shape[1]):
                gamma[a, b] = float(ascending_sum(residual[lag:] * residual[: len(residual) - lag] * x[lag:, a] * x[: len(x) - lag, b]))
        meat += gamma if lag == 0 else weight * (gamma + gamma.T)
    covariance = inv @ meat @ inv
    eigenvalues = np.linalg.eigvalsh(covariance)
    tolerance = 1e-12 * max(1.0, float(np.max(np.abs(covariance))))
    if float(np.min(eigenvalues)) < -tolerance:
        raise Reject("HAC_COVARIANCE")
    return (covariance + covariance.T) / 2


def tone_fit(ch0: np.ndarray, start: int, stop: int, f_ref: float) -> dict[str, Any]:
    idx = np.arange(start, stop, dtype=np.float64)
    phase = 2 * math.pi * f_ref * idx / FS
    x = np.column_stack((np.cos(phase), -np.sin(phase), np.ones(len(idx))))
    g = np.empty((3, 3), dtype=np.float64)
    h = np.empty(3, dtype=np.float64)
    for row in range(3):
        h[row] = float(ascending_sum(x[:, row] * ch0[start:stop]))
        for col in range(3):
            g[row, col] = float(ascending_sum(x[:, row] * x[:, col]))
    beta = cholesky_solve(g, h)
    residual = ch0[start:stop] - x @ beta
    covariance = hac_covariance(x, residual)
    amplitude2 = float(beta[0] ** 2 + beta[1] ** 2)
    if amplitude2 <= 0:
        raise Reject("ZERO_DRIVE_CONTROL")
    gradient = np.asarray([-beta[1] / amplitude2, beta[0] / amplitude2, 0.0], dtype=np.float64)
    phase_variance = float(gradient @ covariance @ gradient)
    if phase_variance < -1e-18:
        raise Reject("HAC_COVARIANCE")
    return {"amplitude": math.sqrt(amplitude2), "phase_rad": math.atan2(beta[1], beta[0]), "phase_standard_uncertainty_rad": math.sqrt(max(0.0, phase_variance))}


def joint_drive_reference_fit(ch0: np.ndarray, start: int, stop: int, f_ref: float) -> dict[str, float]:
    """Fit C1 and C2 jointly so the gauge-relative phase covariance is complete."""
    idx = np.arange(start, stop, dtype=np.float64)
    phase = 2 * math.pi * f_ref * idx / FS
    x = np.column_stack((np.cos(phase), -np.sin(phase), np.cos(2.0 * phase), -np.sin(2.0 * phase), np.ones(len(idx))))
    g = np.empty((5, 5), dtype=np.float64)
    h = np.empty(5, dtype=np.float64)
    observed = ch0[start:stop]
    for row in range(5):
        h[row] = float(ascending_sum(x[:, row] * observed))
        for col in range(5):
            g[row, col] = float(ascending_sum(x[:, row] * x[:, col]))
    beta = cholesky_solve(g, h)
    residual = observed - x @ beta
    covariance = hac_covariance(x, residual)
    drive_amplitude2 = float(beta[0] ** 2 + beta[1] ** 2)
    reference_amplitude2 = float(beta[2] ** 2 + beta[3] ** 2)
    if drive_amplitude2 <= 0 or reference_amplitude2 <= 0:
        raise Reject("ZERO_DRIVE_CONTROL")
    drive_gradient = np.asarray([-beta[1] / drive_amplitude2, beta[0] / drive_amplitude2, 0.0, 0.0, 0.0], dtype=np.float64)
    reference_gradient = np.asarray([0.0, 0.0, -beta[3] / reference_amplitude2, beta[2] / reference_amplitude2, 0.0], dtype=np.float64)
    error_gradient = drive_gradient - 0.5 * reference_gradient
    drive_variance = float(drive_gradient @ covariance @ drive_gradient)
    reference_variance = float(reference_gradient @ covariance @ reference_gradient)
    drive_reference_covariance = float(drive_gradient @ covariance @ reference_gradient)
    error_variance = float(error_gradient @ covariance @ error_gradient)
    if min(drive_variance, reference_variance, error_variance) < -1e-18:
        raise Reject("HAC_COVARIANCE")
    return {
        "drive_amplitude": math.sqrt(drive_amplitude2),
        "drive_phase_rad": math.atan2(beta[1], beta[0]),
        "drive_phase_standard_uncertainty_rad": math.sqrt(max(0.0, drive_variance)),
        "reference_amplitude": math.sqrt(reference_amplitude2),
        "reference_phase_rad": math.atan2(beta[3], beta[2]),
        "reference_phase_standard_uncertainty_rad": math.sqrt(max(0.0, reference_variance)),
        "drive_reference_phase_covariance_rad2": drive_reference_covariance,
        "gauge_phase_standard_uncertainty_rad": 0.5 * math.sqrt(max(0.0, reference_variance)),
        "error_phase_standard_uncertainty_rad": math.sqrt(max(0.0, error_variance)),
    }


def drive_fit(ch0: np.ndarray, n_gate: int, phase_command: float, f_ref: float, u_skew: float, u_drive_cal: float) -> dict[str, float]:
    preparation_samples = int(round(FS * 32768 / f_ref))
    if n_gate - preparation_samples < 0:
        raise Reject("PREPARATION_WINDOW")
    joint_tones = joint_drive_reference_fit(ch0, n_gate - 100_000, n_gate, f_ref)
    amplitude, phase_drive = joint_tones["drive_amplitude"], joint_tones["drive_phase_rad"]
    reference_amplitude, reference_phase = joint_tones["reference_amplitude"], joint_tones["reference_phase_rad"]
    reference_ratio = reference_amplitude / amplitude
    if not 0.23 <= reference_ratio <= 0.27:
        raise Reject("SOURCE_REFERENCE_MISSING")
    base_gauge = 0.5 * reference_phase
    candidates = (float(wrap(base_gauge)), float(wrap(base_gauge + math.pi)))
    gauge = min(candidates, key=lambda value: abs(float(wrap(phase_drive - value - phase_command))))
    error = float(wrap(phase_drive - gauge - phase_command))
    u95_drive = 1.96 * math.sqrt(joint_tones["error_phase_standard_uncertainty_rad"] ** 2 + u_skew ** 2 + u_drive_cal ** 2)
    if abs(error) + u95_drive > 0.010:
        raise Reject("DRIVE_PHASE_FIDELITY")
    preparation_indices = np.arange(n_gate - preparation_samples, n_gate, dtype=np.float64)
    preparation_predicted = amplitude * np.cos(2 * math.pi * f_ref * preparation_indices / FS + phase_drive) + reference_amplitude * np.cos(2 * math.pi * (2.0 * f_ref) * preparation_indices / FS + reference_phase)
    preparation_residual_ratio_max = float(np.max(np.abs(ch0[n_gate - preparation_samples : n_gate] - preparation_predicted)) / amplitude)
    if preparation_residual_ratio_max > 0.05:
        raise Reject("SOURCE_PREPARATION_NOT_CONTINUOUS")
    post_indices = np.arange(n_gate, SAMPLES, dtype=np.float64)
    predicted = amplitude * np.cos(2 * math.pi * f_ref * post_indices / FS + phase_drive) + reference_amplitude * np.cos(2 * math.pi * (2.0 * f_ref) * post_indices / FS + reference_phase)
    residual_ratio_max = float(np.max(np.abs(ch0[n_gate:] - predicted)) / amplitude)
    if residual_ratio_max > 0.05:
        raise Reject("SOURCE_MONITOR_NOT_CONTINUOUS")
    starts = list(range(n_gate, SAMPLES - 100_000 + 1, 100_000))
    if starts[-1] != SAMPLES - 100_000:
        starts.append(SAMPLES - 100_000)
    persistence: list[float] = []
    reference_persistence: list[float] = []
    phase_errors: list[float] = []
    reference_phase_errors: list[float] = []
    for start in starts:
        try:
            post_drive = tone_fit(ch0, start, start + 100_000, f_ref)
            post_reference = tone_fit(ch0, start, start + 100_000, 2.0 * f_ref)
        except Reject as exc:
            if exc.code == "ZERO_DRIVE_CONTROL":
                raise Reject("SOURCE_MONITOR_NOT_CONTINUOUS") from exc
            raise
        persistence.append(post_drive["amplitude"] / amplitude)
        reference_persistence.append(post_reference["amplitude"] / reference_amplitude)
        phase_errors.append(float(wrap(post_drive["phase_rad"] - gauge - phase_command)))
        reference_phase_errors.append(float(wrap(post_reference["phase_rad"] - reference_phase)))
    if (
        min(persistence) < 0.98 or max(persistence) > 1.02
        or min(reference_persistence) < 0.98 or max(reference_persistence) > 1.02
        or max(abs(value) for value in phase_errors) > 0.010
        or max(abs(value) for value in reference_phase_errors) > 0.010
    ):
        raise Reject("SOURCE_MONITOR_NOT_CONTINUOUS")
    return {"amplitude": amplitude, "phase_rad": phase_drive, "phase_fit_standard_uncertainty_rad": joint_tones["drive_phase_standard_uncertainty_rad"], "reference_phase_fit_standard_uncertainty_rad": joint_tones["reference_phase_standard_uncertainty_rad"], "gauge_phase_fit_standard_uncertainty_rad": joint_tones["gauge_phase_standard_uncertainty_rad"], "drive_reference_phase_covariance_rad2": joint_tones["drive_reference_phase_covariance_rad2"], "error_phase_fit_standard_uncertainty_rad": joint_tones["error_phase_standard_uncertainty_rad"], "phase_skew_standard_uncertainty_rad": u_skew, "phase_drive_cal_standard_uncertainty_rad": u_drive_cal, "u95_drive_rad": u95_drive, "reference_amplitude": reference_amplitude, "reference_amplitude_ratio": reference_ratio, "reference_phase_rad": reference_phase, "gauge_phase_rad": gauge, "error_rad": error, "preparation_first_sample": n_gate - preparation_samples, "preparation_last_sample": n_gate - 1, "preparation_residual_ratio_max": preparation_residual_ratio_max, "continuity_first_sample": n_gate, "continuity_last_sample": SAMPLES - 1, "continuity_residual_ratio_max": residual_ratio_max, "continuity_segment_count": len(starts), "post_gate_amplitude_ratio_min": min(persistence), "post_gate_amplitude_ratio_max": max(persistence), "post_gate_reference_amplitude_ratio_min": min(reference_persistence), "post_gate_reference_amplitude_ratio_max": max(reference_persistence), "post_gate_phase_error_rad_max_abs": max(abs(value) for value in phase_errors), "post_gate_reference_phase_error_rad_max_abs": max(abs(value) for value in reference_phase_errors), "qualified_preparation_samples": preparation_samples}


def project(ch1: np.ndarray, starts: np.ndarray, f_ref: float) -> tuple[np.ndarray, np.ndarray]:
    hann = np.asarray([0.5 - 0.5 * math.cos(2 * math.pi * k / 2047) for k in range(WINDOW)], dtype=np.float64)
    z = np.empty(len(starts), dtype=np.complex128)
    condition = np.empty(len(starts), dtype=np.float64)
    k = np.arange(WINDOW, dtype=np.float64)
    for offset in range(0, len(starts), 128):
        batch = starts[offset : offset + 128]
        indices = batch[:, None].astype(np.float64) + k[None, :]
        phase = 2 * math.pi * f_ref * indices / FS
        x0 = np.cos(phase)
        x1 = -np.sin(phase)
        x2 = np.ones_like(x0)
        y = np.stack([ch1[int(s) : int(s) + WINDOW] for s in batch])
        wx0, wx1, wx2 = x0 * hann, x1 * hann, x2 * hann
        g00 = ascending_sum(x0 * wx0, axis=1)
        g01 = ascending_sum(x0 * wx1, axis=1)
        g02 = ascending_sum(x0 * wx2, axis=1)
        g11 = ascending_sum(x1 * wx1, axis=1)
        g12 = ascending_sum(x1 * wx2, axis=1)
        g22 = ascending_sum(x2 * wx2, axis=1)
        h0 = ascending_sum(wx0 * y, axis=1)
        h1 = ascending_sum(wx1 * y, axis=1)
        h2 = ascending_sum(wx2 * y, axis=1)
        for local in range(len(batch)):
            g = np.array([[g00[local], g01[local], g02[local]], [g01[local], g11[local], g12[local]], [g02[local], g12[local], g22[local]]])
            beta = cholesky_solve(g, np.array([h0[local], h1[local], h2[local]]))
            z[offset + local] = complex(beta[0], beta[1])
            condition[offset + local] = np.linalg.cond(g, 2)
    return z, condition


def median(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    n = len(ordered)
    if not n:
        raise Reject("EMPTY_STATISTIC")
    return float(ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2)


def mad(values: np.ndarray) -> float:
    center = median(values)
    return 1.4826 * median(np.abs(values - center))


def unwrap_exact(values: np.ndarray) -> np.ndarray:
    out = np.empty_like(values, dtype=np.float64)
    out[0] = values[0]
    for index in range(1, len(values)):
        delta = float(wrap(values[index] - values[index - 1]))
        out[index] = out[index - 1] + delta
    return out


def nw_fit(times: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u = times - float(ascending_sum(times) / len(times))
    x = np.column_stack((np.ones(len(times)), u))
    g = np.array([[ascending_sum(x[:, a] * x[:, b]) for b in range(2)] for a in range(2)], dtype=np.float64)
    h = np.array([ascending_sum(x[:, a] * values) for a in range(2)], dtype=np.float64)
    beta = cholesky_solve(g, h)
    residual = values - x @ beta
    cov = hac_covariance(x, residual)
    r2 = 1.0
    centered = values - float(ascending_sum(values) / len(values))
    sst = float(ascending_sum(centered * centered))
    if sst > 0:
        r2 = 1 - float(ascending_sum(residual * residual)) / sst
    return beta, cov, r2


def arm_metrics(z: np.ndarray, starts: np.ndarray, n_gate: int, sigma: float, f_ref: float) -> dict[str, Any]:
    amp = np.abs(z)
    phase = np.angle(z)
    good = amp / sigma >= max(10.0, 1.96 / 0.050)
    first = int(np.argmax(good))
    if not good[first]:
        raise Reject("SNR")
    end = first
    while end < len(good) and good[end]:
        end += 1
    amp = amp[first:end]
    phase = phase[first:end]
    times = (starts[first:end].astype(np.float64) + 1023.5) / FS
    if len(amp) < 9:
        raise Reject("USABLE_REGION")
    unwrapped = unwrap_exact(phase)
    phase_hop = np.diff(unwrapped)
    if np.any(phase_hop <= -math.pi / 2) or np.any(phase_hop >= math.pi / 2):
        raise Reject("UNWRAP_GATE")
    phase_beta, phase_cov, _ = nw_fit(times, unwrapped)
    log_amp = np.log(amp)
    amp_beta, amp_cov, r2 = nw_fit(times, log_amp)
    frequency = f_ref + phase_beta[1] / (2 * math.pi)
    slope = amp_beta[1]
    if frequency <= 0 or slope >= 0:
        raise Reject("FREQUENCY_DECAY_SIGN")
    tau = -1.0 / slope
    q = math.pi * frequency * tau
    u95_f = 1.96 * math.sqrt(max(0.0, phase_cov[1, 1])) / (2 * math.pi)
    u95_tau = 1.96 * math.sqrt(max(0.0, amp_cov[1, 1])) / (slope * slope)
    usable = math.floor(frequency * (times[-1] - times[0]))
    if usable < 256 or r2 < 0.95 or log_amp[0] - log_amp[-1] < 0.25 or u95_f / frequency > 2e-6 or u95_tau / tau > 0.10:
        raise Reject("ARM_QUALITY")
    if np.any(1.96 * sigma / amp > 0.050):
        raise Reject("PHASE_UNCERTAINTY")
    selected_starts = starts[first:end]
    return {"z": z[first:end], "absolute_starts": selected_starts, "relative_starts": selected_starts - n_gate, "amplitude": amp, "phase": phase, "times": times, "frequency_hz": frequency, "tau_s": tau, "q": q, "u95_frequency_hz": u95_f, "u95_tau_s": u95_tau, "r2_log_amplitude": r2, "usable_cycles": usable}


def norm2(values: np.ndarray) -> float:
    return math.sqrt(float(ascending_sum(np.abs(values) ** 2)))


def circular_mean(values: np.ndarray) -> float:
    sin_sum = float(ascending_sum(np.sin(values)))
    cos_sum = float(ascending_sum(np.cos(values)))
    if sin_sum == 0 and cos_sum == 0:
        raise Reject("CIRCULAR_MEAN")
    return math.atan2(sin_sum, cos_sum)


def _quick_frequency_tau(z: np.ndarray, times: np.ndarray, f_ref: float) -> tuple[float, float]:
    phase = unwrap_exact(np.angle(z))
    amplitude = np.abs(z)
    mean_t = float(ascending_sum(times) / len(times))
    u = times - mean_t
    denominator = float(ascending_sum(u * u))
    phase_slope = float(ascending_sum(u * (phase - float(ascending_sum(phase) / len(phase)))) / denominator)
    log_amplitude = np.log(amplitude)
    amplitude_slope = float(ascending_sum(u * (log_amplitude - float(ascending_sum(log_amplitude) / len(log_amplitude)))) / denominator)
    if amplitude_slope >= 0:
        raise Reject("FREQUENCY_DECAY_SIGN")
    return f_ref + phase_slope / (2 * math.pi), -1.0 / amplitude_slope


def _relation_values(z0: np.ndarray, zp: np.ndarray, times: np.ndarray, controls: list[np.ndarray], f_ref: float) -> np.ndarray:
    f0, tau0 = _quick_frequency_tau(z0, times, f_ref)
    fp, taup = _quick_frequency_tau(zp, times, f_ref)
    count = len(z0)
    den_z = 0.5 * (norm2(zp) + norm2(z0))
    den_a = 0.5 * (norm2(np.abs(zp)) + norm2(np.abs(z0)))
    if any(len(value) != count for value in controls):
        raise Reject("CONTROL_ALIGNMENT")
    feed = max(norm2(value) / math.sqrt(count) for value in controls) / min(norm2(z0) / math.sqrt(count), norm2(zp) / math.sqrt(count))
    return np.asarray([
        norm2(zp + z0) / den_z,
        norm2(np.abs(zp) - np.abs(z0)) / den_a,
        abs(fp - f0) / (0.5 * (fp + f0)),
        abs(taup - tau0) / (0.5 * (taup + tau0)),
        circular_mean(wrap(np.angle(zp) - np.angle(z0))),
        feed,
    ], dtype=np.float64)


def relation_metrics(a0: dict[str, Any], api: dict[str, Any], controls: list[tuple[np.ndarray, np.ndarray]], thresholds: dict[str, Any], f_ref: float) -> dict[str, Any]:
    grid = np.intersect1d(a0["relative_starts"], api["relative_starts"], assume_unique=True)
    for _, control_grid in controls:
        grid = np.intersect1d(grid, control_grid, assume_unique=True)
    common_window_count_before_blocking = len(grid)
    blocks = common_window_count_before_blocking // BLOCK
    if blocks < 8 or np.any(np.diff(grid) != HOP):
        raise Reject("CONTROL_ALIGNMENT")
    used_window_count = blocks * BLOCK
    discarded_window_count = common_window_count_before_blocking - used_window_count
    grid = grid[:used_window_count]

    def select(values: np.ndarray, source_grid: np.ndarray) -> np.ndarray:
        indices = np.searchsorted(source_grid, grid)
        if np.any(indices >= len(source_grid)) or np.any(source_grid[indices] != grid):
            raise Reject("CONTROL_ALIGNMENT")
        return values[indices]

    z0 = select(a0["z"], a0["relative_starts"])
    zp = select(api["z"], api["relative_starts"])
    aligned_controls = [select(values, control_grid) for values, control_grid in controls]
    count = len(grid)
    den_z = 0.5 * (norm2(zp) + norm2(z0))
    den_a = 0.5 * (norm2(np.abs(zp)) + norm2(np.abs(z0)))
    times = (grid.astype(np.float64) + 1023.5) / FS
    full = _relation_values(z0, zp, times, aligned_controls, f_ref)
    full_delta = float(full[4])
    phase_residual = float(wrap(full_delta - math.pi))
    metrics = {"neg": float(full[0]), "amplitude": float(full[1]), "frequency": float(full[2]), "decay": float(full[3]), "phase": abs(phase_residual), "feedthrough": float(full[5])}
    deleted = np.empty((blocks, 6), dtype=np.float64)
    for block in range(blocks):
        mask = np.ones(blocks * BLOCK, dtype=bool)
        mask[block * BLOCK : (block + 1) * BLOCK] = False
        deleted[block] = _relation_values(z0[mask], zp[mask], times[mask], [value[mask] for value in aligned_controls], f_ref)
        difference = float(wrap(deleted[block, 4] - full_delta))
        if abs(abs(difference) - math.pi) <= 1e-15:
            raise Reject("CIRCULAR_JACKKNIFE_TIE")
        deleted[block, 4] = full_delta + difference - math.pi
    mean_deleted = np.mean(deleted, axis=0)
    se = np.sqrt((blocks - 1) / blocks * np.sum((deleted - mean_deleted) ** 2, axis=0))
    keys = ("neg", "amplitude", "frequency", "decay", "phase", "feedthrough")
    calibration_u95 = thresholds["calibration_u95"]
    u95 = {key: math.sqrt((1.96 * float(se[index])) ** 2 + decimal(calibration_u95[key], f"thresholds.calibration_u95.{key}") ** 2) for index, key in enumerate(keys)}
    for key, value in metrics.items():
        if value + u95[key] > decimal(thresholds[key], f"thresholds.{key}"):
            raise Reject({"neg": "COMPLEX_NEGATION", "amplitude": "AMPLITUDE_MISMATCH", "frequency": "FREQUENCY_MISMATCH", "decay": "DECAY_MISMATCH", "phase": "HALF_TURN_PHASE_MISMATCH", "feedthrough": "SOURCE_FEEDTHROUGH"}[key])
    delta = full_delta
    residual = phase_residual
    phase_u = u95["phase"]
    confidence = {"antipode_in_set": abs(residual) <= phase_u, "zero_excluded": abs(float(wrap(delta))) > phase_u, "plus_pi_over_2_excluded": abs(float(wrap(delta - math.pi / 2))) > phase_u, "minus_pi_over_2_excluded": abs(float(wrap(delta + math.pi / 2))) > phase_u, "residual_zero_in_set": abs(residual) <= phase_u}
    if not all(confidence.values()):
        raise Reject("PHASE_CONFIDENCE_SET")
    return {"epsilon": metrics, "u95": u95, "jackknife_blocks": blocks, "common_window_count_before_blocking": common_window_count_before_blocking, "common_window_count": count, "discarded_incomplete_block_windows": discarded_window_count, "common_relative_start_first": int(grid[0]), "common_relative_start_last": int(grid[-1]), "confidence_set": confidence}


def read_record(path: Path, scales: list[float]) -> tuple[np.memmap, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = np.memmap(path, dtype="<i2", mode="r", shape=(SAMPLES, CHANNELS))
    try:
        if np.any(raw == -32768) or np.any(raw == 32767):
            raise Reject("ADC_SATURATION")
        ch0 = raw[:, 0].astype(np.float64) * scales[0]
        ch1 = raw[:, 1].astype(np.float64) * scales[1]
        ch2 = np.asarray(raw[:, 2], dtype=np.int16)
        ch3 = raw[:, 3].astype(np.float64) * scales[3]
        return raw, ch0, ch1, ch2, ch3
    except Exception:
        if raw._mmap is not None:
            raw._mmap.close()
        raise


def interval_environment(meta: dict[str, Any], ch3: np.ndarray, n_admit: int, last_center: int) -> dict[str, float]:
    selected = [record for record in meta["environment"]["_records"] if n_admit <= record[0] <= last_center]
    if not selected:
        raise Reject("ENVIRONMENT_CADENCE")
    temperatures = [record[1] for record in selected]
    humidities = [record[2] for record in selected]
    segment = ch3[n_admit : last_center + 1]
    mean = float(ascending_sum(segment) / len(segment))
    centered = segment - mean
    rms = math.sqrt(float(ascending_sum(centered * centered) / len(centered)))
    peak = float(np.max(np.abs(centered)))
    if rms > 0.050 or peak > 0.500:
        raise Reject("VIBRATION_MISMATCH")
    return {"temperature_c": sum(temperatures) / len(temperatures), "humidity_rh": sum(humidities) / len(humidities), "vibration_rms_m_s2": rms, "vibration_peak_m_s2": peak, "record_count": len(selected)}


def analyze_bundle(bundle_path: Path, identity_directory: Path | None = None) -> dict[str, Any]:
    bundle = load_json_strict(bundle_path)
    exact_fields(bundle, {"schema", "roles"}, "bundle")
    if bundle["schema"] != "p0.raw-bundle-manifest.v1" or not isinstance(bundle["roles"], dict) or set(bundle["roles"]) != set(ROLE_ORDER):
        raise Reject("ROLE_SUBSTITUTION")
    base = bundle_path.parent.resolve()
    records: dict[str, dict[str, Any]] = {}
    projected: dict[str, np.ndarray] = {}
    metadata_paths: set[Path] = set()
    payload_paths: set[Path] = set()
    payload_hashes: set[str] = set()
    for role in ROLE_ORDER:
        entry = bundle["roles"][role]
        if not isinstance(entry, dict):
            raise Reject("ROLE_SUBSTITUTION", role)
        exact_fields(entry, {"path", "sha256"}, f"roles.{role}")
        lower_hash(entry["sha256"], f"roles.{role}.sha256")
        meta_path = safe_relative(base, entry["path"], f"roles.{role}.path")
        if meta_path in metadata_paths or not meta_path.is_file() or sha256_file(meta_path) != entry["sha256"]:
            raise Reject("HASH_MISMATCH" if meta_path.is_file() else "ROLE_SUBSTITUTION", role)
        metadata_paths.add(meta_path)
        meta = load_json_strict(meta_path)
        payload_path, meta = validate_metadata(meta, meta_path.parent)
        if meta["role"] != role:
            raise Reject("ROLE_SUBSTITUTION", role)
        if payload_path in payload_paths or meta["payload"]["sha256"] in payload_hashes:
            raise Reject("PAYLOAD_ALIAS", role)
        payload_paths.add(payload_path)
        payload_hashes.add(meta["payload"]["sha256"])
        records[role] = {"meta": meta, "metadata_path": meta_path, "metadata_sha256": entry["sha256"], "payload_path": payload_path}

    first = records[ROLE_ORDER[0]]["meta"]
    common = {
        "calibration": first["custody"]["calibration_sha256"],
        "assignment": first["custody"]["assignment_commitment_sha256"],
        "evidence_class": first["evidence_class"],
        "f_ref": first["_f_ref"],
        "scales": first["payload"]["_scales"],
        "thresholds": canonical_bytes(first["thresholds"]),
        "instrument": canonical_bytes(first["instrument"]),
        "export_adapter": canonical_bytes({key: first["export"][key] for key in ("adapter_id", "adapter_sha256", "lossless_assertions")}),
        "source_setup": canonical_bytes({key: first["source"][key] for key in ("model", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "frequency_hz", "amplitude_vpp", "offset_v", "reference_frequency_hz", "reference_amplitude_vpp", "reference_offset_v", "reference_phase_command_rad", "dual_channel_phase_locked", "monitor_network", "output_mode", "load_mode", "qualified_preparation_cycles", "source_remains_on_through_record", "output_ohms", "setup_queryback_sha256")}),
        "clock_identity": first["clock"]["identity"],
    }
    for role in ROLE_ORDER:
        meta = records[role]["meta"]
        if meta["custody"]["calibration_sha256"] != common["calibration"]:
            raise Reject("CALIBRATION_MISMATCH")
        if meta["custody"]["assignment_commitment_sha256"] != common["assignment"]:
            raise Reject("ASSIGNMENT_MISMATCH")
        if meta["evidence_class"] != common["evidence_class"]:
            raise Reject("EVIDENCE_CLASS_MISMATCH")
        if meta["_f_ref"] != common["f_ref"]:
            raise Reject("TIMEBASE_MISMATCH")
        if meta["payload"]["_scales"] != common["scales"]:
            raise Reject("SCALE_MISMATCH")
        if canonical_bytes(meta["thresholds"]) != common["thresholds"]:
            raise Reject("THRESHOLD_CUSTODY")
        if canonical_bytes(meta["instrument"]) != common["instrument"]:
            raise Reject("INSTRUMENT_CONFIGURATION_MISMATCH")
        if canonical_bytes({key: meta["export"][key] for key in ("adapter_id", "adapter_sha256", "lossless_assertions")}) != common["export_adapter"]:
            raise Reject("EXPORT_ADAPTER_MISMATCH")
        if canonical_bytes({key: meta["source"][key] for key in ("model", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "frequency_hz", "amplitude_vpp", "offset_v", "reference_frequency_hz", "reference_amplitude_vpp", "reference_offset_v", "reference_phase_command_rad", "dual_channel_phase_locked", "monitor_network", "output_mode", "load_mode", "qualified_preparation_cycles", "source_remains_on_through_record", "output_ohms", "setup_queryback_sha256")}) != common["source_setup"]:
            raise Reject("SOURCE_CONFIGURATION_MISMATCH")
        if meta["clock"]["identity"] != common["clock_identity"]:
            raise Reject("TIMEBASE_MISMATCH")
        expected_phase = math.pi if role == "arm_pi" else 0.0
        if decimal(meta["source"]["phase_command_rad"], "source.phase_command_rad") != expected_phase:
            raise Reject("SOURCE_PHASE")

    for role in ROLE_ORDER:
        payload_path = records[role]["payload_path"]
        raw = np.memmap(payload_path, dtype="<i2", mode="r", shape=(SAMPLES, CHANNELS))
        try:
            ch2 = np.asarray(raw[:, 2], dtype=np.int16)
            n_gate, n_series_open, n_contact, n_iso, n_admit = decode_witness(ch2, records[role]["meta"]["witness"])
            records[role].update({"n_gate": n_gate, "n_series_open": n_series_open, "n_contact": n_contact, "n_iso": n_iso, "n_admit": n_admit})
        finally:
            del ch2
            if raw._mmap is not None:
                raw._mmap.close()
            del raw

    matched_guard = max(records[role]["n_admit"] - records[role]["n_gate"] for role in ROLE_ORDER)
    arm_ch3: dict[str, np.ndarray] = {}
    for role in ROLE_ORDER:
        meta = records[role]["meta"]
        raw, ch0, ch1, ch2, ch3 = read_record(records[role]["payload_path"], meta["payload"]["_scales"])
        try:
            n_gate = records[role]["n_gate"]
            drive = drive_fit(
                ch0,
                n_gate,
                decimal(meta["source"]["phase_command_rad"], "phase"),
                common["f_ref"],
                nonnegative_decimal(meta["source"]["phase_skew_standard_uncertainty_rad"], "source.phase_skew_standard_uncertainty_rad"),
                nonnegative_decimal(meta["source"]["phase_drive_cal_standard_uncertainty_rad"], "source.phase_drive_cal_standard_uncertainty_rad"),
            )
            starts = np.arange(n_gate + matched_guard, SAMPLES - WINDOW + 1, HOP, dtype=np.int64)
            if role in ("arm_0", "arm_pi"):
                arm_ch3[role] = ch3
            z, condition = project(ch1, starts, common["f_ref"])
            z = z * np.exp(-1j * drive["gauge_phase_rad"])
            projected[role] = z
            records[role].update({"drive": drive, "starts": starts, "condition_max": float(np.max(condition))})
        finally:
            del ch0, ch1, ch2, ch3
            if raw._mmap is not None:
                raw._mmap.close()
            del raw
    drive_difference = float(wrap(records["arm_pi"]["drive"]["error_rad"] - records["arm_0"]["drive"]["error_rad"]))
    drive_u95_sum = records["arm_pi"]["drive"]["u95_drive_rad"] + records["arm_0"]["drive"]["u95_drive_rad"]
    drive_pair_lhs = abs(drive_difference) + drive_u95_sum
    if drive_pair_lhs > 0.010:
        raise Reject("DRIVE_PHASE_MATCH")
    nonoverlapping_controls = {role: projected[role][::BLOCK] for role in ROLE_ORDER[2:]}
    noise_i = [mad(nonoverlapping_controls[role].real) for role in ROLE_ORDER[2:]]
    noise_q = [mad(nonoverlapping_controls[role].imag) for role in ROLE_ORDER[2:]]
    if max(noise_i) == 0 and max(noise_q) == 0:
        raise Reject("ZERO_MAD")
    quant = decimal(records["arm_0"]["meta"]["payload"]["scale_per_code"][1], "scale") / math.sqrt(12)
    sigma = max(noise_i + [quant]) + max(noise_q + [quant])
    a0 = arm_metrics(projected["arm_0"], records["arm_0"]["starts"], records["arm_0"]["n_gate"], sigma, common["f_ref"])
    api = arm_metrics(projected["arm_pi"], records["arm_pi"]["starts"], records["arm_pi"]["n_gate"], sigma, common["f_ref"])
    thresholds = records["arm_0"]["meta"]["thresholds"]
    metrics = relation_metrics(a0, api, [(projected[role], records[role]["starts"] - records[role]["n_gate"]) for role in ROLE_ORDER[2:]], thresholds, common["f_ref"])
    environments: dict[str, dict[str, float]] = {}
    for role, arm in (("arm_0", a0), ("arm_pi", api)):
        last_center = int(math.floor(arm["times"][-1] * FS))
        environments[role] = interval_environment(records[role]["meta"], arm_ch3[role], records[role]["n_admit"], last_center)
    if abs(environments["arm_0"]["temperature_c"] - environments["arm_pi"]["temperature_c"]) > 0.20:
        raise Reject("TEMPERATURE_MISMATCH")
    if abs(environments["arm_0"]["humidity_rh"] - environments["arm_pi"]["humidity_rh"]) > 2.0:
        raise Reject("HUMIDITY_MISMATCH")
    if abs(environments["arm_0"]["vibration_rms_m_s2"] - environments["arm_pi"]["vibration_rms_m_s2"]) > 0.010:
        raise Reject("VIBRATION_MISMATCH")
    identity_root = (identity_directory or Path(__file__).resolve().parent).resolve()
    schema_path = identity_root / "P0_BUILD_READINESS_SCHEMAS.json"
    fixture_path = identity_root / "P0_SCIENTIFIC_FIXTURES.json"
    if not schema_path.is_file() or not fixture_path.is_file():
        raise Reject("IDENTITY_MISSING")
    dependency = dependency_identity()
    result = {
        "schema": RESULT_SCHEMA,
        "scientific_pass": True,
        "physical_claim_authorized": False,
        "claim_token": "SYNTHETIC_ANALYZER_PASS" if common["evidence_class"] == "SYNTHETIC" else "NO_CLAIM__EXECUTION_AUTHORITY_REQUIRED",
        "claim_ceiling": "NON_EXECUTING_P0_BUILD_READINESS_ONLY",
        "input_custody": {"bundle_sha256": sha256_file(bundle_path), "metadata_sha256": {role: records[role]["metadata_sha256"] for role in ROLE_ORDER}, "payload_sha256": {role: records[role]["meta"]["payload"]["sha256"] for role in ROLE_ORDER}, "assignment_commitment_sha256": common["assignment"], "calibration_sha256": common["calibration"], "thresholds_sha256": records["arm_0"]["meta"]["thresholds"]["sha256"]},
        "implementation_custody": {"analyzer_sha256": sha256_file(Path(__file__).resolve()), "schema_sha256": sha256_file(schema_path), "fixture_sha256": sha256_file(fixture_path), "dependency_identity": dependency, "dependency_sha256": sha256_bytes(canonical_bytes(dependency))},
        "f_ref_hz": common["f_ref"],
        "matched_guard_samples": matched_guard,
        "timing": {role: {key: records[role][key] for key in ("n_gate", "n_series_open", "n_contact", "n_iso", "n_admit")} for role in ROLE_ORDER},
        "environment": environments,
        "noise_sigma": sigma,
        "noise_nonoverlapping_window_counts": {role: len(nonoverlapping_controls[role]) for role in ROLE_ORDER[2:]},
        "drive_phase": {
            "arm_0": {key: records["arm_0"]["drive"][key] for key in ("error_rad", "phase_fit_standard_uncertainty_rad", "reference_phase_fit_standard_uncertainty_rad", "gauge_phase_fit_standard_uncertainty_rad", "drive_reference_phase_covariance_rad2", "error_phase_fit_standard_uncertainty_rad", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "u95_drive_rad", "preparation_first_sample", "preparation_last_sample", "preparation_residual_ratio_max")},
            "arm_pi": {key: records["arm_pi"]["drive"][key] for key in ("error_rad", "phase_fit_standard_uncertainty_rad", "reference_phase_fit_standard_uncertainty_rad", "gauge_phase_fit_standard_uncertainty_rad", "drive_reference_phase_covariance_rad2", "error_phase_fit_standard_uncertainty_rad", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "u95_drive_rad", "preparation_first_sample", "preparation_last_sample", "preparation_residual_ratio_max")},
            "matched_error_difference_rad": drive_difference,
            "matched_u95_sum_rad": drive_u95_sum,
            "matched_acceptance_lhs_rad": drive_pair_lhs,
        },
        "arms": {"arm_0": {key: a0[key] for key in ("frequency_hz", "tau_s", "q", "u95_frequency_hz", "u95_tau_s", "r2_log_amplitude", "usable_cycles")}, "arm_pi": {key: api[key] for key in ("frequency_hz", "tau_s", "q", "u95_frequency_hz", "u95_tau_s", "r2_log_amplitude", "usable_cycles")}},
        "relation_metrics": metrics,
    }
    return result


def synthetic_record(role: str) -> np.ndarray:
    n = np.arange(SAMPLES, dtype=np.float64)
    phase_offset = math.pi if role == "arm_pi" else 0.0
    phase = 2 * math.pi * F_REF * n / FS + phase_offset
    raw = np.zeros((SAMPLES, CHANNELS), dtype="<i2")
    reference_phase = 2 * math.pi * (2.0 * F_REF) * n / FS
    raw[:, 0] = np.rint(8000 * np.cos(phase) + 2000 * np.cos(reference_phase)).astype(np.int16)
    noise = 3.0 * np.sin(2 * math.pi * 7919 * n / FS) + 2.0 * np.sin(2 * math.pi * 12347 * n / FS)
    if role in ("arm_0", "arm_pi"):
        elapsed = np.maximum(0.0, (n - N_CMD) / FS)
        envelope = np.where(n < N_CMD, 9000.0, 9000.0 * np.exp(-elapsed / 0.18))
        raw[:, 1] = np.rint(envelope * np.cos(phase) + noise).astype(np.int16)
    else:
        raw[:, 1] = np.rint(noise).astype(np.int16)
    witness = np.full(SAMPLES, 1000 + 7000, dtype=np.int16)
    witness[N_CMD : N_CMD + 250] = 1000 + 6000
    witness[N_CMD + 250 : N_CMD + 1500] = 1000
    witness[N_CMD + 1500 :] = 1000 + 8000
    raw[:, 2] = witness
    role_phase = ROLE_ORDER.index(role) * 0.173
    raw[:, 3] = np.rint(50 * np.sin(2 * math.pi * 37 * n / FS + role_phase)).astype(np.int16)
    return raw


def canonical_measurement_decimal(value: float) -> str:
    text = f"{value:.15f}".rstrip("0").rstrip(".")
    return text if text else "0"


def synthetic_environment_values(temperature: str = "25", humidity: str = "40") -> tuple[int, int, str, str]:
    temperature_value = environment_decimal(temperature, "synthetic.temperature")
    humidity_value = environment_decimal(humidity, "synthetic.humidity")
    temperature_ticks = max(0, min(65535, round((temperature_value + 45.0) * 65535.0 / 175.0)))
    humidity_ticks = max(0, min(65535, round((humidity_value + 6.0) * 65535.0 / 125.0)))
    return (
        temperature_ticks,
        humidity_ticks,
        canonical_measurement_decimal(sht4x_temperature(temperature_ticks)),
        canonical_measurement_decimal(sht4x_humidity(humidity_ticks)),
    )


def synthetic_environment_bytes(temperature: str = "25", humidity: str = "40") -> bytes:
    temperature_ticks, humidity_ticks, temperature_text, humidity_text = synthetic_environment_values(temperature, humidity)
    temperature_word = temperature_ticks.to_bytes(2, "big")
    humidity_word = humidity_ticks.to_bytes(2, "big")
    lines = [ENVIRONMENT_HEADER]
    for index in range(0, SAMPLES, 100_000):
        seconds, remainder = divmod(index, FS)
        lines.append(
            f"{index},{index * 1000},2037-07-16T00:00:{seconds:02d}.{remainder:06d}Z,"
            f"00000001,fd,{temperature_ticks:04x},{sht4x_crc8(temperature_word):02x},"
            f"{humidity_ticks:04x},{sht4x_crc8(humidity_word):02x},{temperature_text},{humidity_text}"
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def base_metadata(role: str, payload_name: str, payload_hash: str, environment_name: str, environment_hash: str, environment_bytes: int) -> dict[str, Any]:
    phase = math.pi if role == "arm_pi" else 0.0
    frozen = {"neg": "0.020", "amplitude": "0.020", "frequency": "0.000001", "decay": "0.020", "phase": "0.020", "feedthrough": "0.020", "calibration_u95": {"neg": "0.001", "amplitude": "0.001", "frequency": "0.0000001", "decay": "0.001", "phase": "0.001", "feedthrough": "0.001"}}
    frozen_hash = sha256_bytes(canonical_bytes(frozen))
    witness_body = {"centroids_code": [1000 + 1000 * i for i in range(16)], "sigma_code": "10", "gate_search_start": 1_000_000, "gate_search_stop": 1_200_000, "stable_off_samples": 1000, "guard_samples": 10000, "max_transition_samples": 14500}
    witness_hash = sha256_bytes(canonical_bytes(witness_body))
    _, _, temperature_text, humidity_text = synthetic_environment_values()
    clock_mapping_hash = sha256_bytes(canonical_bytes({"alignment": "nearest_raw_sample_index", "monotonic_epoch_ns": 0, "raw_epoch_sample": 0, "sample_rate_hz": FS}))
    calibration_hash = sha256_bytes(canonical_bytes({"clock_identity": "SYNTHETIC-1MHZ-MASTER", "environment_cadence_hz": "10", "environment_sensor_serial_hex": "00000001", "environment_measurement_command_hex": "fd", "clock_mapping_sha256": clock_mapping_hash, "witness_calibration_sha256": witness_hash}))
    return {
        "schema": SCHEMA,
        "run_id": f"SYNTH-{role.upper()}",
        "evidence_class": "SYNTHETIC",
        "role": role,
        "instrument": {"manufacturer": "SYNTHETIC", "model": "P0-CANONICAL-RAW-GENERATOR", "serial": "NONE", "firmware": "NONE", "driver": "numpy-1.26.4", "configuration_queryback_sha256": "0" * 64},
        "export": {"adapter_id": "P0-DN2-SDK-EXPORT-ADAPTER-V1", "adapter_sha256": "1" * 64, "native_file_sha256": payload_hash, "native_file_bytes": PAYLOAD_BYTES, "lossless_assertions": {"sample_loss": False, "reordering": False, "averaging": False, "filtering": False, "resampling": False, "clipping_concealment": False, "unit_ambiguity": False}},
        "payload": {"path": payload_name, "sha256": payload_hash, "bytes": PAYLOAD_BYTES, "dtype": "int16", "endian": "little", "layout": "sample-major-interleaved", "channels": ["CH0_SOURCE", "CH1_CARRIER", "CH2_WITNESS", "CH3_VIBRATION"], "samples_per_channel": SAMPLES, "sample_rate_hz": FS, "scale_per_code": ["0.00001", "0.00001", "0.0001", "0.0001"], "offset": ["0", "0", "0", "0"]},
        "clock": {"identity": "SYNTHETIC-1MHZ-MASTER", "frequency_hz": "32768", "sample_rate_hz": "1000000", "channel_skew_seconds": "0", "record_start_mode": "SOFTWARE_PREARM_FREE_RUN", "external_trigger_connected": False, "phase_gauge": "CH0_SOURCE", "alignment": "CH2_WITNESS"},
        "source": {"model": "SIGLENT SDG1032X", "phase_command_rad": format(phase, ".17g"), "phase_skew_standard_uncertainty_rad": "0.0001", "phase_drive_cal_standard_uncertainty_rad": "0.0001", "frequency_hz": "32768", "amplitude_vpp": "0.4", "offset_v": "0", "reference_frequency_hz": "65536", "reference_amplitude_vpp": "0.1", "reference_offset_v": "0", "reference_phase_command_rad": "0", "dual_channel_phase_locked": True, "monitor_network": "PASSIVE_100K_PLUS_100K_DUAL_TONE_SUM", "output_mode": "CONTINUOUS_SINE", "load_mode": "HIGH_Z", "qualified_preparation_cycles": 32768, "source_remains_on_through_record": True, "output_ohms": "50", "setup_queryback_sha256": "2" * 64},
        "witness": {**witness_body, "calibration_sha256": witness_hash},
        "environment": {"cadence_hz": "10", "temperature_c": temperature_text, "humidity_rh": humidity_text, "vibration_rms_m_s2": "0.003535533905932738", "vibration_peak_m_s2": "0.005", "sensor_serial_hex": "00000001", "measurement_command_hex": "fd", "clock_mapping_sha256": clock_mapping_hash, "calibration_sha256": calibration_hash, "record_path": environment_name, "record_sha256": environment_hash, "record_bytes": environment_bytes, "record_count": len(range(0, SAMPLES, 100_000))},
        "custody": {"calibration_sha256": calibration_hash, "assignment_commitment_sha256": "5" * 64, "assignment_revealed": True, "thresholds_frozen_before_primary": True, "primary_observed": True},
        "thresholds": {**frozen, "sha256": frozen_hash},
    }


def materialize_synthetic(directory: Path) -> Path:
    roles: dict[str, dict[str, str]] = {}
    for role in ROLE_ORDER:
        payload_name = f"{role}.raw"
        payload_path = directory / payload_name
        synthetic_record(role).tofile(payload_path)
        environment_name = f"{role}.environment.csv"
        environment_path = directory / environment_name
        environment_path.write_bytes(synthetic_environment_bytes())
        meta = base_metadata(role, payload_name, sha256_file(payload_path), environment_name, sha256_file(environment_path), environment_path.stat().st_size)
        meta_name = f"{role}.json"
        meta_path = directory / meta_name
        meta_path.write_bytes(canonical_bytes(meta))
        roles[role] = {"path": meta_name, "sha256": sha256_file(meta_path)}
    bundle = {"schema": "p0.raw-bundle-manifest.v1", "roles": roles}
    bundle_path = directory / "bundle.json"
    bundle_path.write_bytes(canonical_bytes(bundle))
    return bundle_path


RAW_ADVERSARIES = {
    "negative_scale_transform": "SCALE_MISMATCH",
    "payload_reuse": "PAYLOAD_ALIAS",
    "mixed_evidence_class": "EVIDENCE_CLASS_MISMATCH",
    "assignment_not_revealed": "ASSIGNMENT_NOT_REVEALED",
    "precommand_not_drive": "CH2_ILLEGAL_STATE",
    "gate_first_sequence_skipped": "GATE_FIRST_SEQUENCE",
    "source_left_on_raw": "SOURCE_OFF_NOT_WITNESSED",
    "source_muted_preparation_middle_raw": "SOURCE_PREPARATION_NOT_CONTINUOUS",
    "source_muted_at_gate_raw": "SOURCE_MONITOR_NOT_CONTINUOUS",
    "source_muted_gate_guard_only_raw": "SOURCE_MONITOR_NOT_CONTINUOUS",
    "source_muted_late_1300000_raw": "SOURCE_MONITOR_NOT_CONTINUOUS",
    "source_muted_late_2000000_raw": "SOURCE_MONITOR_NOT_CONTINUOUS",
    "source_muted_late_3000000_raw": "SOURCE_MONITOR_NOT_CONTINUOUS",
    "reference_tone_missing_raw": "SOURCE_REFERENCE_MISSING",
    "guard_before_series_open_raw": "SERIES_OPEN_SEQUENCE",
    "matched_drive_phase_mismatch_raw": "DRIVE_PHASE_MATCH",
    "matched_temperature_delta_raw": "TEMPERATURE_MISMATCH",
    "noncanonical_metadata_raw": "NONCANONICAL_JSON",
    "threshold_hash_mutation_raw": "HASH_MISMATCH",
    "instrument_configuration_drift": "INSTRUMENT_CONFIGURATION_MISMATCH",
    "export_adapter_drift": "EXPORT_ADAPTER_MISMATCH",
    "source_configuration_drift": "SOURCE_CONFIGURATION_MISMATCH",
    "source_amplitude_out_of_contract": "SOURCE_SETUP",
    "environment_leading_zero_index": "ENVIRONMENT_FORMAT",
    "environment_plus_decimal": "NUMBER_ENCODING",
    "environment_exponent_decimal": "NUMBER_ENCODING",
    "environment_repeated_timestamp": "ENVIRONMENT_CADENCE",
    "environment_crc_mutation": "ENVIRONMENT_CRC",
    "environment_monotonic_drift": "ENVIRONMENT_CADENCE",
    "environment_sensor_identity_drift": "ENVIRONMENT_SENSOR_IDENTITY",
    "negative_zero_metadata": "NUMBER_ENCODING",
}


def plain_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_pairs, parse_constant=lambda value: (_ for _ in ()).throw(Reject("NONFINITE_JSON", value)))


def write_mutated_metadata(directory: Path, bundle: dict[str, Any], role: str, meta: dict[str, Any] | None = None, raw_bytes: bytes | None = None) -> None:
    meta_path = directory / bundle["roles"][role]["path"]
    meta_path.write_bytes(canonical_bytes(meta) if raw_bytes is None else raw_bytes)
    bundle["roles"][role]["sha256"] = sha256_file(meta_path)
    (directory / "bundle.json").write_bytes(canonical_bytes(bundle))


def execute_bundle_adversary(directory: Path, case: str, identity_directory: Path, expected: str) -> dict[str, Any]:
    bundle_path = directory / "bundle.json"
    tracked = [bundle_path, *[directory / f"{role}.json" for role in ROLE_ORDER], *[directory / f"{role}.environment.csv" for role in ROLE_ORDER]]
    originals = {path: path.read_bytes() for path in tracked}
    extra = directory / "adversary.raw"
    try:
        bundle = plain_json(bundle_path)
        role = "arm_0"
        meta_path = directory / bundle["roles"][role]["path"]
        meta = plain_json(meta_path)
        if case in ("truncated_binary", "extra_samples"):
            meta["payload"]["bytes"] = PAYLOAD_BYTES + (-2 if case == "truncated_binary" else 2)
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "wrong_channel_count":
            meta["payload"]["channels"] = meta["payload"]["channels"][:3]
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "wrong_dtype":
            meta["payload"]["dtype"] = "float32"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "wrong_endian":
            meta["payload"]["endian"] = "big"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "nonfinite_scale_metadata":
            data = canonical_bytes(meta).replace(b'"0.00001"', b'NaN', 1)
            write_mutated_metadata(directory, bundle, role, raw_bytes=data)
        elif case == "missing_calibration":
            del meta["custody"]["calibration_sha256"]
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "malformed_json":
            write_mutated_metadata(directory, bundle, role, raw_bytes=b"{\n")
        elif case == "duplicate_json_key":
            data = canonical_bytes(meta)
            write_mutated_metadata(directory, bundle, role, raw_bytes=b'{\n  "role": "arm_0",' + data[1:])
        elif case == "unknown_field":
            meta["unexpected"] = "forbidden"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "path_traversal":
            bundle["roles"][role]["path"] = "../escape.json"
            bundle_path.write_bytes(canonical_bytes(bundle))
        elif case == "hash_mutation":
            bundle["roles"][role]["sha256"] = "0" * 64
            bundle_path.write_bytes(canonical_bytes(bundle))
        elif case == "manifest_role_substitution":
            bundle["roles"]["arm_x"] = bundle["roles"].pop("arm_0")
            bundle_path.write_bytes(canonical_bytes(bundle))
        elif case == "assignment_reveal_before_custody_closure":
            meta["custody"]["assignment_revealed"] = True
            meta["custody"]["primary_observed"] = False
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "threshold_mutation_after_primary_data":
            meta["custody"]["thresholds_frozen_before_primary"] = False
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "negative_scale_transform":
            meta["payload"]["scale_per_code"][0] = "-0.00001"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "payload_reuse":
            pi_role = "arm_pi"
            pi_meta = plain_json(directory / bundle["roles"][pi_role]["path"])
            pi_meta["payload"]["path"] = meta["payload"]["path"]
            pi_meta["payload"]["sha256"] = meta["payload"]["sha256"]
            pi_meta["export"]["native_file_sha256"] = meta["payload"]["sha256"]
            write_mutated_metadata(directory, bundle, pi_role, pi_meta)
        elif case == "mixed_evidence_class":
            meta["evidence_class"] = "PHYSICAL"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "assignment_not_revealed":
            meta["custody"]["assignment_revealed"] = False
            write_mutated_metadata(directory, bundle, role, meta)
        elif case in ("precommand_not_drive", "gate_first_sequence_skipped", "source_left_on_raw", "source_muted_preparation_middle_raw", "source_muted_at_gate_raw", "source_muted_gate_guard_only_raw", "source_muted_late_1300000_raw", "source_muted_late_2000000_raw", "source_muted_late_3000000_raw", "reference_tone_missing_raw", "guard_before_series_open_raw", "matched_drive_phase_mismatch_raw"):
            if case == "matched_drive_phase_mismatch_raw":
                role = "arm_pi"
                meta_path = directory / bundle["roles"][role]["path"]
                meta = plain_json(meta_path)
            raw = synthetic_record(role)
            if case == "precommand_not_drive":
                raw[:N_CMD, 2] = 1000
            elif case == "gate_first_sequence_skipped":
                raw[N_CMD : N_CMD + 250, 2] = 9000
            elif case == "source_left_on_raw":
                raw[N_CMD:, 2] = 8000
            elif case == "source_muted_preparation_middle_raw":
                raw[N_CMD - 800_000 : N_CMD - 790_000, 0] = 0
            elif case == "source_muted_at_gate_raw":
                raw[N_CMD:, 0] = 0
            elif case == "source_muted_gate_guard_only_raw":
                raw[N_CMD : N_CMD + 10_000, 0] = 0
            elif case.startswith("source_muted_late_"):
                raw[int(case.split("_")[3]) :, 0] = 0
            elif case == "guard_before_series_open_raw":
                raw[N_CMD + 300 : N_CMD + 310, 2] = 9000
            elif case == "matched_drive_phase_mismatch_raw":
                n = np.arange(SAMPLES, dtype=np.float64)
                raw[:, 0] = np.rint(8000 * np.cos(2 * math.pi * F_REF * n / FS + math.pi + 0.0095) + 2000 * np.cos(2 * math.pi * (2.0 * F_REF) * n / FS)).astype(np.int16)
            else:
                n = np.arange(SAMPLES, dtype=np.float64)
                phase_offset = math.pi if role == "arm_pi" else 0.0
                raw[:, 0] = np.rint(8000 * np.cos(2 * math.pi * F_REF * n / FS + phase_offset)).astype(np.int16)
            raw.tofile(extra)
            payload_hash = sha256_file(extra)
            meta["payload"].update({"path": extra.name, "sha256": payload_hash})
            meta["export"].update({"native_file_sha256": payload_hash})
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "matched_temperature_delta_raw":
            role = "arm_pi"
            meta = plain_json(directory / bundle["roles"][role]["path"])
            env_path = directory / meta["environment"]["record_path"]
            env_path.write_bytes(synthetic_environment_bytes("30", "40"))
            _, _, temperature_text, humidity_text = synthetic_environment_values("30", "40")
            meta["environment"].update({"temperature_c": temperature_text, "humidity_rh": humidity_text, "record_sha256": sha256_file(env_path), "record_bytes": env_path.stat().st_size})
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "noncanonical_metadata_raw":
            write_mutated_metadata(directory, bundle, role, raw_bytes=canonical_bytes(meta).replace(b"\n", b"\r\n"))
        elif case == "threshold_hash_mutation_raw":
            meta["thresholds"]["neg"] = "0.019"
            write_mutated_metadata(directory, bundle, role, meta)
        elif case in ("instrument_configuration_drift", "export_adapter_drift", "source_configuration_drift", "source_amplitude_out_of_contract"):
            role = "arm_pi"
            meta = plain_json(directory / bundle["roles"][role]["path"])
            if case == "instrument_configuration_drift":
                meta["instrument"]["configuration_queryback_sha256"] = "9" * 64
            elif case == "export_adapter_drift":
                meta["export"]["adapter_sha256"] = "9" * 64
            elif case == "source_amplitude_out_of_contract":
                meta["source"]["amplitude_vpp"] = "0.399"
            else:
                meta["source"]["setup_queryback_sha256"] = "9" * 64
            write_mutated_metadata(directory, bundle, role, meta)
        elif case in ("environment_leading_zero_index", "environment_plus_decimal", "environment_exponent_decimal", "environment_repeated_timestamp", "environment_crc_mutation", "environment_monotonic_drift", "environment_sensor_identity_drift"):
            env_path = directory / meta["environment"]["record_path"]
            lines = env_path.read_text(encoding="utf-8").splitlines()
            fields = lines[1].split(",")
            if case == "environment_leading_zero_index":
                fields[0] = "00"
            elif case == "environment_plus_decimal":
                fields[9] = "+25"
            elif case == "environment_exponent_decimal":
                fields[9] = "250e-1"
            elif case == "environment_repeated_timestamp":
                second = lines[2].split(",")
                second[2] = fields[2]
                lines[2] = ",".join(second)
            elif case == "environment_crc_mutation":
                fields[6] = "00" if fields[6] != "00" else "01"
            elif case == "environment_monotonic_drift":
                fields[1] = "1"
            else:
                fields[3] = "00000002"
            lines[1] = ",".join(fields)
            env_path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
            meta["environment"].update({"record_sha256": sha256_file(env_path), "record_bytes": env_path.stat().st_size})
            write_mutated_metadata(directory, bundle, role, meta)
        elif case == "negative_zero_metadata":
            meta["payload"]["offset"][0] = "-0"
            write_mutated_metadata(directory, bundle, role, meta)
        else:
            raise AssertionError(f"unknown adversary: {case}")
        try:
            analyze_bundle(bundle_path, identity_directory)
        except Reject as exc:
            if exc.code != expected:
                raise AssertionError(f"{case}: {exc.code} != {expected}") from exc
            return {"case": case, "outcome": "PASS", "execution": "actual_bundle_analyzer", "rejected_by": exc.code}
        raise AssertionError(f"{case}: did not reject")
    finally:
        for path, data in originals.items():
            path.write_bytes(data)
        if extra.exists():
            extra.unlink()


def run_raw_adversary_suite(directory: Path, identity_directory: Path) -> list[dict[str, Any]]:
    return [execute_bundle_adversary(directory, case, identity_directory, expected) for case, expected in RAW_ADVERSARIES.items()]


def analyze_control_record(data: bytes) -> None:
    try:
        record = json.loads(data.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=lambda value: (_ for _ in ()).throw(Reject("NONFINITE_JSON", value)))
    except UnicodeDecodeError as exc:
        raise Reject("UTF8", "control") from exc
    except json.JSONDecodeError as exc:
        raise Reject("MALFORMED_JSON", "control") from exc
    if not isinstance(record, dict) or data != canonical_bytes(record):
        raise Reject("NONCANONICAL_JSON", "control")
    exact_fields(record, {"schema", "control_id", "evidence_class", "physical_claim_requested", "source", "path", "transients", "witness", "payload", "environment", "controls", "epsilon"}, "control")
    if record["schema"] != "p0.synthetic-control-evidence.v1" or record["evidence_class"] != "SYNTHETIC" or record["physical_claim_requested"] is not False:
        raise Reject("CONTROL_SCHEMA")
    control_id = record["control_id"]
    if control_id not in set(POSITIVE_CASES) | set(SCIENTIFIC_NEGATIVES):
        raise Reject("CONTROL_ID")
    source = record["source"]
    exact_fields(source, {"stable_state_code", "residual_drive_ratio"}, "control.source")
    nonnegative_int(source["stable_state_code"], "control.source.stable_state_code")
    if source["stable_state_code"] != 8 or nonnegative_decimal(source["residual_drive_ratio"], "control.source.residual_drive_ratio") > 0.001:
        raise Reject("SOURCE_OFF_NOT_WITNESSED")
    path = record["path"]
    exact_fields(path, {"post_barrier_active_elements", "termination_id"}, "control.path")
    nonnegative_int(path["post_barrier_active_elements"], "control.path.post_barrier_active_elements")
    if path["post_barrier_active_elements"] != 0:
        raise Reject("POST_BARRIER_BUFFER_FORBIDDEN")
    if path["termination_id"] != "50R0-TNPW0805":
        raise Reject("TERMINATION_IDENTITY_MISMATCH")
    transients = record["transients"]
    exact_fields(transients, {"settled_after_admit", "detector_memory_seconds", "bounce_end_sample", "n_admit", "timing_mismatch_samples"}, "control.transients")
    if not isinstance(transients["settled_after_admit"], bool):
        raise Reject("TYPE", "control.transients.settled_after_admit")
    for key in ("bounce_end_sample", "n_admit", "timing_mismatch_samples"):
        nonnegative_int(transients[key], f"control.transients.{key}")
    if transients["settled_after_admit"] is not True:
        raise Reject("TRANSIENT_AFTER_ADMIT")
    if nonnegative_decimal(transients["detector_memory_seconds"], "control.transients.detector_memory_seconds") > 0.000010:
        raise Reject("DETECTOR_MEMORY_OVER_10US")
    if transients["bounce_end_sample"] > transients["n_admit"]:
        raise Reject("CONTACT_BOUNCE_AFTER_ADMIT")
    if transients["timing_mismatch_samples"] > 1:
        raise Reject("TIMING_MISMATCH_CONTROL")
    witness = record["witness"]
    exact_fields(witness, {"sample_count", "guard_samples", "invalid_samples", "ambiguous_samples", "post_off_reentry_samples"}, "control.witness")
    for key in witness:
        nonnegative_int(witness[key], f"control.witness.{key}")
    if witness["sample_count"] <= 0:
        raise Reject("CH2_WITNESS_MISSING")
    if witness["guard_samples"] < 10_000:
        raise Reject("GUARD_INTERVAL_SHORT")
    if witness["invalid_samples"] > 0:
        raise Reject("CH2_ILLEGAL_STATE")
    if witness["ambiguous_samples"] > 0:
        raise Reject("CH2_AMBIGUOUS_LEVEL")
    if witness["post_off_reentry_samples"] > 0:
        raise Reject("CH2_POST_OFF_REENTRY")
    payload = record["payload"]
    exact_fields(payload, {"channel_roles", "sample_rate_hz", "channel_skew_seconds", "clipped_samples", "saturated_samples"}, "control.payload")
    nonnegative_int(payload["sample_rate_hz"], "control.payload.sample_rate_hz")
    nonnegative_int(payload["clipped_samples"], "control.payload.clipped_samples")
    nonnegative_int(payload["saturated_samples"], "control.payload.saturated_samples")
    if payload["channel_roles"] != ["CH0_SOURCE", "CH1_CARRIER", "CH2_WITNESS", "CH3_VIBRATION"]:
        raise Reject("CHANNEL_ROLE_MISMATCH")
    if payload["sample_rate_hz"] != FS:
        raise Reject("TIMEBASE_MISMATCH")
    if nonnegative_decimal(payload["channel_skew_seconds"], "control.payload.channel_skew_seconds") > 0.0000001:
        raise Reject("CHANNEL_SKEW_OVER_LIMIT")
    if payload["clipped_samples"] > 0:
        raise Reject("CLIPPING")
    if payload["saturated_samples"] > 0:
        raise Reject("ADC_SATURATION")
    environment = record["environment"]
    exact_fields(environment, {"cadence_samples", "temperature_delta_c", "humidity_delta_rh", "vibration_rms_delta_m_s2"}, "control.environment")
    nonnegative_int(environment["cadence_samples"], "control.environment.cadence_samples")
    if environment["cadence_samples"] != 100_000:
        raise Reject("ENVIRONMENT_CADENCE")
    if nonnegative_decimal(environment["temperature_delta_c"], "control.environment.temperature_delta_c") > 0.20:
        raise Reject("TEMPERATURE_MISMATCH")
    if nonnegative_decimal(environment["humidity_delta_rh"], "control.environment.humidity_delta_rh") > 2.0:
        raise Reject("HUMIDITY_MISMATCH")
    if nonnegative_decimal(environment["vibration_rms_delta_m_s2"], "control.environment.vibration_rms_delta_m_s2") > 0.010:
        raise Reject("VIBRATION_MISMATCH")
    controls = record["controls"]
    exact_fields(controls, {"zero_drive_response_ratio", "resonator_removed_response_ratio", "dummy_c0_feedthrough_ratio", "off_resonance_response_ratio", "reference_leakage_ratio", "fixed_phase_concentration"}, "control.controls")
    for key, code in (
        ("zero_drive_response_ratio", "ZERO_DRIVE_CONTROL"),
        ("resonator_removed_response_ratio", "RESONATOR_REMOVED_CONTROL"),
        ("dummy_c0_feedthrough_ratio", "DUMMY_C0_FEEDTHROUGH"),
        ("off_resonance_response_ratio", "OFF_RESONANCE_RESPONSE"),
        ("reference_leakage_ratio", "REFERENCE_FEEDTHROUGH"),
    ):
        if nonnegative_decimal(controls[key], f"control.controls.{key}") > 0.020:
            raise Reject(code)
    if nonnegative_decimal(controls["fixed_phase_concentration"], "control.controls.fixed_phase_concentration") > 0.950:
        raise Reject("FIXED_PHASE_CONTROL")
    epsilon = record["epsilon"]
    exact_fields(epsilon, {"neg", "amplitude", "frequency", "decay", "phase", "feedthrough"}, "control.epsilon")
    values = {key: nonnegative_decimal(epsilon[key], f"control.epsilon.{key}") for key in epsilon}
    for key, limit, code in (("neg", 0.02, "COMPLEX_NEGATION"), ("amplitude", 0.02, "AMPLITUDE_MISMATCH"), ("frequency", 1e-6, "FREQUENCY_MISMATCH"), ("decay", 0.02, "DECAY_MISMATCH"), ("phase", 0.02, "HALF_TURN_PHASE_MISMATCH"), ("feedthrough", 0.02, "SOURCE_FEEDTHROUGH")):
        if values[key] > limit:
            raise Reject(code)


def run_semantic_suite() -> list[dict[str, Any]]:
    outcomes: list[dict[str, Any]] = []
    base = {
        "schema": "p0.synthetic-control-evidence.v1",
        "control_id": "",
        "evidence_class": "SYNTHETIC",
        "physical_claim_requested": False,
        "source": {"stable_state_code": 8, "residual_drive_ratio": "0"},
        "path": {"post_barrier_active_elements": 0, "termination_id": "50R0-TNPW0805"},
        "transients": {"settled_after_admit": True, "detector_memory_seconds": "0.000010", "bounce_end_sample": 1_100_000, "n_admit": 1_100_000, "timing_mismatch_samples": 0},
        "witness": {"sample_count": SAMPLES, "guard_samples": 10_000, "invalid_samples": 0, "ambiguous_samples": 0, "post_off_reentry_samples": 0},
        "payload": {"channel_roles": ["CH0_SOURCE", "CH1_CARRIER", "CH2_WITNESS", "CH3_VIBRATION"], "sample_rate_hz": FS, "channel_skew_seconds": "0.0000001", "clipped_samples": 0, "saturated_samples": 0},
        "environment": {"cadence_samples": 100_000, "temperature_delta_c": "0.20", "humidity_delta_rh": "2", "vibration_rms_delta_m_s2": "0.010"},
        "controls": {"zero_drive_response_ratio": "0.020", "resonator_removed_response_ratio": "0.020", "dummy_c0_feedthrough_ratio": "0.020", "off_resonance_response_ratio": "0.020", "reference_leakage_ratio": "0.020", "fixed_phase_concentration": "0.950"},
        "epsilon": {"neg": "0", "amplitude": "0", "frequency": "0", "decay": "0", "phase": "0", "feedthrough": "0"},
    }
    for case in POSITIVE_CASES:
        record = copy.deepcopy(base)
        record["control_id"] = case
        analyze_control_record(canonical_bytes(record))
        outcomes.append({"case": case, "class": "positive", "execution": "summary_schema_and_decision_law", "outcome": "PASS"})
    mutations: dict[str, tuple[tuple[str, ...], Any]] = {
        "zero_drive": (("controls", "zero_drive_response_ratio"), "0.021"),
        "resonator_removed": (("controls", "resonator_removed_response_ratio"), "0.021"),
        "dummy_c0_feedthrough": (("controls", "dummy_c0_feedthrough_ratio"), "0.021"),
        "source_left_on": (("source", "stable_state_code"), 7),
        "off_resonance_response": (("controls", "off_resonance_response_ratio"), "0.021"),
        "detector_impulse_memory": (("transients", "detector_memory_seconds"), "0.000011"),
        "controller_buffer_replay": (("path", "post_barrier_active_elements"), 1),
        "analog_switch_charge_transient": (("transients", "settled_after_admit"), False),
        "relay_bounce_transient": (("transients", "bounce_end_sample"), 1_100_001),
        "source_leakage_after_guard": (("epsilon", "feedthrough"), "0.021"),
        "reference_leakage": (("controls", "reference_leakage_ratio"), "0.021"),
        "amplitude_mismatch": (("epsilon", "amplitude"), "0.021"),
        "frequency_mismatch": (("epsilon", "frequency"), "0.0000011"),
        "decay_mismatch": (("epsilon", "decay"), "0.021"),
        "pi_2_phase": (("epsilon", "phase"), "1.5707963267948966"),
        "fixed_random_phases": (("controls", "fixed_phase_concentration"), "0.951"),
        "timing_mismatch": (("transients", "timing_mismatch_samples"), 2),
        "wrong_termination": (("path", "termination_id"), "WRONG"),
        "wrong_guard_interval": (("witness", "guard_samples"), 9_999),
        "ch2_illegal_code": (("witness", "invalid_samples"), 1),
        "ch2_nearest_code_ambiguity": (("witness", "ambiguous_samples"), 1),
        "post_off_state_reentry": (("witness", "post_off_reentry_samples"), 1),
        "missing_witness": (("witness", "sample_count"), 0),
        "channel_swap": (("payload", "channel_roles"), ["CH1_CARRIER", "CH0_SOURCE", "CH2_WITNESS", "CH3_VIBRATION"]),
        "timebase_drift": (("payload", "sample_rate_hz"), 999_999),
        "channel_skew_violation": (("payload", "channel_skew_seconds"), "0.00000011"),
        "clipping": (("payload", "clipped_samples"), 1),
        "adc_saturation": (("payload", "saturated_samples"), 1),
        "environment_cadence_failure": (("environment", "cadence_samples"), 99_999),
        "temperature_mismatch": (("environment", "temperature_delta_c"), "0.21"),
        "humidity_mismatch": (("environment", "humidity_delta_rh"), "2.01"),
        "vibration_mismatch": (("environment", "vibration_rms_delta_m_s2"), "0.011"),
    }
    for case, expected in SCIENTIFIC_NEGATIVES.items():
        state = copy.deepcopy(base)
        state["control_id"] = case
        path, value = mutations[case]
        target = state
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        try:
            analyze_control_record(canonical_bytes(state))
        except Reject as exc:
            if exc.code != expected:
                raise AssertionError(f"{case}: {exc.code} != {expected}") from exc
            outcomes.append({"case": case, "class": "scientific_negative", "execution": "summary_schema_and_decision_law__not_raw_analyzer_evidence", "outcome": "PASS", "rejected_by": exc.code})
        else:
            raise AssertionError(f"{case}: did not reject")
    label_probe = copy.deepcopy(base)
    label_probe["epsilon"]["amplitude"] = "0.021"
    observed: list[str] = []
    for label in ("amplitude_mismatch", "zero_drive"):
        label_probe["control_id"] = label
        try:
            analyze_control_record(canonical_bytes(label_probe))
        except Reject as exc:
            observed.append(exc.code)
    if observed != ["AMPLITUDE_MISMATCH", "AMPLITUDE_MISMATCH"]:
        raise AssertionError(f"control label influenced decision: {observed}")
    domain_probe = copy.deepcopy(base)
    domain_probe["control_id"] = "zero_drive"
    domain_probe["controls"]["zero_drive_response_ratio"] = "-0.001"
    try:
        analyze_control_record(canonical_bytes(domain_probe))
    except Reject as exc:
        if exc.code != "NUMBER_DOMAIN":
            raise AssertionError(f"negative-domain probe: {exc.code} != NUMBER_DOMAIN") from exc
        outcomes.append({"case": "negative_control_domain_invariant", "class": "invariant_probe", "execution": "summary_schema_and_decision_law", "outcome": "PASS", "rejected_by": exc.code})
    else:
        raise AssertionError("negative-domain probe: did not reject")
    return outcomes


def run_malformed_suite(directory: Path, identity_directory: Path) -> list[dict[str, Any]]:
    return [{**execute_bundle_adversary(directory, case, identity_directory, expected), "class": "malformed_or_custody_negative"} for case, expected in MALFORMED_NEGATIVES.items()]


def fixture_document() -> dict[str, Any]:
    return {
        "schema": FIXTURE_SCHEMA,
        "generator": "p0_scientific_analyzer.py",
        "canonical_payload": {"channels": CHANNELS, "samples_per_channel": SAMPLES, "bytes_per_payload": PAYLOAD_BYTES, "role_count": len(ROLE_ORDER), "materialized_replay_bytes": PAYLOAD_BYTES * len(ROLE_ORDER), "dtype": "signed-int16", "endian": "little", "layout": "sample-major-interleaved"},
        "positive": list(POSITIVE_CASES),
        "scientific_negative": [{"case": case, "expected_rejection": code} for case, code in SCIENTIFIC_NEGATIVES.items()],
        "malformed_or_custody_negative": [{"case": case, "expected_rejection": code} for case, code in MALFORMED_NEGATIVES.items()],
        "raw_adversary": [{"case": case, "expected_rejection": code} for case, code in RAW_ADVERSARIES.items()],
        "scope_law": {"semantic_controls": "summary schema and decision-law conformance only", "raw_adversaries": "actual canonical raw-bundle analyzer execution", "topology_only_cases": "require separately bound non-analyzer adjudication receipts"},
        "claim_law": {"synthetic_physical_claim": False, "maximum_token": "SYNTHETIC_ANALYZER_PASS"},
    }


def schema_document() -> dict[str, Any]:
    return {
        "schema": "p0.build-readiness-schemas.v1",
        "canonical_json": {"encoding": "UTF-8", "newline": "LF-final", "keys": "lexicographic", "indent_spaces": 2, "duplicate_keys": "reject", "unknown_fields": "reject", "nonfinite": "reject", "boolean_as_number": "reject", "hashes": "lowercase-sha256", "negative_zero": "reject-by-canonical-form"},
        "enforced_objects": {
            "p0.raw-bundle-manifest.v1": {"exact_fields": ["roles", "schema"], "role_entry_fields": ["path", "sha256"], "roles": list(ROLE_ORDER), "metadata_hashes_required": True, "unique_payloads_required": True},
            SCHEMA: {"exact_fields": ["clock", "custody", "environment", "evidence_class", "export", "instrument", "payload", "role", "run_id", "schema", "source", "thresholds", "witness"], "source_exact_fields": ["model", "phase_command_rad", "phase_skew_standard_uncertainty_rad", "phase_drive_cal_standard_uncertainty_rad", "frequency_hz", "amplitude_vpp", "offset_v", "reference_frequency_hz", "reference_amplitude_vpp", "reference_offset_v", "reference_phase_command_rad", "dual_channel_phase_locked", "monitor_network", "output_mode", "load_mode", "qualified_preparation_cycles", "source_remains_on_through_record", "output_ohms", "setup_queryback_sha256"], "source_frozen_setup": {"c1_amplitude_vpp": "0.4", "c1_offset_v": "0", "c2_amplitude_vpp": "0.1", "c2_offset_v": "0", "load_mode": "HIGH_Z", "physical_output_ohms": "50"}, "cross_record_equal": ["assignment_commitment_sha256", "calibration_sha256", "clock_identity", "evidence_class", "export_adapter_identity", "f_ref_hz", "instrument_identity_and_configuration", "scale_per_code", "source_setup_except_phase", "thresholds"], "assignment_reveal_required": True},
            "p0.environment-record.v1": {"format": "strict-csv", "header": ENVIRONMENT_HEADER, "cadence_samples": 100000, "monotonic_cadence_ns": 100000000, "timestamp_cadence_microseconds": 100000, "timestamp_year": "four-digit-year-not-hard-coded", "sensor_identity": "lowercase-8-hex-bound-to-metadata", "measurement_command_hex": "fd", "crc8": {"polynomial": "0x31", "initial": "0xff", "word_order": "big-endian"}, "canonical_index": "base-10-no-leading-zero", "canonical_decimal": "no-plus-no-leading-zero-no-negative-zero", "temperature_conversion": "-45+175*ticks/65535", "humidity_conversion": "-6+125*ticks/65535", "temperature_c": [20.0, 30.0], "humidity_rh": [20.0, 60.0]},
            "p0.synthetic-control-evidence.v1": {"exact_fields": ["control_id", "controls", "environment", "epsilon", "evidence_class", "path", "payload", "physical_claim_requested", "schema", "source", "transients", "witness"], "decision_independent_of_control_id": True, "scientific_negative_execution": "summary_schema_and_decision_law_only__not_raw_analyzer_evidence"},
            RESULT_SCHEMA: {"physical_claim_authorized": False, "required_custody": ["analyzer_sha256", "assignment_commitment_sha256", "bundle_sha256", "calibration_sha256", "dependency_sha256", "fixture_sha256", "metadata_sha256", "payload_sha256", "schema_sha256", "thresholds_sha256"], "drive_phase_uncertainty": {"joint_design_columns": ["cos(f_ref)", "-sin(f_ref)", "cos(2*f_ref)", "-sin(2*f_ref)", "constant"], "hac_lag": 7, "derived_error": "wrap(phi_C1-0.5*phi_C2-delta_command)", "fit_gradient": "g_C1-0.5*g_C2", "cross_covariance_included": True, "expanded_law": "1.96*sqrt(u_error_fit^2+u_skew^2+u_drive_cal^2)"}},
        },
        "identities": ["p0.build-readiness-contract.v1", "p0.component-identity.v1", "p0.document-identity.v1", "p0.netlist.v1", "p0.channel-map.v1", "p0.source-off-topology.v1", "p0.instrument-configuration.v1", "p0.native-export-adapter-receipt.v1", SCHEMA, "p0.calibration-packet.v1", "p0.arm-assignment-commitment.v1", "p0.environment-record.v1", RESULT_SCHEMA, "p0.control-outcome.v1", "p0.adjudication.v1", "p0.build-readiness-manifest.v1", "p0.contact-attestation.v1"],
        "forbidden_acquisition_fields": ["expected_result", "expected_phase_relation", "winner", "pass_hint"],
    }


def reference_document(identity_directory: Path) -> dict[str, Any]:
    semantic = run_semantic_suite()
    with tempfile.TemporaryDirectory(prefix="p0-scientific-") as temp:
        temp_path = Path(temp)
        bundle = materialize_synthetic(temp_path)
        raw_result = analyze_bundle(bundle, identity_directory)
        malformed = run_malformed_suite(temp_path, identity_directory)
        raw_adversaries = run_raw_adversary_suite(temp_path, identity_directory)
    dependency = dependency_identity()
    return {
        "schema": "p0.analyzer-reference-results.v1",
        "fixture_count": len(POSITIVE_CASES) + len(SCIENTIFIC_NEGATIVES) + len(MALFORMED_NEGATIVES),
        "positive_count": len(POSITIVE_CASES),
        "scientific_negative_count": len(SCIENTIFIC_NEGATIVES),
        "malformed_or_custody_negative_count": len(MALFORMED_NEGATIVES),
        "raw_adversary_count": len(RAW_ADVERSARIES),
        "semantic_outcomes": semantic,
        "malformed_outcomes": malformed,
        "raw_adversary_outcomes": raw_adversaries,
        "raw_numerical_reference": raw_result,
        "artifact_custody": {"analyzer_sha256": sha256_file(Path(__file__).resolve()), "fixture_sha256": sha256_file(identity_directory / "P0_SCIENTIFIC_FIXTURES.json"), "schema_sha256": sha256_file(identity_directory / "P0_BUILD_READINESS_SCHEMAS.json"), "dependency_identity": dependency, "dependency_sha256": sha256_bytes(canonical_bytes(dependency))},
        "physical_claim_authorized": False,
        "claim_ceiling": "NON_EXECUTING_P0_BUILD_READINESS_ONLY",
    }


def build(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fixtures = fixture_document()
    schemas = schema_document()
    (output / "P0_SCIENTIFIC_FIXTURES.json").write_bytes(canonical_bytes(fixtures))
    (output / "P0_BUILD_READINESS_SCHEMAS.json").write_bytes(canonical_bytes(schemas))
    results = reference_document(output)
    (output / "P0_ANALYZER_REFERENCE_RESULTS.json").write_bytes(canonical_bytes(results))


def self_test(full_raw: bool) -> dict[str, Any]:
    semantic = run_semantic_suite()
    raw_result: dict[str, Any] | None = None
    raw_adversaries: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="p0-scientific-") as temp:
        temp_path = Path(temp)
        bundle = materialize_synthetic(temp_path)
        malformed = run_malformed_suite(temp_path, Path(__file__).resolve().parent)
        if full_raw:
            raw_result = analyze_bundle(bundle, Path(__file__).resolve().parent)
            raw_adversaries = run_raw_adversary_suite(temp_path, Path(__file__).resolve().parent)
    return {"self_test": "PASS", "positive": len(POSITIVE_CASES), "scientific_negative": len(SCIENTIFIC_NEGATIVES), "malformed_or_custody_negative": len(MALFORMED_NEGATIVES), "raw_adversary_count": len(raw_adversaries), "full_raw": full_raw, "raw_result": raw_result, "raw_adversaries": raw_adversaries, "semantic_hash": sha256_bytes(canonical_bytes(semantic + malformed + raw_adversaries))}


def verify(directory: Path, full_raw: bool) -> dict[str, Any]:
    expected = {"P0_SCIENTIFIC_FIXTURES.json": canonical_bytes(fixture_document()), "P0_BUILD_READINESS_SCHEMAS.json": canonical_bytes(schema_document())}
    for name, data in expected.items():
        path = directory / name
        if not path.is_file() or path.read_bytes() != data:
            raise Reject("COMMITTED_BYTE_MISMATCH", name)
    if not full_raw:
        raise Reject("FULL_RAW_REQUIRED")
    expected_result = canonical_bytes(reference_document(directory))
    result_path = directory / "P0_ANALYZER_REFERENCE_RESULTS.json"
    if not result_path.is_file() or result_path.read_bytes() != expected_result:
        raise Reject("COMMITTED_BYTE_MISMATCH", result_path.name)
    result = self_test(True)
    result["verify"] = "PASS"
    result["fixture_sha256"] = sha256_file(directory / "P0_SCIENTIFIC_FIXTURES.json")
    result["schema_sha256"] = sha256_file(directory / "P0_BUILD_READINESS_SCHEMAS.json")
    result["result_sha256"] = sha256_file(result_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    p_self = sub.add_parser("self-test")
    p_self.add_argument("--full-raw", action="store_true")
    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    p_verify.add_argument("--full-raw", action="store_true")
    p_analyze = sub.add_parser("analyze")
    p_analyze.add_argument("bundle", type=Path)
    p_analyze.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.mode == "build":
            build(args.output)
            value = {"build": "PASS", "output": str(args.output.resolve())}
        elif args.mode == "self-test":
            value = self_test(args.full_raw)
        elif args.mode == "verify":
            value = verify(args.directory, args.full_raw)
        else:
            value = analyze_bundle(args.bundle.resolve())
            if args.output:
                args.output.write_bytes(canonical_bytes(value))
        print(canonical_bytes(value).decode("utf-8"), end="")
        return 0
    except Reject as exc:
        print(canonical_bytes({"status": "REJECT", "code": exc.code, "detail": exc.detail}).decode("utf-8"), end="", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
