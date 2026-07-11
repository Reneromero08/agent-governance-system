#!/usr/bin/env python3
"""Target-packaged Gate A executor with exact k10temp custody.

The reviewed executor remains available in the bundle as
``gate_a_engineering_smoke_executor_base``.  This module is packaged under the
canonical ``gate_a_engineering_smoke_executor`` name and changes only the
pre-runtime temperature observation: it selects one exact k10temp hwmon input,
retains its closed receipt, and fails before frequency checks or runtime entry
when observation or the 68 C veto fails.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

import gate_a_engineering_smoke_executor_base as base
import gate_a_temperature_custody as temperature_custody
from gate_a_engineering_smoke_executor_base import *  # noqa: F401,F403

_write_all = base._write_all

NATIVE_TEMPERATURE_SCHEMA_ID = "CAT_CAS_PHASE6B6_GATE_A_NATIVE_TEMPERATURE_RECEIPT_V1"
NATIVE_TEMPERATURE_RECEIPT_FILE = "TEMPERATURE_RECEIPTS.jsonl"
NATIVE_TEMPERATURE_HWMON_ROOT = "/sys/class/hwmon"
NATIVE_TEMPERATURE_PHASES = ("pre_capture", "post_capture")
NATIVE_TEMPERATURE_RECEIPT_KEYS = frozenset({
    "schema_id",
    "phase",
    "hwmon_root",
    "required_driver_name",
    "required_temperature_input",
    "millidegrees_per_c",
    "enumerated_hwmon_count",
    "k10temp_candidate_count",
    "selected_hwmon_entry",
    "selected_driver_name",
    "selected_temperature_path",
    "raw_temperature_text",
    "raw_temperature_sha256",
    "raw_millidegrees_c",
    "normalized_temperature_c",
    "veto_temperature_c",
    "observation_complete",
    "veto_passed",
    "failure",
    "observation_tsc",
})
NATIVE_TEMPERATURE_FAILURES = frozenset({
    "TEMPERATURE_CONTRACT_INVALID",
    "HWMON_ROOT_UNOBSERVABLE",
    "HWMON_ENUMERATION_EMPTY",
    "HWMON_ENUMERATION_LIMIT",
    "HWMON_ENUMERATION_UNOBSERVABLE",
    "DRIVER_NAME_UNOBSERVABLE",
    "DRIVER_NAME_EMPTY",
    "TEMPERATURE_PATH_INVALID",
    "K10TEMP_CANDIDATE_COUNT",
    "TEMPERATURE_INPUT_UNOBSERVABLE",
    "TEMPERATURE_RAW_HASH_FAILURE",
    "TEMPERATURE_INTEGER_MALFORMED",
    "TEMPERATURE_IMPLAUSIBLE",
    "TEMPERATURE_RAW_FORMAT_FAILURE",
    "TEMPERATURE_VETO",
})


def validate_native_temperature_receipt(
    receipt: dict[str, Any],
    *,
    expected_phase: str,
    require_pass: bool,
    expected_hwmon_root: str = NATIVE_TEMPERATURE_HWMON_ROOT,
) -> None:
    """Validate the native runtime's closed thermal custody record."""

    base.require(
        isinstance(receipt, dict) and set(receipt) == NATIVE_TEMPERATURE_RECEIPT_KEYS,
        "native temperature receipt key set mismatch",
    )
    base.require(receipt["schema_id"] == NATIVE_TEMPERATURE_SCHEMA_ID, "native temperature schema mismatch")
    base.require(expected_phase in NATIVE_TEMPERATURE_PHASES, "native temperature expected phase is not closed")
    base.require(receipt["phase"] == expected_phase, "native temperature phase mismatch")
    base.require(receipt["hwmon_root"] == expected_hwmon_root, "native temperature hwmon root mismatch")
    base.require(receipt["required_driver_name"] == temperature_custody.DRIVER_NAME, "native temperature driver mismatch")
    base.require(receipt["required_temperature_input"] == temperature_custody.TEMPERATURE_INPUT, "native temperature input mismatch")
    base.require(receipt["millidegrees_per_c"] == temperature_custody.MILLIDEGREES_PER_C, "native temperature scale mismatch")
    base.require(receipt["veto_temperature_c"] == temperature_custody.VETO_C, "native temperature veto threshold mismatch")
    enumerated = receipt["enumerated_hwmon_count"]
    candidates = receipt["k10temp_candidate_count"]
    base.require(isinstance(enumerated, int) and not isinstance(enumerated, bool) and 0 <= enumerated <= 64, "native hwmon count malformed")
    base.require(isinstance(candidates, int) and not isinstance(candidates, bool) and 0 <= candidates <= enumerated, "native k10temp count malformed")
    base.require(
        isinstance(receipt["observation_tsc"], int)
        and not isinstance(receipt["observation_tsc"], bool)
        and receipt["observation_tsc"] > 0,
        "native temperature observation TSC malformed",
    )
    base.require(type(receipt["observation_complete"]) is bool, "native temperature completeness malformed")
    base.require(type(receipt["veto_passed"]) is bool, "native temperature veto result malformed")
    base.require(
        receipt["failure"] is None or isinstance(receipt["failure"], str),
        "native temperature failure malformed",
    )

    selected = receipt["selected_hwmon_entry"]
    selected_driver = receipt["selected_driver_name"]
    selected_path = receipt["selected_temperature_path"]
    if selected is None:
        base.require(selected_driver is None and selected_path is None, "native temperature partial selection")
    else:
        base.require(
            isinstance(selected, str)
            and re.fullmatch(re.escape(expected_hwmon_root) + r"/hwmon[0-9]+", selected) is not None,
            "native temperature selected hwmon path mismatch",
        )
        base.require(selected_driver == temperature_custody.DRIVER_NAME, "native temperature selected driver mismatch")
        base.require(
            selected_path == str(PurePosixPath(selected) / temperature_custody.TEMPERATURE_INPUT),
            "native temperature selected input path mismatch",
        )

    raw_text = receipt["raw_temperature_text"]
    raw_digest = receipt["raw_temperature_sha256"]
    raw_millidegrees = receipt["raw_millidegrees_c"]
    normalized = receipt["normalized_temperature_c"]
    if raw_text is None:
        base.require(raw_digest is None, "native temperature raw digest without raw text")
    else:
        base.require(isinstance(raw_text, str) and raw_text != "", "native temperature raw text malformed")
        try:
            raw_bytes = raw_text.encode("ascii")
        except UnicodeEncodeError as exc:
            raise base.ExecutorError("native temperature raw text is not ASCII") from exc
        base.require(hashlib.sha256(raw_bytes).hexdigest() == raw_digest, "native temperature raw digest mismatch")
    if raw_millidegrees is not None:
        base.require(raw_text is not None, "native temperature integer lacks raw text")
        stripped = raw_text.strip()
        base.require(re.fullmatch(r"[+-]?[0-9]+", stripped) is not None, "native temperature raw integer malformed")
        base.require(type(raw_millidegrees) is int and int(stripped, 10) == raw_millidegrees, "native temperature integer mismatch")
    if normalized is not None:
        base.require(raw_millidegrees is not None, "native normalized temperature lacks integer")
        expected_normalized = raw_millidegrees / temperature_custody.MILLIDEGREES_PER_C
        base.require(
            isinstance(normalized, (int, float))
            and not isinstance(normalized, bool)
            and math.isfinite(normalized)
            and math.isclose(normalized, expected_normalized, rel_tol=0.0, abs_tol=1e-12),
            "native normalized temperature mismatch",
        )

    if receipt["observation_complete"]:
        base.require(enumerated > 0 and candidates == 1 and selected is not None, "native complete observation lacks exact k10temp selection")
        base.require(raw_millidegrees is not None and normalized is not None, "native complete observation lacks temperature value")
        base.require(
            temperature_custody.MIN_PLAUSIBLE_C <= normalized <= temperature_custody.MAX_PLAUSIBLE_C,
            "native temperature is implausible",
        )
        expected_pass = raw_millidegrees < int(
            temperature_custody.VETO_C * temperature_custody.MILLIDEGREES_PER_C
        )
        expected_failure = None if expected_pass else "TEMPERATURE_VETO"
        base.require(receipt["veto_passed"] is expected_pass, "native temperature veto result mismatch")
        base.require(receipt["failure"] == expected_failure, "native temperature completion failure mismatch")
    else:
        base.require(receipt["veto_passed"] is False, "incomplete native temperature observation passed")
        base.require(receipt["failure"] in NATIVE_TEMPERATURE_FAILURES - {"TEMPERATURE_VETO"}, "native temperature failure is not closed")
    if receipt["failure"] is not None:
        base.require(receipt["failure"] in NATIVE_TEMPERATURE_FAILURES, "native temperature failure code mismatch")
    if require_pass:
        base.require(receipt["observation_complete"] is True, "native temperature observation incomplete")
        base.require(receipt["veto_passed"] is True, "native temperature veto")


def verify_native_temperature_receipts(
    runtime_root: Path,
    result: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = runtime_root / NATIVE_TEMPERATURE_RECEIPT_FILE
    base.require(path.is_file() and not path.is_symlink(), "native temperature receipt file missing")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise base.ExecutorError("native temperature receipt file unreadable") from exc
    base.require(raw != b"" and raw.endswith(b"\n"), "native temperature receipt file is incomplete")
    lines = raw.splitlines()
    base.require(len(lines) == 2 and all(lines), "native temperature receipt count mismatch")
    try:
        receipts = [json.loads(line.decode("ascii")) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise base.ExecutorError("native temperature receipt JSON malformed") from exc
    for receipt, phase in zip(receipts, NATIVE_TEMPERATURE_PHASES, strict=True):
        validate_native_temperature_receipt(
            receipt,
            expected_phase=phase,
            require_pass=True,
        )
    pre, post = receipts
    base.require(
        (
            pre["hwmon_root"],
            pre["selected_hwmon_entry"],
            pre["selected_driver_name"],
            pre["selected_temperature_path"],
        )
        == (
            post["hwmon_root"],
            post["selected_hwmon_entry"],
            post["selected_driver_name"],
            post["selected_temperature_path"],
        ),
        "native temperature sensor identity changed across capture",
    )
    base.require(pre["observation_tsc"] < post["observation_tsc"], "native temperature observation order mismatch")
    capture = result["capture"]
    base.require(pre["observation_tsc"] < capture["origin_tsc"], "native pre-capture receipt was not observed before capture")
    base.require(post["observation_tsc"] > capture["last_sample_tsc"], "native post-capture receipt was not observed after capture")
    return pre, post


def verify_retained_runtime_evidence(
    plan: base.FrozenPlan,
    result: dict[str, Any],
) -> None:
    base.verify_retained_runtime_evidence(plan, result)
    verify_native_temperature_receipts(plan.output_root / "runtime", result)


class WorkerRuntime(base.WorkerRuntime):
    """Reviewed worker bridge plus native k10temp receipt verification."""

    def verify_evidence(
        self,
        plan: base.FrozenPlan,
        result: dict[str, Any],
    ) -> None:
        verify_retained_runtime_evidence(plan, result)


class LocalPreflight(base.LocalPreflight):
    """Production preflight with exact target-specific k10temp discovery."""

    def __init__(self, *, hwmon_root: Path = temperature_custody.HWMON_ROOT):
        self.hwmon_root = hwmon_root
        self._temperature_receipt: dict[str, Any] | None = None

    def observe_temperature_receipt(self) -> dict[str, Any]:
        receipt = temperature_custody.observe_temperature(
            "pre_runtime",
            hwmon_root=self.hwmon_root,
        )
        self._temperature_receipt = receipt
        return dict(receipt)

    def temperature_c(self) -> float:
        receipt = self.observe_temperature_receipt()
        try:
            temperature_custody.validate_temperature_receipt(
                receipt,
                expected_phase="pre_runtime",
                require_pass=True,
            )
        except temperature_custody.TemperatureCustodyError as exc:
            raise base.ExecutorError(str(exc)) from exc
        return temperature_custody.normalized_temperature_c(receipt)

    def temperature_receipt(self) -> dict[str, Any]:
        base.require(self._temperature_receipt is not None, "temperature receipt unavailable")
        return dict(self._temperature_receipt)


class JsonEvidenceStore(base.JsonEvidenceStore):
    """Base evidence store plus the pre-runtime thermal custody artifact."""

    def temperature_receipt(self, receipt: dict[str, Any]) -> None:
        temperature_custody.validate_temperature_receipt(
            receipt,
            expected_phase="pre_runtime",
            require_pass=False,
        )
        self._exclusive_json("TEMPERATURE_PREFLIGHT_RECEIPT.json", receipt)


def _retain_temperature_receipt(
    surfaces: base.ExecutionSurfaces,
) -> float:
    observe_method = getattr(surfaces.preflight, "observe_temperature_receipt", None)
    if callable(observe_method):
        receipt = observe_method()
        temperature_custody.validate_temperature_receipt(
            receipt,
            expected_phase="pre_runtime",
            require_pass=False,
        )
        evidence_method = getattr(surfaces.evidence, "temperature_receipt", None)
        base.require(callable(evidence_method), "temperature evidence sink unavailable")
        evidence_method(receipt)
        surfaces.evidence.event({
            "event": "temperature_preflight",
            "temperature_receipt_sha256": temperature_custody.receipt_sha256(receipt),
            "temperature_c": receipt["normalized_temperature_c"],
            "observation_complete": receipt["observation_complete"],
            "veto_passed": receipt["veto_passed"],
            "failure": receipt["failure"],
        })
        # The closed receipt is durable before this pass/fail enforcement.
        try:
            temperature_custody.validate_temperature_receipt(
                receipt,
                expected_phase="pre_runtime",
                require_pass=True,
            )
        except temperature_custody.TemperatureCustodyError as exc:
            raise base.ExecutorError(str(exc)) from exc
        return temperature_custody.normalized_temperature_c(receipt)

    # Compatibility for dependency-injected qualification fakes.  The live
    # LocalPreflight above always exposes and retains the closed receipt.
    temperature = surfaces.preflight.temperature_c()
    surfaces.evidence.event({
        "event": "temperature_preflight",
        "temperature_c": temperature,
        "temperature_receipt_sha256": None,
        "legacy_injected_surface": True,
    })
    base.require(base.math.isfinite(temperature) and temperature < base.TEMPERATURE_VETO_C, "temperature veto")
    return temperature


def execute_once(
    *,
    authority_validation: dict[str, Any],
    authority_sha256: str,
    execution_bundle_sha256: str,
    schedule: dict[str, Any],
    output_root: Path,
    surfaces: base.ExecutionSurfaces,
) -> dict[str, Any]:
    """Consume one exact authority and run only after retained k10temp custody."""

    base.require(set(authority_validation) == {
        "status", "reviewed_adapter_head", "independent_review_id", "execution_bundle_sha256",
    }, "authority validation result is not closed")
    base.require(authority_validation["status"] == "GATE_A_EXECUTION_AUTHORITY_EXACT", "exact authority validation required")
    base.require(isinstance(authority_validation["reviewed_adapter_head"], str) and len(authority_validation["reviewed_adapter_head"]) == 40, "reviewed adapter head missing")
    base.require(isinstance(authority_validation["independent_review_id"], int) and authority_validation["independent_review_id"] > 0, "independent review binding missing")
    base.require(authority_validation["execution_bundle_sha256"] == execution_bundle_sha256, "authority validation bundle binding mismatch")
    base.require(len(authority_sha256) == 64 and all(c in "0123456789abcdef" for c in authority_sha256), "authority digest malformed")
    base.require(len(execution_bundle_sha256) == 64 and all(c in "0123456789abcdef" for c in execution_bundle_sha256), "bundle digest malformed")
    base.validate_frozen_schedule(schedule)
    state = surfaces.preflight.inspect_namespace(output_root)
    base.require(state is base.NamespaceState.ABSENT, f"output namespace not provably absent (state={state.value})")

    plan = base.FrozenPlan(
        authority_sha256=authority_sha256,
        execution_bundle_sha256=execution_bundle_sha256,
        output_root=output_root,
    )
    surfaces.claims.claim(authority_sha256, plan)

    initial_preflight = {
        "namespace_state": state.value,
        "preflight_complete": False,
        "frequency_writes": 0,
        "voltage_writes": 0,
        "msr_reads": 0,
        "msr_writes": 0,
    }
    evidence_started = False
    try:
        surfaces.evidence.begin(plan, initial_preflight)
        evidence_started = True
        base._retain_process_receipt(surfaces, phase="pre_runtime")
        _retain_temperature_receipt(surfaces)
        frequencies = {str(core): surfaces.preflight.frequency_khz(core) for core in (base.SENDER_CORE, base.RECEIVER_CORE)}
        surfaces.evidence.event({"event": "frequency_preflight", "frequency_khz": frequencies})
        base.require(all(value == base.REQUIRED_FREQUENCY_KHZ for value in frequencies.values()), "frequency veto")
        surfaces.evidence.event({"event": "preflight_complete", "preflight_complete": True})
        surfaces.evidence.event({"event": "runtime_start", "runtime_execution_count": 1, "automatic_retry": False})
        result: dict[str, Any] | None = None
        runtime_error: Exception | None = None
        try:
            result = surfaces.runtime.execute(plan)
        except Exception as exc:
            runtime_error = exc
        post_runtime_error: Exception | None = None
        try:
            base._retain_process_receipt(surfaces, phase="post_runtime")
        except Exception as exc:
            post_runtime_error = exc
        if runtime_error is not None:
            if post_runtime_error is not None:
                raise base.ExecutorError(
                    f"runtime failed ({runtime_error}); post-runtime process custody failed ({post_runtime_error})"
                ) from runtime_error
            raise runtime_error
        if post_runtime_error is not None:
            raise post_runtime_error
        base.require(result is not None, "runtime returned no result")
        base.validate_runtime_result(result, plan)
        surfaces.runtime.verify_evidence(plan, result)
        surfaces.evidence.complete(result)
        return result
    except Exception as exc:
        if evidence_started:
            surfaces.evidence.fail(str(exc))
        if isinstance(exc, temperature_custody.TemperatureCustodyError):
            raise base.ExecutorError(str(exc)) from exc
        raise
