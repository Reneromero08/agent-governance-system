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

from pathlib import Path
from typing import Any

import gate_a_engineering_smoke_executor_base as base
import gate_a_temperature_custody as temperature_custody
from gate_a_engineering_smoke_executor_base import *  # noqa: F401,F403


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
        temperature_custody.validate_temperature_receipt(
            receipt,
            expected_phase="pre_runtime",
            require_pass=True,
        )
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
        temperature_custody.validate_temperature_receipt(
            receipt,
            expected_phase="pre_runtime",
            require_pass=True,
        )
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
        raise
