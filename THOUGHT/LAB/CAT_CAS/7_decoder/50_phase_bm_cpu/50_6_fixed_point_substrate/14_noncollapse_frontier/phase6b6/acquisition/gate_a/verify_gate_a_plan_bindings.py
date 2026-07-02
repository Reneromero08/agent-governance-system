#!/usr/bin/env python3
"""Verify the committed Gate A run-plan blob and frozen target namespace."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = next((p for p in Path(__file__).resolve().parents if (p / ".git").exists()), None)
if REPO is None:
    raise SystemExit("repository root not found")

CANDIDATE_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE.json"
NAMESPACE_PATH = HERE / "GATE_A_TARGET_NAMESPACE_BINDING.json"
EXPECTED_NAMESPACE_SHA256 = "5b3090f642d28492e182630e6349eccd8181704f08129d40d886c8f529dfd50e"
EXPECTED_RUN_PLAN_BLOB = "f7352d9c92a5275fc426e791b5509fd8aae85251"


class BindingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON object required: {path}")
    return value


def digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    require(result.returncode == 0, f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def validate(candidate: dict[str, Any], namespace: dict[str, Any]) -> None:
    plan = candidate["plan_bindings"]
    require(plan["run_plan_git_blob_sha1"] == EXPECTED_RUN_PLAN_BLOB, "candidate run-plan blob mismatch")
    require(plan["target_namespace_binding_sha256"] == EXPECTED_NAMESPACE_SHA256, "candidate namespace digest mismatch")

    unsigned = copy.deepcopy(namespace)
    declared = unsigned.pop("binding_sha256")
    require(declared == EXPECTED_NAMESPACE_SHA256, "namespace declared digest mismatch")
    require(digest(unsigned) == EXPECTED_NAMESPACE_SHA256, "namespace content digest mismatch")
    require(namespace == {
        "base_main_commit": "9c41637992536f43d10d152ec176a3577aef1623",
        "binding_sha256": EXPECTED_NAMESPACE_SHA256,
        "must_be_absent_before_deployment": True,
        "remote_execution_root": "/root/catcas_phase6b6_gate_a_smoke_9c416379",
        "remote_output_root": "/root/catcas_phase6b6_gate_a_smoke_9c416379/evidence",
        "remove_only_after_verified_copy_back": True,
        "schema_id": "CAT_CAS_PHASE6B6_GATE_A_TARGET_NAMESPACE_BINDING_V1",
        "status": "FROZEN__NO_EXECUTION_AUTHORITY",
    }, "namespace object mismatch")

    run_plan_path = plan["run_plan_path"]
    namespace_path = plan["target_namespace_binding_path"]
    require(git("rev-parse", f"HEAD:{run_plan_path}") == EXPECTED_RUN_PLAN_BLOB, "committed run-plan blob mismatch")
    require(git("rev-parse", f"HEAD:{namespace_path}"), "committed namespace blob missing")


def self_test(candidate: dict[str, Any], namespace: dict[str, Any]) -> dict[str, bool]:
    cases: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    changed = copy.deepcopy(candidate)
    changed["plan_bindings"]["run_plan_git_blob_sha1"] = "0" * 40
    cases.append(("run_plan_blob_changed", changed, copy.deepcopy(namespace)))
    changed = copy.deepcopy(candidate)
    changed["plan_bindings"]["target_namespace_binding_sha256"] = "0" * 64
    cases.append(("namespace_digest_changed", changed, copy.deepcopy(namespace)))
    changed_namespace = copy.deepcopy(namespace)
    changed_namespace["remote_execution_root"] = "/root/wrong"
    cases.append(("execution_root_changed", copy.deepcopy(candidate), changed_namespace))
    changed_namespace = copy.deepcopy(namespace)
    changed_namespace["remove_only_after_verified_copy_back"] = False
    cases.append(("cleanup_weakened", copy.deepcopy(candidate), changed_namespace))

    results: dict[str, bool] = {}
    for name, changed_candidate, changed_namespace in cases:
        try:
            validate(changed_candidate, changed_namespace)
        except BindingError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"plan-binding mutation accepted: {results}")
    return results


def main() -> int:
    candidate = load(CANDIDATE_PATH)
    namespace = load(NAMESPACE_PATH)
    validate(candidate, namespace)
    results = self_test(candidate, namespace)
    print(json.dumps({
        "status": "PHASE6B6_GATE_A_PLAN_BINDINGS_VALID",
        "run_plan_git_blob_sha1": EXPECTED_RUN_PLAN_BLOB,
        "target_namespace_binding_sha256": EXPECTED_NAMESPACE_SHA256,
        "mutation_results": results,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as exc:
        print(f"plan-binding verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
