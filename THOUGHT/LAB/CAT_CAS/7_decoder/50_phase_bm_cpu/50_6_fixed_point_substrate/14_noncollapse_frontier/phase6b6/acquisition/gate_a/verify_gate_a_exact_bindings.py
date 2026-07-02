#!/usr/bin/env python3
"""Verify that every Gate A candidate binding is frozen by schema constants."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
CANDIDATE_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_AUTHORITY_CANDIDATE.json"
SCHEMA_PATH = HERE / "schemas" / "gate_a_engineering_smoke_authority_candidate.schema.json"


class BindingError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise BindingError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def validate(candidate: dict[str, Any], schema: dict[str, Any]) -> None:
    required = schema["required"]
    properties = schema["properties"]
    require(schema["additionalProperties"] is False, "schema top level is open")
    require(set(candidate) == set(required), "candidate top-level key set mismatch")
    require(set(properties) == set(required), "schema property set mismatch")
    for key in required:
        definition = properties[key]
        require("const" in definition, f"candidate field is not frozen by const: {key}")
        require(candidate[key] == definition["const"], f"candidate differs from schema const: {key}")


def self_test(candidate: dict[str, Any], schema: dict[str, Any]) -> dict[str, bool]:
    mutations: list[tuple[str, dict[str, Any]]] = []
    changed = copy.deepcopy(candidate)
    changed["evidence_bindings"]["portable_archive_sha256"] = "0" * 64
    mutations.append(("archive_digest_changed", changed))
    changed = copy.deepcopy(candidate)
    changed["architecture_binding"]["architecture_path"] = "wrong"
    mutations.append(("architecture_path_changed", changed))
    changed = copy.deepcopy(candidate)
    changed["authority"]["engineering_smoke_authorized"] = True
    mutations.append(("authority_escalated", changed))
    changed = copy.deepcopy(candidate)
    changed["smoke_schedule"]["schedule_sha256"] = "0" * 64
    mutations.append(("schedule_digest_changed", changed))
    changed = copy.deepcopy(candidate)
    changed["unexpected"] = True
    mutations.append(("extra_property_added", changed))
    results: dict[str, bool] = {}
    for name, mutation in mutations:
        try:
            validate(mutation, schema)
        except BindingError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"one or more binding mutations were accepted: {results}")
    return results


def main() -> int:
    candidate = load(CANDIDATE_PATH)
    schema = load(SCHEMA_PATH)
    validate(candidate, schema)
    results = self_test(candidate, schema)
    print(json.dumps({
        "status": "PHASE6B6_GATE_A_EXACT_BINDINGS_VALID",
        "top_level_fields": len(candidate),
        "all_fields_const_bound": True,
        "mutation_results": results,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BindingError as exc:
        print(f"binding verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
