#!/usr/bin/env python3
"""Dependency-free validation of the Gate A schedule schema subset."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
SCHEDULE_PATH = HERE / "GATE_A_ENGINEERING_SMOKE_SCHEDULE.json"
SCHEMA_PATH = HERE / "schemas" / "gate_a_engineering_smoke_schedule.schema.json"


class SchemaError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"object required: {path}")
    return value


def type_matches(value: Any, kind: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(kind, False)


def resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any]:
    require(ref.startswith("#/"), f"unsupported ref: {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        require(isinstance(node, dict) and part in node, f"unresolved ref: {ref}")
        node = node[part]
    require(isinstance(node, dict), f"ref is not schema object: {ref}")
    return node


def validate(value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$") -> None:
    if "$ref" in schema:
        validate(value, resolve_ref(root, schema["$ref"]), root, path)
        return
    if "const" in schema:
        require(value == schema["const"], f"const mismatch at {path}")
    if "type" in schema:
        kinds = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        require(any(type_matches(value, kind) for kind in kinds), f"type mismatch at {path}: {kinds}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        require(all(key in value for key in required), f"missing required property at {path}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            require(set(value) <= set(properties), f"extra property at {path}: {set(value) - set(properties)}")
        for key, child in value.items():
            if key in properties:
                validate(child, properties[key], root, f"{path}.{key}")
    if isinstance(value, list):
        if "minItems" in schema:
            require(len(value) >= schema["minItems"], f"minItems failed at {path}")
        if schema.get("uniqueItems") is True:
            encoded = [json.dumps(item, sort_keys=True, separators=(",", ":")) for item in value]
            require(len(encoded) == len(set(encoded)), f"uniqueItems failed at {path}")
        if "items" in schema:
            for index, item in enumerate(value):
                validate(item, schema["items"], root, f"{path}[{index}]")


def self_test(schedule: dict[str, Any], schema: dict[str, Any]) -> dict[str, bool]:
    cases: list[tuple[str, dict[str, Any]]] = []
    changed = copy.deepcopy(schedule)
    changed["unexpected"] = True
    cases.append(("extra_top_level_property", changed))
    changed = copy.deepcopy(schedule)
    del changed["slot_definitions"]["T"]["executed"]["sign"]
    cases.append(("missing_executed_field", changed))
    changed = copy.deepcopy(schedule)
    changed["slot_definitions"]["D0"]["executed"]["drive_on"] = "false"
    cases.append(("wrong_boolean_type", changed))
    changed = copy.deepcopy(schedule)
    changed["preconditions"].append(changed["preconditions"][0])
    cases.append(("duplicate_precondition", changed))
    changed = copy.deepcopy(schedule)
    changed["slot_sequence"][0] = "A0P"
    cases.append(("sequence_const_changed", changed))

    results: dict[str, bool] = {}
    for name, mutation in cases:
        try:
            validate(mutation, schema, schema)
        except SchemaError:
            results[name] = True
        else:
            results[name] = False
    require(all(results.values()), f"schema mutation accepted: {results}")
    return results


def main() -> int:
    schedule = load(SCHEDULE_PATH)
    schema = load(SCHEMA_PATH)
    require(schema["$id"] == "CAT_CAS_PHASE6B6_GATE_A_ENGINEERING_SMOKE_SCHEDULE_SCHEMA_V1", "schedule schema ID mismatch")
    validate(schedule, schema, schema)
    results = self_test(schedule, schema)
    print(json.dumps({
        "status": "PHASE6B6_GATE_A_SCHEDULE_SCHEMA_VALID",
        "schema_id": schema["$id"],
        "mutation_results": results,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SchemaError as exc:
        print(f"schedule schema verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
