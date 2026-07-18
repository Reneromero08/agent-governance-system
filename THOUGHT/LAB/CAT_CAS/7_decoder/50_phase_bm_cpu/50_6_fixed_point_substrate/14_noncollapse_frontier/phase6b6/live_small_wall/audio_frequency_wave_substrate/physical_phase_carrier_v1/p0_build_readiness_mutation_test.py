#!/usr/bin/env python3
"""Exhaustively prove candidate-root tamper detection for byte mutations.

This is deliberately not a semantic mutation test: the unchanged expected root
must reject every changed candidate byte before domain validation.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import p0_build_readiness_validator as validator  # noqa: E402

RESULT = ROOT / validator.MUTATION_RESULT


def walk(value: Any, path: tuple[Any, ...] = ()) -> Iterator[tuple[str, tuple[Any, ...]]]:
    if isinstance(value, dict):
        for key, member in value.items():
            yield "delete_dict_key", path + (key,)
            yield from walk(member, path + (key,))
    elif isinstance(value, list):
        for index, member in enumerate(value):
            yield "delete_list_item", path + (index,)
            yield from walk(member, path + (index,))
    else:
        yield "mutate_scalar", path


def parent_at(value: Any, path: tuple[Any, ...]) -> tuple[Any, Any]:
    current = value
    for step in path[:-1]:
        current = current[step]
    return current, path[-1]


def changed_scalar(value: Any) -> Any:
    if isinstance(value, bool):
        return not value
    if value is None:
        return "MUTATION"
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.125
    if isinstance(value, str):
        return value + "__MUTATION"
    raise TypeError(type(value).__name__)


def structured_mutations(name: str, data: bytes) -> Iterator[tuple[str, bytes]]:
    value = json.loads(data.decode("utf-8"))
    for kind, path in walk(value):
        mutated = copy.deepcopy(value)
        parent, key = parent_at(mutated, path)
        if kind == "delete_dict_key":
            del parent[key]
        elif kind == "delete_list_item":
            del parent[key]
        else:
            parent[key] = changed_scalar(parent[key])
        yield f"{name}:{kind}:{'/'.join(map(str, path))}", validator.canonical(mutated)


def line_mutations(name: str, data: bytes) -> Iterator[tuple[str, bytes]]:
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        mutated = list(lines)
        body = line[:-1] if line.endswith(b"\n") else line
        ending = b"\n" if line.endswith(b"\n") else b""
        mutated[index] = body + b" [MUTATION]" + ending
        yield f"{name}:line:{index + 1}", b"".join(mutated)


def execute() -> dict[str, Any]:
    baseline = validator.read_snapshot()
    summary = validator.validate_candidate(baseline)
    root = summary["candidate_root"]
    counts = {"delete_dict_key": 0, "delete_list_item": 0, "mutate_scalar": 0, "text_line": 0}
    total = 0
    rejected = 0
    accepted_examples: list[str] = []
    for name in validator.CANDIDATE_NAMES:
        if name in validator.PRETTY_JSON:
            cases = structured_mutations(name, baseline[name])
        else:
            cases = line_mutations(name, baseline[name])
        for identity, mutated_bytes in cases:
            total += 1
            if ":delete_dict_key:" in identity:
                counts["delete_dict_key"] += 1
            elif ":delete_list_item:" in identity:
                counts["delete_list_item"] += 1
            elif ":mutate_scalar:" in identity:
                counts["mutate_scalar"] += 1
            else:
                counts["text_line"] += 1
            snapshot = dict(baseline)
            snapshot[name] = mutated_bytes
            try:
                validator.validate_candidate(snapshot, expected_root=root)
            except validator.Failure:
                rejected += 1
            else:
                if len(accepted_examples) < 10:
                    accepted_examples.append(identity)
    return {
        "accepted": total - rejected,
        "accepted_examples": accepted_examples,
        "candidate_root": root,
        "class_counts": counts,
        "evidence_class": "ROOT_BINDING_TAMPER_DETECTION_ONLY__NOT_SEMANTIC_VALIDATION",
        "mutations_rejected": rejected,
        "mutations_total": total,
        "schema": "p0.build-readiness-mutation-results.v1",
        "status": "PASS" if total >= 1000 and rejected == total else "FAIL",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("build", "verify"))
    args = parser.parse_args(argv)
    try:
        actual = execute()
        if actual["status"] != "PASS":
            raise validator.Failure(f"mutation escape: {actual}")
        if args.mode == "build":
            temporary = RESULT.with_suffix(RESULT.suffix + ".tmp")
            temporary.write_bytes(validator.canonical(actual))
            temporary.replace(RESULT)
        else:
            if not RESULT.is_file() or RESULT.read_bytes() != validator.canonical(actual):
                raise validator.Failure("mutation result committed-byte mismatch")
        print(validator.canonical(actual).decode("utf-8"), end="")
        return 0
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(validator.canonical({"reason": str(exc), "status": "FAIL"}).decode("utf-8"), end="", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
