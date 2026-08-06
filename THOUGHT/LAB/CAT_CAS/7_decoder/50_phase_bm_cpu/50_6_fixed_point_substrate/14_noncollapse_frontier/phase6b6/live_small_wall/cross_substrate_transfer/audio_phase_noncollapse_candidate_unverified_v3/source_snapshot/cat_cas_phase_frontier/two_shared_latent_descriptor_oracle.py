#!/usr/bin/env python3
"""Independent public-topology oracle for bounded two-port descriptors."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


FEATURE = {"collision": 1, "cyclic_separation": 2}
SCOPE = {"port_a": 1, "port_b": 2, "joint": 3}
AXIS = {"x": 1, "y": 2, "z": 3, "controlled_phase": 4}
OFFSET = 1469598103934665603
PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def fail(message: str) -> None:
    raise RuntimeError(message)


def fnv_word(checksum: int, value: int) -> int:
    for byte in range(4):
        checksum ^= (value >> (8 * byte)) & 0xFF
        checksum = (checksum * PRIME) & MASK64
    return checksum


def semantic_word(module: dict[str, Any]) -> int:
    return (
        FEATURE[str(module["feature"])]
        | (SCOPE[str(module["scope"])] << 2)
        | (AXIS[str(module["axis"])] << 4)
        | (int(module["separation"]) << 7)
        | (int(module["strength"]) << 11)
        | (int(module["chirp"]) << 16)
    )


def validate_module(module: dict[str, Any]) -> None:
    feature = str(module["feature"])
    scope = str(module["scope"])
    axis = str(module["axis"])
    separation = int(module["separation"])
    strength = int(module["strength"])
    chirp = int(module["chirp"])
    if feature not in FEATURE or scope not in SCOPE or axis not in AXIS:
        fail("unknown descriptor enum")
    if not 1 <= strength <= 16 or not 1 <= chirp <= 16:
        fail("strength/chirp outside grid17 nonzero range")
    if feature == "collision" and separation != 0:
        fail("collision descriptor has separation")
    if feature == "cyclic_separation" and not 1 <= separation <= 8:
        fail("cyclic separation outside quotient range")
    if scope == "joint" and axis != "controlled_phase":
        fail("joint descriptor lacks controlled phase")
    if scope != "joint" and axis == "controlled_phase":
        fail("local descriptor uses joint axis")


def compile_independently(
    modules: list[dict[str, Any]],
) -> dict[str, object]:
    if not 4 <= len(modules) <= 8:
        fail("program length outside bounded compiler ceiling")
    words: list[int] = []
    local_a = 0
    local_b = 0
    joints: list[int] = []
    non_diagonal = False
    for index, module in enumerate(modules):
        validate_module(module)
        word = semantic_word(module)
        if word in words:
            fail("duplicate semantic descriptor")
        words.append(word)
        scope = str(module["scope"])
        if scope == "port_a":
            local_a += 1
        elif scope == "port_b":
            local_b += 1
        else:
            joints.append(index)
        if (
            scope != "joint"
            and str(module["axis"]) in {"x", "y"}
        ):
            non_diagonal = True
    if local_a == 0 or local_b == 0:
        fail("program omits a typed local consumer")
    if len(joints) < 2:
        fail("program lacks two joint consumers")
    if not non_diagonal:
        fail("program lacks declared non-diagonal local structure")
    stage_cut = joints[0] + 1
    if stage_cut >= len(modules):
        fail("derived stage cut leaves no continuation")
    if not any(index >= stage_cut for index in joints[1:]):
        fail("no joint consumer follows the stage boundary")
    checksum = OFFSET
    for value in (1, len(modules), stage_cut, *words):
        checksum = fnv_word(checksum, value)
    reverse_order = list(range(len(modules) - 1, -1, -1))
    return {
        "module_count": len(modules),
        "stage_cut": stage_cut,
        "checksum": checksum,
        "semantic_words": words,
        "inverse_indices": reverse_order,
        "local_a_count": local_a,
        "local_b_count": local_b,
        "joint_count": len(joints),
    }


def main() -> int:
    if len(sys.argv) != 3:
        fail(
            "usage: two_shared_latent_descriptor_oracle.py "
            "PROGRAMS_JSON SERVICE_RESULT_JSON"
        )
    programs_payload = json.loads(
        Path(sys.argv[1]).read_text(encoding="utf-8")
    )
    service_result = json.loads(
        Path(sys.argv[2]).read_text(encoding="utf-8")
    )
    programs = programs_payload["programs"]
    compiled = [
        compile_independently(list(program["modules"]))
        for program in programs
    ]
    oracle_checksums = [entry["checksum"] for entry in compiled]
    production_checksums = list(
        service_result["compiler"]["checksums"]
    )
    if oracle_checksums != production_checksums:
        fail("independent topology checksums disagree with service")
    if [entry["module_count"] for entry in compiled] != [6, 5, 7]:
        fail("unexpected bounded program family sizes")
    if [entry["stage_cut"] for entry in compiled] != [2, 2, 3]:
        fail("unexpected independently derived stage cuts")
    result = {
        "result": "PASS",
        "production_backend_imported": False,
        "production_protocol_imported": False,
        "production_compiler_called": False,
        "oracle_scope": (
            "PUBLIC_DESCRIPTOR_ENUM_RANGE_STRUCTURAL_VALIDATION_"
            "STAGE_CUT_CANONICAL_FNV64_AND_REVERSE_INDEX_ORDER"
        ),
        "programs": compiled,
        "checks": {
            "service_checksums_match": True,
            "three_distinct_families": len(set(oracle_checksums)) == 3,
            "two_joint_consumers_each": all(
                int(entry["joint_count"]) >= 2
                for entry in compiled
            ),
            "both_local_ports_each": all(
                int(entry["local_a_count"]) >= 1
                and int(entry["local_b_count"]) >= 1
                for entry in compiled
            ),
            "inverse_is_reverse_topology": all(
                entry["inverse_indices"]
                == list(
                    range(
                        int(entry["module_count"]) - 1,
                        -1,
                        -1,
                    )
                )
                for entry in compiled
            ),
        },
        "full_285_necklace_recurrence_reimplemented": False,
        "classical_reference_parity_established": False,
        "terminal": False,
    }
    if not all(result["checks"].values()):
        fail("independent descriptor oracle check failed")
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
