"""Immutable Phase 6B.6 binding to the qualified V2 physical interface."""

from __future__ import annotations

import math
import hashlib
import json
import re
from pathlib import Path
from typing import Any


QUALIFIED_V2_SOURCE = {
    "reviewed_source": "ba48125d15009a044bb869b5716c412b1a8baa1b",
    "generated_contracts": "500f7dfcd198e6e70dc3f999248aa61224d530cd",
    "corrective_evidence": "9291d61ab3eb8d27e2bff347f1ec90a046726228",
    "source_bundle_sha256": "bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f",
    "physical_interface_source_path": "14_noncollapse_frontier/holo_runtime_v2/combined_pdn_hardware.c",
    "physical_interface_source_sha256": "c95e90c3344a05d67799f44158036f316da66faf0fd66e47336ae045e8b4c976",
    "v2_source_bundle_manifest_schema": "CAT_CAS_PHASE6_V2_SOURCE_BUNDLE_MANIFEST_V1",
}

MODE_NAMES = ("basis", "rotation", "residual", "mini")
V2_HARDWARE_SOURCE_PATH = Path(__file__).resolve().parents[2] / "holo_runtime_v2" / "combined_pdn_hardware.c"


def digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tone_hz(index: int) -> float:
    if index < 0 or index >= 12:
        raise ValueError("tone index outside qualified V2 range")
    low = math.log(20.0)
    high = math.log(1500.0)
    x = index / 11.0
    return math.exp(low + (high - low) * x) * (1.0 + 0.013 * math.sin(2.399963 * (index + 1)))


def _u64(value: int) -> int:
    return value & ((1 << 64) - 1)


def _code_rand(state: int) -> tuple[int, int]:
    x = state
    x = _u64(x ^ _u64(x << 13))
    x = _u64(x ^ (x >> 7))
    x = _u64(x ^ _u64(x << 17))
    return x, x


def codebook() -> dict[str, tuple[int, ...]]:
    weights = (4, 5, 6, 7)
    state = 0x243F6A8885A308D3 ^ 7
    best: list[list[int]] | None = None
    best_distance = -1
    for _ in range(4000):
        candidate: list[list[int]] = []
        for weight in weights:
            row = [1] * 12
            pool = list(range(12))
            for i in range(weight):
                state, rnd = _code_rand(state)
                j = i + int(rnd % (12 - i))
                pool[i], pool[j] = pool[j], pool[i]
                row[pool[i]] = -1
            candidate.append(row)
        distance = 99
        for i in range(4):
            for j in range(i + 1, 4):
                hamming = sum(a != b for a, b in zip(candidate[i], candidate[j]))
                distance = min(distance, hamming)
        if distance > best_distance:
            best_distance = distance
            best = [row[:] for row in candidate]
    if best is None:
        raise AssertionError("codebook generation failed")
    return {name: tuple(best[index]) for index, name in enumerate(MODE_NAMES)}


def _source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_modes(source: str) -> tuple[str, ...]:
    match = re.search(r'const char \*modes\[\] = \{([^}]+)\};', source)
    if not match:
        raise ValueError("qualified V2 mode table not found")
    modes = tuple(re.findall(r'"([^"]+)"', match.group(1)))
    if modes != MODE_NAMES:
        raise ValueError("qualified V2 mode table changed")
    return modes


def _verify_source_semantics(source: str) -> None:
    required = (
        "double low = log(20), high = log(1500), x = index / 11.0;",
        "1 + .013 * sin(2.399963 * (index + 1))",
        "int weights[4] = {4, 5, 6, 7}",
        "code_rng = 0x243F6A8885A308D3ULL ^ 7ULL;",
        "iteration < 4000",
        "return mode >= 0 && source >= 0 && source < 12 ? codebook[mode][source] : 0;",
    )
    missing = [snippet for snippet in required if snippet not in source]
    if missing:
        raise ValueError("qualified V2 physical interface source semantics changed")


def extract_qualified_v2_table(source_path: Path | None = None) -> dict[str, Any]:
    path = source_path or V2_HARDWARE_SOURCE_PATH
    source = path.read_text(encoding="utf-8")
    source_sha = _source_sha256(path)
    if source_path is None and source_sha != QUALIFIED_V2_SOURCE["physical_interface_source_sha256"]:
        raise ValueError("qualified V2 physical interface source digest mismatch")
    _verify_source_semantics(source)
    modes = _extract_modes(source)
    table = {
        "schema_id": "CAT_CAS_PHASE6B6_IMPORTED_V2_TONE_CODEWORD_TABLE_V1",
        "source": {**QUALIFIED_V2_SOURCE, "extracted_artifact_sha256": source_sha},
        "tone_hz_formula": "exp(log(20)+(log(1500)-log(20))*index/11)*(1+.013*sin(2.399963*(index+1)))",
        "mode_names": modes,
        "mode_to_codeword_mapping": {mode: index for index, mode in enumerate(modes)},
        "tones": [
            {
                "physical_tone_index": index,
                "frequency_hz": tone_hz(index),
                "codeword_source_index": index,
                "mode_signs": {mode: signs[index] for mode, signs in codebook().items()},
            }
            for index in range(12)
        ],
        "codebook": codebook(),
        "codebook_rows": [{"mode": mode, "row": codebook()[mode]} for mode in modes],
    }
    table["tone_codeword_table_sha256"] = digest(table)
    return table


def verify_v2_table_binding(source_path: Path | None = None, expected_table: dict[str, Any] | None = None) -> dict[str, Any]:
    extracted = extract_qualified_v2_table(source_path)
    expected = expected_table or TONE_CODEWORD_TABLE
    if extracted["tone_codeword_table_sha256"] != expected["tone_codeword_table_sha256"]:
        raise ValueError("Phase 6B.6 V2 extracted table binding mismatch")
    return extracted


def tone_codeword_table() -> dict[str, Any]:
    return extract_qualified_v2_table()


def reconstructed_tone_codeword_table() -> dict[str, Any]:
    table = {
        "schema_id": "CAT_CAS_PHASE6B6_IMPORTED_V2_TONE_CODEWORD_TABLE_V1",
        "source": QUALIFIED_V2_SOURCE,
        "tone_hz_formula": "exp(log(20)+(log(1500)-log(20))*index/11)*(1+.013*sin(2.399963*(index+1)))",
        "mode_names": MODE_NAMES,
        "mode_to_codeword_mapping": {mode: index for index, mode in enumerate(MODE_NAMES)},
        "tones": [
            {
                "physical_tone_index": index,
                "frequency_hz": tone_hz(index),
                "codeword_source_index": index,
                "mode_signs": {mode: signs[index] for mode, signs in codebook().items()},
            }
            for index in range(12)
        ],
        "codebook": codebook(),
        "codebook_rows": [{"mode": mode, "row": codebook()[mode]} for mode in MODE_NAMES],
    }
    table["tone_codeword_table_sha256"] = digest(table)
    return table


TONE_CODEWORD_TABLE = tone_codeword_table()

PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT = {
    "schema_id": "CAT_CAS_PHASE6B6_PRE_ACQUISITION_V2_EQUIVALENCE_REQUIREMENT_V1",
    "required_before": "ACQUISITION_AUTHORITY",
    "non_hardware_qualification_must_emit_independent_table_from": QUALIFIED_V2_SOURCE["physical_interface_source_path"],
    "comparison_method": "byte_for_byte_or_value_for_value",
    "comparison_scope": (
        "12 tone frequencies",
        "4 mode names",
        "4 complete codeword rows",
        "mode-to-row mapping",
        "source SHA-256",
        "extracted table digest",
    ),
    "source_sha256": QUALIFIED_V2_SOURCE["physical_interface_source_sha256"],
    "imported_table_digest": TONE_CODEWORD_TABLE["tone_codeword_table_sha256"],
    "current_python_reproduction_is_independent_c_extraction": False,
    "evidence_package_created": False,
}
