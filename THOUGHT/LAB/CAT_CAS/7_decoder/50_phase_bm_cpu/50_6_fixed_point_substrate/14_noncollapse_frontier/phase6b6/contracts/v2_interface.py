"""Immutable Phase 6B.6 binding to the qualified V2 physical interface."""

from __future__ import annotations

import math
from typing import Any

from .contract import digest


QUALIFIED_V2_SOURCE = {
    "reviewed_source": "ba48125d15009a044bb869b5716c412b1a8baa1b",
    "generated_contracts": "500f7dfcd198e6e70dc3f999248aa61224d530cd",
    "corrective_evidence": "9291d61ab3eb8d27e2bff347f1ec90a046726228",
    "source_bundle_sha256": "bec71b2369587e68a88e9e2b5cb47837a07d5cdef6f13990417e0c0928e85f2f",
    "v2_source_bundle_manifest_schema": "CAT_CAS_PHASE6_V2_SOURCE_BUNDLE_MANIFEST_V1",
}

MODE_NAMES = ("basis", "rotation", "residual", "mini")


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


def tone_codeword_table() -> dict[str, Any]:
    table = {
        "schema_id": "CAT_CAS_PHASE6B6_IMPORTED_V2_TONE_CODEWORD_TABLE_V1",
        "source": QUALIFIED_V2_SOURCE,
        "tone_hz_formula": "exp(log(20)+(log(1500)-log(20))*index/11)*(1+.013*sin(2.399963*(index+1)))",
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
    }
    table["tone_codeword_table_sha256"] = digest(table)
    return table


TONE_CODEWORD_TABLE = tone_codeword_table()
