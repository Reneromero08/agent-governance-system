"""Receiver-only schedule projection for ordinary V2 decoding."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

SENDER_PRIVATE_FIELDS = frozenset({
    "sender_codeword_source_index",
    "sender_theta_idx",
    "scramble_key_digest",
})


def load_receiver_schedule(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            projected = {
                key: value for key, value in row.items()
                if key not in SENDER_PRIVATE_FIELDS
            }
            if SENDER_PRIVATE_FIELDS & projected.keys():
                raise AssertionError("sender-private field escaped projection")
            rows.append(projected)
    return rows


def physical_gate_digest(gate: list[int]) -> str:
    return hashlib.sha256(bytes(gate)).hexdigest()
