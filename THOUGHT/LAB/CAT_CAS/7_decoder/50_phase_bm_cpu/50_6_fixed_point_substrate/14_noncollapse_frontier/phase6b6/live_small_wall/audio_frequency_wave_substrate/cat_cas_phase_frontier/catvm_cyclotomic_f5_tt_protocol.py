#!/usr/bin/env python3
"""Protocol-only framing for the cyclotomic TT CATVM boundary."""

from __future__ import annotations

import json


REQUEST_BYTES = 1024
RESPONSE_BYTES = 4096


def fail(message: str) -> None:
    raise RuntimeError(message)


def fixed_packet(payload: dict[str, object], size: int) -> bytes:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    if len(encoded) > size:
        fail("CATVM cyclotomic packet overflow")
    return encoded + b" " * (size - len(encoded))


def parse_packet(
    packet: bytes, expected_bytes: int
) -> dict[str, object]:
    if len(packet) != expected_bytes:
        fail("CATVM cyclotomic packet length invalid")
    value = json.loads(packet.rstrip(b" ").decode("ascii"))
    if not isinstance(value, dict):
        fail("CATVM cyclotomic packet type invalid")
    return value
