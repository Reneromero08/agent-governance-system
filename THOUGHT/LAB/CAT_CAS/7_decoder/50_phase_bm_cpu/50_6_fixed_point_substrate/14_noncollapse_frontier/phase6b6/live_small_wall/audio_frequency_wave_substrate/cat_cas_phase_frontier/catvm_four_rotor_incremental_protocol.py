#!/usr/bin/env python3
"""Protocol-only framing for the incremental four-rotor CATVM."""

from __future__ import annotations

import json


PROTOCOL_VERSION = 1
REQUEST_BYTES = 1024
RESPONSE_BYTES = 8192


def fail(message: str) -> None:
    raise RuntimeError(message)


def fixed_packet(payload: dict[str, object], size: int) -> bytes:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    if len(encoded) > size:
        fail("CATVM four-rotor packet overflow")
    return encoded + b" " * (size - len(encoded))


def parse_packet(
    packet: bytes, expected_bytes: int
) -> dict[str, object]:
    if len(packet) != expected_bytes:
        fail("CATVM four-rotor packet length invalid")
    value = json.loads(packet.rstrip(b" ").decode("ascii"))
    if not isinstance(value, dict):
        fail("CATVM four-rotor packet type invalid")
    if value.get("version") != PROTOCOL_VERSION:
        fail("CATVM four-rotor protocol version invalid")
    return value


def request(command: str, transaction_id: int, **fields: object) -> dict[str, object]:
    return {
        "version": PROTOCOL_VERSION,
        "command": command,
        "transaction_id": transaction_id,
        **fields,
    }
