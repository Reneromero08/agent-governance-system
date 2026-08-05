#!/usr/bin/env python3
"""Fixed binary protocol for the bounded Rotor-6 open-momentum CATVM."""

from __future__ import annotations

import struct


MAGIC = 0x4D4F4341
REQUEST = struct.Struct("<IIIIQ")
RESPONSE = struct.Struct("<IIIIiQQQ")

RUN_INPLACE = 1
REUSE_INPLACE = 2
PROJECT_HIDDEN_DURING_FORWARD = 3
MISSING_INVERSE = 4
WRONG_INVERSE = 5
REORDERED_INVERSE = 6
NULL_CARRIER = 7
RUN_SNAPSHOT = 8
STOP = 9
EARLY_RESPONSE_DURING_FORWARD = 10
WRONG_TYPE_DURING_FORWARD = 11
WRONG_OWNER_DURING_FORWARD = 12
PING = 13
NOOP = 14
WRONG_REFLECTION_INVERSE = 15

STATUS_OK = 0
STATUS_DENIED = 1

BOUNDARY_VALID = 1
RESTORED = 2
REUSE_FLAG = 4
SNAPSHOT_RELOADED = 8
CONTROL_DISCRIMINATED = 16


def request(command: int, generation: int, family: int, nonce: int) -> bytes:
    return REQUEST.pack(MAGIC, command, generation, family, nonce)


def response(payload: bytes) -> dict[str, int]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid open-momentum CATVM response size")
    (
        magic,
        status,
        command,
        generation,
        boundary,
        flags,
        receipt,
        resource,
    ) = RESPONSE.unpack(payload)
    if magic != MAGIC:
        raise RuntimeError("invalid open-momentum CATVM response magic")
    return {
        "status": status,
        "command": command,
        "generation": generation,
        "boundary": boundary,
        "flags": flags,
        "receipt": receipt,
        "resource": resource,
    }
