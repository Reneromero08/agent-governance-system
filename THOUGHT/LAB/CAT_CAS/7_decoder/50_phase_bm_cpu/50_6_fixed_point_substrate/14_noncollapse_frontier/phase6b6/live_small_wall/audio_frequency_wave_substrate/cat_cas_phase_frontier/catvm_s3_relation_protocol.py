#!/usr/bin/env python3
"""Fixed binary protocol for the bounded S3 relation CATVM."""

from __future__ import annotations

import struct


MAGIC = 0x53334341
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

STATUS_OK = 0
STATUS_DENIED = 1

BOUNDARY_VALID = 1
RESTORED = 2
REUSE_FLAG = 4
SNAPSHOT_RELOADED = 8


def request(command: int, generation: int, depth: int, family: int, nonce: int) -> bytes:
    packed = (depth & 0xFFFF) | ((family & 0xFFFF) << 16)
    return REQUEST.pack(MAGIC, command, generation, packed, nonce)


def response(payload: bytes) -> dict[str, int]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid S3 CATVM response size")
    magic, status, command, generation, boundary, flags, receipt, resource = RESPONSE.unpack(payload)
    if magic != MAGIC:
        raise RuntimeError("invalid S3 CATVM response magic")
    return {
        "status": status,
        "command": command,
        "generation": generation,
        "boundary": boundary,
        "flags": flags,
        "receipt": receipt,
        "resource": resource,
    }
