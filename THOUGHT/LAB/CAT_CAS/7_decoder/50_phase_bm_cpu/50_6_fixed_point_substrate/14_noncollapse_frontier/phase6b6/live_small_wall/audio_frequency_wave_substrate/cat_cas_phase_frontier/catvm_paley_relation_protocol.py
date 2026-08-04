#!/usr/bin/env python3
"""Fixed binary protocol for the atomic Paley relation CATVM service."""

from __future__ import annotations

import struct


MAGIC = 0x43505241
REQUEST = struct.Struct("<IIIIQ")
RESPONSE = struct.Struct("<IIIIiQQQ")

RUN = 1
REUSE = 2
PROJECT_HIDDEN = 3
MISSING_INVERSE = 4
WRONG_INVERSE = 5
REORDERED_INVERSE = 6
NULL_CARRIER = 7
SNAPSHOT = 8
STOP = 9

STATUS_OK = 0
STATUS_DENIED = 1
STATUS_ERROR = 2

BOUNDARY_VALID = 1
RESTORED = 2
REUSE_FLAG = 4


def request(command: int, generation: int, topology: int, family: int, nonce: int) -> bytes:
    return REQUEST.pack(MAGIC, command, generation, (topology & 0xFFFF) | (family << 16), nonce)


def response(payload: bytes) -> dict[str, int]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid Paley CATVM response size")
    magic, status, command, generation, boundary, flags, receipt, resource = RESPONSE.unpack(payload)
    if magic != MAGIC:
        raise RuntimeError("invalid Paley CATVM response magic")
    return {
        "status": status,
        "command": command,
        "generation": generation,
        "boundary": boundary,
        "flags": flags,
        "receipt": receipt,
        "resource": resource,
    }
