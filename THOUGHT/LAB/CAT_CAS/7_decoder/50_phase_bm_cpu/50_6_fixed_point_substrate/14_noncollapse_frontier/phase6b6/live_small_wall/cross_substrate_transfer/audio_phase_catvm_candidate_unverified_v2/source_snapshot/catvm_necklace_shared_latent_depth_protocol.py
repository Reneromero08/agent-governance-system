#!/usr/bin/env python3
"""Wire protocol for topology-rematerialized shared-latent CATVM."""

from __future__ import annotations

import struct


MAGIC = 0x43564450
REQUEST = struct.Struct("<IIIIQQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQ")

INITIALIZE = 1
BEGIN = 2
PROJECT = 3
CONTINUE = 4
REUSE = 5
MISSING_INVERSE = 6
REORDERED_INVERSE = 7
WRONG_INVERSE_VARIANT = 8
WRONG_OWNER = 9
NULL_CARRIER = 10
SNAPSHOT = 11
STOP = 12

STATUS_OK = 0
STATUS_DENIED = 1
STATUS_ERROR = 2

BOUNDARY_VALID = 1
RESTORED = 2
STAGE_RESIDENT = 4
REUSE_FLAG = 8


def parameter(depth: int, variant: int) -> int:
    if not (0 <= depth <= 0xFFFF and 0 <= variant <= 0xFFFF):
        raise ValueError("depth/variant exceeds wire parameter")
    return depth | (variant << 16)


def request(
    command: int,
    generation: int,
    depth: int,
    variant: int,
    lease: int,
    nonce: int,
) -> bytes:
    return REQUEST.pack(
        MAGIC,
        command,
        generation,
        parameter(depth, variant),
        lease,
        nonce,
    )


def response(payload: bytes) -> dict[str, object]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid depth CATVM response size")
    fields = RESPONSE.unpack(payload)
    if fields[0] != MAGIC:
        raise RuntimeError("invalid depth CATVM response magic")
    return {
        "status": fields[1],
        "command": fields[2],
        "generation": fields[3],
        "flags": fields[4],
        "lease": fields[5],
        "receipt": fields[6],
        "boundary": list(fields[7:14]),
        "restoration_error": fields[14],
        "norm_error": fields[15],
        "native_operations": fields[16],
    }
