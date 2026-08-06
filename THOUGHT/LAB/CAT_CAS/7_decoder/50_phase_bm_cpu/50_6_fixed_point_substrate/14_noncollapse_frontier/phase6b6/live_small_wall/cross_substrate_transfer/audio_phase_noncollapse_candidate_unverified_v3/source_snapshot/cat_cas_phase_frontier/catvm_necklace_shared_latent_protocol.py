#!/usr/bin/env python3
"""Fixed wire protocol for the shared-latent necklace CATVM."""

from __future__ import annotations

import struct


MAGIC = 0x43564C50
REQUEST = struct.Struct("<IIIIQQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQ")

INITIALIZE = 1
BEGIN = 2
PROJECT = 3
CONTINUE = 4
REUSE = 5
MISSING_INVERSE = 6
REORDERED_INVERSE = 7
WRONG_SEMANTIC = 8
WRONG_TYPE = 9
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


def request(
    command: int,
    generation: int,
    lease: int,
    nonce: int,
) -> bytes:
    return REQUEST.pack(
        MAGIC,
        command,
        generation,
        0,
        lease,
        nonce,
    )


def response(payload: bytes) -> dict[str, object]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid shared-latent response size")
    fields = RESPONSE.unpack(payload)
    if fields[0] != MAGIC:
        raise RuntimeError("invalid shared-latent response magic")
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
