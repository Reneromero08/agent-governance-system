#!/usr/bin/env python3
"""Fixed CATVM protocol for the bosonic Givens phase backend."""

from __future__ import annotations

import struct

MAGIC = 0x43564247
REQUEST = struct.Struct("<IIQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQQ")

INITIALIZE = 1
DIRECT_BEGIN = 2
SNAPSHOT_BEGIN = 3
BEGIN_PRIMARY = 4
PROJECT_INTERMEDIATE = 5
CONTINUE_PRIMARY = 6
REUSE = 7
MISSING_INVERSE = 8
WRONG_INVERSE = 9
REORDERED_INVERSE = 10
NULL_CARRIER = 11
STOP = 12
DIRECT_CONTINUE = 13
SNAPSHOT_CONTINUE = 14

STATUS_OK = 0
STATUS_DENIED = 1

BOUNDARY_VALID = 1
RESTORED = 2
STAGE_RESIDENT = 4
SNAPSHOT_RELOAD = 8
REUSE_FLAG = 16


def request(command: int, nonce: int) -> bytes:
    return REQUEST.pack(MAGIC, command, nonce)


def response(payload: bytes) -> dict[str, object]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid CATVM response size")
    fields = RESPONSE.unpack(payload)
    if fields[0] != MAGIC:
        raise RuntimeError("invalid CATVM response magic")
    return {
        "status": fields[1],
        "command": fields[2],
        "generation": fields[3],
        "flags": fields[4],
        "receipt": fields[5],
        "state_hash": fields[6],
        "boundary": list(fields[7:14]),
        "restoration_error": fields[14],
        "norm_error": fields[15],
        "native_operations": fields[16],
        "snapshot_reload_bytes": fields[17],
    }
