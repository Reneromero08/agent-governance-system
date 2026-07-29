#!/usr/bin/env python3
"""Fixed-ABI public descriptor protocol for the bounded two-port CATVM."""

from __future__ import annotations

import struct


MAGIC = 0x43564C50
REQUEST = struct.Struct("<IIIIQQ")
RESPONSE = struct.Struct("<IIIIIQQ7dddQ")

INITIALIZE = 1
BEGIN = 2
PROJECT = 3
CONTINUE = 4
MISSING_INVERSE = 6
REORDERED_INVERSE = 7
WRONG_SEMANTIC = 8
WRONG_TYPE = 9
NULL_CARRIER = 10
SNAPSHOT = 11
STOP = 12
WRONG_OWNER_A = 13
WRONG_OWNER_B = 14
UNDERMERGE = 15
DUPLICATE_PORT = 16
STALE_INTERNAL_GENERATION = 17
WRONG_INTERNAL_LEASE = 18
DECLARE = 19
APPEND = 20
SEAL = 21
STALE_EPOCH = 22
WRONG_CHECKSUM = 23
WRONG_SLOT = 24
STALE_BOUND_GENERATION = 25

STATUS_OK = 0
STATUS_DENIED = 1
STATUS_ERROR = 2

BOUNDARY_VALID = 1
RESTORED = 2
STAGE_RESIDENT = 4
REUSE = 8

FEATURES = {"collision": 1, "cyclic_separation": 2}
SCOPES = {"port_a": 1, "port_b": 2, "joint": 3}
AXES = {"x": 1, "y": 2, "z": 3, "controlled_phase": 4}


def request(
    command: int,
    generation: int,
    lease: int,
    nonce: int,
    reserved: int = 0,
) -> bytes:
    return REQUEST.pack(
        MAGIC,
        command,
        generation,
        reserved,
        lease,
        nonce,
    )


def response(payload: bytes) -> dict[str, object]:
    if len(payload) != RESPONSE.size:
        raise RuntimeError("invalid descriptor response size")
    fields = RESPONSE.unpack(payload)
    if fields[0] != MAGIC:
        raise RuntimeError("invalid descriptor response magic")
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


def declare_control(slot: int, count: int) -> int:
    if not 0 <= slot < 256 or not 0 <= count < 256:
        raise ValueError("declare field outside uint8")
    return slot | (count << 8)


def pack_module(
    module: dict[str, object],
    slot: int,
    index: int,
) -> int:
    feature = FEATURES[str(module["feature"])]
    scope = SCOPES[str(module["scope"])]
    axis = AXES[str(module["axis"])]
    separation = int(module["separation"])
    strength = int(module["strength"])
    chirp = int(module["chirp"])
    if not 0 <= slot <= 3 or not 0 <= index <= 7:
        raise ValueError("descriptor slot/index outside packed range")
    if not 0 <= separation <= 15:
        raise ValueError("descriptor separation outside packed range")
    if not 0 <= strength <= 31 or not 0 <= chirp <= 31:
        raise ValueError("descriptor strength/chirp outside packed range")
    return (
        feature
        | (scope << 2)
        | (axis << 4)
        | (separation << 7)
        | (strength << 11)
        | (chirp << 16)
        | (slot << 21)
        | (index << 23)
    )


def semantic_word(packed: int) -> int:
    return packed & 0x001FFFFF
