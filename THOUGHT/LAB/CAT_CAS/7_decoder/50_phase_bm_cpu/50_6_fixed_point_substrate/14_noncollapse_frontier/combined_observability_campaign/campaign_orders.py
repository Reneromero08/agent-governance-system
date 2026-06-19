#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import random
from typing import Any

MODES = ("basis", "rotation", "residual", "mini")
PHASES = (0, 2, 4, 6)
ROUTES = ("v4s5", "v2s3")
SEEDS = tuple(range(6))
FAMILIES = ("real", "wrong", "pseudo", "order_sham", "silent", "scramble")


def keyed_rng(*parts: object) -> random.Random:
    key = "|".join(str(part) for part in parts).encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def frozen_orders(nbin: int = 12) -> dict[str, list[int]]:
    fwd = list(range(nbin))
    result = {"FWD": fwd, "REV": list(reversed(fwd))}
    for name in ("RND1", "RND2"):
        rng = keyed_rng("phase6b5e", name, nbin)
        for _ in range(10000):
            candidate = fwd.copy()
            rng.shuffle(candidate)
            distances = [sum(a != b for a, b in zip(candidate, old)) for old in result.values()]
            if candidate not in result.values() and min(distances) >= 10:
                result[name] = candidate
                break
        else:
            raise RuntimeError(f"unable to freeze {name}")
    return result


def partition(seed: int) -> str:
    if seed <= 2:
        return "train"
    if seed == 3:
        return "validation"
    if seed == 4:
        return "stress"
    return "test"


def permutation(key: str, nbin: int = 12) -> list[int]:
    values = list(range(nbin))
    keyed_rng(key).shuffle(values)
    return values


def symbol(session_id: str, order_name: str, orders: dict[str, list[int]], family: str, trial: int) -> dict[str, Any]:
    rng = keyed_rng(session_id, order_name, family, trial)
    actual = rng.choice(MODES)
    declared = actual
    declared_order = order_name
    code_perm = list(range(12))
    if family == "wrong":
        declared = rng.choice([mode for mode in MODES if mode != actual])
    elif family == "pseudo":
        declared = rng.choice(MODES)
        code_perm = permutation(f"pseudo|{session_id}|{order_name}|{trial}")
    elif family == "order_sham":
        names = tuple(orders)
        declared_order = names[(names.index(order_name) + 1) % len(names)]
    return {
        "family": family,
        "trial": trial,
        "actual_mode": actual,
        "declared_mode": declared,
        "theta_idx": rng.randrange(8),
        "executed_tone_order": order_name,
        "declared_tone_order": declared_order,
        "tone_execution_indices": orders[order_name],
        "codeword_bin_permutation": code_perm,
        "drive_on": family != "silent",
        "shared_schedule": family != "scramble",
    }


def gauge_symbols(orders: dict[str, list[int]]) -> list[dict[str, Any]]:
    return [
        {
            "family": "gauge_preamble",
            "actual_mode": mode,
            "declared_mode": mode,
            "theta_idx": theta,
            "executed_tone_order": "FWD",
            "declared_tone_order": "FWD",
            "tone_execution_indices": orders["FWD"],
            "codeword_bin_permutation": list(range(12)),
            "drive_on": True,
            "shared_schedule": True,
        }
        for mode in MODES
        for theta in (0, 4)
    ]
