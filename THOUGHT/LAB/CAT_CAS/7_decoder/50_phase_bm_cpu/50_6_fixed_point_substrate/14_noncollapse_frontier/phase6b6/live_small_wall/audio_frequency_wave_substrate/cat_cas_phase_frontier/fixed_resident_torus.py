"""Mutable CAT_CAS frontier: fixed-resident reversible phase path process.

This is development code, not a frozen evidence package.

For an odd prime cyclic width M and nonzero shift w, the simultaneous phase
map

    y[s] = x[s] * x[s-w]

is a bijection on p-th-root relations for every odd phase modulus p.  The map
therefore needs one resident torus rather than a fresh O(M) layer per program
step.  Its inverse is derived from the same public shift and performs only
complex phase products, conjugations, and fixed integer phase powers.

The sole discrete conversion remains ``collapse_boundary``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ENGINE_VERSION = "0.1.0-development"
DEFAULT_PHASE_MODULI = (3, 5, 7, 11, 13)
ROOT_DISTANCE_MAX = 2.0e-9
ROOT_MARGIN_MIN = 0.15
RESTORATION_MAX = 2.0e-11
PHASE_LOCK_ITERATIONS = 1


def canonical_bytes(document: Any) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, math.isqrt(value) + 1, 2))


def _pairwise_coprime(values: Iterable[int]) -> bool:
    items = tuple(values)
    return all(
        math.gcd(items[left], items[right]) == 1
        for left in range(len(items))
        for right in range(left + 1, len(items))
    )


@dataclass(frozen=True)
class PhaseProgram:
    name: str
    residue_modulus: int
    phase_moduli: tuple[int, ...]
    weights: tuple[int, ...]
    target_residue: int

    def validate(self) -> None:
        if not self.name or len(self.name) > 128:
            raise ValueError("program name is missing or too long")
        if not _is_prime(self.residue_modulus) or self.residue_modulus == 2:
            raise ValueError("residue modulus must be an odd prime")
        if not self.phase_moduli:
            raise ValueError("phase moduli are required")
        if (
            any(not _is_prime(value) or value == 2 for value in self.phase_moduli)
            or not _pairwise_coprime(self.phase_moduli)
        ):
            raise ValueError("phase moduli must be pairwise-coprime odd primes")
        if not self.weights:
            raise ValueError("at least one public shift is required")
        if any(
            not isinstance(weight, int)
            or not 0 < weight < self.residue_modulus
            for weight in self.weights
        ):
            raise ValueError("every public shift must be nonzero and in range")
        if not 0 <= self.target_residue < self.residue_modulus:
            raise ValueError("target residue is outside the torus")

    def document(self) -> dict[str, Any]:
        return {
            "geometry": {
                "phase_moduli": list(self.phase_moduli),
                "residue_modulus": self.residue_modulus,
            },
            "name": self.name,
            "public_shifts": list(self.weights),
            "target_residue": self.target_residue,
        }

    @property
    def fingerprint(self) -> str:
        self.validate()
        return sha256_bytes(canonical_bytes(self.document()))


def carrier_shape(program: PhaseProgram) -> tuple[int, int, int]:
    program.validate()
    return len(program.phase_moduli), 2, program.residue_modulus


def borrowed_carrier(program: PhaseProgram, identity: int = 0) -> np.ndarray:
    """Construct a deterministic noncanonical carrier for development probes."""

    phase_count, channel_count, residue_count = carrier_shape(program)
    p = np.arange(phase_count, dtype=np.float64)[:, None, None]
    residue = np.arange(residue_count, dtype=np.float64)[None, None, :]
    phase = (
        0.173
        + 0.071 * p
        + 0.037 * residue
        + 0.023 * np.sin(0.17 * residue + identity * 0.07)
    )
    amplitude = (
        0.71
        + 0.08 * np.cos(0.13 * residue + 0.05 * p)
    )
    rail = amplitude * np.exp(1j * phase)
    result = np.repeat(rail, channel_count, axis=1)
    return np.asarray(result, dtype=np.complex128)


def _as_carrier(program: PhaseProgram, carrier: np.ndarray) -> np.ndarray:
    candidate = np.asarray(carrier, dtype=np.complex128)
    if candidate.shape != carrier_shape(program):
        raise ValueError("borrowed carrier has the wrong fixed geometry")
    if (
        not np.all(np.isfinite(candidate))
        or float(np.min(np.abs(candidate))) <= 1.0e-14
    ):
        raise ValueError("borrowed carrier must be finite and nonzero")
    return np.array(candidate, copy=True)


def _relations(carrier: np.ndarray) -> np.ndarray:
    relation = carrier[:, 1, :] * np.conjugate(carrier[:, 0, :])
    magnitude = np.abs(relation)
    if float(np.min(magnitude)) <= 1.0e-15:
        raise ValueError("carrier relation collapsed")
    return relation / magnitude


def _initial_relations(program: PhaseProgram) -> np.ndarray:
    result = np.ones(
        (len(program.phase_moduli), program.residue_modulus),
        dtype=np.complex128,
    )
    for phase_index, modulus in enumerate(program.phase_moduli):
        result[phase_index, 0] = np.exp(2j * math.pi / modulus)
    return result


def _seed(program: PhaseProgram, carrier: np.ndarray) -> np.ndarray:
    working = _as_carrier(program, carrier)
    if float(np.max(np.abs(_relations(working) - 1.0))) > 1.0e-12:
        raise ValueError("borrowed twin rails lost their common-mode relation")
    working[:, 1, :] *= _initial_relations(program)
    return working


def _lock(row: np.ndarray, modulus: int) -> np.ndarray:
    """Fixed p-fold phase dynamic; it does not label or select a root."""

    locked = row / np.abs(row)
    for _ in range(PHASE_LOCK_ITERATIONS):
        locked *= np.exp(-1j * np.imag(locked**modulus) / float(modulus))
        locked /= np.abs(locked)
    return locked


def _lock_rows(
    rows: np.ndarray, moduli: tuple[int, ...]
) -> np.ndarray:
    locked = rows / np.abs(rows)
    powers = np.asarray(moduli, dtype=np.int64)[:, None]
    divisors = np.asarray(moduli, dtype=np.float64)[:, None]
    for _ in range(PHASE_LOCK_ITERATIONS):
        locked *= np.exp(-1j * np.imag(locked**powers) / divisors)
        locked /= np.abs(locked)
    return locked


def _forward_relation(
    relation: np.ndarray, shift: int, modulus: int
) -> np.ndarray:
    return _lock(relation * np.roll(relation, shift), modulus)


@lru_cache(maxsize=None)
def _inverse_kernel(width: int, shift: int, modulus: int) -> np.ndarray:
    """Return public circulant inverse coefficients for I + P^w."""

    inverse_two = pow(2, -1, modulus)
    kernel = np.zeros((width, width), dtype=np.int64)
    for output in range(width):
        for distance in range(width):
            source = (output - distance * shift) % width
            sign = 1 if distance % 2 == 0 else -1
            kernel[output, source] = (inverse_two * sign) % modulus
    kernel.setflags(write=False)
    return kernel


@lru_cache(maxsize=None)
def _inverse_kernels(
    width: int, shift: int, moduli: tuple[int, ...]
) -> np.ndarray:
    kernels = np.stack(
        tuple(
            _inverse_kernel(width, shift, modulus)
            for modulus in moduli
        ),
        axis=0,
    )
    kernels.setflags(write=False)
    return kernels


def _inverse_relation(
    relation: np.ndarray, shift: int, modulus: int
) -> np.ndarray:
    """Invert y[s] = x[s] x[s-w] on one odd cyclic phase group.

    Along the unique cycle ``0,w,2w,...`` the inverse is

        x = (1/2) * (I - P^w + P^(2w) - ... + P^((M-1)w)) * y.

    The fixed public coefficient network is applied as complex phase powers and
    products. It never converts a relation to an exponent label.
    """

    width = relation.size
    kernel = _inverse_kernel(width, shift, modulus)
    recovered = np.prod(relation[None, :] ** kernel, axis=1)
    return _lock(recovered, modulus)


def _forward(carrier: np.ndarray, shift: int, moduli: tuple[int, ...]) -> None:
    before = _relations(carrier)
    after = _lock_rows(
        before * np.roll(before, shift, axis=1),
        moduli,
    )
    carrier[:, 1, :] = after * carrier[:, 0, :]


def _inverse(carrier: np.ndarray, shift: int, moduli: tuple[int, ...]) -> None:
    before = _relations(carrier)
    kernels = _inverse_kernels(before.shape[1], shift, moduli)
    after = _lock_rows(
        np.prod(before[:, None, :] ** kernels, axis=2),
        moduli,
    )
    carrier[:, 1, :] = after * carrier[:, 0, :]


@dataclass(frozen=True)
class PhaseExecution:
    program: PhaseProgram
    result_latch: np.ndarray
    restored_carrier: np.ndarray
    displacement_l2: float
    restoration_max_abs: float
    resident_complex_cells: int
    public_program_words: int
    history_factor_count: int
    inverse_mode: str


def execute_catalytic(
    program: PhaseProgram,
    carrier: np.ndarray,
    *,
    forward_mode: str = "correct",
    inverse_mode: str = "correct",
) -> PhaseExecution:
    """Borrow, evolve, latch, reverse, and restore one fixed torus."""

    program.validate()
    borrowed = _as_carrier(program, carrier)
    working = _seed(program, borrowed)
    if forward_mode == "correct":
        forward_shifts = program.weights
    elif forward_mode == "remove_last":
        forward_shifts = program.weights[:-1]
    elif forward_mode == "scramble_geometry":
        forward_shifts = tuple(
            (weight * weight + 1) % program.residue_modulus or 1
            for weight in program.weights
        )
    elif forward_mode == "removed":
        forward_shifts = ()
    else:
        raise ValueError("unknown forward mode")

    for shift in forward_shifts:
        _forward(working, shift, program.phase_moduli)
    displacement = float(np.linalg.norm(working - borrowed))
    latch = np.array(
        _relations(working)[:, program.target_residue], copy=True
    )

    if inverse_mode == "correct":
        inverse_shifts = tuple(reversed(forward_shifts))
    elif inverse_mode == "wrong_program":
        inverse_shifts = tuple(
            reversed(
                tuple(
                    (shift + 1) % program.residue_modulus or 1
                    for shift in forward_shifts
                )
            )
        )
    elif inverse_mode == "omitted":
        inverse_shifts = ()
    else:
        raise ValueError("unknown inverse mode")
    for shift in inverse_shifts:
        _inverse(working, shift, program.phase_moduli)
    if inverse_mode != "omitted":
        working[:, 1, :] *= np.conjugate(_initial_relations(program))

    return PhaseExecution(
        program=program,
        result_latch=latch,
        restored_carrier=working,
        displacement_l2=displacement,
        restoration_max_abs=float(np.max(np.abs(working - borrowed))),
        resident_complex_cells=int(working.size),
        public_program_words=len(program.weights),
        history_factor_count=0,
        inverse_mode=inverse_mode,
    )


@dataclass(frozen=True)
class BoundaryResult:
    valid: bool
    residues: tuple[int, ...] | None
    count_mod_crt: int | None
    root_distances: tuple[float, ...]
    root_margins: tuple[float, ...]


def _crt(residues: tuple[int, ...], moduli: tuple[int, ...]) -> int:
    product = math.prod(moduli)
    total = 0
    for residue, modulus in zip(residues, moduli):
        partial = product // modulus
        total += residue * partial * pow(partial, -1, modulus)
    return total % product


def collapse_boundary(
    program: PhaseProgram, result_latch: np.ndarray
) -> BoundaryResult:
    """The sole phase-to-discrete conversion."""

    latch = np.asarray(result_latch, dtype=np.complex128)
    if latch.shape != (len(program.phase_moduli),):
        raise ValueError("result latch has the wrong geometry")
    residues: list[int] = []
    distances: list[float] = []
    margins: list[float] = []
    for value, modulus in zip(latch, program.phase_moduli):
        phasor = value / abs(value)
        roots = np.exp(2j * math.pi * np.arange(modulus) / modulus)
        root_distances = np.abs(roots - phasor)
        order = np.argsort(root_distances, kind="stable")
        residues.append(int(order[0]))
        distances.append(float(root_distances[order[0]]))
        margins.append(
            float(root_distances[order[1]] - root_distances[order[0]])
        )
    valid = bool(
        max(distances, default=math.inf) <= ROOT_DISTANCE_MAX
        and min(margins, default=-math.inf) >= ROOT_MARGIN_MIN
    )
    decoded = tuple(residues) if valid else None
    return BoundaryResult(
        valid=valid,
        residues=decoded,
        count_mod_crt=(
            None if decoded is None else _crt(decoded, program.phase_moduli)
        ),
        root_distances=tuple(distances),
        root_margins=tuple(margins),
    )


def maximum_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(left, dtype=np.complex128)
                - np.asarray(right, dtype=np.complex128)
            )
        )
    )


def source_no_smuggle() -> dict[str, Any]:
    source = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    native_names = {
        "_forward_relation",
        "_inverse_relation",
        "_forward",
        "_inverse",
        "execute_catalytic",
    }
    forbidden = {
        "argmax",
        "expected_result",
        "oracle",
        "solve",
        "enumerate_candidates",
    }
    calls: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in native_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        calls.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):
                        calls.add(child.func.attr)
    hits = sorted(calls & forbidden)
    return {
        "complete_path_modes": 0,
        "decoded_feedback": False,
        "forbidden_call_hits": hits,
        "history_factor_count": 0,
        "passed": not hits,
    }


def engine_fingerprint() -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "engine_version": ENGINE_VERSION,
                "source_sha256": sha256_bytes(Path(__file__).read_bytes()),
            }
        )
    )


if not source_no_smuggle()["passed"]:
    raise RuntimeError("fixed-resident phase engine contains a forbidden path")
