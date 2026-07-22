from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SITE_COUNT = 5
SAMPLE_RATE_HZ = 48_000
SAMPLE_COUNT = 256
MODE_COUNT = 1 << SITE_COUNT
ACTIVE_BINS = np.asarray(tuple(range(8, 8 + MODE_COUNT)), dtype=np.int64)
GEOMETRY_SAMPLES = np.asarray(
    tuple((11 + 7 * index) % SAMPLE_COUNT for index in range(MODE_COUNT)),
    dtype=np.int64,
)
RESTORATION_MAX = 2.0e-12
REUSE_RESPONSE_MAX = 1.0e-12
WRONG_RESTORATION_MIN = 1.0e-3
DISPLACEMENT_MIN = 1.0
MATERIALITY_MIN = 1.0e-3
CLAIM_CEILING = "BOUNDED_SOFTWARE_RECURSIVE_SPECTRAL_PHASE_REFERENCE_ONLY"


@dataclass(frozen=True)
class SpectralPhaseLaw:
    relation_phase_scale: float = math.pi / 64.0
    softmax_beta: float = 6.0
    unique_gap_min: float = 0.5
    response_coherence_min: float = 0.75
    geometry_coherence_min: float = 0.999999999
    maximum_abs_coupling: float = 2.0
    maximum_abs_field: float = 2.0

    def validate(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("spectral phase law contains a nonfinite value")
        if self.relation_phase_scale <= 0.0:
            raise ValueError("relation phase scale must be positive")
        if self.softmax_beta <= 0.0:
            raise ValueError("softmax beta must be positive")
        if self.unique_gap_min <= 0.0:
            raise ValueError("unique gap minimum must be positive")
        if not 0.0 < self.response_coherence_min <= 1.0:
            raise ValueError("response coherence minimum outside (0, 1]")
        if not 0.0 < self.geometry_coherence_min <= 1.0:
            raise ValueError("geometry coherence minimum outside (0, 1]")
        if min(self.maximum_abs_coupling, self.maximum_abs_field) <= 0.0:
            raise ValueError("problem bounds must be positive")
        maximum_penalty = (
            SITE_COUNT * (SITE_COUNT - 1) * self.maximum_abs_coupling
            + 2.0 * SITE_COUNT * self.maximum_abs_field
        )
        if self.relation_phase_scale * maximum_penalty >= math.pi:
            raise ValueError("phase penalty can wrap inside the frozen problem bounds")


DEFAULT_LAW = SpectralPhaseLaw()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def metric(value: float) -> float:
    return float(f"{float(value):.12g}")


def borrowed_carrier() -> np.ndarray:
    times = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ
    sample_index = np.arange(SAMPLE_COUNT, dtype=np.float64)
    phase = 0.173 * sample_index + 0.23 * np.sin(
        2.0 * math.pi * 41.0 * times
    )
    amplitude = 0.65 + 0.25 * np.cos(2.0 * math.pi * 29.0 * times)
    carrier = np.asarray(amplitude * np.exp(1j * phase), dtype=np.complex128)
    if carrier.shape != (SAMPLE_COUNT,) or float(np.min(np.abs(carrier))) <= 0.0:
        raise ValueError("borrowed carrier is outside the frozen envelope")
    return carrier


def problem_identity_sha256(coupling: np.ndarray, field: np.ndarray) -> str:
    return sha256_bytes(
        canonical_bytes(
            {
                "coupling_matrix_J": np.asarray(coupling, dtype=np.float64).tolist(),
                "field_vector_h": np.asarray(field, dtype=np.float64).tolist(),
            }
        )
    )


def recursive_antipodal_phase_modes(site_count: int = SITE_COUNT) -> np.ndarray:
    """Construct the complete recursive S1 phase tree without decoded spin state."""
    modes = np.ones((site_count, 1), dtype=np.complex128)
    for site in range(site_count):
        antipodal = np.array(modes, copy=True)
        antipodal[site] *= np.exp(1j * math.pi)
        modes = np.concatenate((modes, antipodal), axis=1)
    if modes.shape != (site_count, 1 << site_count):
        raise RuntimeError("recursive phase tree has the wrong shape")
    return modes


PHASE_MODES = recursive_antipodal_phase_modes()


def validate_problem(
    coupling: np.ndarray, field: np.ndarray, law: SpectralPhaseLaw
) -> tuple[np.ndarray, np.ndarray]:
    law.validate()
    coupling = np.asarray(coupling, dtype=np.float64)
    field = np.asarray(field, dtype=np.float64)
    if coupling.shape != (SITE_COUNT, SITE_COUNT) or field.shape != (SITE_COUNT,):
        raise ValueError("five-site problem shape required")
    if not np.all(np.isfinite(coupling)) or not np.all(np.isfinite(field)):
        raise ValueError("problem must be finite")
    if not np.allclose(coupling, coupling.T, atol=0.0, rtol=0.0):
        raise ValueError("coupling matrix must be exactly symmetric")
    if not np.allclose(np.diag(coupling), 0.0, atol=0.0, rtol=0.0):
        raise ValueError("coupling diagonal must be exactly zero")
    if float(np.max(np.abs(coupling))) > law.maximum_abs_coupling:
        raise ValueError("coupling exceeds frozen spectral phase bound")
    if float(np.max(np.abs(field))) > law.maximum_abs_field:
        raise ValueError("field exceeds frozen spectral phase bound")
    return np.array(coupling, copy=True), np.array(field, copy=True)


def canonical_geometry() -> np.ndarray:
    times = np.arange(SAMPLE_COUNT, dtype=np.float64) / SAMPLE_RATE_HZ
    beams: list[np.ndarray] = []
    for site in range(SITE_COUNT):
        leaf_phase = (
            2.0 * math.pi * (47.0 + 5.0 * site) * times
        ) + (0.31 + 0.07 * site)
        middle_phase = (
            2.0 * math.pi * (233.0 + 13.0 * site) * times
        ) + (-0.22 + 0.05 * site)
        middle_phase = middle_phase + (0.31 + 0.035 * site) * np.sin(
            leaf_phase
        )
        root_phase = (
            2.0 * math.pi * (1237.0 + 71.0 * site) * times
        ) + (0.17 - 0.04 * site)
        root_phase = root_phase + (0.62 + 0.045 * site) * np.sin(
            middle_phase
        )
        beams.append(np.exp(1j * root_phase))
    geometry = np.asarray(beams, dtype=np.complex128)
    if geometry.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("canonical recursive geometry has the wrong shape")
    return geometry


def geometry_anchors(beams: np.ndarray) -> np.ndarray:
    beams = np.asarray(beams, dtype=np.complex128)
    if beams.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("recursive geometry has the wrong shape")
    selected = beams[:, GEOMETRY_SAMPLES]
    magnitudes = np.abs(selected)
    if float(np.min(magnitudes)) <= 0.0:
        raise ValueError("recursive geometry contains a zero phase anchor")
    return selected / magnitudes


def as_carrier_bank(borrowed: np.ndarray) -> np.ndarray:
    borrowed = np.asarray(borrowed, dtype=np.complex128)
    if borrowed.shape == (SAMPLE_COUNT,):
        borrowed = np.repeat(borrowed[np.newaxis, :], SITE_COUNT, axis=0)
    if borrowed.shape != (SITE_COUNT, SAMPLE_COUNT):
        raise ValueError("borrowed carrier bank has the wrong shape")
    if not np.all(np.isfinite(borrowed)) or float(np.min(np.abs(borrowed))) <= 0.0:
        raise ValueError("borrowed carrier must be finite and nonzero")
    return np.array(borrowed, copy=True)


def normalized_active(spectrum: np.ndarray) -> np.ndarray:
    active = np.asarray(spectrum[:, ACTIVE_BINS], dtype=np.complex128)
    magnitudes = np.abs(active)
    if float(np.min(magnitudes)) <= 1.0e-15:
        raise ValueError("active spectral mode collapsed")
    return active / magnitudes


def common_merit_phase(
    spectrum: np.ndarray, anchors: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    active = normalized_active(spectrum)
    expected = anchors * PHASE_MODES
    samples = active / expected
    means = np.mean(samples, axis=0)
    coherence = np.abs(means)
    if float(np.min(coherence)) <= 1.0e-15:
        raise ValueError("recursive phase relation lost its common carrier")
    return means / coherence, coherence


@dataclass(frozen=True)
class NativeExecution:
    borrowed: np.ndarray
    coupling: np.ndarray
    field: np.ndarray
    actual_beams: np.ndarray
    program_beams: np.ndarray
    displaced: np.ndarray
    seed_operator: np.ndarray
    relation_history: np.ndarray
    relation_labels: tuple[str, ...]
    displacement_l2: float
    law: SpectralPhaseLaw


@dataclass(frozen=True)
class BoundaryProjection:
    label: str
    responses: tuple[complex, ...]
    coherence: tuple[float, ...]
    raw_spins: tuple[int, ...]
    spins: tuple[int, ...] | None
    mode_penalties: tuple[float, ...]
    best_mode_index: int
    second_mode_gap: float
    best_mode_concentration: float
    minimum_geometry_coherence: float
    valid: bool

    def document(self) -> dict[str, Any]:
        return {
            "best_mode_concentration": metric(self.best_mode_concentration),
            "best_mode_index": self.best_mode_index,
            "coherence": [metric(value) for value in self.coherence],
            "label": self.label,
            "minimum_geometry_coherence": metric(self.minimum_geometry_coherence),
            "mode_penalties": [metric(value) for value in self.mode_penalties],
            "raw_spin_shadow": list(self.raw_spins),
            "responses": [
                {"imag": metric(value.imag), "real": metric(value.real)}
                for value in self.responses
            ],
            "second_mode_gap": metric(self.second_mode_gap),
            "spins": None if self.spins is None else list(self.spins),
            "valid": self.valid,
        }


def seed_recursive_spectral_tree(
    spectrum: np.ndarray,
    anchors: np.ndarray,
    *,
    enabled: bool,
) -> tuple[np.ndarray, np.ndarray]:
    current = np.array(spectrum, dtype=np.complex128, copy=True)
    operator = np.ones_like(current)
    if not enabled:
        return current, operator
    original = current[:, ACTIVE_BINS]
    if float(np.min(np.abs(original))) <= 1.0e-12:
        raise ValueError("borrowed carrier cannot support the frozen spectral modes")
    seed_amplitude = float(np.sqrt(np.mean(np.abs(original) ** 2)))
    target = seed_amplitude * anchors * PHASE_MODES
    operator[:, ACTIVE_BINS] = target / original
    current *= operator
    return current, operator


def relational_phase_operator(
    spectrum: np.ndarray,
    anchors: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    relation_enabled: np.ndarray,
    law: SpectralPhaseLaw,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    current = np.array(spectrum, dtype=np.complex128, copy=True)
    history: list[np.ndarray] = []
    labels: list[str] = []
    enabled = np.asarray(relation_enabled, dtype=np.bool_)
    if enabled.shape != (SITE_COUNT, SITE_COUNT):
        raise ValueError("relation enable map has the wrong shape")
    reference_anchors = geometry_anchors(canonical_geometry())
    active = normalized_active(current)
    for left in range(SITE_COUNT):
        for right in range(left + 1, SITE_COUNT):
            if not enabled[left, right] or coupling[left, right] == 0.0:
                continue
            relation = (
                active[left]
                * np.conjugate(active[right])
                * np.conjugate(reference_anchors[left])
                * reference_anchors[right]
            )
            alignment = np.real(relation)
            strength = float(coupling[left, right])
            penalty = abs(strength) - strength * alignment
            rotation = np.exp(-1j * law.relation_phase_scale * penalty)
            operator = np.ones_like(current)
            operator[:, ACTIVE_BINS] = rotation[np.newaxis, :]
            current *= operator
            history.append(operator)
            labels.append(f"pair_{left}_{right}")
            active = normalized_active(current)
    for site in range(SITE_COUNT):
        if field[site] == 0.0:
            continue
        merit, _ = common_merit_phase(current, reference_anchors)
        active = normalized_active(current)
        local_orientation = (
            active[site]
            * np.conjugate(reference_anchors[site])
            * np.conjugate(merit)
        )
        alignment = np.real(local_orientation)
        strength = float(field[site])
        penalty = abs(strength) - strength * alignment
        rotation = np.exp(-1j * law.relation_phase_scale * penalty)
        operator = np.ones_like(current)
        operator[:, ACTIVE_BINS] = rotation[np.newaxis, :]
        current *= operator
        history.append(operator)
        labels.append(f"field_{site}")
    if not history:
        history.append(np.ones_like(current))
        labels.append("identity")
    return current, np.stack(history), tuple(labels)


def execute_native_cycle(
    borrowed: np.ndarray,
    coupling: np.ndarray,
    field: np.ndarray,
    *,
    law: SpectralPhaseLaw = DEFAULT_LAW,
    program_beams: np.ndarray | None = None,
    actual_beams: np.ndarray | None = None,
    relation_enabled: np.ndarray | None = None,
    transform_enabled: bool = True,
) -> NativeExecution:
    coupling, field = validate_problem(coupling, field, law)
    borrowed_bank = as_carrier_bank(borrowed)
    canonical = canonical_geometry()
    program = canonical if program_beams is None else np.asarray(program_beams, dtype=np.complex128)
    actual = program if actual_beams is None else np.asarray(actual_beams, dtype=np.complex128)
    if program.shape != canonical.shape or actual.shape != canonical.shape:
        raise ValueError("recursive geometry bank has the wrong shape")
    program = np.array(program, copy=True)
    actual = np.array(actual, copy=True)
    anchors = geometry_anchors(actual)
    enabled = (
        np.ones((SITE_COUNT, SITE_COUNT), dtype=np.bool_)
        if relation_enabled is None
        else np.asarray(relation_enabled, dtype=np.bool_)
    )
    enabled = np.array(enabled, copy=True)
    np.fill_diagonal(enabled, False)
    spectrum = np.fft.fft(borrowed_bank, axis=1, norm="ortho")
    seeded, seed_operator = seed_recursive_spectral_tree(
        spectrum, anchors, enabled=transform_enabled
    )
    evolved, history, labels = relational_phase_operator(
        seeded, anchors, coupling, field, enabled, law
    )
    displaced = np.fft.ifft(evolved, axis=1, norm="ortho")
    return NativeExecution(
        borrowed=borrowed_bank,
        coupling=coupling,
        field=field,
        actual_beams=actual,
        program_beams=program,
        displaced=displaced,
        seed_operator=seed_operator,
        relation_history=history,
        relation_labels=labels,
        displacement_l2=float(np.linalg.norm(displaced - borrowed_bank)),
        law=law,
    )


def project_boundary(
    execution: NativeExecution,
    label: str,
    *,
    query_beams: np.ndarray | None = None,
) -> BoundaryProjection:
    query = execution.program_beams if query_beams is None else np.asarray(query_beams, dtype=np.complex128)
    anchors = geometry_anchors(query)
    spectrum = np.fft.fft(execution.displaced, axis=1, norm="ortho")
    active = normalized_active(spectrum)
    expected = anchors * PHASE_MODES
    merit_samples = active / expected
    merit_means = np.mean(merit_samples, axis=0)
    geometry_coherence = np.abs(merit_means)
    safe = np.maximum(geometry_coherence, 1.0e-300)
    merit = merit_means / safe
    merit_angles = np.angle(merit)
    penalties = -merit_angles / execution.law.relation_phase_scale
    order = np.argsort(penalties, kind="stable")
    best = int(order[0])
    gap = float(penalties[order[1]] - penalties[order[0]])
    shifted = penalties - penalties[best]
    weights = np.exp(-execution.law.softmax_beta * shifted)
    weights /= float(np.sum(weights))
    observed_modes = active / (anchors * merit[np.newaxis, :])
    responses_array = observed_modes @ weights
    responses = tuple(complex(value) for value in responses_array)
    coherence = tuple(float(abs(value)) for value in responses)
    raw_spins = tuple(1 if value.real >= 0.0 else -1 for value in responses)
    valid = bool(
        float(np.min(geometry_coherence)) >= execution.law.geometry_coherence_min
        and gap >= execution.law.unique_gap_min
        and min(coherence) >= execution.law.response_coherence_min
    )
    return BoundaryProjection(
        label=label,
        responses=responses,
        coherence=coherence,
        raw_spins=raw_spins,
        spins=raw_spins if valid else None,
        mode_penalties=tuple(float(value) for value in penalties),
        best_mode_index=best,
        second_mode_gap=gap,
        best_mode_concentration=float(weights[best]),
        minimum_geometry_coherence=float(np.min(geometry_coherence)),
        valid=valid,
    )


def restore_carrier(execution: NativeExecution, mode: str = "correct") -> np.ndarray:
    if mode == "omitted":
        return np.array(execution.displaced, copy=True)
    spectrum = np.fft.fft(execution.displaced, axis=1, norm="ortho")
    if mode == "correct":
        for operator in execution.relation_history[::-1]:
            spectrum *= np.conjugate(operator)
        spectrum /= execution.seed_operator
    elif mode == "wrong_phase":
        for operator in execution.relation_history[::-1]:
            wrong = np.array(operator, copy=True)
            wrong[:, ACTIVE_BINS] = np.roll(
                np.conjugate(operator[:, ACTIVE_BINS]), 1, axis=1
            )
            spectrum *= wrong
        spectrum /= execution.seed_operator
    elif mode == "wrong_seed":
        for operator in execution.relation_history[::-1]:
            spectrum *= np.conjugate(operator)
    else:
        raise ValueError("unknown restoration mode")
    return np.fft.ifft(spectrum, axis=1, norm="ortho")


def maximum_abs_error(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def execution_environment() -> dict[str, str]:
    return {
        "numpy_version": np.__version__,
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
    }


def machine_contract(law: SpectralPhaseLaw = DEFAULT_LAW) -> dict[str, Any]:
    return {
        "active_frequency_bins": [int(value) for value in ACTIVE_BINS],
        "boundary_only_operations": [
            "mode phase comparison",
            "spectral gap",
            "softmax projection",
            "antipodal sign extraction",
        ],
        "claim_ceiling": CLAIM_CEILING,
        "execution_environment": execution_environment(),
        "law": {key: metric(value) for key, value in asdict(law).items()},
        "mode_count": MODE_COUNT,
        "native_relations": [
            "recursive antipodal phase-mode seeding",
            "pairwise conjugate phase penalty",
            "local field phase penalty",
            "reversible spectral phase rotation",
        ],
        "sample_count": SAMPLE_COUNT,
        "source_closure": "self-contained v3 source plus frozen CPython/NumPy environment",
        "site_count": SITE_COUNT,
    }


def machine_fingerprint(law: SpectralPhaseLaw = DEFAULT_LAW) -> str:
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return sha256_bytes(
        canonical_bytes(
            {
                "contract": machine_contract(law),
                "phase_modes": [
                    [
                        {"imag": metric(value.imag), "real": metric(value.real)}
                        for value in row
                    ]
                    for row in PHASE_MODES
                ],
                "source_sha256": source_sha,
            }
        )
    )


def source_forbidden_patterns() -> tuple[str, ...]:
    return (
        "exact_oracle(",
        "ising_energy(",
        "@ spins",
        "J @",
        "optimum_state",
        "expected_result",
        "problem_sha256",
    )


def assert_no_native_smuggle() -> None:
    source = Path(__file__).read_text(encoding="utf-8")
    native = source[source.index("def relational_phase_operator(") : source.index("def project_boundary(")]
    found = [pattern for pattern in source_forbidden_patterns() if pattern in native]
    if found:
        raise RuntimeError(f"forbidden native feedback pattern: {found}")


assert_no_native_smuggle()
