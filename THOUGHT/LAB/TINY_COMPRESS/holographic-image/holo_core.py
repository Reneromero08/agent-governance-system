"""Generic .holo core: dimensional Shannon compression primitives.

This module is domain-neutral. It does not know about images, audio, text, or
activations. Domain adapters provide an observation matrix X with shape
(samples, observed_dim), then this core measures the information spectrum,
projects into a lower-dimensional basis, renders back, and reports distortion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


KPolicy = Literal["participation", "shannon", "variance", "fixed"]


@dataclass(frozen=True)
class HoloSpectrum:
    """Information spectrum of an observation matrix."""

    eigenvalues: np.ndarray
    cumulative_variance: np.ndarray
    participation_dimension: float
    shannon_dimension: float


@dataclass(frozen=True)
class HoloProjection:
    """Low-dimensional representation plus render basis."""

    coordinates: np.ndarray
    basis: np.ndarray
    mean: np.ndarray
    spectrum: HoloSpectrum
    k: int


@dataclass(frozen=True)
class HoloVerification:
    """Distortion metrics between source observations and render."""

    mse: float
    rmse: float
    relative_error: float
    variance_retained: float


def analyze_spectrum(observations: np.ndarray, eps: float = 1e-12) -> HoloSpectrum:
    """Measure active information dimensions for observations.

    Args:
        observations: Matrix shaped (samples, observed_dim).
        eps: Floor for numerical stability and zero-spectrum handling.

    Returns:
        HoloSpectrum with descending covariance eigenvalues.
    """
    x = _as_observation_matrix(observations)
    centered = x - x.mean(axis=0, keepdims=True)

    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    if x.shape[0] > 1:
        eigenvalues = (singular_values ** 2) / (x.shape[0] - 1)
    else:
        eigenvalues = singular_values ** 2

    eigenvalues = eigenvalues[eigenvalues > eps]
    if eigenvalues.size == 0:
        eigenvalues = np.array([eps], dtype=np.float64)

    total = float(eigenvalues.sum())
    probabilities = eigenvalues / total
    cumulative = np.cumsum(probabilities)

    participation = (total * total) / float(np.square(eigenvalues).sum())
    entropy = -float(np.sum(probabilities * np.log(probabilities + eps)))
    shannon = float(np.exp(entropy))

    return HoloSpectrum(
        eigenvalues=eigenvalues.astype(np.float64),
        cumulative_variance=cumulative.astype(np.float64),
        participation_dimension=float(participation),
        shannon_dimension=shannon,
    )


def choose_k(
    spectrum: HoloSpectrum,
    policy: KPolicy = "variance",
    variance_target: float = 0.95,
    fixed_k: int | None = None,
) -> int:
    """Choose retained dimensions from an information spectrum."""
    max_k = int(spectrum.eigenvalues.size)

    if policy == "fixed":
        if fixed_k is None:
            raise ValueError("fixed_k is required when policy='fixed'")
        k = fixed_k
    elif policy == "participation":
        k = int(np.ceil(spectrum.participation_dimension))
    elif policy == "shannon":
        k = int(np.ceil(spectrum.shannon_dimension))
    elif policy == "variance":
        if not 0.0 < variance_target <= 1.0:
            raise ValueError("variance_target must be in (0, 1]")
        k = int(np.searchsorted(spectrum.cumulative_variance, variance_target) + 1)
    else:
        raise ValueError(f"unknown k policy: {policy}")

    return max(1, min(int(k), max_k))


def project(
    observations: np.ndarray,
    policy: KPolicy = "variance",
    variance_target: float = 0.95,
    fixed_k: int | None = None,
) -> HoloProjection:
    """Project observations into a lower-dimensional .holo representation."""
    x = _as_observation_matrix(observations)
    mean = x.mean(axis=0, keepdims=True)
    centered = x - mean

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    spectrum = analyze_spectrum(x)
    k = choose_k(spectrum, policy=policy, variance_target=variance_target, fixed_k=fixed_k)

    basis = vt[:k]
    coordinates = centered @ basis.T

    return HoloProjection(
        coordinates=coordinates.astype(np.float32),
        basis=basis.astype(np.float32),
        mean=mean.reshape(-1).astype(np.float32),
        spectrum=spectrum,
        k=k,
    )


def render(projection: HoloProjection, render_k: int | None = None) -> np.ndarray:
    """Render observations from a .holo projection.

    render_k can be lower than projection.k for progressive essence-to-detail
    reconstruction.
    """
    k = projection.k if render_k is None else max(1, min(render_k, projection.k))
    return projection.coordinates[:, :k] @ projection.basis[:k] + projection.mean


def verify(source: np.ndarray, reconstructed: np.ndarray, spectrum: HoloSpectrum | None = None) -> HoloVerification:
    """Compare source observations to a rendered reconstruction."""
    x = _as_observation_matrix(source).astype(np.float64)
    y = _as_observation_matrix(reconstructed).astype(np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: source {x.shape}, reconstructed {y.shape}")

    diff = x - y
    mse = float(np.mean(np.square(diff)))
    rmse = float(np.sqrt(mse))
    denom = float(np.linalg.norm(x))
    rel = float(np.linalg.norm(diff) / denom) if denom > 0.0 else 0.0

    retained = 1.0
    if spectrum is not None:
        retained = _estimate_variance_retained(x, y)

    return HoloVerification(
        mse=mse,
        rmse=rmse,
        relative_error=rel,
        variance_retained=float(retained),
    )


def _estimate_variance_retained(source: np.ndarray, reconstructed: np.ndarray) -> float:
    source_var = float(np.var(source))
    residual_var = float(np.var(source - reconstructed))
    if source_var <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - residual_var / source_var))


def _as_observation_matrix(observations: np.ndarray) -> np.ndarray:
    x = np.asarray(observations, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("observations must have shape (samples, observed_dim)")
    if x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("observations must be non-empty")
    return x

