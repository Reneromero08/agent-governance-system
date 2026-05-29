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


@dataclass(frozen=True)
class HoloRateModel:
    """First-order payload model for a linear .holo projection.

    The model deliberately stays simple: a retained dimension stores one basis
    vector of observed_dim values plus one coordinate per sample. This makes the
    marginal engineering cost of each dimension explicit.
    """

    samples: int
    observed_dim: int
    bits_per_coordinate: int = 16
    bits_per_basis_value: int = 16
    metadata_bits: int = 0

    def bits_for_k(self, k: int) -> int:
        if k < 0:
            raise ValueError("k must be non-negative")
        coord_bits = self.samples * k * self.bits_per_coordinate
        basis_bits = self.observed_dim * k * self.bits_per_basis_value
        mean_bits = self.observed_dim * self.bits_per_basis_value
        return int(coord_bits + basis_bits + mean_bits + self.metadata_bits)

    def marginal_bits(self) -> int:
        return int(
            self.samples * self.bits_per_coordinate
            + self.observed_dim * self.bits_per_basis_value
        )


@dataclass(frozen=True)
class HoloActionCurve:
    """Rate-distortion/action curve over retained dimensions."""

    k_values: np.ndarray
    retained_information: np.ndarray
    tail_information: np.ndarray
    payload_bits: np.ndarray
    marginal_information_per_bit: np.ndarray
    action: np.ndarray
    best_k: int


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


def rate_distortion_action(
    spectrum: HoloSpectrum,
    rate_model: HoloRateModel,
    rate_weight: float,
    distortion_weight: float = 1.0,
    eps: float = 1e-12,
) -> HoloActionCurve:
    """Compute the engineering action for every retained dimension.

    The normalized action is:

        A(k) = distortion_weight * T(k) + rate_weight * B(k)

    where T(k) is discarded spectral mass and B(k) is payload bits. This is the
    practical form of "the math is the engineering": dimensions are retained
    only when the information they preserve justifies their storage cost.
    """
    if rate_weight < 0.0:
        raise ValueError("rate_weight must be non-negative")
    if distortion_weight <= 0.0:
        raise ValueError("distortion_weight must be positive")

    p = spectral_probabilities(spectrum, eps=eps)
    k_values = np.arange(1, p.size + 1, dtype=np.int64)
    retained = np.cumsum(p)
    tail = 1.0 - retained
    payload_bits = np.array([rate_model.bits_for_k(int(k)) for k in k_values], dtype=np.float64)
    marginal_bits = float(rate_model.marginal_bits())
    marginal = p / max(marginal_bits, eps)
    action = distortion_weight * tail + rate_weight * payload_bits
    best_k = int(k_values[int(np.argmin(action))])

    return HoloActionCurve(
        k_values=k_values,
        retained_information=retained.astype(np.float64),
        tail_information=tail.astype(np.float64),
        payload_bits=payload_bits,
        marginal_information_per_bit=marginal.astype(np.float64),
        action=action.astype(np.float64),
        best_k=best_k,
    )


def choose_k_by_action(
    spectrum: HoloSpectrum,
    rate_model: HoloRateModel,
    rate_weight: float,
    distortion_weight: float = 1.0,
) -> int:
    """Select k by minimizing the linear .holo rate-distortion action."""
    return rate_distortion_action(
        spectrum,
        rate_model,
        rate_weight=rate_weight,
        distortion_weight=distortion_weight,
    ).best_k


def spectral_probabilities(spectrum: HoloSpectrum, eps: float = 1e-12) -> np.ndarray:
    """Return normalized nonzero eigenvalue mass."""
    eigenvalues = np.asarray(spectrum.eigenvalues, dtype=np.float64)
    total = float(eigenvalues.sum())
    if total <= eps:
        return np.ones(1, dtype=np.float64)
    return eigenvalues / total


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
