#!/usr/bin/env python3
"""Focused tests for the .holo dimensional math.

These are not image-quality tests. They test the codec math directly:

1. low-rank data recovers the true active dimension;
2. isotropic data is correctly identified as incompressible;
3. the engineering action prefers fewer dimensions when rate cost rises.
"""

import numpy as np

import holo_core as holo


def make_low_rank(samples: int = 96, observed_dim: int = 24, rank: int = 4) -> np.ndarray:
    rng = np.random.default_rng(7)
    latent = rng.normal(size=(samples, rank))
    basis = rng.normal(size=(rank, observed_dim))
    return latent @ basis


def make_isotropic(samples: int = 512, observed_dim: int = 24) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.normal(size=(samples, observed_dim))


def assert_low_rank_recovery() -> dict:
    x = make_low_rank()
    projection = holo.project(x, policy="variance", variance_target=0.999)
    rendered = holo.render(projection)
    verification = holo.verify(x, rendered, projection.spectrum)

    assert projection.k == 4, f"expected k=4, got {projection.k}"
    assert verification.relative_error < 1e-5, verification
    assert verification.variance_retained > 0.999999, verification

    return {
        "k": projection.k,
        "D_pr": projection.spectrum.participation_dimension,
        "D_shannon": projection.spectrum.shannon_dimension,
        "relative_error": verification.relative_error,
    }


def assert_isotropic_not_compressible() -> dict:
    x = make_isotropic()
    spectrum = holo.analyze_spectrum(x)
    k95 = holo.choose_k(spectrum, policy="variance", variance_target=0.95)
    occupancy = spectrum.shannon_dimension / x.shape[1]

    assert occupancy > 0.85, f"expected high occupancy, got {occupancy}"
    assert k95 >= 20, f"expected k95 near full dimension, got {k95}"

    return {
        "k95": k95,
        "D_pr": spectrum.participation_dimension,
        "D_shannon": spectrum.shannon_dimension,
        "occupancy": occupancy,
    }


def assert_action_is_rate_sensitive() -> dict:
    x = make_low_rank(rank=6)
    spectrum = holo.analyze_spectrum(x)
    rate_model = holo.HoloRateModel(samples=x.shape[0], observed_dim=x.shape[1])

    cheap = holo.rate_distortion_action(spectrum, rate_model, rate_weight=1e-8)
    medium = holo.rate_distortion_action(spectrum, rate_model, rate_weight=1e-4)
    expensive = holo.rate_distortion_action(spectrum, rate_model, rate_weight=1e-3)

    assert cheap.best_k > medium.best_k > expensive.best_k, (
        cheap.best_k,
        medium.best_k,
        expensive.best_k,
    )
    assert cheap.best_k >= 5, cheap.best_k
    assert medium.best_k <= 3, medium.best_k
    assert expensive.best_k == 1, expensive.best_k

    return {
        "cheap_best_k": cheap.best_k,
        "medium_best_k": medium.best_k,
        "expensive_best_k": expensive.best_k,
        "cheap_action": float(cheap.action[cheap.best_k - 1]),
        "medium_action": float(medium.action[medium.best_k - 1]),
        "expensive_action": float(expensive.action[expensive.best_k - 1]),
    }


def main() -> None:
    results = {
        "low_rank": assert_low_rank_recovery(),
        "isotropic": assert_isotropic_not_compressible(),
        "rate_sensitive": assert_action_is_rate_sensitive(),
    }
    print(results)


if __name__ == "__main__":
    main()
