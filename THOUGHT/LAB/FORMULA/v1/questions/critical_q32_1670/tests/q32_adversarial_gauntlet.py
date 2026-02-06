"""
Q32: Adversarial Gauntlet (Phase 1)

This script tries to BREAK the "meaning field" operationalization under hostile synthetic conditions.
It is intentionally stricter than q32_meaning_field_tests.py:

- Sweeps correlation strength, bias magnitude, sample sizes
- Swaps noise families (Gaussian/Laplace/Student-t) while keeping the measurement spec pinned
- Runs negative controls (shuffled checks must destroy signal)
- Runs an intervention test (adding independent checks collapses echo-chambers)

Run
  python THOUGHT/LAB/FORMULA/questions/32/q32_adversarial_gauntlet.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class Scenario:
    truth: float
    noise: float
    bias: float


def mean(x: np.ndarray) -> float:
    return float(np.mean(x))


def std(x: np.ndarray) -> float:
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return s + EPS


def se(x: np.ndarray) -> float:
    return std(x) / math.sqrt(len(x))


def kernel_gaussian(z: float) -> float:
    return float(math.exp(-0.5 * z * z))


def kernel_laplace(z: float) -> float:
    return float(math.exp(-abs(z)))


KERNELS: Dict[str, Callable[[float], float]] = {
    "gaussian": kernel_gaussian,
    "laplace": kernel_laplace,
}


def generate_independent(s: Scenario, n: int, rng: np.random.Generator, family: str) -> np.ndarray:
    if family == "gaussian":
        return s.truth + rng.normal(0.0, s.noise, size=n)
    if family == "laplace":
        return s.truth + rng.laplace(0.0, s.noise, size=n)
    if family == "student_t":
        # Heavy-tail with df=3 (finite variance, nasty tails)
        return s.truth + rng.standard_t(df=3, size=n) * s.noise
    raise ValueError(f"Unknown family: {family}")


def generate_echo_chamber(s: Scenario, n: int, rng: np.random.Generator, family: str, rho: float) -> np.ndarray:
    """
    Echo chamber: observations share a latent common shock (correlation) and cluster around a biased center.

    rho in [0,1): correlation strength via shared shock.
    """
    center = s.truth + s.bias
    shared = rng.normal(0.0, s.noise * 0.1, size=1)[0]

    if family == "gaussian":
        indep = rng.normal(0.0, s.noise * 0.1, size=n)
    elif family == "laplace":
        indep = rng.laplace(0.0, s.noise * 0.1, size=n)
    elif family == "student_t":
        indep = rng.standard_t(df=3, size=n) * (s.noise * 0.1)
    else:
        raise ValueError(f"Unknown family: {family}")

    return center + rho * shared + math.sqrt(max(0.0, 1.0 - rho * rho)) * indep


def E(mu_hat: float, check: np.ndarray, kernel: str) -> float:
    z = abs(mu_hat - mean(check)) / (se(check) + EPS)
    return KERNELS[kernel](z)


def R_grounded(obs: np.ndarray, check: np.ndarray, kernel: str, use_se: bool) -> float:
    mu_hat = mean(obs)
    grad_S = (std(obs) / math.sqrt(len(obs))) if use_se else std(obs)
    return E(mu_hat, check, kernel) / (grad_S + EPS)


def R_ungrounded(obs: np.ndarray) -> float:
    # Explicitly echo-chamber vulnerable.
    s = std(obs)
    E_internal = 1.0 / (1.0 + s)
    return E_internal / s


def M(R: float) -> float:
    return float(math.log(R + EPS))


def summarize(name: str, values: List[float]) -> str:
    a = np.array(values, dtype=float)
    return f"{name}: mean={a.mean():.3f} std={a.std():.3f} min={a.min():.3f} max={a.max():.3f}"


def test_sweep_discrimination(rng: np.random.Generator) -> None:
    """
    Sweep parameters; require that grounded M generally ranks independent > echo.
    This is a strict test, but it only applies when the "echo chamber" is actually a
    *false basin* (i.e., biased away from truth and worse than the independent sample).

    If the echo cluster is centered on truth (bias≈0), it is not a false meaning basin and
    it is acceptable (even expected) for it to look highly resonant.
    """
    families = ["gaussian", "laplace", "student_t"]
    rhos = [0.0, 0.3, 0.6, 0.85]
    biases = [0.0, 1.0, 3.0, 6.0]
    ns = [10, 20, 40]
    kernels = ["gaussian", "laplace"]

    results = []

    for family in families:
        for kernel in kernels:
            for rho in rhos:
                for bias_mag in biases:
                    for n in ns:
                        pair_wins = []
                        eligible = 0
                        for _ in range(120):
                            s = Scenario(
                                truth=float(rng.uniform(-10, 10)),
                                noise=float(rng.uniform(0.8, 2.0)),
                                bias=float(rng.choice([-1.0, 1.0]) * bias_mag),
                            )

                            ind = generate_independent(s, n, rng, family)
                            echo = generate_echo_chamber(s, n, rng, family, rho=rho)
                            check = generate_independent(s, max(30, n), rng, family)

                            M_ind = M(R_grounded(ind, check, kernel=kernel, use_se=True))
                            M_echo = M(R_grounded(echo, check, kernel=kernel, use_se=True))

                            # Only score discrimination when echo is a wrong basin (worse than independent).
                            err_ind = abs(mean(ind) - s.truth)
                            err_echo = abs(mean(echo) - s.truth)
                            if err_echo > err_ind + 0.5 * s.noise:
                                eligible += 1
                                pair_wins.append(1.0 if M_ind > M_echo else 0.0)

                        # Skip configurations that do not generate enough wrong-basin cases.
                        if eligible < 30:
                            continue

                        pw = float(np.mean(pair_wins))
                        results.append(((family, kernel, rho, bias_mag, n), pw))

    assert len(results) > 50, "FAIL: sweep did not produce enough eligible wrong-basin cases"

    # Aggregate: median pairwise discrimination should be strong (on wrong-basin cases).
    pw_values = [pw for _, pw in results]
    median_pw = float(np.median(pw_values))
    p10_pw = float(np.percentile(pw_values, 10))

    print("\n[Q32:G1] Discrimination sweep")
    print(f"  median P(M_ind > M_echo) = {median_pw:.3f}")
    print(f"  10th percentile          = {p10_pw:.3f}")

    # We require strong median discrimination and a reasonable 10th percentile.
    assert median_pw >= 0.80, "FAIL: median discrimination below spec gate"
    if p10_pw < 0.65:
        worst = sorted(results, key=lambda x: x[1])[:10]
        print("  WORST 10 CONFIGS (family, kernel, rho, bias_mag, n -> pw):")
        for cfg, pw in worst:
            print(f"    {cfg} -> {pw:.3f}")
        raise AssertionError("FAIL: tail cases too weak; field not robust under sweep")


def test_negative_control_shuffle(rng: np.random.Generator) -> None:
    """
    Negative control:
    If check sets are unrelated to the claim, grounded discrimination must collapse.
    """
    n_trials = 300
    n = 25
    family = "gaussian"
    kernel = "gaussian"

    good = []
    bad = []

    for _ in range(n_trials):
        s = Scenario(
            truth=float(rng.uniform(-10, 10)),
            noise=float(rng.uniform(0.8, 2.0)),
            bias=float(rng.normal(0.0, 3.0)),
        )
        ind = generate_independent(s, n, rng, family)
        echo = generate_echo_chamber(s, n, rng, family, rho=0.8)

        check_good = generate_independent(s, 40, rng, family)
        # Bad check: unrelated truth (shuffle truth)
        s_bad = Scenario(truth=float(rng.uniform(-10, 10)), noise=s.noise, bias=s.bias)
        check_bad = generate_independent(s_bad, 40, rng, family)

        good.append(1.0 if M(R_grounded(ind, check_good, kernel, use_se=True)) > M(R_grounded(echo, check_good, kernel, use_se=True)) else 0.0)
        bad.append(1.0 if M(R_grounded(ind, check_bad, kernel, use_se=True)) > M(R_grounded(echo, check_bad, kernel, use_se=True)) else 0.0)

    good_pw = float(np.mean(good))
    bad_pw = float(np.mean(bad))

    print("\n[Q32:G2] Negative control (shuffled checks)")
    print(f"  P_ind>echo with GOOD checks = {good_pw:.3f}")
    print(f"  P_ind>echo with BAD checks  = {bad_pw:.3f}")

    assert good_pw >= 0.80, "FAIL: good-check discrimination too weak"
    assert bad_pw <= 0.60, "FAIL: negative control did not collapse (tautology/leak suspected)"


def test_independence_intervention(rng: np.random.Generator) -> None:
    """
    Intervention test:
    Start with echo-chamber-only check (biased); then add fresh independent check data.
    Echo-chamber M should drop (collapse), independent M should stabilize or improve.

    This is the causal core: the field must respond correctly to independence injections.
    """
    n_trials = 250
    n_obs = 25
    family = "gaussian"
    kernel = "gaussian"

    echo_drops = []
    ind_changes = []

    for _ in range(n_trials):
        s = Scenario(
            truth=float(rng.uniform(-10, 10)),
            noise=float(rng.uniform(0.8, 2.0)),
            bias=float(rng.normal(0.0, 3.0)),
        )
        ind = generate_independent(s, n_obs, rng, family)
        echo = generate_echo_chamber(s, n_obs, rng, family, rho=0.9)

        # Biased check (correlated evidence loop): centered at truth+bias
        check_echo = (s.truth + s.bias) + rng.normal(0.0, s.noise * 0.2, size=25)
        # Fresh independent check injections
        check_fresh = generate_independent(s, 40, rng, family)

        M_echo_before = M(R_grounded(echo, check_echo, kernel, use_se=True))
        M_echo_after = M(R_grounded(echo, np.concatenate([check_echo, check_fresh]), kernel, use_se=True))
        echo_drops.append(M_echo_after - M_echo_before)

        M_ind_before = M(R_grounded(ind, check_echo, kernel, use_se=True))
        M_ind_after = M(R_grounded(ind, np.concatenate([check_echo, check_fresh]), kernel, use_se=True))
        ind_changes.append(M_ind_after - M_ind_before)

    echo_drops = np.array(echo_drops)
    ind_changes = np.array(ind_changes)

    print("\n[Q32:G3] Independence intervention")
    print(f"  mean dM echo (after - before) = {echo_drops.mean():.3f}")
    print(f"  mean dM ind  (after - before) = {ind_changes.mean():.3f}")
    print(f"  P(dM_echo < 0) = {float(np.mean(echo_drops < 0)):.3f}")

    assert float(np.mean(echo_drops < 0)) >= 0.70, "FAIL: echo-chamber does not reliably collapse under independence injection"
    assert ind_changes.mean() >= echo_drops.mean() + 0.5, "FAIL: independent case does not improve relative to echo under intervention"


def main() -> int:
    rng = np.random.default_rng(123)

    test_sweep_discrimination(rng)
    test_negative_control_shuffle(rng)
    test_independence_intervention(rng)

    print("\n[Q32] ADVERSARIAL GAUNTLET PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
