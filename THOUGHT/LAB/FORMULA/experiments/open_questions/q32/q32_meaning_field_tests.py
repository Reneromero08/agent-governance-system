"""
Q32: Meaning as Field — Falsifiable Tests

Goal
Prove (in a falsifiable, mechanical way) that "meaning as field" is not just metaphor by
demonstrating that a resonance field M := log(R) behaves like an empirically grounded,
distributed control signal on a semiosphere.

Core falsifier
If echo-chambers (correlated consensus) can sustain high M under independence stress,
then this is not a "meaning field" (it collapses into social mirroring).

We deliberately compare:
  - UNGROUNDED resonance: uses only internal agreement/dispersion (echo-chamber vulnerable)
  - GROUNDED resonance: anchors E against independent empirical checks (echo-chamber collapses)

Run
  python THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_meaning_field_tests.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

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
    # ddof=1 so n=1 returns nan; guard with EPS.
    s = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    return s + EPS


def generate_independent(s: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    return s.truth + rng.normal(0.0, s.noise, size=n)


def generate_echo_chamber(s: Scenario, n: int, rng: np.random.Generator) -> np.ndarray:
    # Tight cluster around a biased center (high agreement, potentially wrong).
    center = s.truth + s.bias
    return center + rng.normal(0.0, s.noise * 0.1, size=n)


def E_from_empirical_check(mu_hat: float, check: np.ndarray) -> float:
    """
    Empirical grounding: compare the claim mu_hat against independent check data.

    We use a dimensionless residual z = |mu_hat - mean(check)| / std(check),
    and a Gaussian kernel E = exp(-z^2/2) as a bounded compatibility score in [0,1].
    """
    # We care about whether the *claim* mu_hat matches the independently measured mean.
    # The relevant uncertainty is the standard error of the check mean, not the raw spread.
    se = std(check) / math.sqrt(len(check))
    z = abs(mu_hat - mean(check)) / (se + EPS)
    return float(math.exp(-0.5 * z * z))


def R_grounded(obs: np.ndarray, check: np.ndarray, use_se: bool = False) -> float:
    """
    Grounded resonance:
      R = E(check) / ∇S

    ∇S must represent uncertainty scale, not just "spread of raw observations".
    For dynamics tests, using standard error (SE) makes ∇S shrink with more data.
    """
    mu_hat = mean(obs)
    s = std(obs)
    grad_S = (s / math.sqrt(len(obs))) if use_se else s
    return E_from_empirical_check(mu_hat, check) / (grad_S + EPS)


def R_ungrounded(obs: np.ndarray) -> float:
    """
    Ungrounded resonance (echo-chamber vulnerable): uses only internal dispersion.
    This matches the known failure mode: tight-but-wrong clusters can look "high R".
    """
    s = std(obs)
    E_internal = 1.0 / (1.0 + s)
    return E_internal / s


def M_from_R(R: float) -> float:
    return float(math.log(R + EPS))


def test_1_echo_chamber_falsifier(rng: np.random.Generator) -> None:
    """
    Falsifiable claim:
      With empirical grounding, echo-chambers should NOT sustain high M.
      Without grounding, echo-chambers CAN sustain high M (known vulnerability).
    """
    n_trials = 400
    n_obs = 30
    n_check = 40

    grounded_ind = []
    grounded_echo = []
    ungrounded_ind = []
    ungrounded_echo = []

    for _ in range(n_trials):
        s = Scenario(
            truth=float(rng.uniform(-10, 10)),
            noise=float(rng.uniform(0.8, 2.0)),
            bias=float(rng.normal(0.0, 3.0)),
        )

        ind_obs = generate_independent(s, n_obs, rng)
        echo_obs = generate_echo_chamber(s, n_obs, rng)
        check = generate_independent(s, n_check, rng)

        grounded_ind.append(M_from_R(R_grounded(ind_obs, check)))
        grounded_echo.append(M_from_R(R_grounded(echo_obs, check)))

        ungrounded_ind.append(M_from_R(R_ungrounded(ind_obs)))
        ungrounded_echo.append(M_from_R(R_ungrounded(echo_obs)))

    grounded_ind = np.array(grounded_ind)
    grounded_echo = np.array(grounded_echo)
    ungrounded_ind = np.array(ungrounded_ind)
    ungrounded_echo = np.array(ungrounded_echo)

    # Discrimination: grounded_ind should exceed grounded_echo in most paired draws.
    pair_wins = float(np.mean(grounded_ind > grounded_echo))

    print("\n[Q32:T1] Echo-chamber falsifier")
    print(f"  P(M_grounded(ind) > M_grounded(echo)) = {pair_wins:.3f}")
    print(f"  mean(M_grounded ind)  = {grounded_ind.mean():.3f}")
    print(f"  mean(M_grounded echo) = {grounded_echo.mean():.3f}")
    print(f"  mean(M_ungrounded ind)  = {ungrounded_ind.mean():.3f}")
    print(f"  mean(M_ungrounded echo) = {ungrounded_echo.mean():.3f}")

    # Hard gates (tunable, but must be strict enough to be falsifiable).
    assert pair_wins > 0.80, "FAIL: grounded field does not separate independent truth from echo-chamber"
    assert ungrounded_echo.mean() > ungrounded_ind.mean() + 1.0, (
        "FAIL: expected the ungrounded echo-chamber vulnerability to show up (echo should look 'more resonant')"
    )


def test_2_phase_transition_gate(rng: np.random.Generator) -> None:
    """
    Operationalize "nonlinear time" as thresholded phase transitions under accumulating evidence.

    We use ∇S as SE (std/sqrt(n)) so adding observations shrinks uncertainty scale.
    Gate opens when M crosses a fixed threshold.
    """
    n_trials = 200
    max_steps = 60
    check_batch = 30
    threshold_M = -0.2  # log(R) threshold (chosen to be falsifiable)

    opens_ind = 0
    opens_echo = 0
    open_step_ind = []
    open_step_echo = []

    for _ in range(n_trials):
        s = Scenario(
            truth=float(rng.uniform(-10, 10)),
            noise=float(rng.uniform(0.9, 2.2)),
            bias=float(rng.normal(0.0, 3.0)),
        )

        ind_stream = generate_independent(s, max_steps, rng)
        echo_stream = generate_echo_chamber(s, max_steps, rng)
        check = generate_independent(s, check_batch, rng)

        ind_opened = False
        echo_opened = False

        for t in range(5, max_steps + 1):
            ind_obs = ind_stream[:t]
            echo_obs = echo_stream[:t]

            M_ind = M_from_R(R_grounded(ind_obs, check, use_se=True))
            M_echo = M_from_R(R_grounded(echo_obs, check, use_se=True))

            if (not ind_opened) and (M_ind > threshold_M):
                ind_opened = True
                opens_ind += 1
                open_step_ind.append(t)

            if (not echo_opened) and (M_echo > threshold_M):
                echo_opened = True
                opens_echo += 1
                open_step_echo.append(t)

        if not ind_opened:
            open_step_ind.append(max_steps + 1)
        if not echo_opened:
            open_step_echo.append(max_steps + 1)

    open_rate_ind = opens_ind / n_trials
    open_rate_echo = opens_echo / n_trials

    print("\n[Q32:T2] Phase transition gate")
    print(f"  Gate threshold M > {threshold_M:.2f}")
    print(f"  Open rate (independent)  = {open_rate_ind:.3f}")
    print(f"  Open rate (echo-chamber) = {open_rate_echo:.3f}")
    print(f"  Median open step (ind)   = {int(np.median(open_step_ind))}")
    print(f"  Median open step (echo)  = {int(np.median(open_step_echo))}")

    assert open_rate_ind > 0.70, "FAIL: independent cases should usually crystallize (gate opens)"
    assert open_rate_echo < 0.30, "FAIL: echo-chambers should rarely crystallize under grounding"


def test_3_propagation_gluing(rng: np.random.Generator) -> None:
    """
    Propagation/gluing test:
    Two overlapping patches that track the same truth should glue into a stable global meaning;
    patches tracking different centers should not glue under grounding.
    """
    n_trials = 200
    n_patch = 25
    n_overlap = 10
    n_check = 40

    threshold_M = -0.2  # same gate threshold as Test 2
    success_same = 0
    success_diff = 0
    eligible_same = 0
    eligible_diff = 0

    for _ in range(n_trials):
        base = Scenario(
            truth=float(rng.uniform(-10, 10)),
            noise=float(rng.uniform(0.8, 2.0)),
            bias=float(rng.normal(0.0, 3.0)),
        )

        overlap = generate_independent(base, n_overlap, rng)

        # Same-truth patches
        A = np.concatenate([overlap, generate_independent(base, n_patch, rng)])
        B = np.concatenate([overlap, generate_independent(base, n_patch, rng)])
        check = generate_independent(base, n_check, rng)

        # Use SE so "field intensity" reflects uncertainty shrinking with more evidence.
        M_A = M_from_R(R_grounded(A, check, use_se=True))
        M_B = M_from_R(R_grounded(B, check, use_se=True))
        M_AB = M_from_R(R_grounded(np.concatenate([A, B]), check, use_se=True))

        # Different-truth patches (B is biased away via echo chamber)
        B_bad = np.concatenate([overlap, generate_echo_chamber(base, n_patch, rng)])
        M_B_bad = M_from_R(R_grounded(B_bad, check, use_se=True))
        M_AB_bad = M_from_R(R_grounded(np.concatenate([A, B_bad]), check, use_se=True))

        # Gluing criterion (field-like propagation):
        # - If both patches are above threshold, the glued context should also be above threshold.
        # - If one patch is below threshold, the glued context should remain below (no false propagation).
        if (M_A > threshold_M) and (M_B > threshold_M):
            eligible_same += 1
            if M_AB > threshold_M:
                success_same += 1

        if (M_A > threshold_M) and (M_B_bad < threshold_M):
            eligible_diff += 1
            if M_AB_bad < threshold_M:
                success_diff += 1

    rate_same = success_same / max(1, eligible_same)
    rate_diff = success_diff / max(1, eligible_diff)

    print("\n[Q32:T3] Propagation / gluing")
    print(f"  Gate threshold M > {threshold_M:.2f}")
    print(f"  Eligible (same-truth) = {eligible_same}/{n_trials}")
    print(f"  Eligible (mixed)      = {eligible_diff}/{n_trials}")
    print(f"  Glue pass rate (same-truth) = {rate_same:.3f}")
    print(f"  Glue pass rate (mixed)      = {rate_diff:.3f}")

    assert eligible_same > 30, "FAIL: not enough eligible same-truth cases to evaluate gluing"
    assert eligible_diff > 30, "FAIL: not enough eligible mixed cases to evaluate non-propagation"
    assert rate_same > 0.80, "FAIL: same-truth patches should glue most of the time"
    assert rate_diff > 0.80, "FAIL: mixed/echo should NOT propagate (glued context stays below threshold)"


def main() -> int:
    rng = np.random.default_rng(42)

    test_1_echo_chamber_falsifier(rng)
    test_2_phase_transition_gate(rng)
    test_3_propagation_gluing(rng)

    print("\n[Q32] ALL TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
