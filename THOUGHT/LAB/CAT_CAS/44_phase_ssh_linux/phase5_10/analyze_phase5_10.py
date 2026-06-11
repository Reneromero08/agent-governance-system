#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EXP44 Phase 5.10 analysis pipeline (host-independent, buildable now).

This module is the verification-side analysis for boundary state preparation
(see the SPEC files in this directory; this code owns NONE of the .md). It is
deliberately host-independent: it ingests basin-scan CSVs produced on the
Phenom (or synthetic fixtures) and runs the frozen-threshold basin classifier,
the transition-matrix estimator, the dwell-time loop-area extrapolator, the
parametric-scaling null comparator, and the statistics battery.

Discipline (binding, per the Exp 44 manifesto / the 5.10 gate spec):
  - FROZEN THRESHOLDS: basin classes are assigned from thresholds loaded from
    basin_thresholds_frozen.json. Thresholds are NEVER tuned post-hoc here; if
    the frozen file is absent, the classifier refuses to invent thresholds
    against the data (Gate-3 protection).
  - REAL NULLS (M-5): every selection lift is compared against a label-reshuffle
    empirical null; a null that does not separate KILLS the selection claim.
  - REAL STATISTICS (M-6): binomial (Wilson) confidence intervals on basin
    rates, an empirical permutation p-value, and Bonferroni/Holm correction
    across selectors. Numbers are never reported bare.
  - Path(__file__) for outputs (M-7): no hardcoded experiment paths.
  - No ceremonial anything: the classifier is label-blind on the feature
    vector; the parametric-scaling comparator is the central physical/logical
    discriminator (G-4) and is allowed to downgrade a clean selection.

Outputs are written under  <this_dir>/_generated/  (derived from __file__).

This file contains only analysis. It does not run silicon and does not touch
any phase3* / phase3b* artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# numpy is used only for vector arithmetic in the loop-area fit; everything
# statistical is implemented explicitly so the math is auditable.
import numpy as np

# ----------------------------------------------------------------------------
# Output location (M-7): derived from __file__, never a hardcoded lab path.
# ----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = THIS_DIR / "_generated"
DEFAULT_FROZEN_THRESHOLDS = THIS_DIR / "basin_thresholds_frozen.json"

BASIN_CLASSES = ("collapsed", "mid", "high")


# ============================================================================
# Frozen-threshold basin classifier
# ============================================================================
@dataclass
class FrozenThresholds:
    """Frozen basin-classification thresholds.

    The classifier is a label-blind cut on a single calibrated carrier metric
    (default: boundary_thickness). Two cut points partition the metric into
    collapsed / mid / high. These thresholds are CALIBRATED ELSEWHERE on a
    dedicated calibration set and FROZEN; this object only reads them.
    """

    metric: str
    collapsed_max: float  # metric <= collapsed_max  -> collapsed
    high_min: float       # metric >= high_min       -> high  (else mid)
    source: str = "frozen_file"
    calibration_run: str = "unspecified"

    def classify(self, value: float) -> str:
        if value <= self.collapsed_max:
            return "collapsed"
        if value >= self.high_min:
            return "high"
        return "mid"

    @staticmethod
    def from_file(path: Path) -> "FrozenThresholds":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return FrozenThresholds(
            metric=str(data["metric"]),
            collapsed_max=float(data["collapsed_max"]),
            high_min=float(data["high_min"]),
            source=str(data.get("source", str(path))),
            calibration_run=str(data.get("calibration_run", "unspecified")),
        )

    def to_dict(self) -> dict:
        return {
            "metric": self.metric,
            "collapsed_max": self.collapsed_max,
            "high_min": self.high_min,
            "source": self.source,
            "calibration_run": self.calibration_run,
        }


def freeze_thresholds_from_calibration(
    values: Sequence[float],
    metric: str,
    calibration_run: str,
    out_path: Path,
    quantiles: Tuple[float, float] = (1.0 / 3.0, 2.0 / 3.0),
) -> FrozenThresholds:
    """Derive thresholds from a CALIBRATION set and freeze them to disk.

    This is the ONLY sanctioned way to create thresholds. It must be run on a
    dedicated calibration set BEFORE any selection data is seen (the spec's
    binding freeze rule / Gate-3). The default rule is empirical terciles of
    the calibration metric, which is label-blind. Callers selecting thresholds
    against selection outcomes are violating the freeze rule, not this function.
    """
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("cannot freeze thresholds from an empty calibration set")
    lo_q, hi_q = quantiles
    collapsed_max = float(np.quantile(arr, lo_q))
    high_min = float(np.quantile(arr, hi_q))
    thr = FrozenThresholds(
        metric=metric,
        collapsed_max=collapsed_max,
        high_min=high_min,
        source=f"calibration_terciles({lo_q:.3f},{hi_q:.3f})",
        calibration_run=calibration_run,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(thr.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    return thr


# ============================================================================
# Record model
# ============================================================================
@dataclass
class BasinRecord:
    """A single catalytic run with its carrier metric and context covariates."""

    run_id: str
    selector: str                 # prelude / state-preparation label
    metric_value: float           # the calibrated carrier metric
    basin_observed: str           # frozen-threshold classification (filled in)
    restoration_failures: int = 0
    physical_state: str = "default"   # e.g. VID/P-state bucket
    covariate: Optional[float] = None  # parametric-scaling covariate (temp/rail)
    basin_reported: Optional[str] = None  # label present in source CSV (audit)


# Column-name candidates by role, so we can ingest both synthetic and the real
# 5.9V matrices without per-file hardcoding.
_METRIC_CANDIDATES = (
    "boundary_thickness", "carrier_amplitude", "stable_thickness",
    "spike_free_thickness", "mean_radius", "d_eff", "metric_value",
)
_SELECTOR_CANDIDATES = ("selector", "prelude", "prelude_type", "selector_label")
_BASIN_CANDIDATES = ("basin", "basin_class", "basin_observed")
_RESTORE_CANDIDATES = ("restoration_failures", "restore_failures", "rest_fail")
_COVARIATE_CANDIDATES = (
    "temp_c", "temperature", "die_temp", "k10temp", "wall_power",
    "rail", "covariate", "cooling",
)
_PHYS_CANDIDATES = (
    "physical_state", "pstate", "vid_decoded", "decoded_voltage", "vid_request",
)


def _pick(header: Sequence[str], candidates: Sequence[str]) -> Optional[str]:
    lower = {h.lower(): h for h in header}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def load_basin_csv(
    path: Path,
    thresholds: FrozenThresholds,
    metric_override: Optional[str] = None,
) -> List[BasinRecord]:
    """Load a basin-scan CSV and (re)classify every row with FROZEN thresholds.

    The basin column already present in the source (if any) is preserved as
    basin_reported for audit, but basin_observed is recomputed here from the
    frozen thresholds so classification cannot drift between source files.
    """
    rows: List[BasinRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        metric_col = metric_override or _pick(header, _METRIC_CANDIDATES)
        if metric_col is None or metric_col not in header:
            # Fall back to the frozen metric name if it is a column.
            if thresholds.metric in header:
                metric_col = thresholds.metric
            else:
                raise ValueError(
                    f"{path}: no usable carrier-metric column found "
                    f"(looked for {_METRIC_CANDIDATES} and '{thresholds.metric}')"
                )
        sel_col = _pick(header, _SELECTOR_CANDIDATES)
        basin_col = _pick(header, _BASIN_CANDIDATES)
        restore_col = _pick(header, _RESTORE_CANDIDATES)
        cov_col = _pick(header, _COVARIATE_CANDIDATES)
        phys_col = _pick(header, _PHYS_CANDIDATES)
        run_col = "run_id" if "run_id" in header else None

        for i, raw in enumerate(reader):
            try:
                metric_value = float(raw[metric_col])
            except (TypeError, ValueError):
                continue  # skip un-parseable rows rather than guess
            selector = (raw.get(sel_col) if sel_col else None) or "unlabeled"
            run_id = (raw.get(run_col) if run_col else None) or f"row{i}"
            restoration = 0
            if restore_col and raw.get(restore_col) not in (None, ""):
                try:
                    restoration = int(float(raw[restore_col]))
                except (TypeError, ValueError):
                    restoration = 0
            covariate = None
            if cov_col and raw.get(cov_col) not in (None, ""):
                try:
                    covariate = float(raw[cov_col])
                except (TypeError, ValueError):
                    covariate = None
            phys = (raw.get(phys_col) if phys_col else None) or "default"
            reported = raw.get(basin_col) if basin_col else None
            rows.append(
                BasinRecord(
                    run_id=run_id,
                    selector=str(selector),
                    metric_value=metric_value,
                    basin_observed=thresholds.classify(metric_value),
                    restoration_failures=restoration,
                    physical_state=str(phys),
                    covariate=covariate,
                    basin_reported=str(reported) if reported is not None else None,
                )
            )
    return rows


# ============================================================================
# Statistics: Wilson binomial CI, permutation null, multiple-comparison
# ============================================================================
def wilson_ci(successes: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% default).

    Wilson is used rather than the normal-approx Wald interval because basin
    rates are often near 0 or 1 with small n, where Wald is badly miscalibrated.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((p * (1 - p) / n) + (z * z) / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def _basin_indicator(records: Sequence[BasinRecord], target_basin: str) -> np.ndarray:
    return np.array(
        [1.0 if r.basin_observed == target_basin else 0.0 for r in records],
        dtype=float,
    )


def permutation_null_lift(
    selector_flags: np.ndarray,
    target_flags: np.ndarray,
    observed_lift: float,
    n_perm: int,
    rng: random.Random,
) -> Tuple[float, Tuple[float, float]]:
    """Empirical label-reshuffle null for a selection lift (M-5).

    selector_flags : 1.0 where the run used the candidate selector, else 0.0
    target_flags   : 1.0 where the run landed in the target basin, else 0.0
    observed_lift  : P(target | selector) - P(target | other)  (the real effect)

    Returns (p_value, null_lift_95ci). The null shuffles the selector labels
    across runs and recomputes the lift; a selection that does not exceed this
    reshuffled null is killed. This is a real null: it CAN return p ~ 1.0.
    """
    flags = list(selector_flags)
    target = target_flags
    n = len(flags)
    n_sel = int(round(sum(flags)))
    if n_sel == 0 or n_sel == n:
        return (1.0, (0.0, 0.0))
    null_lifts = np.empty(n_perm, dtype=float)
    idx = list(range(n))
    for k in range(n_perm):
        rng.shuffle(idx)
        sel_mask = np.zeros(n, dtype=bool)
        for j in idx[:n_sel]:
            sel_mask[j] = True
        in_sel = target[sel_mask]
        out_sel = target[~sel_mask]
        p_in = in_sel.mean() if in_sel.size else 0.0
        p_out = out_sel.mean() if out_sel.size else 0.0
        null_lifts[k] = p_in - p_out
    # One-sided p: probability the null produces a lift >= observed.
    p_value = float((np.sum(null_lifts >= observed_lift) + 1) / (n_perm + 1))
    lo, hi = np.quantile(null_lifts, [0.025, 0.975])
    return (p_value, (float(lo), float(hi)))


def holm_bonferroni(pvals: Sequence[float], alpha: float = 0.05) -> List[bool]:
    """Holm-Bonferroni step-down rejections across selectors (M-6).

    Returns a parallel list of booleans: True = reject null (selection survives
    correction) at family-wise alpha. Controls family-wise error across the
    multiple selectors tested, which was the explicit 5.9V gap.
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    reject = [False] * m
    for rank, idx in enumerate(order):
        adj_alpha = alpha / (m - rank)
        if pvals[idx] <= adj_alpha:
            reject[idx] = True
        else:
            break  # step-down: once one fails, all larger p-values fail
    return reject


# ============================================================================
# Transition-matrix estimator  P(basin | selector, physical_state)
# ============================================================================
@dataclass
class SelectorBasinStats:
    selector: str
    physical_state: str
    n_runs: int
    counts: Dict[str, int]
    rates: Dict[str, float]
    cis: Dict[str, Tuple[float, float]]
    restoration_failures: int


def estimate_transition_matrix(
    records: Sequence[BasinRecord],
) -> Dict[Tuple[str, str], SelectorBasinStats]:
    """Estimate P(basin | selector, physical_state) with binomial CIs."""
    groups: Dict[Tuple[str, str], List[BasinRecord]] = {}
    for r in records:
        groups.setdefault((r.selector, r.physical_state), []).append(r)
    out: Dict[Tuple[str, str], SelectorBasinStats] = {}
    for key, recs in groups.items():
        n = len(recs)
        counts = {b: 0 for b in BASIN_CLASSES}
        for r in recs:
            counts[r.basin_observed] = counts.get(r.basin_observed, 0) + 1
        rates = {b: (counts[b] / n if n else 0.0) for b in BASIN_CLASSES}
        cis = {b: wilson_ci(counts[b], n) for b in BASIN_CLASSES}
        rest = sum(r.restoration_failures for r in recs)
        out[key] = SelectorBasinStats(
            selector=key[0],
            physical_state=key[1],
            n_runs=n,
            counts=counts,
            rates=rates,
            cis=cis,
            restoration_failures=rest,
        )
    return out


# ============================================================================
# Dwell-time LOOP-AREA A(tau) extrapolator (thermal-bistability test)
# ============================================================================
@dataclass
class LoopAreaFit:
    """Hysteresis loop-area extrapolation for the leakage-thermal saddle-node.

    For a thermal bistability the up-sweep and down-sweep branches enclose a
    hysteresis loop whose area A(tau) depends on the sweep dwell time tau. A
    genuine bistability has a NON-ZERO area in the quasi-static limit
    (tau -> inf): A(tau) = A_inf + C / tau (rate-dependent loops collapse onto
    a finite static loop). A merely rate-dependent (dynamic, non-bistable)
    response extrapolates to A_inf ~ 0. We fit A vs 1/tau and report the
    intercept A_inf with its standard error; A_inf significantly > 0 is the
    signature, A_inf ~ 0 is the kill.
    """

    a_inf: float
    a_inf_stderr: float
    slope_c: float
    r_squared: float
    n_points: int
    verdict: str

    def to_dict(self) -> dict:
        return {
            "a_inf_intercept": self.a_inf,
            "a_inf_stderr": self.a_inf_stderr,
            "slope_C_per_invtau": self.slope_c,
            "r_squared": self.r_squared,
            "n_points": self.n_points,
            "verdict": self.verdict,
        }


def loop_area(up_branch: Sequence[float], down_branch: Sequence[float]) -> float:
    """Signed-magnitude area between up-sweep and down-sweep branches.

    Both branches are sampled at the same ordered control points; the loop area
    is the trapezoidal integral of |up - down| over the (normalized) control
    axis. Returns a non-negative scalar.
    """
    up = np.asarray(list(up_branch), dtype=float)
    down = np.asarray(list(down_branch), dtype=float)
    if up.size != down.size or up.size < 2:
        return 0.0
    diff = np.abs(up - down)
    x = np.linspace(0.0, 1.0, up.size)
    return float(np.trapz(diff, x)) if hasattr(np, "trapz") else float(np.trapz(diff, x))


def fit_loop_area_extrapolation(
    taus: Sequence[float],
    areas: Sequence[float],
    a_inf_tol: float = 1e-9,
) -> LoopAreaFit:
    """Fit A(tau) = A_inf + C*(1/tau) and report the tau->inf intercept.

    A weighted-free OLS of area on (1/tau). The intercept is A_inf. We report
    its standard error so the tau->inf area carries a real uncertainty (M-6),
    and a verdict that distinguishes a finite static loop (bistable candidate)
    from a vanishing one (dynamic-only).
    """
    tau = np.asarray(list(taus), dtype=float)
    area = np.asarray(list(areas), dtype=float)
    mask = tau > 0
    tau, area = tau[mask], area[mask]
    n = tau.size
    if n < 2:
        return LoopAreaFit(0.0, float("inf"), 0.0, 0.0, n, "INSUFFICIENT_SWEEPS")
    x = 1.0 / tau
    # OLS with intercept: area = b0 + b1*x. b0 = A_inf.
    xbar = x.mean()
    ybar = area.mean()
    sxx = float(np.sum((x - xbar) ** 2))
    if sxx <= 0:
        # All sweeps at one tau: cannot extrapolate, report the mean area.
        return LoopAreaFit(
            float(ybar), float("inf"), 0.0, 0.0, n, "SINGLE_TAU_NO_EXTRAP"
        )
    sxy = float(np.sum((x - xbar) * (area - ybar)))
    slope = sxy / sxx
    intercept = float(ybar - slope * xbar)
    resid = area - (intercept + slope * x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((area - ybar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    # Standard error of the intercept (OLS).
    if n > 2:
        sigma2 = ss_res / (n - 2)
        se_intercept = math.sqrt(sigma2 * (1.0 / n + xbar * xbar / sxx))
    else:
        se_intercept = float("inf")
    # Verdict: is the quasi-static loop finite (bistable) or vanishing?
    if intercept <= a_inf_tol:
        verdict = "NO_STATIC_LOOP_DYNAMIC_ONLY"
    elif se_intercept != float("inf") and intercept > 2.0 * se_intercept:
        verdict = "FINITE_STATIC_LOOP_BISTABILITY_CANDIDATE"
    else:
        verdict = "STATIC_LOOP_WEAK_UNDERPOWERED"
    return LoopAreaFit(intercept, se_intercept, slope, r2, n, verdict)


# ============================================================================
# Parametric-scaling null comparator (G-4, the central physical/logical test)
# ============================================================================
@dataclass
class ParametricScalingResult:
    """Does the effect size move with a physical covariate (temp/cooling/rail)?

    A true substrate basin SCALES parametrically with a physical condition; a
    logical / cache-replacement attractor is INVARIANT to it. We regress the
    per-bucket effect size on the covariate and test the slope against a
    permutation null on the covariate assignment.
    """

    covariate_name: str
    slope: float
    slope_null_ci: Tuple[float, float]
    p_value: float
    n_buckets: int
    verdict: str  # SCALES_PHYSICAL | INVARIANT_LOGICAL | UNDERPOWERED

    def to_dict(self) -> dict:
        return {
            "covariate": self.covariate_name,
            "slope": self.slope,
            "slope_null_ci_lo": self.slope_null_ci[0],
            "slope_null_ci_hi": self.slope_null_ci[1],
            "p_value": self.p_value,
            "n_buckets": self.n_buckets,
            "verdict": self.verdict,
        }


def parametric_scaling_null(
    covariates: Sequence[float],
    effect_sizes: Sequence[float],
    covariate_name: str,
    n_perm: int,
    rng: random.Random,
) -> ParametricScalingResult:
    """Test whether effect size scales with a physical covariate vs a null.

    covariates   : physical covariate per bucket (die temp / cooling / rail)
    effect_sizes : selection effect size per bucket (e.g. selection lift)

    The null shuffles which effect size is paired with which covariate and
    recomputes the regression slope; an effect that does not scale beyond this
    null is the INVARIANT_LOGICAL downgrade (G-4 kill). This comparator is
    allowed to DOWNGRADE an otherwise-clean selection.
    """
    cov = np.asarray(list(covariates), dtype=float)
    eff = np.asarray(list(effect_sizes), dtype=float)
    n = cov.size
    if n < 3 or eff.size != n or float(np.var(cov)) == 0.0:
        return ParametricScalingResult(
            covariate_name, 0.0, (0.0, 0.0), 1.0, n, "UNDERPOWERED"
        )

    def _slope(c: np.ndarray, e: np.ndarray) -> float:
        cbar, ebar = c.mean(), e.mean()
        scc = float(np.sum((c - cbar) ** 2))
        if scc <= 0:
            return 0.0
        return float(np.sum((c - cbar) * (e - ebar)) / scc)

    observed_slope = _slope(cov, eff)
    null_slopes = np.empty(n_perm, dtype=float)
    idx = list(range(n))
    for k in range(n_perm):
        rng.shuffle(idx)
        null_slopes[k] = _slope(cov, eff[idx])
    p_value = float(
        (np.sum(np.abs(null_slopes) >= abs(observed_slope)) + 1) / (n_perm + 1)
    )
    lo, hi = np.quantile(null_slopes, [0.025, 0.975])
    if p_value <= 0.05 and abs(observed_slope) > 0:
        verdict = "SCALES_PHYSICAL"
    else:
        verdict = "INVARIANT_LOGICAL"
    return ParametricScalingResult(
        covariate_name, observed_slope, (float(lo), float(hi)),
        p_value, n, verdict,
    )


# ============================================================================
# Selection analysis: tie statistics + nulls + correction together
# ============================================================================
@dataclass
class SelectionResult:
    selector: str
    target_basin: str
    physical_state: str
    n_runs: int
    n_target_in_selector: int
    p_target_given_selector: float
    p_target_given_other: float
    selection_lift: float
    lift_ci: Tuple[float, float]
    null_p_value: float
    null_lift_ci: Tuple[float, float]
    holm_reject: bool
    restoration_failures: int
    verdict: str

    def to_row(self) -> dict:
        return {
            "selector": self.selector,
            "target_basin": self.target_basin,
            "physical_state": self.physical_state,
            "n_runs": self.n_runs,
            "n_target_in_selector": self.n_target_in_selector,
            "p_target_given_selector": round(self.p_target_given_selector, 6),
            "p_target_given_other": round(self.p_target_given_other, 6),
            "selection_lift": round(self.selection_lift, 6),
            "lift_ci_lo": round(self.lift_ci[0], 6),
            "lift_ci_hi": round(self.lift_ci[1], 6),
            "null_p_value": round(self.null_p_value, 6),
            "null_lift_ci_lo": round(self.null_lift_ci[0], 6),
            "null_lift_ci_hi": round(self.null_lift_ci[1], 6),
            "holm_reject_null": self.holm_reject,
            "restoration_failures": self.restoration_failures,
            "verdict": self.verdict,
        }


def analyze_selection(
    records: Sequence[BasinRecord],
    target_basin: str,
    n_perm: int = 2000,
    alpha: float = 0.05,
    seed: int = 44,
    min_repeats: int = 20,
) -> List[SelectionResult]:
    """For every selector, estimate the lift toward target_basin and test it.

    Implements the 5.10C / G-3 selection test: each selector's lift toward the
    target basin (vs all other selectors pooled) is measured, given a binomial
    CI, tested against a label-reshuffle empirical null, and the family of
    selectors is Holm-Bonferroni corrected. min_repeats encodes the spec's
    Gate-5 floor (>=20 repeats per prelude); below it the verdict is forced to
    UNDERPOWERED regardless of point estimate.
    """
    rng = random.Random(seed)
    selectors = sorted({r.selector for r in records})
    target_flags = _basin_indicator(records, target_basin)
    results: List[SelectionResult] = []
    raw_pvals: List[float] = []
    pending: List[SelectionResult] = []

    for sel in selectors:
        sel_flags = np.array(
            [1.0 if r.selector == sel else 0.0 for r in records], dtype=float
        )
        n_sel = int(round(sel_flags.sum()))
        in_mask = sel_flags == 1.0
        out_mask = ~in_mask
        p_in = float(target_flags[in_mask].mean()) if in_mask.any() else 0.0
        p_out = float(target_flags[out_mask].mean()) if out_mask.any() else 0.0
        lift = p_in - p_out
        n_target_in = int(round(target_flags[in_mask].sum()))
        # CI on the in-selector basin rate (reported as the lift's anchor).
        ci = wilson_ci(n_target_in, n_sel)
        p_value, null_ci = permutation_null_lift(
            sel_flags, target_flags, lift, n_perm, rng
        )
        rest = sum(
            r.restoration_failures for r in records if r.selector == sel
        )
        raw_pvals.append(p_value)
        pending.append(
            SelectionResult(
                selector=sel,
                target_basin=target_basin,
                physical_state="pooled",
                n_runs=n_sel,
                n_target_in_selector=n_target_in,
                p_target_given_selector=p_in,
                p_target_given_other=p_out,
                selection_lift=lift,
                lift_ci=ci,
                null_p_value=p_value,
                null_lift_ci=null_ci,
                holm_reject=False,  # filled after correction
                restoration_failures=rest,
                verdict="PENDING",
            )
        )

    rejects = holm_bonferroni(raw_pvals, alpha=alpha)
    for res, rej in zip(pending, rejects):
        res.holm_reject = bool(rej)
        if res.n_runs < min_repeats:
            res.verdict = "UNDERPOWERED_BELOW_GATE5_FLOOR"
        elif res.restoration_failures > 0:
            res.verdict = "VOID_RESTORATION_FAILURE"
        elif rej and res.selection_lift > 0:
            res.verdict = "SELECTION_SURVIVES_NULL_AND_CORRECTION"
        elif res.null_p_value <= alpha and res.selection_lift > 0:
            res.verdict = "SELECTION_WEAK_FAILS_CORRECTION"
        else:
            res.verdict = "NO_SELECTION_CONSISTENT_WITH_NULL"
        results.append(res)
    return results


# ============================================================================
# Writers
# ============================================================================
def write_transition_matrix_csv(
    matrix: Dict[Tuple[str, str], SelectorBasinStats], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "selector", "physical_state", "n_runs",
        "collapsed_count", "mid_count", "high_count",
        "p_collapsed", "p_mid", "p_high",
        "ci_collapsed_lo", "ci_collapsed_hi",
        "ci_mid_lo", "ci_mid_hi",
        "ci_high_lo", "ci_high_hi",
        "restoration_failures",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for stats in sorted(matrix.values(), key=lambda s: (s.selector, s.physical_state)):
            writer.writerow({
                "selector": stats.selector,
                "physical_state": stats.physical_state,
                "n_runs": stats.n_runs,
                "collapsed_count": stats.counts["collapsed"],
                "mid_count": stats.counts["mid"],
                "high_count": stats.counts["high"],
                "p_collapsed": round(stats.rates["collapsed"], 6),
                "p_mid": round(stats.rates["mid"], 6),
                "p_high": round(stats.rates["high"], 6),
                "ci_collapsed_lo": round(stats.cis["collapsed"][0], 6),
                "ci_collapsed_hi": round(stats.cis["collapsed"][1], 6),
                "ci_mid_lo": round(stats.cis["mid"][0], 6),
                "ci_mid_hi": round(stats.cis["mid"][1], 6),
                "ci_high_lo": round(stats.cis["high"][0], 6),
                "ci_high_hi": round(stats.cis["high"][1], 6),
                "restoration_failures": stats.restoration_failures,
            })


def write_selection_csv(results: Sequence[SelectionResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not results:
        out_path.write_text("", encoding="utf-8")
        return
    fields = list(results[0].to_row().keys())
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for res in results:
            writer.writerow(res.to_row())


# ============================================================================
# High-level driver for a single subphase CSV
# ============================================================================
@dataclass
class SubphaseAnalysis:
    name: str
    source_csv: str
    n_records: int
    thresholds: dict
    transition_matrix_csv: str
    selection_csv: str
    best_selection: Optional[dict]
    loop_area: Optional[dict] = None
    parametric_scaling: Optional[dict] = None


def analyze_subphase(
    name: str,
    csv_path: Path,
    thresholds: FrozenThresholds,
    out_dir: Path,
    target_basin: str = "high",
    n_perm: int = 2000,
    seed: int = 44,
) -> SubphaseAnalysis:
    records = load_basin_csv(csv_path, thresholds)
    matrix = estimate_transition_matrix(records)
    results = analyze_selection(
        records, target_basin=target_basin, n_perm=n_perm, seed=seed
    )
    tm_csv = out_dir / f"{name}_transition_matrix.csv"
    sel_csv = out_dir / f"{name}_selection.csv"
    write_transition_matrix_csv(matrix, tm_csv)
    write_selection_csv(results, sel_csv)
    best = None
    if results:
        best_res = max(results, key=lambda r: (r.holm_reject, r.selection_lift))
        best = best_res.to_row()
    return SubphaseAnalysis(
        name=name,
        source_csv=str(csv_path),
        n_records=len(records),
        thresholds=thresholds.to_dict(),
        transition_matrix_csv=str(tm_csv),
        selection_csv=str(sel_csv),
        best_selection=best,
    )


# ============================================================================
# Self-test: synthetic deterministic data + (optional) real 5.9V ingest
# ============================================================================
def _synth_records(seed: int = 44) -> List[BasinRecord]:
    """Deterministic synthetic basin data with ONE planted selection.

    'thermal_prelude' is planted to bias the 'high' basin AND to do so in a way
    that SCALES with die temperature: its metric is centered well below the high
    cut at cold temperatures and is pushed across the cut as temperature rises,
    so the per-temperature 'high' rate climbs monotonically with the covariate.
    This is a genuine substrate-style positive control for the G-4 parametric-
    scaling test (a logical attractor would be temperature-invariant). A 'quiet'
    baseline and a 'shuffled_control' sit near the middle; every other selector
    is null. The metric is generated WITHOUT reference to the frozen thresholds,
    so classification stays label-blind.
    """
    rng = random.Random(seed)
    selectors_mu = {
        "quiet": 100.0,
        "cache_prelude": 100.0,
        "syscall_prelude": 105.0,
        "shuffled_control": 100.0,
        "thermal_prelude": 95.0,    # planted base; temperature carries it 'high'
    }
    records: List[BasinRecord] = []
    for sel, mu in selectors_mu.items():
        for rep in range(36):  # 6 temp buckets x 6 reps; > Gate-5 floor of 20
            # covariate: synthetic die temp spanning a cold->warm band.
            temp = 40.0 + (rep % 6) * 8.0   # {40,48,56,64,72,80}
            val = max(0.0, rng.gauss(mu, 12.0))
            if sel == "thermal_prelude":
                # Gentle monotonic temperature drive: at cold temps the metric
                # sits near 'mid' (sub-threshold), and the 'high' rate climbs
                # gradually as temperature rises rather than saturating. This
                # makes the per-bucket high-rate scale with the covariate (the
                # G-4 physical signature); a logical attractor would be flat.
                val += (temp - 40.0) * 1.1
            records.append(
                BasinRecord(
                    run_id=f"{sel}_R{rep}",
                    selector=sel,
                    metric_value=val,
                    basin_observed="",  # filled by classifier below
                    restoration_failures=0,
                    physical_state="vid_default",
                    covariate=temp,
                )
            )
    return records


def _synth_loop_sweeps(seed: int = 44) -> Tuple[List[float], List[float]]:
    """Deterministic dwell-time sweeps with a planted finite static loop.

    Area(tau) = A_inf + C/tau with A_inf > 0  (a real bistability signature).
    """
    rng = random.Random(seed + 1)
    a_inf_true, c_true = 12.0, 220.0
    taus = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    areas = []
    for tau in taus:
        noise = rng.gauss(0.0, 0.4)
        areas.append(a_inf_true + c_true / tau + noise)
    return taus, areas


def run_self_test(out_dir: Path, real_csv: Optional[Path] = None) -> dict:
    """End-to-end self-test. Returns a JSON-able summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"synthetic": {}, "real": None, "loop_area": {}, "parametric": {}}

    # --- 1. Freeze thresholds from a SYNTHETIC CALIBRATION set (label-blind) ---
    cal_records = _synth_records(seed=7)  # separate seed: calibration != selection
    cal_values = [r.metric_value for r in cal_records]
    thr = freeze_thresholds_from_calibration(
        cal_values,
        metric="boundary_thickness",
        calibration_run="synthetic_calibration_seed7",
        out_path=out_dir / "basin_thresholds_frozen.json",
    )
    report["thresholds"] = thr.to_dict()

    # --- 2. Classify the (independent) selection set with FROZEN thresholds ---
    sel_records = _synth_records(seed=44)
    for r in sel_records:
        r.basin_observed = thr.classify(r.metric_value)
    matrix = estimate_transition_matrix(sel_records)
    write_transition_matrix_csv(matrix, out_dir / "synthetic_transition_matrix.csv")
    results = analyze_selection(sel_records, target_basin="high", n_perm=2000, seed=44)
    write_selection_csv(results, out_dir / "synthetic_selection.csv")
    report["synthetic"] = {
        "n_records": len(sel_records),
        "selectors": sorted({r.selector for r in sel_records}),
        "selection": [res.to_row() for res in results],
    }

    # --- 3. Parametric-scaling comparator on the synthetic effect ---------
    #   Per-run (covariate, in-target-basin indicator) for the planted selector.
    #   Regressing the indicator on temperature run-by-run (rather than on a
    #   handful of bucket means) keeps the permutation null properly powered:
    #   a physical effect's high-rate rises with temperature; a logical one is
    #   flat. We feed every planted run, not pre-averaged buckets.
    planted = [
        r for r in sel_records
        if r.selector == "thermal_prelude" and r.covariate is not None
    ]
    cov = [r.covariate for r in planted]
    eff = [1.0 if r.basin_observed == "high" else 0.0 for r in planted]
    psr = parametric_scaling_null(
        cov, eff, covariate_name="synthetic_die_temp_c",
        n_perm=2000, rng=random.Random(99),
    )
    report["parametric"] = psr.to_dict()

    # --- 4. Loop-area extrapolation -------------------------------------
    taus, areas = _synth_loop_sweeps(seed=44)
    fit = fit_loop_area_extrapolation(taus, areas)
    report["loop_area"] = fit.to_dict()

    # --- 5. Optional: ingest the REAL 5.9V matrix ------------------------
    if real_csv is not None and real_csv.exists():
        # Real data carries its own basin labels; reclassify with frozen
        # thresholds derived from the REAL metric distribution (a real
        # calibration set would be a dedicated run; here we use the real
        # file's own terciles as a stand-in calibration, FROZEN before the
        # selection analysis reads any selector outcome).
        real_raw = load_basin_csv(
            real_csv,
            thresholds=FrozenThresholds(
                metric="boundary_thickness", collapsed_max=0.0, high_min=1e18,
                source="placeholder_pre_calibration",
            ),
        )
        real_vals = [r.metric_value for r in real_raw]
        real_thr = freeze_thresholds_from_calibration(
            real_vals,
            metric="boundary_thickness",
            calibration_run=f"real_5_9v_terciles::{real_csv.name}",
            out_path=out_dir / "basin_thresholds_frozen_real.json",
        )
        real_records = load_basin_csv(real_csv, real_thr)
        real_matrix = estimate_transition_matrix(real_records)
        write_transition_matrix_csv(
            real_matrix, out_dir / "real_5_9v_transition_matrix.csv"
        )
        real_results = analyze_selection(
            real_records, target_basin="high", n_perm=2000, seed=44
        )
        write_selection_csv(real_results, out_dir / "real_5_9v_selection.csv")
        # Audit: how well do frozen-threshold labels agree with the source's?
        agree = sum(
            1 for r in real_records
            if r.basin_reported is not None and r.basin_reported == r.basin_observed
        )
        labeled = sum(1 for r in real_records if r.basin_reported is not None)
        report["real"] = {
            "source": str(real_csv),
            "n_records": len(real_records),
            "selectors": sorted({r.selector for r in real_records}),
            "frozen_thresholds": real_thr.to_dict(),
            "label_agreement_vs_source": (
                f"{agree}/{labeled}" if labeled else "n/a"
            ),
            "selection": [res.to_row() for res in real_results],
            "restoration_failures_total": sum(
                r.restoration_failures for r in real_records
            ),
        }

    (out_dir / "self_test_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def _find_real_csv() -> Optional[Path]:
    """Locate the real 70-row 5.9V basin matrix relative to __file__ (M-7).

    No hardcoded lab path: we walk up to the 44_phase_ssh_linux root from
    __file__ and join the known relative results path.
    """
    base = THIS_DIR.parent  # .../44_phase_ssh_linux
    candidate = (
        base / "phase5_9" / "results" / "k10_voltage_probe"
        / "p4_vid5_phase6_basin_repro"
        / "phase5_9v_phase6_basin_repro_audit.csv"
    )
    return candidate if candidate.exists() else None


def _print_report(report: dict) -> None:
    print("=" * 70)
    print("PHASE 5.10 ANALYSIS SELF-TEST")
    print("=" * 70)
    thr = report.get("thresholds", {})
    print(f"[thresholds] metric={thr.get('metric')} "
          f"collapsed_max={thr.get('collapsed_max'):.3f} "
          f"high_min={thr.get('high_min'):.3f} "
          f"(source={thr.get('source')})")

    print("\n[SYNTHETIC selection -> 'high' basin]  (planted: thermal_prelude)")
    for row in report["synthetic"]["selection"]:
        print(
            f"  {row['selector']:<18} n={row['n_runs']:<3} "
            f"lift={row['selection_lift']:+.3f} "
            f"CI[{row['lift_ci_lo']:.2f},{row['lift_ci_hi']:.2f}] "
            f"null_p={row['null_p_value']:.4f} "
            f"holm={row['holm_reject_null']} -> {row['verdict']}"
        )

    ps = report.get("parametric", {})
    print(f"\n[parametric-scaling G-4] covariate={ps.get('covariate')} "
          f"slope={ps.get('slope'):+.4f} p={ps.get('p_value'):.4f} "
          f"-> {ps.get('verdict')}")

    la = report.get("loop_area", {})
    print(f"\n[loop-area A(tau->inf)] intercept={la.get('a_inf_intercept'):.3f} "
          f"+/- {la.get('a_inf_stderr'):.3f}  R2={la.get('r_squared'):.4f} "
          f"-> {la.get('verdict')}")

    if report.get("real"):
        r = report["real"]
        print(f"\n[REAL 5.9V ingest] {Path(r['source']).name}  "
              f"n={r['n_records']}  "
              f"label_agreement={r['label_agreement_vs_source']}  "
              f"restoration_failures={r['restoration_failures_total']}")
        print(f"  selectors: {', '.join(r['selectors'])}")
        for row in r["selection"]:
            print(
                f"  {row['selector']:<22} n={row['n_runs']:<3} "
                f"lift={row['selection_lift']:+.3f} "
                f"null_p={row['null_p_value']:.4f} "
                f"holm={row['holm_reject_null']} -> {row['verdict']}"
            )
    else:
        print("\n[REAL 5.9V ingest] not found (synthetic-only run)")
    print("=" * 70)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EXP44 Phase 5.10 analysis pipeline")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic + real-data self-test")
    parser.add_argument("--basin-csv", type=str, default=None,
                        help="analyze a single basin-scan CSV")
    parser.add_argument("--thresholds", type=str, default=str(DEFAULT_FROZEN_THRESHOLDS),
                        help="path to basin_thresholds_frozen.json")
    parser.add_argument("--target-basin", type=str, default="high",
                        choices=list(BASIN_CLASSES))
    parser.add_argument("--out-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=44)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)

    if args.basin_csv:
        thr_path = Path(args.thresholds)
        if not thr_path.exists():
            print(f"[error] frozen thresholds not found: {thr_path}. "
                  f"Freeze them from a calibration set first (Gate-3).")
            return 2
        thr = FrozenThresholds.from_file(thr_path)
        analysis = analyze_subphase(
            name=Path(args.basin_csv).stem,
            csv_path=Path(args.basin_csv),
            thresholds=thr,
            out_dir=out_dir,
            target_basin=args.target_basin,
            seed=args.seed,
        )
        print(json.dumps(analysis.__dict__, indent=2, sort_keys=True, default=str))
        return 0

    # Default: self-test (also runs if --self-test given).
    real = _find_real_csv()
    report = run_self_test(out_dir, real_csv=real)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
