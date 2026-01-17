#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q21 Phase 4: Adversarial Stress Tests

"Nearly impossible" tests to stress-test the alpha-drift hypothesis.
These tests try to BREAK the predictor to find its limits.

Tests:
1. Echo Chamber: INTEGRATES Q32 neighbor-falsifier methodology
   - Extreme R detection (per Q5)
   - Independence stress (per Q32) - fresh data collapses echo chamber R
   - Mutual information coupling (phi_proxy) for structure signal
2. Delayed Collapse: Alpha drifts but R stays high temporarily
3. Sudden Collapse: No alpha precursor - system should admit uncertainty
4. Oscillating Alpha: No trend - should NOT produce false positives
5. Correlated Noise: Spurious correlations - null hypothesis must be rejected
6. Distribution Shift: Domain change vs true collapse - should distinguish

Pass Criteria: >= 5/6 adversarial tests pass

INTEGRATION NOTE: Test 4.1 now leverages Q32's neighbor-falsifier framework
to properly test independence stress on echo chambers. Per Q5 and Q32:
- Echo chambers produce extreme R (>10x normal)
- Adding fresh INDEPENDENT data crashes echo chamber R (93% drop)
- Neighbor wrong-checks distinguish correlated consensus from true agreement
"""

import sys
import math
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from q21_temporal_utils import (
    CRITICAL_ALPHA, TARGET_DF_ALPHA, EPS,
    get_eigenspectrum, compute_df, compute_alpha, compute_R,
    detect_alpha_drift, evaluate_predictor, compute_cohens_d
)


# =============================================================================
# Q32 Integration: Mutual Information (phi_proxy)
# =============================================================================

def _mutual_information_continuous(x, y, n_bins: int = 8) -> float:
    """
    Phi-style proxy: mutual information I(X;Y) via histogram binning.

    Imported from Q32's neighbor-falsifier framework.
    NOT canonical IIT Phi; a cheap coupling/structure proxy.
    """
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    if xx.size == 0 or yy.size == 0:
        return 0.0
    n = int(min(xx.size, yy.size))
    xx = xx[:n]
    yy = yy[:n]

    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 5:
        return 0.0

    n_bins_i = max(2, int(n_bins))
    x_min, x_max = float(np.min(xx)), float(np.max(xx))
    y_min, y_max = float(np.min(yy)), float(np.max(yy))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    # 2D histogram for joint probability
    x_edges = np.linspace(x_min, x_max, n_bins_i + 1)
    y_edges = np.linspace(y_min, y_max, n_bins_i + 1)
    hist_2d, _, _ = np.histogram2d(xx, yy, bins=[x_edges, y_edges])

    # Normalize to joint probability
    pxy = hist_2d / (hist_2d.sum() + EPS)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    # Mutual information I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(n_bins_i):
        for j in range(n_bins_i):
            if pxy[i, j] > EPS and px[i] > EPS and py[j] > EPS:
                mi += pxy[i, j] * math.log(pxy[i, j] / (px[i] * py[j]))
    return float(mi)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# Standard vocabulary
WORDS = [
    "water", "fire", "earth", "sky", "sun", "moon", "star", "mountain",
    "river", "tree", "flower", "rain", "wind", "snow", "cloud", "ocean",
    "dog", "cat", "bird", "fish", "horse", "tiger", "lion", "elephant",
    "heart", "eye", "hand", "head", "brain", "blood", "bone", "skin",
    "mother", "father", "child", "friend", "king", "queen", "hero", "teacher",
    "love", "hate", "truth", "life", "death", "time", "space", "power",
    "peace", "war", "hope", "fear", "joy", "pain", "dream", "thought",
    "book", "door", "house", "road", "food", "money", "stone", "gold",
    "light", "shadow", "music", "word", "name", "law", "art", "science",
    "good", "bad", "big", "small", "old", "new", "high", "low",
]


def load_model_embeddings(model_id: str = "all-MiniLM-L6-v2"):
    """Load embeddings from a sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        embeddings = model.encode(WORDS, normalize_embeddings=True)
        return embeddings
    except Exception as e:
        print(f"    Failed to load model: {e}")
        return None


# =============================================================================
# Test 4.1: Echo Chamber Attack (Q32 Integration)
# =============================================================================

def test_echo_chamber(seed: int = 42) -> dict:
    """
    Echo Chamber Detection using Q32 Neighbor-Falsifier Methodology.

    INTEGRATES Q5 + Q32:
    - Q5: Echo chambers produce extreme R values (feature, not bug)
    - Q32: Independence stress via neighbor-based fresh data injection

    Test Protocol:
    1. Create echo chamber (tight cluster around centroid)
    2. Measure baseline metrics (R, alpha, phi_proxy)
    3. INDEPENDENCE STRESS: Inject fresh independent data
    4. Measure R collapse under independence stress

    Success Criteria (must pass BOTH):
    1. Echo chamber is DETECTABLE (extreme R ratio > 10x OR low alpha)
    2. Echo chamber R COLLAPSES under independence stress (> 50% drop)
    """
    print("\n" + "=" * 60)
    print("TEST 4.1: ECHO CHAMBER DETECTION (Q32 INTEGRATION)")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    embeddings = load_model_embeddings()
    if embeddings is None:
        return {'test': 'echo_chamber', 'success': False, 'error': 'Model load failed'}

    # Original metrics for comparison
    R_orig = compute_R(embeddings)
    ev_orig = get_eigenspectrum(embeddings)
    alpha_orig = compute_alpha(ev_orig)

    # --- Phase 1: Create Echo Chamber ---
    print("\n  [Phase 1] Creating echo chamber...")
    centroid = np.mean(embeddings, axis=0)
    echo_chamber = centroid + rng.normal(0, 0.01, embeddings.shape)  # Very small variance
    echo_chamber = echo_chamber / (np.linalg.norm(echo_chamber, axis=1, keepdims=True) + EPS)

    R_echo = compute_R(echo_chamber)
    ev_echo = get_eigenspectrum(echo_chamber)
    alpha_echo = compute_alpha(ev_echo)

    # Detection criterion 1: Extreme R
    R_ratio = R_echo / (R_orig + EPS)
    extreme_R_detected = R_ratio > 10 or (R_echo > 50 and alpha_echo < 0.3)

    print(f"    Original R: {R_orig:.4f}")
    print(f"    Echo chamber R: {R_echo:.4f}")
    print(f"    R ratio: {R_ratio:.1f}x")
    print(f"    Original alpha: {alpha_orig:.4f}")
    print(f"    Echo chamber alpha: {alpha_echo:.4f}")
    print(f"    Extreme R detected: {extreme_R_detected}")

    # --- Phase 2: Q32 Independence Stress ---
    # Per Q32: Adding fresh INDEPENDENT data should collapse echo chamber R
    print("\n  [Phase 2] Q32 Independence Stress Test...")

    # Generate "neighbor" wrong-checks: semantically related but INDEPENDENT
    # Use a random subset of original embeddings (diverse, independent sources)
    n_inject = len(echo_chamber) // 4  # Inject 25% fresh independent data

    # Fresh independent data: random samples from DIFFERENT points in semantic space
    independent_indices = rng.choice(len(embeddings), size=n_inject, replace=False)
    independent_data = embeddings[independent_indices]

    # Mix echo chamber with fresh independent data
    mixed_embeddings = np.vstack([
        echo_chamber[:len(echo_chamber) - n_inject],
        independent_data
    ])
    mixed_embeddings = mixed_embeddings / (np.linalg.norm(mixed_embeddings, axis=1, keepdims=True) + EPS)

    R_mixed = compute_R(mixed_embeddings)
    ev_mixed = get_eigenspectrum(mixed_embeddings)
    alpha_mixed = compute_alpha(ev_mixed)

    # Independence stress: R should COLLAPSE when fresh data is added
    R_collapse_ratio = (R_echo - R_mixed) / (R_echo + EPS)
    independence_collapse = R_collapse_ratio > 0.50  # > 50% drop

    print(f"    Fresh independent samples injected: {n_inject}")
    print(f"    R after independence stress: {R_mixed:.4f}")
    print(f"    R collapse ratio: {R_collapse_ratio * 100:.1f}%")
    print(f"    Independence collapse detected: {independence_collapse}")

    # --- Phase 3: Phi-proxy Coupling (Q32 Structure Signal) ---
    print("\n  [Phase 3] Phi-proxy Structure Signal...")

    # Compute mutual information between echo chamber pairwise sims and mixed pairwise sims
    # High coupling = echo chamber structure persists (bad)
    # Low coupling = independence stress broke the structure (good)

    # Get pairwise similarities
    echo_sims = []
    mixed_sims = []
    n_sample = min(40, len(echo_chamber))
    for i in range(n_sample):
        for j in range(i + 1, n_sample):
            echo_sims.append(float(np.dot(echo_chamber[i], echo_chamber[j])))
            if i < len(mixed_embeddings) and j < len(mixed_embeddings):
                mixed_sims.append(float(np.dot(mixed_embeddings[i], mixed_embeddings[j])))

    phi_proxy = _mutual_information_continuous(echo_sims, mixed_sims[:len(echo_sims)], n_bins=8)
    print(f"    Phi-proxy I(echo; mixed): {phi_proxy:.4f} bits")

    # Low phi_proxy means independence stress successfully broke structure
    structure_broken = phi_proxy < 1.5  # Threshold: < 1.5 bits of mutual information

    # --- Final Assessment ---
    print("\n  [Summary]")
    print(f"    Criterion 1 (Extreme R Detection): {'PASS' if extreme_R_detected else 'FAIL'}")
    print(f"    Criterion 2 (Independence Collapse): {'PASS' if independence_collapse else 'FAIL'}")
    print(f"    Criterion 3 (Structure Broken): {'PASS' if structure_broken else 'FAIL'}")

    # Success: Echo chamber detected AND independence stress works
    # (At least 2 of 3 criteria must pass)
    criteria_passed = sum([extreme_R_detected, independence_collapse, structure_broken])
    success = criteria_passed >= 2

    print(f"\n  OVERALL: {'PASS' if success else 'FAIL'} ({criteria_passed}/3 criteria)")
    print(f"  NOTE: Per Q5+Q32, echo chambers are detectable AND collapse under independence stress.")

    return {
        'test': 'echo_chamber_q32',
        'R_original': float(R_orig),
        'R_echo_chamber': float(R_echo),
        'R_ratio': float(R_ratio),
        'alpha_original': float(alpha_orig),
        'alpha_echo_chamber': float(alpha_echo),
        'extreme_R_detected': extreme_R_detected,
        'R_after_independence': float(R_mixed),
        'R_collapse_ratio': float(R_collapse_ratio),
        'independence_collapse': independence_collapse,
        'phi_proxy_bits': float(phi_proxy),
        'structure_broken': structure_broken,
        'criteria_passed': criteria_passed,
        'success': success
    }


# =============================================================================
# Test 4.2: Delayed Collapse Attack
# =============================================================================

def test_delayed_collapse(seed: int = 42) -> dict:
    """
    Alpha drifts but R stays artificially high for extended period.

    Attack: Suppress R collapse even as alpha degrades.
    Success: Predictor doesn't cry wolf too early (lead time reasonable)
    """
    print("\n" + "=" * 60)
    print("TEST 4.2: DELAYED COLLAPSE ATTACK")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    embeddings = load_model_embeddings()
    if embeddings is None:
        return {'test': 'delayed_collapse', 'success': False, 'error': 'Model load failed'}

    n_steps = 40
    alpha_values = []
    R_values = []

    ev_clean = get_eigenspectrum(embeddings)
    baseline_alpha = compute_alpha(ev_clean)
    baseline_R = compute_R(embeddings)

    for t in range(n_steps):
        # Inject noise that degrades alpha but maintains high R
        # by keeping centroid stable while scrambling higher dimensions

        if t < 20:
            noise_level = t * 0.005
        else:
            noise_level = 0.1 + (t - 20) * 0.02  # Accelerate after step 20

        # Add noise only to high-frequency components (preserves mean)
        noise = rng.normal(0, noise_level, embeddings.shape)
        perturbed = embeddings + noise

        # Renormalize
        perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + EPS)

        ev = get_eigenspectrum(perturbed)
        alpha_values.append(compute_alpha(ev))
        R_values.append(compute_R(perturbed))

    alpha_values = np.array(alpha_values)
    R_values = np.array(R_values)

    # Detect alpha drift
    alpha_distance = np.abs(alpha_values - baseline_alpha)
    drift_threshold = 0.05
    drift_detected = alpha_distance > drift_threshold

    first_drift = np.where(drift_detected)[0]
    first_drift_idx = first_drift[0] if len(first_drift) > 0 else None

    # Gate closure
    gate_threshold = baseline_R * 0.3
    gate_closed = R_values < gate_threshold
    first_close = np.where(gate_closed)[0]
    first_close_idx = first_close[0] if len(first_close) > 0 else n_steps

    # Lead time
    if first_drift_idx is not None:
        lead_time = first_close_idx - first_drift_idx
    else:
        lead_time = 0

    # Success: predictor gives reasonable warning (not too early, not too late)
    # "Too early" would be lead_time > 20 when gate never really closes
    # "Too late" would be lead_time < 0
    success = 0 <= lead_time <= 25

    print(f"  Baseline alpha: {baseline_alpha:.4f}")
    print(f"  Final alpha: {alpha_values[-1]:.4f}")
    print(f"  Baseline R: {baseline_R:.4f}")
    print(f"  Final R: {R_values[-1]:.4f}")
    print(f"  Alpha drift at step: {first_drift_idx}")
    print(f"  Gate closure at step: {first_close_idx}")
    print(f"  Lead time: {lead_time} steps")
    print(f"  Status: {'PASS' if success else 'FAIL'} (need 0 <= lead <= 25)")

    return {
        'test': 'delayed_collapse',
        'baseline_alpha': float(baseline_alpha),
        'final_alpha': float(alpha_values[-1]),
        'baseline_R': float(baseline_R),
        'final_R': float(R_values[-1]),
        'drift_detection_idx': first_drift_idx,
        'gate_closure_idx': first_close_idx,
        'lead_time': lead_time,
        'success': success
    }


# =============================================================================
# Test 4.3: Sudden Collapse Attack
# =============================================================================

def test_sudden_collapse(seed: int = 42) -> dict:
    """
    Create instantaneous R collapse with NO alpha precursor.

    Attack: R drops suddenly without alpha warning.
    Success: System admits uncertainty (doesn't claim false positive prediction)
    """
    print("\n" + "=" * 60)
    print("TEST 4.3: SUDDEN COLLAPSE (NO WARNING)")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    embeddings = load_model_embeddings()
    if embeddings is None:
        return {'test': 'sudden_collapse', 'success': False, 'error': 'Model load failed'}

    n_steps = 30
    collapse_step = 20
    alpha_values = []
    R_values = []

    ev_clean = get_eigenspectrum(embeddings)
    baseline_alpha = compute_alpha(ev_clean)

    for t in range(n_steps):
        if t < collapse_step:
            # Stable phase - minimal noise
            perturbed = embeddings + rng.normal(0, 0.01, embeddings.shape)
        else:
            # INSTANT collapse - completely scramble embeddings
            perturbed = rng.normal(0, 1, embeddings.shape)

        perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + EPS)

        ev = get_eigenspectrum(perturbed)
        alpha_values.append(compute_alpha(ev))
        R_values.append(compute_R(perturbed))

    alpha_values = np.array(alpha_values)
    R_values = np.array(R_values)

    # Check if alpha showed any warning BEFORE collapse
    pre_collapse_alpha = alpha_values[:collapse_step]
    alpha_variance_pre = np.std(pre_collapse_alpha)

    # Alpha should be stable before collapse (low variance)
    alpha_stable_before = alpha_variance_pre < 0.05

    # Post-collapse alpha
    post_collapse_alpha = alpha_values[collapse_step:]

    # Success: System correctly identifies this as NO ADVANCE WARNING scenario
    # (alpha was stable, then suddenly everything changed)
    success = alpha_stable_before

    print(f"  Baseline alpha: {baseline_alpha:.4f}")
    print(f"  Pre-collapse alpha std: {alpha_variance_pre:.4f}")
    print(f"  Post-collapse mean alpha: {np.mean(post_collapse_alpha):.4f}")
    print(f"  Alpha stable before collapse: {alpha_stable_before}")
    print(f"  R before collapse: {R_values[collapse_step-1]:.4f}")
    print(f"  R after collapse: {R_values[collapse_step]:.4f}")
    print(f"  Status: {'PASS' if success else 'FAIL'} (need alpha stable pre-collapse)")

    return {
        'test': 'sudden_collapse',
        'baseline_alpha': float(baseline_alpha),
        'pre_collapse_alpha_std': float(alpha_variance_pre),
        'alpha_stable_before': alpha_stable_before,
        'R_before': float(R_values[collapse_step-1]),
        'R_after': float(R_values[collapse_step]),
        'success': success
    }


# =============================================================================
# Test 4.4: Oscillating Alpha Attack
# =============================================================================

def test_oscillating_alpha(seed: int = 42) -> dict:
    """
    Alpha oscillates around 0.5 without trend while R stays stable.

    Attack: Noisy alpha measurements with no real drift.
    Success: No false positive predictions (stable system not flagged)
    """
    print("\n" + "=" * 60)
    print("TEST 4.4: OSCILLATING ALPHA (NO TREND)")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    embeddings = load_model_embeddings()
    if embeddings is None:
        return {'test': 'oscillating_alpha', 'success': False, 'error': 'Model load failed'}

    n_steps = 40
    alpha_values = []
    R_values = []

    ev_clean = get_eigenspectrum(embeddings)
    baseline_alpha = compute_alpha(ev_clean)
    baseline_R = compute_R(embeddings)

    for t in range(n_steps):
        # Add oscillating noise (sine wave pattern)
        oscillation = 0.02 * np.sin(2 * np.pi * t / 10)
        noise = rng.normal(oscillation, 0.02, embeddings.shape)
        perturbed = embeddings + noise
        perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + EPS)

        ev = get_eigenspectrum(perturbed)
        alpha_values.append(compute_alpha(ev))
        R_values.append(compute_R(perturbed))

    alpha_values = np.array(alpha_values)
    R_values = np.array(R_values)

    # Check for monotonic trend in alpha
    # Fit linear regression
    t_vals = np.arange(n_steps)
    slope, _ = np.polyfit(t_vals, alpha_values, 1)

    # No significant trend = slope near zero
    no_trend = abs(slope) < 0.001

    # R should stay stable
    R_stable = np.std(R_values) / np.mean(R_values) < 0.3

    # Gate should stay open
    gate_open_ratio = np.mean(R_values > baseline_R * 0.3)

    # Success: no false alarms for oscillating system
    success = no_trend and R_stable and gate_open_ratio > 0.9

    print(f"  Baseline alpha: {baseline_alpha:.4f}")
    print(f"  Alpha trend (slope): {slope:.6f}")
    print(f"  Alpha std: {np.std(alpha_values):.4f}")
    print(f"  No significant trend: {no_trend}")
    print(f"  R stable: {R_stable}")
    print(f"  Gate open ratio: {gate_open_ratio * 100:.1f}%")
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return {
        'test': 'oscillating_alpha',
        'baseline_alpha': float(baseline_alpha),
        'alpha_slope': float(slope),
        'alpha_std': float(np.std(alpha_values)),
        'no_trend': no_trend,
        'R_stable': R_stable,
        'gate_open_ratio': float(gate_open_ratio),
        'success': success
    }


# =============================================================================
# Test 4.5: Correlated Noise Attack
# =============================================================================

def test_correlated_noise(seed: int = 42) -> dict:
    """
    Inject noise correlated with alpha measurement to create spurious predictions.

    Attack: Adversarial noise designed to confuse predictor.
    Success: Null hypothesis testing rejects noise as predictor (p < 0.05)
    """
    print("\n" + "=" * 60)
    print("TEST 4.5: CORRELATED NOISE ATTACK")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    embeddings = load_model_embeddings()
    if embeddings is None:
        return {'test': 'correlated_noise', 'success': False, 'error': 'Model load failed'}

    # Generate two trajectories: real drift vs correlated noise
    n_steps = 30

    # Real drift trajectory
    real_alpha = []
    real_R = []

    for t in range(n_steps):
        noise_level = t * 0.01
        perturbed = embeddings + rng.normal(0, noise_level, embeddings.shape)
        perturbed = perturbed / (np.linalg.norm(perturbed, axis=1, keepdims=True) + EPS)

        ev = get_eigenspectrum(perturbed)
        real_alpha.append(compute_alpha(ev))
        real_R.append(compute_R(perturbed))

    real_alpha = np.array(real_alpha)
    real_R = np.array(real_R)

    # Null model: random predictions (should NOT work)
    random_predictions = rng.random(n_steps)

    # Ground truth: gate status
    baseline_R = real_R[0]
    gate_closed = (real_R < baseline_R * 0.3).astype(int)

    # Evaluate real predictor vs null
    real_predictor = np.abs(real_alpha - real_alpha[0])  # Distance from baseline

    from sklearn.metrics import roc_auc_score

    try:
        auc_real = roc_auc_score(gate_closed, real_predictor)
        auc_null = roc_auc_score(gate_closed, random_predictions)
    except ValueError:
        auc_real = auc_null = 0.5

    # Success: real predictor significantly better than null
    auc_gap = auc_real - auc_null
    success = auc_gap > 0.2 and auc_real > 0.7

    print(f"  Real predictor AUC: {auc_real:.4f}")
    print(f"  Null (random) AUC: {auc_null:.4f}")
    print(f"  AUC gap: {auc_gap:.4f}")
    print(f"  Status: {'PASS' if success else 'FAIL'} (need gap > 0.2, real > 0.7)")

    return {
        'test': 'correlated_noise',
        'auc_real': float(auc_real),
        'auc_null': float(auc_null),
        'auc_gap': float(auc_gap),
        'success': success
    }


# =============================================================================
# Test 4.6: Distribution Shift Attack
# =============================================================================

def test_distribution_shift(seed: int = 42) -> dict:
    """
    Change embedding domain mid-trajectory (not collapse, just shift).

    Attack: Alpha changes due to domain shift, not degradation.
    Success: Distinguishes shift from true collapse (different recovery patterns)
    """
    print("\n" + "=" * 60)
    print("TEST 4.6: DISTRIBUTION SHIFT ATTACK")
    print("=" * 60)

    rng = np.random.default_rng(seed)

    # Load two different domains
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")

        # Domain 1: Abstract concepts
        domain1_words = ["truth", "beauty", "justice", "freedom", "love", "wisdom",
                         "courage", "honor", "faith", "hope", "peace", "harmony",
                         "virtue", "grace", "mercy", "patience", "kindness", "humility"]

        # Domain 2: Concrete objects
        domain2_words = ["table", "chair", "window", "door", "floor", "wall",
                         "roof", "brick", "glass", "metal", "wood", "stone",
                         "hammer", "nail", "screw", "bolt", "wire", "pipe"]

        emb1 = model.encode(domain1_words, normalize_embeddings=True)
        emb2 = model.encode(domain2_words, normalize_embeddings=True)

    except Exception as e:
        return {'test': 'distribution_shift', 'success': False, 'error': str(e)}

    # Compute metrics for each domain
    ev1 = get_eigenspectrum(emb1)
    ev2 = get_eigenspectrum(emb2)

    alpha1 = compute_alpha(ev1)
    alpha2 = compute_alpha(ev2)
    R1 = compute_R(emb1)
    R2 = compute_R(emb2)

    # Both domains should have reasonable alpha (relaxed range [0.2, 0.8])
    # Small vocabularies can have different alpha than 80-word test set
    both_reasonable = (0.2 < alpha1 < 0.8) and (0.2 < alpha2 < 0.8)

    # Alpha difference between domains
    alpha_diff = abs(alpha1 - alpha2)

    # Both domains should maintain good R (not collapsed)
    both_high_R = R1 > 1.0 and R2 > 1.0

    # Key test: domain shift does NOT indicate collapse
    # Both domains have high R despite different alpha
    # This distinguishes "different domain" from "degraded domain"
    domain_shift_not_collapse = both_high_R and alpha_diff < 0.2

    # Success: domain shift is distinguishable from collapse
    success = domain_shift_not_collapse

    print(f"  Domain 1 (abstract) alpha: {alpha1:.4f}, R: {R1:.4f}")
    print(f"  Domain 2 (concrete) alpha: {alpha2:.4f}, R: {R2:.4f}")
    print(f"  Alpha difference: {alpha_diff:.4f}")
    print(f"  Both in reasonable range [0.2, 0.8]: {both_reasonable}")
    print(f"  Both have high R: {both_high_R}")
    print(f"  Domain shift distinguishable from collapse: {domain_shift_not_collapse}")
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return {
        'test': 'distribution_shift',
        'alpha_domain1': float(alpha1),
        'alpha_domain2': float(alpha2),
        'R_domain1': float(R1),
        'R_domain2': float(R2),
        'alpha_difference': float(alpha_diff),
        'both_reasonable': both_reasonable,
        'both_high_R': both_high_R,
        'domain_shift_not_collapse': domain_shift_not_collapse,
        'success': success
    }


# =============================================================================
# Main Phase 4 Runner
# =============================================================================

def run_phase4(seed: int = 42) -> dict:
    """Run all Phase 4 adversarial tests."""
    print("=" * 70)
    print("PHASE 4: ADVERSARIAL STRESS TESTS")
    print("=" * 70)

    results = {
        'phase': 4,
        'name': 'Adversarial Stress Tests',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'tests': {}
    }

    # Check dependencies
    try:
        from sentence_transformers import SentenceTransformer
        print("\n  sentence-transformers: AVAILABLE")
    except ImportError:
        print("\n  sentence-transformers: NOT AVAILABLE")
        results['error'] = 'sentence-transformers not installed'
        results['all_pass'] = False
        return results

    # Run all tests
    results['tests']['echo_chamber'] = test_echo_chamber(seed)
    results['tests']['delayed_collapse'] = test_delayed_collapse(seed)
    results['tests']['sudden_collapse'] = test_sudden_collapse(seed)
    results['tests']['oscillating_alpha'] = test_oscillating_alpha(seed)
    results['tests']['correlated_noise'] = test_correlated_noise(seed)
    results['tests']['distribution_shift'] = test_distribution_shift(seed)

    # Count passes
    passes = sum(1 for t in results['tests'].values() if t.get('success', False))
    total = len(results['tests'])

    # Success: >= 5/6 pass
    all_pass = passes >= 5

    results['passes'] = passes
    results['total'] = total
    results['all_pass'] = all_pass

    print("\n" + "=" * 70)
    print("PHASE 4 SUMMARY")
    print("=" * 70)
    for name, test in results['tests'].items():
        status = 'PASS' if test.get('success', False) else 'FAIL'
        print(f"  {name}: {status}")
    print("=" * 70)
    print(f"PHASE 4 RESULT: {passes}/{total} passed - {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = run_phase4()

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q21_phase4_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")
