#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Q21 Phase 2: Synthetic Controlled Transitions

Tests alpha-drift detection on synthetic trajectories with known ground truth.

Tests:
1. Drift Injection: Alpha drifts from 0.5 to 0.3, detection >= 5 steps before R drops
2. Sharp vs Gradual: Both transition types show detectable alpha change
3. Conservation Violation: d(Df*alpha)/dt > 3sigma predicts collapse

Success Criteria:
- Lead time >= 5 steps (alpha drift precedes R crash)
- Detection works for both sharp and gradual transitions
- Conservation violation is predictive
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from q21_temporal_utils import (
    CRITICAL_ALPHA, TARGET_DF_ALPHA, EPS,
    compute_temporal_alpha, compute_df_alpha_trajectory,
    detect_alpha_drift, detect_conservation_violation,
    compute_lead_time, evaluate_predictor, compute_cohens_d,
    generate_drifting_trajectory, generate_collapse_trajectory,
    generate_healthy_trajectory, compute_R, get_eigenspectrum,
    compute_df, compute_alpha
)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')


# =============================================================================
# Test 2.1: Drift Injection
# =============================================================================

def test_drift_injection(seed: int = 42) -> dict:
    """
    Test that alpha drift is detected BEFORE R drops.

    Scenario: Alpha drifts from ~0.5 to 0.3 starting at step 20.
    Success: Alpha drift detected at least 5 steps before R < threshold.
    """
    print("\n" + "=" * 60)
    print("TEST 2.1: DRIFT INJECTION")
    print("=" * 60)

    # Generate drifting trajectory
    n_steps = 60
    drift_start = 25
    trajectory, alpha_schedule = generate_drifting_trajectory(
        n_steps=n_steps,
        drift_start=drift_start,
        alpha_start=CRITICAL_ALPHA,
        alpha_end=0.25,  # Significant drift
        seed=seed
    )

    # Compute alpha and R trajectories
    alpha_result = compute_temporal_alpha(trajectory)
    alpha_traj = alpha_result['alpha_raw']

    R_traj = np.array([compute_R(emb) for emb in trajectory])
    gate_threshold = np.median(R_traj[:drift_start]) * 0.5  # Gate closes at 50% of baseline

    # Gate status: True = closed (R < threshold)
    gate_closed = R_traj < gate_threshold

    # Detect alpha drift
    drift_detection = detect_alpha_drift(
        alpha_traj,
        stable_alpha=np.mean(alpha_traj[:15]),  # Baseline from first 15 steps
        threshold_sigma=2.0,
        baseline_window=15
    )

    # Compute lead time
    lead_result = compute_lead_time(
        drift_detection['drift_detected'],
        gate_closed
    )

    # Find first R drop below threshold
    first_gate_close = np.where(gate_closed)[0]
    first_gate_close_idx = first_gate_close[0] if len(first_gate_close) > 0 else None

    # Results
    detected = drift_detection['first_detection_idx'] is not None
    if detected and first_gate_close_idx is not None:
        actual_lead_time = first_gate_close_idx - drift_detection['first_detection_idx']
    else:
        actual_lead_time = 0

    # Success criteria: lead time >= 5 steps
    success = actual_lead_time >= 5

    print(f"  Drift start (scheduled): step {drift_start}")
    print(f"  Alpha drift detected: step {drift_detection['first_detection_idx']}")
    print(f"  First gate closure: step {first_gate_close_idx}")
    print(f"  Lead time: {actual_lead_time} steps")
    print(f"  Gate threshold: {gate_threshold:.4f}")
    print(f"  R at start: {R_traj[0]:.4f}")
    print(f"  R at end: {R_traj[-1]:.4f}")
    print(f"  Status: {'PASS' if success else 'FAIL'} (need >= 5 steps lead)")

    return {
        'test': 'drift_injection',
        'drift_start': drift_start,
        'alpha_detection_idx': drift_detection['first_detection_idx'],
        'gate_closure_idx': first_gate_close_idx,
        'lead_time': actual_lead_time,
        'gate_threshold': float(gate_threshold),
        'success': success,
        'alpha_trajectory': alpha_traj.tolist(),
        'R_trajectory': R_traj.tolist(),
        'scheduled_alpha': alpha_schedule.tolist()
    }


# =============================================================================
# Test 2.2: Sharp vs Gradual Transitions
# =============================================================================

def test_sharp_vs_gradual(seed: int = 42) -> dict:
    """
    Test detection works for both sharp (step-function) and gradual (linear) drift.

    Success: Both types show positive lead time (detection before collapse).
    """
    print("\n" + "=" * 60)
    print("TEST 2.2: SHARP VS GRADUAL TRANSITIONS")
    print("=" * 60)

    results = {}
    rng = np.random.default_rng(seed)

    for transition_type in ['gradual', 'sharp']:
        print(f"\n  [{transition_type.upper()}]")

        n_steps = 50
        transition_start = 20

        if transition_type == 'gradual':
            # Linear drift over 20 steps
            trajectory, alpha_schedule = generate_drifting_trajectory(
                n_steps=n_steps,
                drift_start=transition_start,
                alpha_start=CRITICAL_ALPHA,
                alpha_end=0.2,
                seed=seed
            )
        else:
            # Sharp transition: instantaneous alpha change
            trajectory = []
            alpha_schedule = []
            dim = 384
            n_samples = 80

            for t in range(n_steps):
                if t < transition_start:
                    alpha_t = CRITICAL_ALPHA
                else:
                    alpha_t = 0.2  # Instant jump

                alpha_schedule.append(alpha_t)

                # Generate embeddings
                k = np.arange(1, dim + 1)
                eigenvalues = (k.astype(float)) ** (-alpha_t)
                eigenvalues = eigenvalues * dim / np.sum(eigenvalues)

                Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
                cov = Q @ np.diag(eigenvalues) @ Q.T

                try:
                    L = np.linalg.cholesky(cov + EPS * np.eye(dim))
                    z = rng.standard_normal((n_samples, dim))
                    embeddings = z @ L.T
                except np.linalg.LinAlgError:
                    embeddings = rng.standard_normal((n_samples, dim))

                trajectory.append(embeddings)

            alpha_schedule = np.array(alpha_schedule)

        # Compute trajectories
        alpha_result = compute_temporal_alpha(trajectory)
        alpha_traj = alpha_result['alpha_raw']
        R_traj = np.array([compute_R(emb) for emb in trajectory])

        # Gate threshold
        gate_threshold = np.median(R_traj[:transition_start]) * 0.6
        gate_closed = R_traj < gate_threshold

        # Detect drift
        drift_detection = detect_alpha_drift(
            alpha_traj,
            stable_alpha=np.mean(alpha_traj[:15]),
            threshold_sigma=2.0,
            baseline_window=15
        )

        # Lead time
        lead_result = compute_lead_time(
            drift_detection['drift_detected'],
            gate_closed
        )

        # First gate closure
        first_gate_close = np.where(gate_closed)[0]
        first_gate_close_idx = first_gate_close[0] if len(first_gate_close) > 0 else n_steps

        if drift_detection['first_detection_idx'] is not None:
            lead_time = first_gate_close_idx - drift_detection['first_detection_idx']
        else:
            lead_time = 0

        success = lead_time > 0

        print(f"    Detection idx: {drift_detection['first_detection_idx']}")
        print(f"    Gate closure idx: {first_gate_close_idx}")
        print(f"    Lead time: {lead_time} steps")
        print(f"    Status: {'PASS' if success else 'FAIL'}")

        results[transition_type] = {
            'detection_idx': drift_detection['first_detection_idx'],
            'gate_closure_idx': first_gate_close_idx,
            'lead_time': lead_time,
            'success': success
        }

    # Overall success: both types pass
    overall_success = results['gradual']['success'] and results['sharp']['success']

    print(f"\n  Overall: {'PASS' if overall_success else 'FAIL'}")

    return {
        'test': 'sharp_vs_gradual',
        'results': results,
        'overall_success': overall_success
    }


# =============================================================================
# Test 2.3: Conservation Violation as Predictor
# =============================================================================

def test_conservation_violation(seed: int = 42) -> dict:
    """
    Test that d(Df*alpha)/dt violation predicts collapse.

    Success: |d(Df*alpha)/dt| > 3*sigma_baseline predicts gate closure.
    """
    print("\n" + "=" * 60)
    print("TEST 2.3: CONSERVATION VIOLATION AS PREDICTOR")
    print("=" * 60)

    # Generate collapse trajectory
    n_steps = 60
    collapse_start = 30
    trajectory, R_values = generate_collapse_trajectory(
        n_steps=n_steps,
        collapse_start=collapse_start,
        collapse_speed=0.15,
        seed=seed
    )

    # Compute Df*alpha trajectory
    df_alpha_result = compute_df_alpha_trajectory(trajectory)
    df_alpha_traj = df_alpha_result['df_alpha_trajectory']

    # Detect conservation violation
    violation_result = detect_conservation_violation(
        df_alpha_traj,
        threshold_sigma=3.0,
        baseline_window=20
    )

    # Gate status
    R_traj = np.array(R_values)
    gate_threshold = np.median(R_traj[:collapse_start]) * 0.5
    gate_closed = R_traj < gate_threshold

    # Find first violation and first gate closure
    first_violation = violation_result['first_violation_idx']
    first_gate_close = np.where(gate_closed)[0]
    first_gate_close_idx = first_gate_close[0] if len(first_gate_close) > 0 else n_steps

    # Lead time
    if first_violation is not None:
        lead_time = first_gate_close_idx - first_violation
    else:
        lead_time = 0

    # Evaluate as predictor
    # Use violation magnitude as continuous predictor
    predictions = violation_result['violation_magnitude']
    ground_truth = gate_closed.astype(int)

    eval_result = evaluate_predictor(predictions, ground_truth)

    # Success criteria: AUC >= 0.7 and lead_time > 0
    success = eval_result.get('auc', 0) >= 0.7 and lead_time > 0

    print(f"  Collapse start: step {collapse_start}")
    print(f"  First violation detected: step {first_violation}")
    print(f"  First gate closure: step {first_gate_close_idx}")
    print(f"  Lead time: {lead_time} steps")
    print(f"  Baseline Df*alpha: {violation_result['baseline_mean']:.4f}")
    print(f"  Final Df*alpha: {df_alpha_traj[-1]:.4f}")
    print(f"  Target (8e): {TARGET_DF_ALPHA:.4f}")
    print(f"  Prediction AUC: {eval_result.get('auc', 0):.4f}")
    print(f"  Status: {'PASS' if success else 'FAIL'}")

    return {
        'test': 'conservation_violation',
        'collapse_start': collapse_start,
        'first_violation_idx': first_violation,
        'gate_closure_idx': first_gate_close_idx,
        'lead_time': lead_time,
        'baseline_df_alpha': float(violation_result['baseline_mean']),
        'final_df_alpha': float(df_alpha_traj[-1]),
        'auc': eval_result.get('auc', 0),
        'success': success,
        'df_alpha_trajectory': df_alpha_traj.tolist(),
        'R_trajectory': R_traj.tolist()
    }


# =============================================================================
# Test 2.4: Effect Size (Cohen's d)
# =============================================================================

def test_effect_size(seed: int = 42) -> dict:
    """
    Test that alpha values before vs after drift show large effect size.

    Success: Cohen's d >= 1.0 (large effect)
    """
    print("\n" + "=" * 60)
    print("TEST 2.4: EFFECT SIZE (COHEN'S D)")
    print("=" * 60)

    # Generate trajectory with clear drift
    n_steps = 60
    drift_start = 25
    trajectory, _ = generate_drifting_trajectory(
        n_steps=n_steps,
        drift_start=drift_start,
        alpha_start=CRITICAL_ALPHA,
        alpha_end=0.2,
        seed=seed
    )

    alpha_result = compute_temporal_alpha(trajectory)
    alpha_traj = alpha_result['alpha_raw']

    # Split into before and after drift
    alpha_before = alpha_traj[:drift_start]
    alpha_after = alpha_traj[drift_start + 10:]  # Skip transition period

    # Compute Cohen's d
    d = compute_cohens_d(alpha_before, alpha_after)

    # Success: |d| >= 1.0 (large effect)
    success = abs(d) >= 1.0

    print(f"  Mean alpha (before): {np.mean(alpha_before):.4f}")
    print(f"  Mean alpha (after): {np.mean(alpha_after):.4f}")
    print(f"  Std alpha (before): {np.std(alpha_before):.4f}")
    print(f"  Std alpha (after): {np.std(alpha_after):.4f}")
    print(f"  Cohen's d: {d:.4f}")
    print(f"  Status: {'PASS' if success else 'FAIL'} (need |d| >= 1.0)")

    return {
        'test': 'effect_size',
        'mean_before': float(np.mean(alpha_before)),
        'mean_after': float(np.mean(alpha_after)),
        'std_before': float(np.std(alpha_before)),
        'std_after': float(np.std(alpha_after)),
        'cohens_d': float(d),
        'success': success
    }


# =============================================================================
# Main Phase 2 Runner
# =============================================================================

def run_phase2(seed: int = 42) -> dict:
    """Run all Phase 2 synthetic drift tests."""
    print("=" * 70)
    print("PHASE 2: SYNTHETIC CONTROLLED TRANSITIONS")
    print("=" * 70)

    results = {
        'phase': 2,
        'name': 'Synthetic Controlled Transitions',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'tests': {}
    }

    # Run all tests
    results['tests']['drift_injection'] = test_drift_injection(seed)
    results['tests']['sharp_vs_gradual'] = test_sharp_vs_gradual(seed)
    results['tests']['conservation_violation'] = test_conservation_violation(seed)
    results['tests']['effect_size'] = test_effect_size(seed)

    # Overall result
    all_pass = all([
        results['tests']['drift_injection']['success'],
        results['tests']['sharp_vs_gradual']['overall_success'],
        results['tests']['conservation_violation']['success'],
        results['tests']['effect_size']['success']
    ])

    results['all_pass'] = all_pass

    print("\n" + "=" * 70)
    print("PHASE 2 SUMMARY")
    print("=" * 70)
    print(f"  Drift Injection: {'PASS' if results['tests']['drift_injection']['success'] else 'FAIL'}")
    print(f"  Sharp vs Gradual: {'PASS' if results['tests']['sharp_vs_gradual']['overall_success'] else 'FAIL'}")
    print(f"  Conservation Violation: {'PASS' if results['tests']['conservation_violation']['success'] else 'FAIL'}")
    print(f"  Effect Size: {'PASS' if results['tests']['effect_size']['success'] else 'FAIL'}")
    print("=" * 70)
    print(f"PHASE 2 RESULT: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 70)

    return results


if __name__ == '__main__':
    results = run_phase2()

    # Save results
    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    path = results_dir / f'q21_phase2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'

    # Remove large arrays for JSON storage (keep summaries)
    for test_name in results['tests']:
        for key in list(results['tests'][test_name].keys()):
            if 'trajectory' in key.lower():
                del results['tests'][test_name][key]

    with open(path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved: {path}")
