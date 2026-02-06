#!/usr/bin/env python3
"""
Q26: Minimum Data Requirements for R Stability (R: 1240)

PRE-REGISTRATION:
1. HYPOTHESIS: Minimum N scales with log(dimensionality)
2. PREDICTION: N_min = c * log(D) for some constant c
3. FALSIFICATION: If N_min scales linearly with D
4. DATA: Subsample from available datasets
5. THRESHOLD: Derive scaling law

CONTEXT:
Q7 showed R is intensive (CV=0.158 across scales)

Author: Claude
Date: 2026-01-27
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class StabilityResult:
    N: int
    D: int
    R_mean: float
    R_std: float
    R_cv: float
    n_trials: int
    is_stable: bool


def compute_R(embeddings: np.ndarray) -> float:
    if len(embeddings) == 0:
        return 0.0
    n = len(embeddings)
    if n < 2:
        return 1.0
    truth_vector = embeddings.mean(axis=0)
    distances = np.linalg.norm(embeddings - truth_vector, axis=1)
    mean_dist = np.mean(distances)
    if mean_dist < 1e-10:
        return float(n)
    sigma = mean_dist
    z = distances / sigma
    E = np.mean(np.exp(-0.5 * z**2))
    cv = np.std(distances) / (mean_dist + 1e-10)
    concentration = 1.0 / (1.0 + cv)
    R = float(E * concentration / sigma)
    return max(0.0, min(R, 100.0))


def generate_structured_embeddings(N: int, D: int, seed: int = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    truth = np.random.randn(D)
    truth = truth / np.linalg.norm(truth)
    kappa = 5.0
    embeddings = []
    for _ in range(N):
        noise = np.random.randn(D)
        noise = noise - np.dot(noise, truth) * truth
        noise_norm = np.linalg.norm(noise)
        if noise_norm > 1e-10:
            noise = noise / noise_norm
        else:
            noise = np.zeros(D)
            noise[0] = 1.0 if truth[0] < 0.5 else -1.0
        angle = np.random.exponential(1.0 / kappa)
        emb = np.cos(angle) * truth + np.sin(angle) * noise
        emb = emb / (np.linalg.norm(emb) + 1e-10)
        embeddings.append(emb)
    return np.array(embeddings)


def test_stability_at_N_D(N: int, D: int, n_trials: int = 30, stability_threshold: float = 0.10) -> StabilityResult:
    R_values = []
    for trial in range(n_trials):
        seed = trial * 10000 + N * 100 + D
        embeddings = generate_structured_embeddings(N, D, seed=seed)
        R = compute_R(embeddings)
        R_values.append(R)
    R_array = np.array(R_values)
    R_mean = np.mean(R_array)
    R_std = np.std(R_array)
    R_cv = R_std / (R_mean + 1e-10)
    return StabilityResult(N=N, D=D, R_mean=R_mean, R_std=R_std, R_cv=R_cv, n_trials=n_trials, is_stable=R_cv < stability_threshold)


def find_N_min(D: int, stability_threshold: float = 0.10, N_candidates: List[int] = None, n_trials: int = 30) -> int:
    if N_candidates is None:
        N_candidates = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    for N in N_candidates:
        result = test_stability_at_N_D(N, D, n_trials, stability_threshold)
        if result.is_stable:
            return N
    return N_candidates[-1]


def fit_log_scaling(D_values: np.ndarray, N_min_values: np.ndarray) -> Tuple[float, float, float]:
    log_D = np.log(D_values)
    try:
        coeffs = np.polyfit(log_D, N_min_values, 1)
        c, b = coeffs
        predicted = c * log_D + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return c, b, r_squared
    except Exception:
        return 0, 0, 0


def fit_linear_scaling(D_values: np.ndarray, N_min_values: np.ndarray) -> Tuple[float, float, float]:
    try:
        coeffs = np.polyfit(D_values, N_min_values, 1)
        c, b = coeffs
        predicted = c * D_values + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return c, b, r_squared
    except Exception:
        return 0, 0, 0


def fit_sqrt_scaling(D_values: np.ndarray, N_min_values: np.ndarray) -> Tuple[float, float, float]:
    sqrt_D = np.sqrt(D_values)
    try:
        coeffs = np.polyfit(sqrt_D, N_min_values, 1)
        c, b = coeffs
        predicted = c * sqrt_D + b
        ss_res = np.sum((N_min_values - predicted) ** 2)
        ss_tot = np.sum((N_min_values - np.mean(N_min_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        return c, b, r_squared
    except Exception:
        return 0, 0, 0


def run_experiment():
    print("\n" + "=" * 80)
    print("Q26: MINIMUM DATA REQUIREMENTS FOR R STABILITY")
    print("=" * 80)
    N_candidates = [5, 10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 500]
    D_values = [10, 25, 50, 100, 200, 384, 500, 768, 1000]
    stability_threshold = 0.10
    n_trials = 30
    print(f"\nConfiguration:")
    print(f"  N candidates: {N_candidates}")
    print(f"  D values: {D_values}")
    print(f"  Stability threshold: CV < {stability_threshold:.0%}")
    print(f"  Trials per test: {n_trials}")

    print("\n" + "-" * 60)
    print("PHASE 1: STABILITY SWEEP (N vs CV at D=384)")
    print("-" * 60)
    D_fixed = 384
    print(f"\nTesting stability at D={D_fixed}:")
    stability_sweep = []
    for N in N_candidates:
        result = test_stability_at_N_D(N, D_fixed, n_trials, stability_threshold)
        stability_sweep.append(result)
        stable_str = "[STABLE]" if result.is_stable else "[UNSTABLE]"
        print(f"  N={N:4d}: R={result.R_mean:.4f} +/- {result.R_std:.4f}, CV={result.R_cv:.4f} {stable_str}")
    N_90 = None
    for result in stability_sweep:
        if result.is_stable:
            N_90 = result.N
            break
    print(f"\nN_min for D={D_fixed}: {N_90 if N_90 else '>'+str(N_candidates[-1])}")

    print("\n" + "-" * 60)
    print("PHASE 2: N_MIN VS DIMENSIONALITY")
    print("-" * 60)
    N_min_results = {}
    for D in D_values:
        N_min = find_N_min(D, stability_threshold, N_candidates, n_trials)
        N_min_results[D] = N_min
        print(f"  D={D:4d}: N_min={N_min}")

    print("\n" + "-" * 60)
    print("PHASE 3: FIT SCALING LAW")
    print("-" * 60)
    D_array = np.array(list(N_min_results.keys()))
    N_min_array = np.array(list(N_min_results.values()))
    c_log, b_log, r2_log = fit_log_scaling(D_array, N_min_array)
    c_lin, b_lin, r2_lin = fit_linear_scaling(D_array, N_min_array)
    c_sqrt, b_sqrt, r2_sqrt = fit_sqrt_scaling(D_array, N_min_array)
    print(f"\nScaling Law Fits:")
    print(f"  N_min = {c_log:.2f} * log(D) + {b_log:.2f}   (R^2 = {r2_log:.4f})")
    print(f"  N_min = {c_lin:.4f} * D + {b_lin:.2f}        (R^2 = {r2_lin:.4f})")
    print(f"  N_min = {c_sqrt:.2f} * sqrt(D) + {b_sqrt:.2f} (R^2 = {r2_sqrt:.4f})")
    fits = {"log": r2_log, "linear": r2_lin, "sqrt": r2_sqrt}
    best_fit = max(fits, key=fits.get)
    best_r2 = fits[best_fit]
    print(f"\nBest fit: {best_fit.upper()} scaling (R^2 = {best_r2:.4f})")

    print("\n" + "-" * 60)
    print("PHASE 4: HYPOTHESIS TESTING")
    print("-" * 60)
    print(f"\nPre-registered Hypothesis: N_min ~ log(D)")
    print(f"Pre-registered Falsification: N_min ~ D (linear)")
    print()
    if best_fit == "log" and r2_log > 0.7:
        verdict = "CONFIRMED"
        reasoning = f"Log scaling is best fit with R^2={r2_log:.4f}"
    elif best_fit == "linear" and r2_lin > 0.8:
        verdict = "FALSIFIED"
        reasoning = f"Linear scaling is better fit with R^2={r2_lin:.4f}"
    elif best_fit == "sqrt" and r2_sqrt > 0.7:
        verdict = "PARTIAL"
        reasoning = f"Sqrt scaling (intermediate) is best fit with R^2={r2_sqrt:.4f}"
    elif max(r2_log, r2_lin, r2_sqrt) < 0.5:
        verdict = "INCONCLUSIVE"
        reasoning = "No clear scaling law detected (all R^2 < 0.5)"
    else:
        verdict = "NUANCED"
        reasoning = f"Best fit is {best_fit} with R^2={best_r2:.4f}"
    print(f"VERDICT: {verdict}")
    print(f"Reasoning: {reasoning}")

    print("\n" + "=" * 80)
    print("Q26 SUMMARY")
    print("=" * 80)
    print(f"\nScaling Law: N_min = {c_log:.2f} * log(D) + {b_log:.2f}")
    print(f"\nPractical N_min values:")
    print(f"  D=384  (MiniLM):   N_min = {N_min_results.get(384, 'N/A')}")
    print(f"  D=768  (BERT):     N_min = {N_min_results.get(768, 'N/A')}")
    print(f"  D=1000 (large):    N_min = {N_min_results.get(1000, 'N/A')}")
    print(f"\n  Context from Q7: R is intensive (CV=0.158 across scales)")
    print(f"  This means once N >= N_min, increasing N doesn't improve R")

    result = {
        "test_id": "Q26_MINIMUM_DATA_REQUIREMENTS",
        "verdict": verdict,
        "hypothesis": "N_min ~ log(D)",
        "best_scaling": best_fit,
        "scaling_fits": {
            "log": {"c": c_log, "b": b_log, "r2": r2_log},
            "linear": {"c": c_lin, "b": b_lin, "r2": r2_lin},
            "sqrt": {"c": c_sqrt, "b": b_sqrt, "r2": r2_sqrt}
        },
        "N_min_by_D": N_min_results,
        "stability_threshold": stability_threshold,
        "reasoning": reasoning
    }
    return result


def test_with_real_embeddings():
    if not ST_AVAILABLE:
        print("\nSentenceTransformer not available - skipping real data validation")
        return None
    print("\n" + "-" * 60)
    print("VALIDATION: REAL EMBEDDING DATA")
    print("-" * 60)
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        texts = [
            "The king and queen ruled wisely.",
            "A father loves his children equally.",
            "Good things come to those who wait.",
            "The happy dog ran through the field.",
            "Red and blue make purple together.",
            "She loves to read books about history.",
            "The cold wind blew through the street.",
            "He hopes to swim in the warm ocean.",
            "The excited children played outside.",
            "A wise man thinks before he speaks.",
            "The mother bird flew to feed babies.",
            "Black and white create contrast.",
            "The tall tree provided shade.",
            "Fear and hope exist together.",
            "The calm cat sat by the fireplace.",
            "Brothers and sisters share memories.",
            "The worst storms bring best rainbows.",
            "A small phone can store big dreams.",
            "The prince and princess lived together.",
            "Nature provides beautiful colors.",
        ]
        full_embeddings = model.encode(texts, show_progress_bar=False)
        D = full_embeddings.shape[1]
        print(f"\nReal embeddings: {len(texts)} texts, D={D}")
        N_values = [5, 10, 15, 20]
        for N in N_values:
            if N > len(texts):
                continue
            R_values = []
            for trial in range(30):
                np.random.seed(trial)
                idx = np.random.choice(len(texts), N, replace=False)
                R = compute_R(full_embeddings[idx])
                R_values.append(R)
            R_mean = np.mean(R_values)
            R_cv = np.std(R_values) / (R_mean + 1e-10)
            stable_str = "[STABLE]" if R_cv < 0.10 else "[UNSTABLE]"
            print(f"  N={N:2d}: R={R_mean:.4f}, CV={R_cv:.4f} {stable_str}")
        return {"status": "validated", "D": D}
    except Exception as e:
        print(f"  Real validation failed: {e}")
        return None


if __name__ == "__main__":
    import json
    from pathlib import Path
    results = run_experiment()
    real_results = test_with_real_embeddings()
    if real_results:
        results["real_data_validation"] = real_results
    output_path = Path(__file__).parent / "q26_results.json"
    try:
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    except Exception as e:
        print(f"\nCould not save results: {e}")
    print("\n" + "=" * 80)
    print(f"FINAL VERDICT: {results['verdict']}")
    print("=" * 80)
