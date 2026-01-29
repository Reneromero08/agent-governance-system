#!/usr/bin/env python3
"""
F.7.9: Monte Carlo Robustness

Tests if the formula is robust to noise in measurements.

If formula is robust: small input noise → small output variance
If formula is brittle: small input noise → large output variance

Prediction: CV < 0.5 with 10% input noise.
Falsification: CV > 1.0 (formula amplifies noise unacceptably).
"""

import numpy as np


def monte_carlo_robustness(n_trials=10000):
    """
    Add measurement noise to E, ∇S, σ, Df and check if R predictions remain stable.

    If formula is robust: small input noise → small output variance
    If formula is brittle: small input noise → large output variance
    """

    # True values (representative of AGS scenario)
    E_true = 1.0
    nabla_S_true = 0.5
    sigma_true = 100
    Df_true = 2.0

    R_samples = []

    for _ in range(n_trials):
        # Add 10% measurement noise
        E = E_true * (1 + 0.1 * np.random.randn())
        nabla_S = nabla_S_true * (1 + 0.1 * np.random.randn())
        sigma = sigma_true * (1 + 0.1 * np.random.randn())
        Df = Df_true * (1 + 0.1 * np.random.randn())

        # Ensure positive values
        E = max(E, 0.01)
        nabla_S = max(nabla_S, 0.01)
        sigma = max(sigma, 1)
        Df = max(Df, 0.1)

        # Calculate R
        R = (E / nabla_S) * (sigma ** Df)
        R_samples.append(R)

    R_mean = np.mean(R_samples)
    R_std = np.std(R_samples)
    R_cv = R_std / R_mean if R_mean > 0 else float('inf')  # Coefficient of variation

    # True R for comparison
    R_true = (E_true / nabla_S_true) * (sigma_true ** Df_true)

    # Sensitivity analysis: which variable contributes most to variance?
    def calc_R(E, nabla_S, sigma, Df):
        return (E / nabla_S) * (sigma ** Df)

    sensitivity = {}

    # E sensitivity
    E_varied = [E_true * (1 + 0.1 * np.random.randn()) for _ in range(1000)]
    E_varied = [max(e, 0.01) for e in E_varied]
    sensitivity['E'] = np.std([calc_R(e, nabla_S_true, sigma_true, Df_true) for e in E_varied])

    # nabla_S sensitivity
    nabla_S_varied = [nabla_S_true * (1 + 0.1 * np.random.randn()) for _ in range(1000)]
    nabla_S_varied = [max(n, 0.01) for n in nabla_S_varied]
    sensitivity['nabla_S'] = np.std([calc_R(E_true, n, sigma_true, Df_true) for n in nabla_S_varied])

    # sigma sensitivity
    sigma_varied = [sigma_true * (1 + 0.1 * np.random.randn()) for _ in range(1000)]
    sigma_varied = [max(s, 1) for s in sigma_varied]
    sensitivity['sigma'] = np.std([calc_R(E_true, nabla_S_true, s, Df_true) for s in sigma_varied])

    # Df sensitivity
    Df_varied = [Df_true * (1 + 0.1 * np.random.randn()) for _ in range(1000)]
    Df_varied = [max(d, 0.1) for d in Df_varied]
    sensitivity['Df'] = np.std([calc_R(E_true, nabla_S_true, sigma_true, d) for d in Df_varied])

    return {
        'R_mean': R_mean,
        'R_std': R_std,
        'R_cv': R_cv,
        'R_true': R_true,
        'bias': (R_mean - R_true) / R_true if R_true > 0 else 0,
        'sensitivity': sensitivity
    }


def monte_carlo_extreme_scenarios(n_trials=1000):
    """
    Test formula stability under extreme parameter combinations.
    """
    scenarios = {
        'low_essence': {'E': 0.1, 'nabla_S': 0.5, 'sigma': 100, 'Df': 2},
        'high_entropy': {'E': 1.0, 'nabla_S': 2.0, 'sigma': 100, 'Df': 2},
        'extreme_compression': {'E': 1.0, 'nabla_S': 0.5, 'sigma': 10000, 'Df': 2},
        'high_fractal': {'E': 1.0, 'nabla_S': 0.5, 'sigma': 100, 'Df': 5},
        'balanced': {'E': 1.0, 'nabla_S': 0.5, 'sigma': 100, 'Df': 2},
    }

    results = {}

    for name, params in scenarios.items():
        R_samples = []

        for _ in range(n_trials):
            # Add 10% noise to all parameters
            E = params['E'] * (1 + 0.1 * np.random.randn())
            nabla_S = params['nabla_S'] * (1 + 0.1 * np.random.randn())
            sigma = params['sigma'] * (1 + 0.1 * np.random.randn())
            Df = params['Df'] * (1 + 0.1 * np.random.randn())

            # Ensure valid values
            E = max(E, 0.01)
            nabla_S = max(nabla_S, 0.01)
            sigma = max(sigma, 1)
            Df = max(Df, 0.1)

            R = (E / nabla_S) * (sigma ** Df)
            R_samples.append(R)

        R_mean = np.mean(R_samples)
        R_std = np.std(R_samples)
        cv = R_std / R_mean if R_mean > 0 else float('inf')

        results[name] = {
            'params': params,
            'R_mean': R_mean,
            'R_std': R_std,
            'cv': cv
        }

    return results


if __name__ == '__main__':
    print("F.7.9: Monte Carlo Robustness")
    print("=" * 50)

    print("\nRunning 10,000 trials with 10% measurement noise...")
    result = monte_carlo_robustness()

    print(f"\nTrue R: {result['R_true']:.4f}")
    print(f"Mean R: {result['R_mean']:.4f}")
    print(f"Std R:  {result['R_std']:.4f}")
    print(f"CV:     {result['R_cv']:.4f}")
    print(f"Bias:   {result['bias']:.4%}")

    print(f"\nSensitivity analysis (std contribution):")
    print("-" * 40)
    total_sens = sum(result['sensitivity'].values())
    for var, val in sorted(result['sensitivity'].items(), key=lambda x: -x[1]):
        pct = val / total_sens * 100 if total_sens > 0 else 0
        print(f"  {var:10s}: {val:12.2f} ({pct:5.1f}%)")

    if result['R_cv'] < 0.5:
        print("\n** VALIDATED: Formula is robust (CV < 0.5)")
    elif result['R_cv'] < 1.0:
        print("\n*  PASS: Formula is moderately robust (CV < 1.0)")
    else:
        print("\nX  FALSIFIED: Formula amplifies noise unacceptably (CV > 1.0)")

    # Extreme scenarios
    print("\n" + "=" * 50)
    print("Extreme Scenario Analysis")
    print("=" * 50)

    extreme_results = monte_carlo_extreme_scenarios()

    print(f"\n{'Scenario':20s} | {'R_mean':>12s} | {'CV':>8s}")
    print("-" * 50)
    for name, data in extreme_results.items():
        print(f"{name:20s} | {data['R_mean']:12.2f} | {data['cv']:8.4f}")

    # Check which scenario is most sensitive
    most_sensitive = max(extreme_results.items(), key=lambda x: x[1]['cv'])
    print(f"\nMost sensitive scenario: {most_sensitive[0]} (CV={most_sensitive[1]['cv']:.4f})")
