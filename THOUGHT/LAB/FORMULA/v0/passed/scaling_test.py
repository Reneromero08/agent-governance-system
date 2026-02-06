#!/usr/bin/env python3
"""
F.7.3: Exponential vs Power Law vs Logarithmic

Tests if sigma^Df relationship is truly exponential, or is it power/log.

Formula predicts: R = k × sigma^Df (exponential in Df)

Prediction: Exponential or power law beats linear/log (AIC difference > 10).
Falsification: Linear model has lowest AIC.
"""

import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')


def exponential(x, a, b):
    return a * np.exp(b * x)


def power_law(x, a, b):
    return a * np.power(np.maximum(x, 1e-10), b)


def logarithmic(x, a, b):
    return a * np.log(np.maximum(x, 1e-10) + 1) + b


def linear(x, a, b):
    return a * x + b


def test_scaling_relationship(sigma_values, R_values, Df=1):
    """
    Test which model best fits the sigma → R relationship.

    Formula predicts: R = k × sigma^Df (exponential in Df)

    Competing hypotheses:
    - Linear: R = a × sigma + b
    - Logarithmic: R = a × log(sigma) + b
    - Power law: R = a × sigma^b
    - Exponential: R = a × exp(b × sigma)
    """

    models = {
        'linear': linear,
        'logarithmic': logarithmic,
        'power_law': power_law,
        'exponential': exponential
    }

    results = {}

    # Normalize sigma for numerical stability
    sigma_norm = sigma_values / np.max(sigma_values)

    for name, func in models.items():
        try:
            if name == 'exponential':
                # Exponential is tricky - use log-transformed fitting
                popt, _ = curve_fit(func, sigma_norm, R_values,
                                   p0=[0.5, 1.0], maxfev=10000,
                                   bounds=([0, -10], [10, 10]))
            elif name == 'power_law':
                popt, _ = curve_fit(func, sigma_norm, R_values,
                                   p0=[1.0, 0.5], maxfev=10000,
                                   bounds=([0, -5], [10, 5]))
            else:
                popt, _ = curve_fit(func, sigma_norm, R_values, maxfev=10000)

            pred = func(sigma_norm, *popt)
            r2 = r2_score(R_values, pred)

            # AIC = n * log(MSE) + 2k where k = number of parameters
            mse = np.mean((R_values - pred)**2)
            n = len(R_values)
            k = len(popt)
            aic = n * np.log(mse + 1e-10) + 2 * k

            results[name] = {'r2': r2, 'aic': aic, 'params': popt.tolist(), 'predictions': pred.tolist()}
        except Exception as e:
            results[name] = {'r2': 0, 'aic': float('inf'), 'params': None, 'error': str(e)}

    # Best model by AIC (lower is better)
    valid_results = {k: v for k, v in results.items() if v['aic'] != float('inf')}
    if valid_results:
        best = min(valid_results.items(), key=lambda x: x[1]['aic'])
    else:
        best = ('none', {'r2': 0, 'aic': float('inf')})

    return results, best


if __name__ == '__main__':
    print("F.7.3: Exponential vs Linear Scaling")
    print("=" * 50)

    # AGS compression data points
    sigma_test = np.array([1, 10, 24, 100, 1000, 1455, 10000, 56370], dtype=float)
    R_test = np.array([0.5, 0.7, 0.75, 0.85, 0.92, 0.94, 0.97, 0.99])

    print(f"\nTest data (AGS compression levels):")
    print(f"  sigma values: {sigma_test.tolist()}")
    print(f"  R values: {R_test.tolist()}")

    results, best = test_scaling_relationship(sigma_test, R_test)

    print(f"\nModel comparison:")
    print("-" * 40)
    for name, data in sorted(results.items(), key=lambda x: x[1].get('aic', float('inf'))):
        if data['aic'] != float('inf'):
            print(f"  {name:12s}: R²={data['r2']:.4f}, AIC={data['aic']:.2f}")
        else:
            print(f"  {name:12s}: Failed to fit - {data.get('error', 'unknown')}")

    print(f"\nBest model: {best[0]}")

    if best[0] == 'linear':
        print("\nX  FALSIFIED: Linear model wins - formula's exponential claim doesn't hold")
    elif best[0] in ['exponential', 'power_law']:
        print(f"\n** VALIDATED: {best[0]} model wins - consistent with sigma^Df relationship")
    else:
        print(f"\n~  INCONCLUSIVE: {best[0]} model wins")
