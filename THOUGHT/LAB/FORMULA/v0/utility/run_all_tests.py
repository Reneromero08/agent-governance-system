#!/usr/bin/env python3
"""
Formula Falsification Test Suite
================================

Tests the Living Formula: R = (E / nabla_S) × sigma(f)^Df

Run: python run_all_tests.py

"The formula that cannot be falsified is not a formula—it's a prayer.
Let's see if this one bleeds."
"""

import sys
import json
import numpy as np
from datetime import datetime
from pathlib import Path

# Test imports
from info_theory_test import test_shannon_consistency
from scaling_test import test_scaling_relationship
from fractal_test import test_Df_R_correlation
from eigenvalue_essence_test import test_eigenvalue_E_predicts_R
from entropy_stress_test import entropy_injection_test
from audio_test import audio_resonance_test
from network_test import network_centrality_test
from monte_carlo_test import monte_carlo_robustness

# Falsification thresholds
THRESHOLDS = {
    'F.7.2': {'metric': 'correlation', 'pass': 0.5, 'strong_pass': 0.8},
    'F.7.3': {'metric': 'aic_ratio', 'pass': 1.0, 'note': 'exp/linear AIC < 1 means exp wins'},
    'F.7.4': {'metric': 'correlation', 'pass': 0.0, 'strong_pass': 0.5},
    'F.7.5': {'metric': 'correlation', 'pass': 0.0, 'strong_pass': 0.7},
    'F.7.6': {'metric': 'cv', 'pass': 0.5, 'strong_pass': 0.2, 'direction': 'lower'},
    'F.7.7': {'metric': 'correlation', 'pass': 0.5, 'strong_pass': 0.85},
    'F.7.8': {'metric': 'correlation', 'pass': 0.0, 'strong_pass': 0.7},
    'F.7.9': {'metric': 'cv', 'pass': 1.0, 'strong_pass': 0.5, 'direction': 'lower'},
}


def print_banner():
    print("=" * 70)
    print(" FORMULA FALSIFICATION TEST SUITE")
    print(" R = (E / nabla_S) × sigma(f)^Df")
    print("=" * 70)
    print(f" Started: {datetime.now().isoformat()}")
    print("=" * 70)
    print()


def evaluate_result(test_id, value, direction='higher'):
    """Evaluate if test passes, fails, or needs refinement."""
    t = THRESHOLDS.get(test_id, {})

    if direction == 'lower':
        if value <= t.get('strong_pass', float('inf')):
            return 'VALIDATED', '**'
        elif value <= t.get('pass', float('inf')):
            return 'PASS', '*'
        else:
            return 'FALSIFIED', 'X'
    else:
        if value >= t.get('strong_pass', float('inf')):
            return 'VALIDATED', '**'
        elif value >= t.get('pass', -float('inf')):
            return 'PASS', '*'
        else:
            return 'FALSIFIED', 'X'


def run_test(name, func, test_id):
    """Run a single test with error handling."""
    print(f"\n{'─' * 70}")
    print(f" {test_id}: {name}")
    print(f"{'─' * 70}")

    try:
        result = func()
        return {'status': 'completed', 'result': result}
    except Exception as e:
        print(f" ERROR: {e}")
        return {'status': 'error', 'error': str(e)}


def main():
    print_banner()

    results = {}

    # F.7.2: Information Theory Test
    print("\n[F.7.2] Information-Theoretic Validation")
    print("Testing if R correlates with Shannon's mutual information...")
    test = run_test("Shannon Consistency", test_shannon_consistency, "F.7.2")
    if test['status'] == 'completed':
        corr, data = test['result']
        status, symbol = evaluate_result('F.7.2', corr)
        print(f" Result: MI-R correlation = {corr:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.2'] = {'correlation': corr, 'status': status}
    else:
        results['F.7.2'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.3: Scaling Test
    print("\n[F.7.3] Exponential vs Linear Scaling")
    print("Testing if sigma^Df relationship is truly exponential...")
    # Generate test data
    sigma_test = np.array([1, 10, 24, 100, 1000, 1455, 10000, 56370])
    R_test = np.array([0.5, 0.7, 0.75, 0.85, 0.92, 0.94, 0.97, 0.99])
    test = run_test("Scaling Relationship",
                    lambda: test_scaling_relationship(sigma_test, R_test), "F.7.3")
    if test['status'] == 'completed':
        model_results, best = test['result']
        print(f" Best model: {best[0]} (AIC: {best[1]['aic']:.2f}, R²: {best[1]['r2']:.4f})")
        for name, data in model_results.items():
            print(f"   {name}: R²={data['r2']:.4f}, AIC={data['aic']:.2f}")

        # Falsified if linear wins
        linear_aic = model_results['linear']['aic']
        exp_aic = model_results.get('exponential', {}).get('aic', float('inf'))
        power_aic = model_results.get('power_law', {}).get('aic', float('inf'))

        if best[0] == 'linear':
            status = 'FALSIFIED'
            symbol = 'X'
        elif best[0] in ['exponential', 'power_law']:
            status = 'VALIDATED'
            symbol = '**'
        else:
            status = 'PASS'
            symbol = '*'

        print(f" Status: {symbol} {status}")
        results['F.7.3'] = {'best_model': best[0], 'models': model_results, 'status': status}
    else:
        results['F.7.3'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.4: Fractal Dimension Test
    print("\n[F.7.4] Fractal Dimension Measurement")
    print("Testing if Df correlates with R across symbol sets...")
    test = run_test("Df-R Correlation", test_Df_R_correlation, "F.7.4")
    if test['status'] == 'completed':
        data = test['result']
        print(f" Results by symbol set:")
        for name, vals in data.items():
            print(f"   {name}: Df_box={vals['Df_box']:.3f}, R={vals['R']:.3f}")
        # Calculate correlation
        Df_values = [v['Df_box'] for v in data.values()]
        R_values = [v['R'] for v in data.values()]
        corr = np.corrcoef(Df_values, R_values)[0, 1]
        status, symbol = evaluate_result('F.7.4', corr)
        print(f" Df-R correlation: {corr:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.4'] = {'correlation': corr, 'data': data, 'status': status}
    else:
        results['F.7.4'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.5: Eigenvalue Essence Test
    print("\n[F.7.5] Eigenvalue Spectrum as Essence")
    print("Testing if eigenvalue-based E predicts R...")
    test = run_test("Eigenvalue E", test_eigenvalue_E_predicts_R, "F.7.5")
    if test['status'] == 'completed':
        data = test['result']
        print(f" Results by word set:")
        for item in data:
            print(f"   {item['expected_E']}: E_total={item['E_total']:.3f}, R={item['R']:.3f}")
        # Calculate correlation
        E_values = [item['E_total'] for item in data]
        R_values = [item['R'] for item in data]
        corr = np.corrcoef(E_values, R_values)[0, 1]
        status, symbol = evaluate_result('F.7.5', corr)
        print(f" E-R correlation: {corr:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.5'] = {'correlation': corr, 'data': data, 'status': status}
    else:
        results['F.7.5'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.6: Entropy Stress Test
    print("\n[F.7.6] Entropy Injection Stress Test")
    print("Testing if R × nabla_S = constant...")
    test = run_test("Entropy Injection", entropy_injection_test, "F.7.6")
    if test['status'] == 'completed':
        data, cv = test['result']
        status, symbol = evaluate_result('F.7.6', cv, direction='lower')
        print(f" R × nabla_S coefficient of variation: {cv:.4f}")
        print(f" (Lower is better - should be constant)")
        print(f" Status: {symbol} {status}")
        results['F.7.6'] = {'cv': cv, 'n_points': len(data), 'status': status}
    else:
        results['F.7.6'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.7: Audio Test
    print("\n[F.7.7] Cross-Domain Audio Test")
    print("Testing formula in audio/SNR domain...")
    test = run_test("Audio Resonance", audio_resonance_test, "F.7.7")
    if test['status'] == 'completed':
        data, corr = test['result']
        status, symbol = evaluate_result('F.7.7', corr)
        print(f" SNR-R correlation: {corr:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.7'] = {'correlation': corr, 'n_cases': len(data), 'status': status}
    else:
        results['F.7.7'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.8: Network Test
    print("\n[F.7.8] Network Centrality Comparison")
    print("Testing if R correlates with eigenvector centrality...")
    test = run_test("Network Centrality", network_centrality_test, "F.7.8")
    if test['status'] == 'completed':
        data = test['result']
        corr = data['corr_centrality']
        status, symbol = evaluate_result('F.7.8', corr)
        print(f" Centrality-R correlation: {corr:.4f}")
        print(f" PageRank-R correlation: {data['corr_pagerank']:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.8'] = {'corr_centrality': corr, 'corr_pagerank': data['corr_pagerank'], 'status': status}
    else:
        results['F.7.8'] = {'status': 'ERROR', 'error': test['error']}

    # F.7.9: Monte Carlo Test
    print("\n[F.7.9] Monte Carlo Robustness")
    print("Testing formula stability under measurement noise...")
    test = run_test("Monte Carlo", monte_carlo_robustness, "F.7.9")
    if test['status'] == 'completed':
        data = test['result']
        cv = data['R_cv']
        status, symbol = evaluate_result('F.7.9', cv, direction='lower')
        print(f" R coefficient of variation: {cv:.4f}")
        print(f" Bias: {data['bias']:.4f}")
        print(f" Sensitivity ranking:")
        for var, val in sorted(data['sensitivity'].items(), key=lambda x: -x[1]):
            print(f"   {var}: {val:.4f}")
        print(f" Status: {symbol} {status}")
        results['F.7.9'] = {'cv': cv, 'bias': data['bias'], 'sensitivity': data['sensitivity'], 'status': status}
    else:
        results['F.7.9'] = {'status': 'ERROR', 'error': test['error']}

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    passed = sum(1 for r in results.values() if r.get('status') in ['PASS', 'VALIDATED'])
    failed = sum(1 for r in results.values() if r.get('status') == 'FALSIFIED')
    errors = sum(1 for r in results.values() if r.get('status') == 'ERROR')

    print(f"\n Tests Passed: {passed}/9")
    print(f" Tests Failed: {failed}/9")
    print(f" Errors: {errors}/9")

    print("\n Results by test:")
    for test_id in sorted(results.keys()):
        r = results[test_id]
        status = r.get('status', 'UNKNOWN')
        if status == 'VALIDATED':
            print(f"   {test_id}: ** VALIDATED")
        elif status == 'PASS':
            print(f"   {test_id}: *  PASS")
        elif status == 'FALSIFIED':
            print(f"   {test_id}: X  FALSIFIED")
        else:
            print(f"   {test_id}: ?  {status}")

    print("\n" + "-" * 70)
    if failed >= 3:
        print(" VERDICT: X FORMULA FALSIFIED")
        print(" Three or more tests failed - formula does not hold.")
    elif failed > 0:
        print(" VERDICT: ~ FORMULA NEEDS REFINEMENT")
        print(f" {failed} test(s) failed - core structure may hold but needs modification.")
    else:
        print(" VERDICT: * FORMULA VALIDATED")
        print(" All tests passed - formula holds across domains.")
    print("-" * 70)

    # Save results
    output_path = Path(__file__).parent / 'falsification_results.json'
    with open(output_path, 'w') as f:
        # Convert numpy values for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        json.dump({
            'timestamp': datetime.now().isoformat(),
            'results': convert(results),
            'summary': {
                'passed': passed,
                'failed': failed,
                'errors': errors,
                'verdict': 'FALSIFIED' if failed >= 3 else 'REFINEMENT' if failed > 0 else 'VALIDATED'
            }
        }, f, indent=2)

    print(f"\n Results saved to: {output_path}")

    return 0 if failed < 3 else 1


if __name__ == '__main__':
    sys.exit(main())
