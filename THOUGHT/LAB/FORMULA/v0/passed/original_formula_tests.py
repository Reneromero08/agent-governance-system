#!/usr/bin/env python3
"""
Original Formula Claim Tests
=============================

Tests the specific claims made in Formula 1.11:
R = (E * I^2) / D

Claims to test:
1. Einstein: E = mc^2 maps to R = m * c^2 / D
2. Newton: F = ma maps to R = m * a^2 / D  <-- CRITICAL: Newton says F=ma, not ma^2!
3. Music: Consonance = freq * harmonics^2 / noise
4. Transformers: Performance = architecture * layers^2 / noise
5. Gravity: F = Gm1m2/r^2 structure

Key Question: Does I^2 (squared information) actually appear in these equations?
"""

import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def test_dimensional_analysis():
    """
    TEST 1: Dimensional Analysis of Physics Mappings

    Check if the claimed mappings preserve dimensional consistency.

    Original claims:
    - Einstein: Energy = mass * c^2 / D
    - Newton: Force = mass * a^2 / D  <-- THIS IS WRONG! F = ma, not ma^2
    - Gravity: F = m1 * (G/r^2)^2 / D  <-- THIS IS WRONG! F = Gm1m2/r^2
    """
    print("=" * 60)
    print("TEST 1: Dimensional Analysis")
    print("=" * 60)

    results = {}

    # Einstein: E = mc^2
    # Claim: R = m * c^2 / D where I = c
    # Check: Does E = mc^2 have I^2 structure? YES! c is squared.
    print("\n[Einstein E = mc^2]")
    print("  Formula claim: R = m * c^2 / D")
    print("  Actual: E = m * c^2")
    print("  Structure match: c IS squared in original")
    print("  Dimensional check: [M][L^2/T^2] = [Energy] CORRECT")
    results['einstein'] = {
        'structure_match': True,
        'dimensional_correct': True,
        'notes': 'c^2 appears in original - good mapping'
    }

    # Newton: F = ma
    # Claim: R = m * a^2 / D where I = a
    # Check: Does F = ma have I^2 structure? NO! a is NOT squared.
    print("\n[Newton F = ma]")
    print("  Formula claim: R = m * a^2 / D")
    print("  Actual: F = m * a  (NOT a^2!)")
    print("  Structure match: FAIL - a is NOT squared in Newton")
    print("  Dimensional check: [M][L/T^2]^2 = [M*L^2/T^4] != [Force]")
    print("  ** FALSIFICATION: Newton's law is F=ma, not F=ma^2 **")
    results['newton'] = {
        'structure_match': False,
        'dimensional_correct': False,
        'notes': 'CRITICAL: Newton says F=ma, formula claims ma^2'
    }

    # Gravity: F = Gm1m2/r^2
    # Claim: R = m1 * (G/r^2)^2 / D
    # Check: The formula is F = G*m1*m2/r^2, not m1*(G/r^2)^2
    print("\n[Gravity F = Gm1m2/r^2]")
    print("  Formula claim: R = m1 * (G/r^2)^2 / D")
    print("  Actual: F = G * m1 * m2 / r^2")
    print("  Structure match: FAIL - G and r^2 are not squared together")
    print("  ** FALSIFICATION: Gravity doesn't have (G/r^2)^2 structure **")
    results['gravity'] = {
        'structure_match': False,
        'dimensional_correct': False,
        'notes': 'Gravity equation structure does not match claim'
    }

    # Schrodinger: ih d/dt |psi> = H |psi>
    # Claim: R = psi * H^2 / D
    # Check: Schrodinger is linear in H, not H^2
    print("\n[Schrodinger ih(d/dt)|psi> = H|psi>]")
    print("  Formula claim: R = psi * H^2 / D")
    print("  Actual: H|psi> (linear in H, NOT H^2)")
    print("  Structure match: FAIL - Hamiltonian is NOT squared")
    results['schrodinger'] = {
        'structure_match': False,
        'notes': 'Schrodinger equation is linear in H'
    }

    # Summary
    print("\n" + "-" * 60)
    print("SUMMARY:")
    matches = sum(1 for r in results.values() if r.get('structure_match', False))
    print(f"  Structure matches: {matches}/{len(results)}")

    if matches < len(results) / 2:
        print("\nX  FALSIFIED: Most physics mappings don't have I^2 structure")
    else:
        print("\n*  PASS: Majority of mappings have I^2 structure")

    return results


def test_music_harmonics():
    """
    TEST 2: Music/Harmonics Claim

    Claim: R = frequency * harmonics^2 / noise

    Test: Generate tones with varying harmonics and noise,
    measure perceived consonance, check if formula predicts it.
    """
    print("\n" + "=" * 60)
    print("TEST 2: Music/Harmonics")
    print("=" * 60)

    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    test_cases = []

    # Vary harmonics (I) and noise (D)
    for n_harmonics in [1, 2, 4, 8, 16]:
        for noise_level in [0.01, 0.05, 0.1, 0.2]:
            # Generate signal with harmonics
            freq = 440  # A4
            signal = np.zeros_like(t)
            for h in range(1, n_harmonics + 1):
                signal += (1/h) * np.sin(2 * np.pi * freq * h * t)
            signal /= np.max(np.abs(signal) + 1e-10)

            # Add noise
            noise = np.random.randn(len(t)) * noise_level
            noisy_signal = signal + noise

            # Measure "consonance" as SNR (simple proxy)
            signal_power = np.var(signal)
            noise_power = np.var(noise)
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100

            # Formula prediction: R = E * I^2 / D
            # E = frequency (constant), I = harmonics, D = noise
            E = freq
            I = n_harmonics
            D = noise_level + 0.001  # Avoid division by zero
            R_formula = (E * I**2) / D

            # Alternative: R = E * I / D (linear, not squared)
            R_linear = (E * I) / D

            test_cases.append({
                'harmonics': n_harmonics,
                'noise': noise_level,
                'snr': snr,
                'R_formula': R_formula,
                'R_linear': R_linear
            })

    # Check correlations
    snr_values = [tc['snr'] for tc in test_cases]
    R_formula_values = [tc['R_formula'] for tc in test_cases]
    R_linear_values = [tc['R_linear'] for tc in test_cases]

    corr_squared = np.corrcoef(snr_values, R_formula_values)[0, 1]
    corr_linear = np.corrcoef(snr_values, R_linear_values)[0, 1]

    print(f"\nCorrelation with SNR:")
    print(f"  I^2 formula: {corr_squared:.4f}")
    print(f"  I^1 (linear): {corr_linear:.4f}")

    # The key question: Does I^2 predict better than I^1?
    if corr_squared > corr_linear + 0.05:
        print("\n** VALIDATED: I^2 (squared) predicts better than linear")
        status = 'VALIDATED'
    elif corr_linear > corr_squared + 0.05:
        print("\nX  FALSIFIED: Linear (I^1) predicts better than I^2")
        status = 'FALSIFIED'
    else:
        print("\n~  INCONCLUSIVE: I^2 and I^1 perform similarly")
        status = 'INCONCLUSIVE'

    return {
        'corr_squared': corr_squared,
        'corr_linear': corr_linear,
        'status': status,
        'test_cases': test_cases
    }


def test_transformer_scaling():
    """
    TEST 3: Transformer Scaling Claim

    Claim: Performance = architecture * layers^2 / noise

    Test with known transformer scaling laws.
    Actual scaling: Loss ~ (parameters)^(-alpha) where alpha ~ 0.076
    This is a POWER LAW, not quadratic in layers.
    """
    print("\n" + "=" * 60)
    print("TEST 3: Transformer Scaling")
    print("=" * 60)

    # Known scaling law data (approximated from Chinchilla/GPT papers)
    # Parameters (billions) vs Loss
    params = np.array([0.1, 0.5, 1, 5, 10, 50, 100, 500])

    # Approximate losses (lower is better) - power law: L ~ N^(-0.076)
    # Using approximate values from scaling papers
    loss = 2.5 * params ** (-0.076)  # Chinchilla-style scaling

    # If formula were true: R = E * I^2 / D
    # Where I = sqrt(params) (layers scale with sqrt of params roughly)
    # Then R ~ params (quadratic in layers = linear in params)

    # Test different scaling hypotheses

    # H1: Formula claim - quadratic in "information" (layers)
    # Layers ~ sqrt(params), so I^2 ~ params
    # R_formula ~ params
    R_formula = params  # Linear in params = quadratic in layers

    # H2: Actual scaling law - power law
    R_powerlaw = params ** 0.076  # Inverse of loss scaling

    # H3: Log scaling
    R_log = np.log(params + 1)

    # Invert loss to get "performance"
    performance = 1 / loss

    # Correlations with performance
    corr_formula = np.corrcoef(performance, R_formula)[0, 1]
    corr_power = np.corrcoef(performance, R_powerlaw)[0, 1]
    corr_log = np.corrcoef(performance, R_log)[0, 1]

    print(f"\nCorrelation with transformer performance:")
    print(f"  Formula (I^2 ~ params): {corr_formula:.4f}")
    print(f"  Power law (N^0.076):    {corr_power:.4f}")
    print(f"  Logarithmic:            {corr_log:.4f}")

    # Fit R^2 for each model
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score

    # Formula model
    lr = LinearRegression()
    lr.fit(R_formula.reshape(-1, 1), performance)
    r2_formula = r2_score(performance, lr.predict(R_formula.reshape(-1, 1)))

    lr.fit(R_powerlaw.reshape(-1, 1), performance)
    r2_power = r2_score(performance, lr.predict(R_powerlaw.reshape(-1, 1)))

    print(f"\nR^2 scores:")
    print(f"  Formula (I^2): {r2_formula:.4f}")
    print(f"  Power law:     {r2_power:.4f}")

    if r2_formula > r2_power:
        print("\n*  PASS: Formula scaling fits transformer data")
        status = 'PASS'
    else:
        print("\nX  FALSIFIED: Power law fits better than I^2 formula")
        status = 'FALSIFIED'

    return {
        'r2_formula': r2_formula,
        'r2_power': r2_power,
        'status': status
    }


def test_information_squared_universality():
    """
    TEST 4: Is I^2 Actually Universal?

    The core claim is that "information squared" appears everywhere.
    Let's check if squaring actually improves predictions across domains.
    """
    print("\n" + "=" * 60)
    print("TEST 4: I^2 Universality")
    print("=" * 60)

    domains_tested = 0
    squared_wins = 0

    # Domain 1: Signal processing (SNR)
    # SNR = signal_power / noise_power
    # If formula: SNR ~ E * I^2 / D
    print("\n[Signal Processing]")
    signal_strengths = np.array([0.1, 0.5, 1, 2, 5, 10])
    noise_levels = np.array([0.1, 0.2, 0.5, 1, 2])

    results = []
    for s in signal_strengths:
        for n in noise_levels:
            snr_actual = s / n
            snr_squared = s**2 / n  # I^2 model
            snr_linear = s / n       # Linear model
            results.append({
                'actual': snr_actual,
                'squared': snr_squared,
                'linear': snr_linear
            })

    actual = [r['actual'] for r in results]
    squared = [r['squared'] for r in results]
    linear = [r['linear'] for r in results]

    corr_sq = np.corrcoef(actual, squared)[0, 1]
    corr_lin = np.corrcoef(actual, linear)[0, 1]

    print(f"  I^2 correlation: {corr_sq:.4f}")
    print(f"  Linear correlation: {corr_lin:.4f}")
    domains_tested += 1
    if corr_sq > corr_lin:
        squared_wins += 1
        print("  Winner: I^2")
    else:
        print("  Winner: Linear")

    # Domain 2: Network effects (Metcalfe's Law)
    # Actual: Value ~ n^2 (n = users)
    # This DOES have squared structure!
    print("\n[Network Effects - Metcalfe's Law]")
    n_users = np.array([10, 100, 1000, 10000, 100000])
    value_actual = n_users ** 2  # Metcalfe
    value_squared = n_users ** 2
    value_linear = n_users

    corr_sq = np.corrcoef(value_actual, value_squared)[0, 1]
    corr_lin = np.corrcoef(value_actual, value_linear)[0, 1]

    print(f"  I^2 correlation: {corr_sq:.4f}")
    print(f"  Linear correlation: {corr_lin:.4f}")
    domains_tested += 1
    if corr_sq > corr_lin:
        squared_wins += 1
        print("  Winner: I^2 (Metcalfe IS n^2)")
    else:
        print("  Winner: Linear")

    # Domain 3: Learning curves
    # Actual: Error ~ 1/sqrt(samples) or 1/samples
    # Not squared!
    print("\n[Machine Learning - Sample Complexity]")
    n_samples = np.array([100, 500, 1000, 5000, 10000, 50000])
    error_actual = 1 / np.sqrt(n_samples)  # Typical learning curve
    perf_actual = 1 - error_actual

    perf_squared = n_samples ** 2  # I^2 prediction
    perf_linear = n_samples  # Linear prediction
    perf_sqrt = np.sqrt(n_samples)  # Sqrt prediction

    corr_sq = np.corrcoef(perf_actual, perf_squared)[0, 1]
    corr_lin = np.corrcoef(perf_actual, perf_linear)[0, 1]
    corr_sqrt = np.corrcoef(perf_actual, perf_sqrt)[0, 1]

    print(f"  I^2 correlation: {corr_sq:.4f}")
    print(f"  Linear correlation: {corr_lin:.4f}")
    print(f"  Sqrt correlation: {corr_sqrt:.4f}")
    domains_tested += 1
    if corr_sq > max(corr_lin, corr_sqrt):
        squared_wins += 1
        print("  Winner: I^2")
    else:
        print("  Winner: NOT I^2")

    # Summary
    print("\n" + "-" * 60)
    print(f"SUMMARY: I^2 wins in {squared_wins}/{domains_tested} domains")

    if squared_wins > domains_tested / 2:
        print("\n*  PASS: I^2 is somewhat universal")
        status = 'PASS'
    else:
        print("\nX  FALSIFIED: I^2 is not universally the best predictor")
        status = 'FALSIFIED'

    return {
        'domains_tested': domains_tested,
        'squared_wins': squared_wins,
        'status': status
    }


def test_physics_equation_structure():
    """
    TEST 5: Do Physics Equations Actually Have R = E*I^2/D Structure?

    Rigorous check of whether fundamental equations can be expressed as:
    R = (E * I^2) / D
    """
    print("\n" + "=" * 60)
    print("TEST 5: Physics Equation Structure Analysis")
    print("=" * 60)

    equations = {
        'Einstein E=mc^2': {
            'formula': 'E = m * c^2',
            'has_squared_term': True,
            'squared_what': 'c (speed of light)',
            'division_by_noise': False,
            'mapping_valid': 'PARTIAL - has I^2 but no D term'
        },
        'Newton F=ma': {
            'formula': 'F = m * a',
            'has_squared_term': False,
            'squared_what': None,
            'division_by_noise': False,
            'mapping_valid': 'FALSE - no squared term'
        },
        'Kinetic Energy': {
            'formula': 'KE = (1/2) * m * v^2',
            'has_squared_term': True,
            'squared_what': 'v (velocity)',
            'division_by_noise': False,
            'mapping_valid': 'PARTIAL - has I^2 but no D term'
        },
        'Gravity': {
            'formula': 'F = G * m1 * m2 / r^2',
            'has_squared_term': True,
            'squared_what': 'r (distance) in denominator',
            'division_by_noise': False,
            'mapping_valid': 'FALSE - r^2 is in denominator, not I^2'
        },
        'Coulomb': {
            'formula': 'F = k * q1 * q2 / r^2',
            'has_squared_term': True,
            'squared_what': 'r (distance) in denominator',
            'division_by_noise': False,
            'mapping_valid': 'FALSE - same as gravity'
        },
        'Ohm V=IR': {
            'formula': 'V = I * R',
            'has_squared_term': False,
            'squared_what': None,
            'division_by_noise': False,
            'mapping_valid': 'FALSE - no squared term'
        },
        'Power P=IV': {
            'formula': 'P = I * V = I^2 * R',
            'has_squared_term': True,
            'squared_what': 'I (current) in P=I^2R form',
            'division_by_noise': False,
            'mapping_valid': 'PARTIAL - has I^2 in one form'
        },
        'Wave Energy': {
            'formula': 'E proportional to A^2',
            'has_squared_term': True,
            'squared_what': 'A (amplitude)',
            'division_by_noise': False,
            'mapping_valid': 'PARTIAL - has I^2 but no D term'
        },
        'Schrodinger': {
            'formula': 'H|psi> = E|psi>',
            'has_squared_term': False,
            'squared_what': None,
            'division_by_noise': False,
            'mapping_valid': 'FALSE - linear in H'
        },
        'Ideal Gas PV=nRT': {
            'formula': 'PV = nRT',
            'has_squared_term': False,
            'squared_what': None,
            'division_by_noise': False,
            'mapping_valid': 'FALSE - all terms linear'
        }
    }

    has_squared = 0
    full_match = 0

    for name, data in equations.items():
        print(f"\n[{name}]")
        print(f"  Formula: {data['formula']}")
        print(f"  Has squared term: {data['has_squared_term']}")
        if data['has_squared_term']:
            print(f"  What's squared: {data['squared_what']}")
            has_squared += 1
        print(f"  Mapping valid: {data['mapping_valid']}")
        if 'TRUE' in data['mapping_valid'] or data['mapping_valid'].startswith('PARTIAL'):
            if 'no D' not in data['mapping_valid']:
                full_match += 1

    print("\n" + "-" * 60)
    print(f"Equations with squared terms: {has_squared}/{len(equations)}")
    print(f"Equations matching R=E*I^2/D structure: {full_match}/{len(equations)}")

    # Key finding: Many equations have squared terms, but...
    # 1. The squared term isn't always "Information"
    # 2. None naturally have a "Dissonance" divisor
    # 3. The mapping is forced, not natural

    print("\nKEY FINDING:")
    print("  - Squared terms exist (c^2, v^2, A^2) but these are")
    print("    specific physical quantities, not generic 'Information'")
    print("  - No physics equation naturally has a 'Dissonance' term")
    print("  - The mapping R=E*I^2/D is a metaphor, not isomorphism")

    if full_match > len(equations) / 2:
        print("\n*  PASS: Most equations fit the structure")
    else:
        print("\nX  FALSIFIED: Most equations don't naturally fit R=E*I^2/D")

    return equations


def main():
    print("=" * 60)
    print("ORIGINAL FORMULA CLAIM TESTS")
    print("R = (E * I^2) / D")
    print("=" * 60)

    results = {}

    # Test 1: Dimensional Analysis
    results['dimensional'] = test_dimensional_analysis()

    # Test 2: Music Harmonics
    results['music'] = test_music_harmonics()

    # Test 3: Transformer Scaling
    results['transformers'] = test_transformer_scaling()

    # Test 4: I^2 Universality
    results['universality'] = test_information_squared_universality()

    # Test 5: Physics Structure
    results['physics'] = test_physics_equation_structure()

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n[Test Results]")
    print(f"  1. Dimensional Analysis: Newton F=ma claim is FALSE (ma, not ma^2)")
    print(f"  2. Music Harmonics: {results['music']['status']}")
    print(f"  3. Transformer Scaling: {results['transformers']['status']}")
    print(f"  4. I^2 Universality: {results['universality']['status']}")
    print(f"  5. Physics Structure: Most equations don't fit R=E*I^2/D")

    print("\n[Critical Falsifications]")
    print("  - Newton's F=ma does NOT have a squared term")
    print("  - Gravity F=Gm1m2/r^2 has r^2 in denominator, not as I^2")
    print("  - Schrodinger equation is linear in H, not H^2")
    print("  - No physics equation has natural 'Dissonance' divisor")

    print("\n[Where the Formula MAY Work]")
    print("  - Domains where squared scaling exists (networks, waves)")
    print("  - As a metaphorical framework, not literal mapping")
    print("  - When I^2 is interpreted as 'recursive amplification'")

    print("\n" + "-" * 60)
    print("VERDICT: The specific physics mappings are FALSIFIED,")
    print("but the formula may work as a qualitative framework.")
    print("-" * 60)


if __name__ == '__main__':
    main()
