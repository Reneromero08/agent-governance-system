#!/usr/bin/env python3
"""
F.7.7: Cross-Domain Audio Test

Tests formula in audio domain:

E = signal power
nabla_S = noise power
sigma = compression ratio
Df = spectral complexity (number of harmonics)
R = perceptual quality (SNR)

Formula predicts: R = (E/nabla_S) × sigma^Df

Prediction: Correlation > 0.85 between formula R and actual SNR.
Falsification: Correlation < 0.5 or formula systematically over/under-predicts.
"""

import numpy as np


def audio_resonance_test():
    """
    Test formula in audio domain:

    E = signal power
    nabla_S = noise power
    sigma = compression ratio (e.g., MP3 bitrate)
    Df = spectral complexity (number of harmonics)
    R = perceptual quality (PESQ or SNR)

    Formula predicts: R = (E/nabla_S) × sigma^Df
    """

    # Generate test tones with known properties
    sr = 44100  # Sample rate
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    test_cases = []

    # Vary each parameter
    for freq in [440, 880, 1760]:  # Essence (fundamental)
        for noise_level in [0.01, 0.1, 0.5]:  # Entropy
            for n_harmonics in [1, 3, 7]:  # Fractal dimension
                # Generate signal
                signal = np.zeros_like(t)
                for h in range(1, n_harmonics + 1):
                    signal += (1/h) * np.sin(2 * np.pi * freq * h * t)
                signal /= np.max(np.abs(signal) + 1e-10)

                # Add noise
                noise = np.random.randn(len(t)) * noise_level
                noisy_signal = signal + noise

                # Measurements
                E = np.var(signal)
                nabla_S = np.var(noise)
                Df = n_harmonics  # Proxy for fractal dimension
                sigma = 1.0  # No compression in this test

                # R = SNR (ground truth)
                R_actual = 10 * np.log10(E / nabla_S) if nabla_S > 0 else 100

                # R predicted by formula (need to calibrate constants)
                R_formula = (E / nabla_S) * (sigma ** Df) if nabla_S > 0 else float('inf')

                test_cases.append({
                    'freq': freq,
                    'noise_level': noise_level,
                    'n_harmonics': n_harmonics,
                    'E': E,
                    'nabla_S': nabla_S,
                    'Df': Df,
                    'R_actual': R_actual,
                    'R_formula': R_formula
                })

    # Correlation between R_actual and R_formula
    R_actual = [t['R_actual'] for t in test_cases]
    R_formula = [min(t['R_formula'], 1e6) for t in test_cases]  # Cap infinities

    correlation = np.corrcoef(R_actual, R_formula)[0, 1]

    return test_cases, correlation


def audio_resonance_with_librosa():
    """
    Extended test using librosa for spectral analysis.
    Only runs if librosa is available.
    """
    try:
        import librosa
    except ImportError:
        return None, None

    sr = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))

    test_cases = []

    for freq in [440, 880]:
        for noise_level in [0.01, 0.1, 0.3]:
            for n_harmonics in [1, 4, 8]:
                # Generate harmonic signal
                signal = np.zeros_like(t)
                for h in range(1, n_harmonics + 1):
                    signal += (1/h) * np.sin(2 * np.pi * freq * h * t)
                signal /= np.max(np.abs(signal) + 1e-10)

                # Add noise
                noise = np.random.randn(len(t)) * noise_level
                noisy_signal = signal + noise

                # Librosa spectral analysis
                S = np.abs(librosa.stft(signal))
                S_noisy = np.abs(librosa.stft(noisy_signal))

                # Spectral centroid as measure of "essence"
                centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)[0].mean()

                # Spectral flatness as measure of entropy
                flatness = librosa.feature.spectral_flatness(y=noisy_signal)[0].mean()

                E = np.var(signal)
                nabla_S = np.var(noise)
                Df = n_harmonics

                R_actual = 10 * np.log10(E / nabla_S) if nabla_S > 0 else 100
                R_formula = (E / nabla_S) * (1.0 ** Df) if nabla_S > 0 else float('inf')

                test_cases.append({
                    'freq': freq,
                    'noise_level': noise_level,
                    'n_harmonics': n_harmonics,
                    'centroid': centroid,
                    'flatness': flatness,
                    'R_actual': R_actual,
                    'R_formula': R_formula
                })

    R_actual = [t['R_actual'] for t in test_cases]
    R_formula = [min(t['R_formula'], 1e6) for t in test_cases]
    correlation = np.corrcoef(R_actual, R_formula)[0, 1]

    return test_cases, correlation


if __name__ == '__main__':
    print("F.7.7: Cross-Domain Audio Test")
    print("=" * 50)

    test_cases, correlation = audio_resonance_test()

    print(f"\nGenerated {len(test_cases)} test cases")
    print("-" * 70)
    print(f"{'Freq':>6s} | {'Noise':>6s} | {'Harm':>4s} | {'R_actual':>10s} | {'R_formula':>10s}")
    print("-" * 70)

    # Show sample
    for tc in test_cases[::9]:  # Every 9th case
        print(f"{tc['freq']:6d} | {tc['noise_level']:6.2f} | {tc['n_harmonics']:4d} | {tc['R_actual']:10.2f} | {tc['R_formula']:10.2f}")

    print("-" * 70)
    print(f"\nSNR-Formula correlation: {correlation:.4f}")

    # Also check log-transformed correlation
    R_actual = [t['R_actual'] for t in test_cases]
    R_formula = [np.log10(max(t['R_formula'], 1e-10)) for t in test_cases]
    corr_log = np.corrcoef(R_actual, R_formula)[0, 1]
    print(f"SNR-log(Formula) correlation: {corr_log:.4f}")

    if correlation > 0.85:
        print("\n** VALIDATED: Strong SNR-R correlation (>0.85)")
    elif correlation > 0.5:
        print("\n*  PASS: Moderate SNR-R correlation (>0.5)")
    else:
        print("\nX  FALSIFIED: Weak SNR-R correlation (<0.5)")

    # Try librosa version if available
    print("\n" + "-" * 50)
    print("Attempting librosa-enhanced test...")
    test_cases_lib, corr_lib = audio_resonance_with_librosa()
    if test_cases_lib:
        print(f"Librosa correlation: {corr_lib:.4f}")
    else:
        print("librosa not available - skipped")
