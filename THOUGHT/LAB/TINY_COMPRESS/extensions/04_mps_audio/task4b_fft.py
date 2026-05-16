"""
MPS vs SVD for Audio Compression with Fourier Pre-Processing

Tests whether MPS beats SVD when compressing frequency-domain audio spectra
instead of time-domain waveforms. The hypothesis is that frequency spectra
have low-entanglement structure (sparse peaks) that MPS exploits better.

Usage:
    python mps_audio_fft.py
"""

import numpy as np
from scipy import signal as scipy_signal
import quimb.tensor as qtn

np.random.seed(42)


# Audio Signal Generators (reused from Task 4)

SAMPLE_RATE = 8000
SEGMENT_LEN = 1024
NUM_SEGMENTS = 3

def sine_wave(duration=1.0, freq=440.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)

def sine_sweep(duration=1.0, f0=200.0, f1=2000.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return scipy_signal.chirp(t, f0=f0, f1=f1, t1=duration, method='linear')

def square_wave(duration=1.0, freq=220.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return scipy_signal.square(2 * np.pi * freq * t)

def triangle_wave(duration=1.0, freq=330.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return scipy_signal.sawtooth(2 * np.pi * freq * t, width=0.5)

def amplitude_modulated(duration=1.0, carrier=440.0, modulator=5.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * carrier * t) * (1 + 0.5 * np.sin(2 * np.pi * modulator * t))

def frequency_modulated(duration=1.0, carrier=440.0, deviation=100.0, mod_rate=8.0, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * carrier * t + deviation / mod_rate * np.sin(2 * np.pi * mod_rate * t))

def noise_burst(duration=1.0, sr=SAMPLE_RATE):
    n = int(sr * duration)
    burst = np.random.randn(n)
    env = np.ones(n)
    fade = int(sr * 0.05)
    env[:fade] = np.linspace(0, 1, fade)
    env[-fade:] = np.linspace(1, 0, fade)
    return burst * env

def harmonic_complex(duration=1.0, base=110.0, harmonics=5, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)
    for h in range(1, harmonics + 1):
        sig += np.sin(2 * np.pi * base * h * t) / h
    return sig / np.max(np.abs(sig))

def piano_like(duration=1.0, base=261.63, sr=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.sin(2 * np.pi * base * t) * np.exp(-3 * t)
    for h in range(2, 6):
        sig += np.sin(2 * np.pi * base * h * t) * np.exp(-5 * t) / h
    return sig / np.max(np.abs(sig))


AUDIO_GENERATORS = {
    'sine_440': lambda: sine_wave(freq=440.0),
    'sine_880': lambda: sine_wave(freq=880.0),
    'sweep_200_2000': lambda: sine_sweep(f0=200, f1=2000),
    'square_220': lambda: square_wave(freq=220.0),
    'triangle_330': lambda: triangle_wave(freq=330.0),
    'am_sine': lambda: amplitude_modulated(),
    'fm_sine': lambda: frequency_modulated(),
    'noise_burst': lambda: noise_burst(),
    'harmonic_complex': lambda: harmonic_complex(),
    'piano_like': lambda: piano_like(),
}


def extract_segments(signal, seg_len=SEGMENT_LEN, num=3):
    total_needed = seg_len * num
    if len(signal) < total_needed:
        signal = np.tile(signal, int(np.ceil(total_needed / len(signal))))
    segments = []
    for i in range(num):
        start = i * seg_len
        segments.append(signal[start:start + seg_len].copy())
    return segments


def fft_magnitude_full(segment):
    """FFT spectrum magnitude vector (length=1024, same as input)."""
    spectrum = np.fft.fft(segment)
    return np.abs(spectrum)


# Compression Methods

def svd_compress(data, k):
    """SVD: reshape (32,32) for 1024-length data, truncate to k."""
    assert len(data) == 1024
    mat = data.reshape(32, 32)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    U_k = U[:, :k].copy()
    S_k = S[:k].copy()
    Vh_k = Vh[:k, :].copy()
    return U_k, S_k, Vh_k


def svd_decompress(U_k, S_k, Vh_k):
    return (U_k * S_k) @ Vh_k


def svd_param_count(k):
    return 32 * k + k + 32 * k  # 65k


def mps_compress(data, chi):
    mps = qtn.MatrixProductState.from_dense(
        data, dims=2, max_bond=chi, method='svd'
    )
    return mps


def mps_decompress(mps):
    return mps.to_dense().flatten()


def mps_param_count(mps):
    return sum(t.size for t in mps.tensors)


def compute_snr(original, reconstructed):
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    if noise_power < 1e-30:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)


# Task 4 time-domain win rates (from REPORT.md)
TASK4_WIN_RATES = {
    'sine_440': 50.0,
    'sine_880': 50.0,
    'sweep_200_2000': 0.0,
    'square_220': 12.0,
    'triangle_330': 11.0,
    'am_sine': 50.0,
    'fm_sine': 0.0,
    'noise_burst': 0.0,
    'harmonic_complex': 17.0,
    'piano_like': 17.0,
}


def run_benchmark():
    print("=" * 80)
    print("MPS vs SVD for Audio with Fourier Pre-Processing (Task 4b)")
    print("=" * 80)

    # Generate audio
    print("\nGenerating audio signals...")
    all_segments = {}
    for name, gen in AUDIO_GENERATORS.items():
        signal = gen()
        segments = extract_segments(signal, SEGMENT_LEN, NUM_SEGMENTS)
        all_segments[name] = segments
        print(f"  {name}: {len(segments)} segments x {SEGMENT_LEN}")

    k_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    chi_values = [1, 2, 4, 6, 8, 12, 16, 24, 32]

    results = {}

    for name, segments in all_segments.items():
        print(f"\n{'=' * 60}")
        print(f"Audio: {name}  |  Domain: FREQUENCY (FFT magnitude)")
        print(f"{'=' * 60}")

        # Compute magnitude spectra once
        spectra = [fft_magnitude_full(seg) for seg in segments]

        # SVD
        svd_rows = []
        for k in k_values:
            seg_snrs = []
            for mag in spectra:
                U_k, S_k, Vh_k = svd_compress(mag, k)
                recon = svd_decompress(U_k, S_k, Vh_k).flatten()
                snr = compute_snr(mag, recon)
                seg_snrs.append(snr)
            mean_snr = np.mean(seg_snrs)
            params = svd_param_count(k)
            cr = SEGMENT_LEN / params
            svd_rows.append((k, params, cr, mean_snr))

        # MPS
        mps_rows = []
        for chi in chi_values:
            seg_snrs = []
            for mag in spectra:
                mps = mps_compress(mag, chi)
                recon = mps_decompress(mps)
                snr = compute_snr(mag, recon)
                seg_snrs.append(snr)
            mean_snr = np.mean(seg_snrs)
            params = mps_param_count(mps_compress(spectra[0], chi))
            cr = SEGMENT_LEN / params
            mps_rows.append((chi, params, cr, mean_snr))

        # Print SVD table
        print(f"\n  SVD on FFT magnitude:")
        print(f"  {'k':>4} {'Params':>7} {'CR':>8} {'SNR(dB)':>10}")
        print(f"  {'-' * 29}")
        for k, params, cr, snr in svd_rows:
            print(f"  {k:>4d} {params:>7d} {cr:>8.2f}x {snr:>10.2f}")

        # Print MPS table
        print(f"\n  MPS on FFT magnitude:")
        print(f"  {'chi':>4} {'Params':>7} {'CR':>8} {'SNR(dB)':>10}")
        print(f"  {'-' * 29}")
        for chi, params, cr, snr in mps_rows:
            print(f"  {chi:>4d} {params:>7d} {cr:>8.2f}x {snr:>10.2f}")

        # Matched CR comparison
        print(f"\n  --- Matched Compression Ratio ---")
        print(f"  {'Method':<12} {'k/chi':>6} {'Params':>7} {'CR':>8} {'SNR(dB)':>10}")
        print(f"  {'-' * 43}")

        all_points = []
        for k, params, cr, snr in svd_rows:
            all_points.append(('SVD', k, params, cr, snr))
        for chi, params, cr, snr in mps_rows:
            all_points.append(('MPS', chi, params, cr, snr))
        all_points.sort(key=lambda x: -x[3])

        mps_points = [(m, c, p, cr, s) for m, c, p, cr, s in all_points if m == 'MPS']
        svd_points = [(m, c, p, cr, s) for m, c, p, cr, s in all_points if m == 'SVD']

        max_cr_ratio = 1.5
        matched = []
        for mp in mps_points:
            best_svd = min(svd_points, key=lambda sp: abs(sp[3] - mp[3]))
            cr_ratio = max(mp[3], best_svd[3]) / max(min(mp[3], best_svd[3]), 1e-10)
            if cr_ratio <= max_cr_ratio:
                matched.append((mp, best_svd))

        seen = set()
        unique_matched = []
        for mp, sp in matched:
            key = (round(mp[3], 1), round(sp[3], 1))
            if key not in seen:
                seen.add(key)
                unique_matched.append((mp, sp))

        mps_wins = 0
        total_comparisons = 0
        for mp, sp in unique_matched:
            mps_snr, svd_snr = mp[4], sp[4]
            winner = "MPS" if mps_snr > svd_snr else "SVD" if svd_snr > mps_snr else "TIE"
            if winner == "MPS":
                mps_wins += 1
            total_comparisons += 1
            gap = mps_snr - svd_snr
            print(f"  {'MPS':<12} {'chi=' + str(mp[1]):>6} {mp[2]:>7d} {mp[3]:>8.2f}x {mp[4]:>10.2f}")
            print(f"  {'SVD':<12} {'k=' + str(sp[1]):>6} {sp[2]:>7d} {sp[3]:>8.2f}x {sp[4]:>10.2f}")
            print(f"  {'Gap':>20} {'':>15} {gap:>+10.2f} dB ({winner})")
            print()

        if total_comparisons > 0:
            win_rate = mps_wins / total_comparisons * 100
            print(f"  MPS wins: {mps_wins}/{total_comparisons} ({win_rate:.0f}%)")
            task4_rate = TASK4_WIN_RATES.get(name, None)
            if task4_rate is not None:
                delta = win_rate - task4_rate
                arrow = "UP" if delta > 0 else "DOWN" if delta < 0 else "SAME"
                print(f"  Task 4 (time-domain): {task4_rate:.0f}%  ->  Delta: {delta:+.0f}pp ({arrow})")
        else:
            print("  No valid comparisons available")
            win_rate = 0.0

        results[name] = {
            'mps_wins': mps_wins,
            'total': total_comparisons,
            'win_rate_fd': win_rate,
            'win_rate_td': TASK4_WIN_RATES.get(name, None),
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Fourier Pre-Processing Impact on MPS vs SVD")
    print("=" * 80)

    total_fd_wins = 0
    total_fd_comp = 0
    total_td_wins = 0
    total_td_comp = 0
    improvements = 0

    for name, r in sorted(results.items()):
        wr_fd = r['win_rate_fd']
        wr_td = r['win_rate_td']
        w = r['mps_wins']
        t = r['total']
        if wr_td is not None:
            delta = wr_fd - wr_td
        else:
            delta = 0
        if delta > 0:
            improvements += 1

        verdict = "MPS WINS" if wr_fd > 50 else "SVD WINS" if wr_fd < 50 else "TIE"
        delta_str = f" (Delta: +{delta:.0f}pp)" if delta > 0 else f" (Delta: {delta:.0f}pp)" if delta < 0 else " (unchanged)"
        print(f"  {name:<25s}: Freq={wr_fd:.0f}% | Time={wr_td:.0f}%{delta_str} -> {verdict}")

        if 'noise' not in name:
            total_fd_wins += w
            total_fd_comp += t

    if total_fd_comp > 0:
        overall_fd = total_fd_wins / total_fd_comp * 100
        overall_td = 16.0  # from Task 4
        overall_delta = overall_fd - overall_td
        print(f"\n  Tonal average:")
        print(f"    Time-domain MPS win rate: {overall_td:.0f}% (from Task 4)")
        print(f"    Frequency-domain MPS win rate: {overall_fd:.0f}%")
        print(f"    Improvement: {overall_delta:+.0f}pp ({improvements}/{len(results)} audio types improved)")
        if overall_fd >= 50:
            print(f"\n  >>> HYPOTHESIS CONFIRMED: MPS beats SVD with Fourier pre-processing")
        else:
            print(f"\n  >>> HYPOTHESIS REFUTED: MPS win rate {overall_fd:.0f}% on frequency-domain data")
            print(f"  >>> (predicted >50%, actual {overall_fd:.0f}%. Delta vs time-domain: {overall_delta:+.0f}pp)")
    return results


if __name__ == '__main__':
    results = run_benchmark()
