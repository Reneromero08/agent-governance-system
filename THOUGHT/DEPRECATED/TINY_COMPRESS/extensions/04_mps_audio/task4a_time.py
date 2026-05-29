"""
MPS vs SVD for Audio Compression

Tests whether MPS (Matrix Product States) outperforms SVD for compressing
genuinely 1D sequential data (audio). Validates the Phase 3f prediction.

Phase 3f found that SVD beats MPS for 2D images because 1D flattening breaks
spatial locality. Prediction: MPS should beat SVD for genuinely 1D audio data
where sequential structure matches the tensor network architecture.

Usage:
    python mps_audio.py
"""

import numpy as np
from scipy import signal as scipy_signal
import quimb.tensor as qtn


# Audio Signal Generators

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
    """Extract num non-overlapping segments of seg_len from signal."""
    total_needed = seg_len * num
    if len(signal) < total_needed:
        signal = np.tile(signal, int(np.ceil(total_needed / len(signal))))
    segments = []
    for i in range(num):
        start = i * seg_len
        segments.append(signal[start:start + seg_len].copy())
    return segments


# Compression Methods

def svd_compress(segment, k):
    """SVD compression: reshape (32,32), truncate to k components."""
    assert len(segment) == 1024
    mat = segment.reshape(32, 32)
    U, S, Vh = np.linalg.svd(mat, full_matrices=False)
    U_k = U[:, :k].copy()
    S_k = S[:k].copy()
    Vh_k = Vh[:k, :].copy()
    return U_k, S_k, Vh_k


def svd_decompress(U_k, S_k, Vh_k):
    """Reconstruct from SVD components."""
    return (U_k * S_k) @ Vh_k


def svd_param_count(k):
    """Number of float values stored for SVD at rank k."""
    return 32 * k + k + 32 * k


def mps_compress(segment, chi):
    """MPS compression: create MPS from dense data, cap bond dimension at chi."""
    assert len(segment) == 1024
    mps = qtn.MatrixProductState.from_dense(
        segment, dims=2, max_bond=chi
    )
    return mps


def mps_decompress(mps):
    """Reconstruct from MPS."""
    return mps.to_dense().flatten()


def mps_param_count(mps):
    """Count total float values in MPS tensors."""
    return sum(t.size for t in mps.tensors)


# SNR Computation

def compute_snr(original, reconstructed):
    """SNR = 10*log10(sum(original^2) / sum((original - reconstructed)^2))"""
    signal_power = np.sum(original ** 2)
    noise_power = np.sum((original - reconstructed) ** 2)
    if noise_power < 1e-30:
        return float('inf')
    return 10.0 * np.log10(signal_power / noise_power)


# Main Benchmark

def run_benchmark():
    """Run full MPS vs SVD comparison."""
    print("=" * 80)
    print("MPS vs SVD for Audio Compression")
    print("=" * 80)

    # Generate audio data
    print("\nGenerating audio signals...")
    all_segments = {}
    for name, gen in AUDIO_GENERATORS.items():
        signal = gen()
        segments = extract_segments(signal, SEGMENT_LEN, NUM_SEGMENTS)
        all_segments[name] = segments
        print(f"  {name}: {len(segments)} segments x {SEGMENT_LEN} samples @ {SAMPLE_RATE} Hz")

    # Try torchaudio built-in dataset
    try:
        import torchaudio
        print("\nTrying torchaudio YESNO dataset...")
        try:
            dataset = torchaudio.datasets.YESNO(root='./_yesno_data', download=True)
            waveform, sample_rate, labels = dataset[0]
            print(f"  YESNO loaded: {waveform.shape}, sr={sample_rate}")
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sample_rate != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
                waveform = resampler(waveform)
            yesno_segments = extract_segments(waveform.numpy().flatten(), SEGMENT_LEN, NUM_SEGMENTS)
            all_segments['yesno'] = yesno_segments
            print(f"  yesno: {len(yesno_segments)} segments")
        except Exception as e:
            print(f"  YESNO failed: {e}, using synthetic only")
    except ImportError:
        print("  torchaudio not available, using synthetic only")

    # Compression parameter sweeps
    k_values = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    chi_values = [1, 2, 4, 6, 8, 12, 16, 24, 32]

    results = {}

    for name, segments in all_segments.items():
        print(f"\n{'-' * 60}")
        print(f"Audio type: {name} ({len(segments)} segments)")
        print(f"{'-' * 60}")

        # SVD Results
        svd_rows = []
        for k in k_values:
            segment_snrs = []
            for seg in segments:
                U_k, S_k, Vh_k = svd_compress(seg, k)
                recon = svd_decompress(U_k, S_k, Vh_k)
                recon_flat = recon.flatten()
                snr = compute_snr(seg, recon_flat)
                segment_snrs.append(snr)
            mean_snr = np.mean(segment_snrs)
            params = svd_param_count(k)
            cr = SEGMENT_LEN / params
            svd_rows.append((k, params, cr, mean_snr))

        # MPS Results
        mps_rows = []
        for chi in chi_values:
            segment_snrs = []
            for seg in segments:
                mps = mps_compress(seg, chi)
                recon = mps_decompress(mps)
                snr = compute_snr(seg, recon)
                segment_snrs.append(snr)
            mean_snr = np.mean(segment_snrs)
            params = mps_param_count(mps_compress(segments[0], chi))
            cr = SEGMENT_LEN / params
            mps_rows.append((chi, params, cr, mean_snr))

        # Aligned Comparison Table
        print(f"\n  {'Method':<8} {'Param':>6} {'CR':>8} {'SNR(dB)':>10} {'k/chi':>6}")
        print(f"  {'-' * 38}")
        for k, params, cr, snr in svd_rows:
            print(f"  {'SVD':<8} {params:>6d} {cr:>8.2f}x {snr:>10.2f} {'k=' + str(k):>6}")
        print()
        for chi, params, cr, snr in mps_rows:
            print(f"  {'MPS':<8} {params:>6d} {cr:>8.2f}x {snr:>10.2f} {'chi=' + str(chi):>6}")

        # Matched CR Comparison
        print(f"\n  --- Matched Compression Ratio Comparison ---")
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

        matched = []
        for mp in mps_points:
            best_svd = min(svd_points, key=lambda sp: abs(sp[3] - mp[3]))
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
            print(f"  {'Gap':>20} {'':>15} {gap:>+10.2f} dB ({winner} wins)")
            print()

        if total_comparisons > 0:
            win_rate = mps_wins / total_comparisons * 100
            print(f"  MPS wins: {mps_wins}/{total_comparisons} ({win_rate:.0f}%)")
        else:
            print("  No valid comparisons available")
            win_rate = 0.0

        results[name] = {
            'mps_wins': mps_wins,
            'total_comparisons': total_comparisons,
            'win_rate': win_rate,
            'svd_rows': [(k, params, cr, snr) for k, params, cr, snr in svd_rows],
            'mps_rows': [(chi, params, cr, snr) for chi, params, cr, snr in mps_rows],
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Does MPS beat SVD for audio?")
    print("=" * 80)

    tonal_wins = 0
    tonal_total = 0
    for name, r in results.items():
        w = r['mps_wins']
        t = r['total_comparisons']
        wr = r['win_rate']
        if t > 0:
            verdict = "MPS WINS" if wr > 50 else "SVD WINS" if wr < 50 else "TIE"
        else:
            verdict = "NO DATA"
        print(f"  {name:<25s}: MPS {w}/{t} wins ({wr:.0f}%) -> {verdict}")
        if 'noise' not in name:
            tonal_wins += w
            tonal_total += t

    if tonal_total > 0:
        overall = tonal_wins / tonal_total * 100
        print(f"\n  {'TONAL AVERAGE':<25s}: {tonal_wins}/{tonal_total} wins ({overall:.0f}%)")
        verdict = "PREDICTION CONFIRMED: MPS beats SVD for 1D audio" if overall >= 50 else "PREDICTION REFUTED: SVD still beats MPS"
        print(f"  >>> {verdict}")
    else:
        print("  No tonal audio data available for comparison")

    return results


if __name__ == '__main__':
    results = run_benchmark()
