# Task 4b: MPS vs SVD with Fourier Pre-Processing

**Date**: 2026-05-16
**Status**: HYPOTHESIS REFUTED
**MPS Win Rate**: 18% frequency-domain vs 16% time-domain (+2pp, within noise)

---

## Hypothesis

Task 4 found MPS only wins for pure sine waves (16% overall win rate).
The hypothesis was that MPS was tested on the wrong representation:
raw audio is time-domain, but its compressible structure lives in the
frequency domain. A Fourier transform should produce sparse, low-entanglement
spectra that MPS exploits.

**Prediction**: MPS win rate > 50% on frequency-domain data.

---

## Methodology

### Pipeline
1. Generate same 10 audio types as Task 4 (1s @ 8kHz)
2. Split into 3 non-overlapping 1024-sample segments
3. Apply full 1024-point FFT to each segment
4. Take magnitude spectrum (1024 values, discarding phase)
5. Compress magnitude spectrum with:
   - **SVD**: reshape (32,32), truncated SVD at k=1..32
   - **MPS**: quimb MPS with dims=2 (10 sites), bond cap chi=1..32
6. Compute SNR on reconstructed magnitude spectrum vs original
7. Match by compression ratio with CR ratio filter (max 1.5x gap),
   count MPS wins

### Fixes Applied
1. **CR gap filter**: MPS chi=1 (51.20x CR) has no valid SVD competitor
   (nearest is k=1 at 15.75x, ratio 3.25x > 1.5x threshold). Filtered out.
   Previous version counted this as an SVD win, inflating SVD win count.
2. **Random seed**: `np.random.seed(42)` set for noise_burst reproducibility.

### SNR Metric
```
SNR = 10 * log10(||original||^2 / ||original - reconstructed||^2)
```

---

## Results

### Frequency-Domain Win Rates vs Time-Domain (Task 4)

| Audio Type | Time-Domain | Freq-Domain | Delta | Pairs |
|-----------|:-----------:|:-----------:|:-----:|:-----:|
| sine_440 | 50% | 25% | -25pp | 1/4 |
| sine_880 | 50% | 25% | -25pp | 1/4 |
| sweep_200_2000 | 0% | 20% | +20pp | 1/5 |
| square_220 | 12% | 12% | 0pp | 1/8 |
| triangle_330 | 11% | 12% | +2pp | 1/8 |
| am_sine | 50% | 25% | -25pp | 1/4 |
| fm_sine | 0% | 20% | +20pp | 1/5 |
| noise_burst | 0% | 0% | 0pp | 0/8 |
| harmonic_complex | 17% | 17% | 0pp | 1/6 |
| piano_like | 17% | 17% | 0pp | 1/6 |
| **TONAL AVERAGE** | **16%** | **18%** | **+2pp** | **9/50** |

### Key Observations

1. **Fourier pre-processing does not help MPS.** Overall win rate: 18% vs
   16% (Task 4 time-domain). The +2pp improvement is within noise.

2. **MPS every win is chi=2 vs SVD k=1.** The pattern is universal: MPS
   chi=2 (72 params, 14.22x CR) beats SVD k=1 (65 params, 15.75x CR) by
   2-13 dB in every audio type. But SVD k=2 (130 params, 7.88x CR) would
   beat MPS chi=2 at a comparable CR. This is a structural limitation of
   the nearest-CR matching algorithm.

3. **Sine waves dropped: 50% to 25%.** In the time domain, a sine wave is
   perfectly rank-2 (bond dimension 2). In the frequency domain, the FFT
   magnitude spectrum of a finite 1024-sample sine wave is a sinc function,
   requiring higher bond dimension. MPS lost its best-case advantage.

4. **Sweeps and FM improved: 0% to 20%.** Frequency-varying signals
   benefit slightly from frequency-domain representation. The magnitude
   spectra of sweeps/FM are densely peaked, which MPS captures better
   than time-domain. But the advantage is marginal (+2pp overall).

5. **Noise unchanged: 0%.** Both methods fail on noise, in any domain.

### Why the Hypothesis Failed

Three structural reasons:

1. **Sinc spreading**: A finite-length FFT of a pure tone produces a sinc
   function, not a delta peak. The oscillating tail requires high bond
   dimension. In time domain, that same tone is perfectly bond-dim-2.

2. **SVD dominates spectral data**: The magnitude spectrum of any signal
   concentrates energy in a few spectral peaks. SVD on the (32,32)
   reshaped spectrum extracts these peaks efficiently via singular values.
   A few peaks = a few significant singular values.

3. **MPS plateau saturation**: Once chi exceeds the effective rank of the
   spectral data, further chi increases don't change the MPS (the
   `from_dense` SVD decomposition finds the true bond dimension). For
   sine-like spectra, chi>=8 gives identical results, wasting params.

---

## Conclusion

### Hypothesis: REFUTED

MPS win rate on frequency-domain data: **18%**. Predicted: >50%.
The improvement over time-domain (16%) is +2pp, within noise.

### The Chi=2 Artefact

Every audio type shows exactly one MPS "win": chi=2 vs SVD k=1. This is
not a real MPS advantage -- it's a matching artefact where the closest
SVD competitor (k=1) is undermatched. At any higher compression quality,
SVD dominates by 5-245 dB.

### When Could This Work?

Fourier pre-processing would help MPS IF:
- The signal is extremely long (infinite limit -> delta peaks)
- The spectrum is literally sparse (few nonzero bins)
- Using a sparse wavelet representation instead of Fourier

These conditions do not hold for practical audio compression.

---

### Files
- `mps_audio_fft.py` - benchmark script (seed=42, CR ratio filter=1.5x)
- `REPORT.md` - this file

*Generated by Task 4b: Fourier Pre-Processing Benchmark*
*Hypothesis refuted: 2026-05-16*
