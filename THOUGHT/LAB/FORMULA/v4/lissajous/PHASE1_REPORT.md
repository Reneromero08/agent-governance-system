# Lissajous Hypothesis — Phase 1+2 Results

Date: 2026-05-14 | Status: **NULL at current depth — requires DEM-level analysis**

---

## Tested

Three operationalizations of "frequency" from existing QEC data:

1. **Syndrome density ratios** between adjacent distances
2. **logR delta** as frequency proxy  
3. **Coherence (1-syn) ratios**

## Results

| Approach | Pearson r | R² | Verdict |
|----------|----------|-----|---------|
| syn ratios | -0.21 | 0.04 | **Null** — ratios nearly constant (~1.1) |
| logR delta | +0.85 | — | Circular — re-measures sigma |
| (1-syn) ratios | monotonic | — | No U-shape — sigma is U-shaped, rationality is monotonic |

## Why

Syndrome density is an aggregate measure. The frequency ratios between
distance levels computed from aggregate syn are all ~1.05-1.18 regardless
of p. There isn't enough variation to distinguish sigma=6.9 from sigma=0.7.

The hypothesis says: *"Each stabilizer has an effective frequency determined
by its error rate, its position in the lattice, and its coupling to neighboring
stabilizers."* This requires per-stabilizer analysis — the DEM correlation
matrix and its eigenvalue spectrum — which existing data doesn't capture.

## What Would Be Needed

1. Extract detector error model for each (p, d) condition
2. Build the stabilizer correlation matrix (which detectors fire together?)
3. Eigendecomposition to find dominant frequency modes
4. Compare the frequency spectrum between d and d+2 to measure rationality
5. Check if frequency rationality predicts sigma

This requires re-processing the existing sweep data to extract DEM-level
information, or fresh simulations with DEM output saved.

## Not Falsified, Not Confirmed

The hypothesis is still specific and falsifiable. It just can't be tested
with aggregate syndrome density — it needs structural analysis of the
stabilizer network at the per-detector level.
