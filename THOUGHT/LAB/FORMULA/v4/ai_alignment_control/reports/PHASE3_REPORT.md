# Phase 3 Report: Symbol Survival

Date: 2026-05-13 | Method: Text corruption chain + sentence embeddings | Status: **COMPLETE — NEGATIVE**

---

## Summary

High-compression symbols show a weak survival advantage at low noise (+0.11 over
low-compression controls, $r = 0.35$), but the effect disappears at medium noise
($r = -0.17$) and near zero at high noise ($r = -0.05$). The formula's prediction
that $\sigma \cdot D_f$ amplifies survival is not confirmed with text corruption
as the noise model.

## Method

- **Symbols**: 10 phrases with assigned $(\sigma, D_f)$ values ranging from
  (1,1) for "Be good" to (5,5) for the constitution core principle
- **Transmission**: 8-generation chain with random word drop and adjacent swap
- **Noise levels**: LOW (10% drop), MED (25% drop, 10% swap), HIGH (50% drop, 25% swap)
- **Chains**: 5 independent chains per symbol per noise level
- **Metric**: Cosine similarity of sentence embeddings (all-MiniLM-L6-v2) to original

## Results

| Noise | Corr($\sigma \cdot D_f$, final_sim) | High $\sigma \cdot D_f$ mean | Low $\sigma \cdot D_f$ mean | Delta |
|-------|-------------------------------------|------------------------------|-----------------------------|-------|
| LOW | +0.35 | 0.646 | 0.540 | **+0.107** |
| MED | -0.17 | 0.334 | 0.436 | -0.101 |
| HIGH | -0.05 | 0.350 | 0.370 | -0.020 |

High-compression symbols have a slight advantage at low noise but the effect
reverses at medium noise. The correlation is weak even at low noise ($r=0.35$).

## Why This Failed

Text corruption (random word drop/swap) disproportionately damages compressed
symbols. A compressed symbol like "Be good" loses half its meaning when one word
drops. An expanded symbol like "Justice is the constant and perpetual will to
render to each his due" retains meaning through redundancy — dropping "perpetual"
or "constant" leaves the core concept intact.

The formula predicts $\sigma \cdot D_f$ amplifies survival through *structured*
redundancy, not random word count. Real cultural transmission uses paraphrasing,
semantic reconstruction, and attractor dynamics — an LLM retelling preserves
semantic intent while varying wording, which is the correct noise model.

## Limitations

1. **Noise model mismatch**: Word corruption is not cultural transmission
2. **$\sigma$ and $D_f$ are human-assigned**: Not measured from the symbols themselves
3. **N=10 symbols, 5 chains each**: Small sample
4. **No LLM-based transmission chain**: Memory exhaustion prevented the intended test

## Next Step

Re-run with an LLM-based transmission chain (paraphrase-and-retell) when GPU
memory is available. This would test the correct noise model: semantic
reconstruction with attractor dynamics rather than random word corruption.

## Files

- `phase3_survival.py` — lightweight text corruption test
- `results/phase3_results.json` — per-symbol trajectories
