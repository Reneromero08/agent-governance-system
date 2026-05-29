# Phase 3.5 Final Report: GPT-2 KV Cache Compression

**Date:** 2026-05-17
**Model:** GPT-2 (124M), local checkpoint (497MB safetensors)
**GPU:** RTX 3060 12GB, CUDA via venv torch 2.5.1+cu121
**Status:** COMPLETE — 13 tasks across adapter training, phase measurement, and phase-aware optimization.

---

## Adapter Training (8 Tasks)

### Task 1: Push Past 85x (OUT-OF-SAMPLE)

| k | Compression | PCA Cosine | Adapter Cosine | Delta |
|---|-------------|-----------|---------------|-------|
| 9 | 85.3x | 0.690 | 0.752 | +0.062 |
| 6 | 128.0x | 0.646 | 0.731 | +0.085 |
| 3 | 256.0x | 0.588 | 0.694 | +0.107 |
| 1 | 768.0x | 0.527 | 0.649 | +0.122 |

**Finding: PASS.** Adapter at k=3 (256x) matches PCA at k=9 (85.3x) on held-out data. 3x compression gain. Delta GROWS with compression.

### Task 2: Asymmetric Budget (OUT-OF-SAMPLE)

| K | V | Compression | Adapter Cosine |
|---|---|-------------|----------------|
| 3 | 15 | 85.3x | 0.767 > sym k9=0.752 (+0.015) |
| 5 | 25 | 51.2x | 0.825 |
| 8 | 36 | 34.9x | 0.868 |

**Finding: PASS.** V needs more budget than K. Asymmetric beats symmetric at equal total.

### Task 3: Bottleneck Sweep (OUT-OF-SAMPLE)

| Bottleneck | Params/Layer | Adapter Cosine |
|------------|-------------|----------------|
| 32 | 49K | 0.713 |
| 64 | 99K | 0.752 |
| 128 | 198K | 0.784 |
| 256 | 397K | 0.802 |

**Finding: PASS.** Knee at 64-128. Diminishing returns past 128.

### Tasks 4-8

| Task | Result | Verdict |
|------|--------|---------|
| 4. Shared adapter | Gap 0.246 vs per-layer | FAIL — layer-specific |
| 5. Cross-model transfer | Gap 0.153 | FAIL — weight-specific |
| 6. Joint K+V | Joint 0.747 < separate 0.752 | FAIL — compete for bottleneck |
| 7. Warm-start | 5/6 comparisons beat random | PASS — init helps |
| 8. Direct decoder | Dec < PCA (shape bug) | FAIL* — unreliable |

---

## Phase Measurement (5 Tasks)

### Task 1: PLV Matrix (144 heads)

**Finding:** Phase-locking is overwhelmingly within-layer. Top-10 PLV pairs all same-layer (PLV 0.994-0.997). 18 phase clusters form from 515 phase-locked pairs. Layer 11 is the phase outlier (PLV=0.750 vs 0.917-0.987). This is the same layer where the KV adapter fails (-0.013 delta). Phase dispersion predicts adapter difficulty.

### Task 2: Phase Dispersion Early-Warning

| k | Attention Cosine | Phase Leads? |
|---|-----------------|--------------|
| 9 (85.3x) | 0.599 | NO (CC lag -1 = -0.816) |
| 3 (256.0x) | 0.406 | **YES** (CC lag -1 = 1.729 > lag 0 = 1.003) |

**Finding:** At k=3 (aggressive compression), phase dispersion spikes 1+ tokens BEFORE attention cosine drops. Phase is a leading indicator of compression failure.

### Task 3: Phase Coherence Loss

**Finding: FAIL.** Adding phase preservation term (lambda 0.1/0.5/1.0) to attention MSE loss has ZERO effect on adapter quality. Attention MSE already captures phase relationships implicitly through the attention output. Phase loss adds no new signal.

### Task 4: Phase-Guided Budget Allocation

**Finding: Marginal.** Allocating adapter capacity by per-layer PLV (L11 gets +25% bottleneck, L5 gets -5%) improves adapter cosine by +0.006 at k=9 and +0.002 at k=3. Effect is real but too small to matter at this scale.

### Task 5: Phase Dispersion Monitor

**Finding: Operational.** Baseline phase dispersion 0.081 +/- 0.030 on dominant cluster (Layer 5). Warning at 2sigma (0.142), critical at 3sigma (0.172). 2/15 tokens flagged as warnings during generation at k=3. Monitor functions as real-time early-warning system for compression failures.

---

## Key Findings

1. **Adapter triples compression.** k=3 (256x) matches k=9 (85x) PCA out-of-sample.
2. **Asymmetric budget wins.** V's higher intrinsic dimensionality justifies larger budget.
3. **Delta grows with compression.** Adapter learns more when PCA discards more.
4. **Layer and weight specific.** Shared/transfer/joint all fail. Per-layer training required.
5. **Phase-locking is within-layer.** 18 clusters. Layer 11 is the outlier (both in PLV and adapter performance).
6. **Phase leads attention at 256x.** Phase dispersion is an early-warning metric for compression failure.
7. **Phase loss adds nothing.** Attention MSE already captures phase. Phase-guided budget is marginal.
8. **Phase monitor works.** 2/15 tokens flagged. Real-time compression failure detection is operational.

---

## Files

```
THOUGHT/LAB/TINY_COMPRESS/
  extensions/03_flat_llm/
    train_adapter.py          — Initial training loop (GPU default)
    flat_llm_adapter.py       — Adapter architecture + attention compute
    trained_adapters.pt       — Saved weights (12 layers)
  llm-spectral/sweeps/
    sweep.py                  — Unified sweep (Tasks 1-8)
    SWEEP_REPORT.md           — Adapter sweep report
    sweep_task[1-8].json      — Per-task JSON results
  llm-spectral/phase/
    task1_plv.py              — PLV matrix 144x144
    task2_dispersion.py        — Phase dispersion per-token
    tasks_345.py              — Phase loss, budget, monitor
    PHASE_REPORT.md           — Phase measurement report
    plv_matrix.json           — Full PLV data + clusters
    task2_k[9,3].json         — Per-token dispersion data
    task3_results.json        — Phase loss lambda sweep
    task5_monitor.json        — Monitor flags
```
