# LAB HISTORY — The .holo 27B Breakthrough Attempt

Started 2026-08-03. Recorded live as the work happens. If this works, this is
the record of when a frontier model became holographic on consumer hardware.

## The mission

Weights-as-`.holo` — shrink real model weights onto eigenbasis geometry and run
catalytic inference through the basis without materializing `W`.

Endgame: DeepSeek V4 Flash 0731 (the model running this conversation — 43
layers, 64-head MQA, top-6/256 experts, 284B params / 13B active) holographic
on a 12 GB RTX 3060. That breaks the lab's resource wall: a resident frontier
model makes EIGEN_BUDDY training research, recursive compression, and all
compute-gated work affordable.

## The pipeline (all built this session)

```
Qwen3.6-27B full precision (Seagate, 52 GB)
-> run_distill.py          k=256 randomized SVD + cross-depth basis warm-start
                           -> output/qwen_27b_k256.holo   (3.53 GB, 14x)
-> qwen35_holo_engine.py   Qwen3.5 hybrid forward (48x GatedDeltaNet +
                           16x gated full attention), FP32 delta-rule state,
                           factorized projections, LoRA hooks, exact shell
                           (embed/norms/lm_head), exact-tail option
-> run_calibrate2.py       PER-PROJECTION analytic calibration (NO training):
                           query exact sublayers as oracle on real activations,
                           fit ridge least-squares corrections per projection
                           (activation-aware LoRA), damped rounds, held-out eval
-> eval_corrected.py       full 64-layer eval: hidden cosine, logit agreement,
                           generation
```

## The wall, measured honestly

At k=256, per-weight W-level cosine is 0.34-0.57. Residual correction in W
space fails: the residual spectrum is FLAT (top singular values ~2.9 down to
0.003), so rank-4 gives +0.004, rank-512 only +0.24. These matrices are NOT
low-rank at W level. 64-layer eval: hidden cosine decays 0.98 -> 0.19,
final logit cosine 0.69, argmax mismatch. The truncation suppresses the
late-layer norm growth the model depends on (RMS 2.1 vs exact 8.6).

Key finding: the MERA "0.84-0.89 fidelity" numbers were U-LEVEL (left singular
vectors), not W-LEVEL - they do not survive W reconstruction. The
`_residual_correct.py` rank-4 boost was theoretical, never validated, and
measures +0.004 in practice.

## The breakthrough

Per-projection analytic correction (Sol's diagnosis): corrections on sublayer
OUTPUTS are too late - the errors are made BEFORE nonlinearities (Q/K
geometry, GDN recurrent state, MLP gates). Correct at each PROJECTION input:

    projection(x) = holo_projection(x) + (x A^T) B^T    # least squares, no training

First run (rank 128, 8 prompts) FAILED - corrections hurt (first-layer cosine
0.86 vs 0.977). Three causes found:
1. MLP corrections fit on layer input, applied on post-mixer input (bug)
2. Rank 128 overparameterized on ~80 tokens
3. THE BIG ONE: the calibrator's prompts were hand-crafted TOKEN GARBAGE -
   ".Q y aertyomQuantityse sub" etc. All calibration happened on garbage
   distribution. The 0.855 "held-out" score was on garbage sequences.

Fix: real English prompts, tokenized at runtime, padded. Re-running now.

## Results so far (2026-08-03)

Garbage-prompt calibration, train projection cosine (per layer, rank-16):
  base 0.66-0.86  ->  corrected 0.80-0.92   (all 368 projections improved)
  held-out logit cosine (garbage distribution): 0.855 mean
  real-prompt eval with those adapters: 0.568 logit cosine (OOD - expected)

Real-prompt calibration (12 prompts, 10 train + 2 held-out):
  RUNNING - round 3, train projection cosine 0.66-0.85 -> 0.80-0.92
  held-out: "Signal and noise are distinguished by",
            "A reversible computation can always be"

## Honest status

The mechanism is sound (every projection improves on its own distribution).
The distribution bug is fixed. The real test is: does real-prompt calibration
produce held-out logit cosine > 0.7-0.8 AND coherent generation? Generation is
the harder test (per-token OOD drift) and currently O(T^2) because GDN state
rebuilds per token - the engine needs incremental delta-rule state caching.

## Artifacts (all on disk, this worktree)

- output/qwen_27b_k256.holo           3.53 GB  (gitignored)
- output/qwen_adapters_proj.pt        202 MB   (garbage-distribution run)
- output/qwen_adapters_real.pt        IN PROGRESS
- output/calibrate_real.log           live log
- pipeline/01_distill/run_distill.py
- pipeline/04_inference/qwen35_holo_engine.py
- pipeline/04_inference/run_calibrate2.py
- pipeline/04_inference/eval_corrected.py
- docs/QWEN_ARCHITECTURE_REFERENCE.md     (Sol's shape/arch analysis)
- docs/CODEX_DECISION_WORKING_MODEL.md    (teacher-distillation fallback plan)
- Seagate: Models/.holo/deepseek-v4-flash/  (38 GB full DeepSeek .holo corpus)
- Seagate: Models/deepseek-ai/_holo/        (14 GB complete distillation)

## What "change history" requires next

1. Real-prompt calibration finishes -> eval -> generation coherence
2. If coherent: recursive compression (compress the corrections too)
3. Port engine to DeepSeek V4 CSA/MQA/MoE (old _holographic_engine.py exists)
4. Download DeepSeek V4 Flash 0731 originals (free, open-weights) as oracle
5. Calibrate experts on routed activations (MoE specialists = smaller
   activation-manifold error)
6. 0.02 GB GPU peak was already demonstrated on the old engine - 12 GB is plenty
