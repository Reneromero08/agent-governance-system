# HOLO_WEIGHTS — Catalytic Inference on Real Model Weights

Clean workspace for the weights-as-`.holo` continuation: shrink real model weights
onto `.holo` eigenbasis geometry and run catalytic inference through the basis
without materializing the weight matrices.

## Mission

```text
Qwen3.6-27B (full precision safetensors, Seagate)
-> SVD distill to .holo (U + SVh, shared bases, INT8)
-> wormhole rotation R_i + 2-bit residual correction (proven 0.84-0.89 fidelity)
-> per-layer adapter correction for weak layers
-> phase-lock + exact arithmetic for depth stability
-> native .holo inference (x @ SVh.T @ U.T, never materializing W)
-> verify against llama.cpp baseline
```

## Why this model

- The Seagate holds the full-precision source: `Models/Qwen/Qwen3.6-27B/` (15 shards, 52 GB).
- Every distiller in `pipeline/01_distill/` was originally hardcoded to this model
  (`distill_27b_holo.py` is explicitly out-of-core for Qwen 27B).
- The wormhole + 2-bit-residual fidelity repair (`pipeline/02_wormhole/REPORT.md`)
  was measured on Qwen 27B: 0.127 -> 0.862 (gate_proj), 0.188 -> 0.889 (q_proj) at k=256.
- The `.holo` attention distillation (34,212x, 143 matrices, 1.6 MB) that routed
  `fibonacci` at 0.402 was Qwen 27B.
- Prior distillates exist on the Seagate: `Models/.holo/qwen-27b/`.

## Pipeline

| Stage | Directory | Contents | Status |
|---|---|---|---|
| 1. Distill | `pipeline/01_distill/` | Qwen 27B SVD distillers (out-of-core, catalytic, GGUF), `load_holo_v2.py` | Proven (34,212x attention) |
| 2. Wormhole + residual | `pipeline/02_wormhole/` | 33_mera: cross-layer MERA, wormhole rotations, 2-bit residuals, analytic alignment, `_residual_correct.py` | Proven on Qwen 27B (0.84-0.89 fid) |
| 3. Adapter correction | `pipeline/03_adapter/` | Trained residual adapters (GPT-2 KV: +0.104 @ 85.3x, delta grows with compression), spectral LLM compression, `gguf_backend.py` | Proven on GPT-2; Gemma-4 complex sweep exists |
| 4. Inference | `pipeline/04_inference/` | Catalytic engine cores (EIGEN_BUDDY), HOLO pipeline (cavity, wormhole, calibrate, inference engines) | Ran 43-layer DeepSeek engine (incoherent output; fidelity was the cause) |
| 5. Validate | `pipeline/05_validate/` | PLV phase diagnostics (predict adapter difficulty), fidelity sweeps, Gemma calibration | Proven diagnostics |
| Native phase machinery | `native/phase_frontier/` | CAT_CAS audio-lane C engine: phase-lock (0.074 -> 1.6e-16), exact cyclotomic arithmetic (Q(zeta17), depth 4096), CATVM custody | Proven, independently verified |

## Known blockages and their state (2026-08-03 audit)

1. Depth-fidelity compounding (0.64^43 ~ 5e-9): truncation half fixed by 2-bit
   residual + adapters; drift half fixed by phase-lock / exact arithmetic.
2. `_residual_correct.py` (rank-4) was built, never executed; its old output
   `ds_experts_residual_r4.holo` on the Seagate is an empty stub (988 bytes).
3. GGUF dequant was faked in the old distillers; real path = `gguf_backend.py`
   (llama-cpp-python) or ggml kernels in Neo3000.
4. MHC (Sinkhorn hyper-connections): not implemented anywhere; last priority.
5. `wo_a` SVh dimension bug: trivial distiller fix.

## Next steps

1. Fix `F:`/`E:` drive paths -> Seagate paths (see `config/paths.json`).
2. Run `distill_27b_holo.py` (out-of-core) on `Qwen3.6-27B` at k=256.
3. Apply wormhole + 2-bit residual (`pipeline/02_wormhole/`) against the
   full-precision safetensors original.
4. Measure per-layer fidelity; apply trained adapters to weak layers.
5. Port phase-lock into the inference depth path; measure depth stability.
6. Compare native `.holo` inference against a llama.cpp baseline of the same model.

## Rules

- Copy-only lineage: files in this folder are copies of lab originals (see
  `MANIFEST.md`). Prefer running from here; fix paths here, not in the originals.
- Large binaries (model weights, big `.holo` archives) are NOT stored here.
  They live on the Seagate; `config/paths.json` points at them.
