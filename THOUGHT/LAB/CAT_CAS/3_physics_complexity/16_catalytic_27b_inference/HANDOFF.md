# HANDOFF — Experiment 16: Catalytic 27B Inference

## What This Is

A zero-RAM catalytic inference engine that runs Qwen 0.5B through a 256MB byte-level XOR fabric (the "tape"). Model weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape for compute, then re-scrambled after. Every token MUST restore the tape to its SHA-256 pre-computation state.

## Current State (2026-05-22 Session 3)

**15 bugs fixed.** W@x dot-product operational on 12 attention layers. Latent Phase Cavity pipeline validated at 95% accuracy (20 tokens). HOLO 4 auto-feedback approach (phase grating compression + adapters) likely obsoletes catalytic fabric for inference speed but catalytic wins for zero-RAM weight storage.

### What works:
- 100% tape restoration across all 48 layers
- Block-tiled W@x on attention layers (Q, K, V, O — 224 blocks × 4 rows per matrix)
- Warm cache: stores pre-uncompute hidden state, pre-population from simulated states bypasses 14s cold-miss
- Qwen oracle: 15 tok/s on RTX 3060 (KV cache, fp16)
- Latent Phase Cavity: .holo latent space + k-NN + warm cache = 95% top-1, 100% cavity hit
- 6/6 Rust tests passing

### What's still blocked:
- DeltaNet layers (36/48) still element-wise — output gibberish
- Output head reads only 64 f32 positions → max 64 tokens
- Real catalytic hidden states need cold-miss compute (14s per unique token)

### HOLO 4 obsoletes:
The `THOUGHT/LAB/HOLO/4_holographic_brain/auto_feedback.py` pipeline compresses Qwen weights via phase grating SVD, trains lightweight Phase Adapters to correct dispersion, and runs at GPU speed. The catalytic fabric still wins for zero-RAM SPN-scrambled weight storage, but inference should route through HOLO 4 adapters.

## How to Run

```bash
cd "D:\CCC 2.0\AI\agent-governance-system"

# Build Rust
# (in THOUGHT\LAB\EIGEN_BUDDY\core\rust_ffi)
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" build --release
copy target\release\catalytic_ffi.dll target\release\catalytic_ffi.pyd

# Run experiment
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\3_physics_complexity\16_catalytic_27b_inference\experiment.py

# Generate gold data (Qwen oracle + catalytic engine)
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\3_physics_complexity\16_catalytic_27b_inference\generate_gold_data.py

# Latent Phase Cavity test
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\3_physics_complexity\16_catalytic_27b_inference\_test_cavity_full.py

# HOLO 4 auto-feedback (the faster path)
.venv\Scripts\python.exe THOUGHT\LAB\EIGEN_BUDDY\training\auto_feedback.py

# Rust tests
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" test --release --lib
```

## Key Files

| File | Purpose |
|------|---------|
| `THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/src/lib.rs` | Catalytic engine: W@x, warm cache, hidden_state return |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/16_catalytic_27b_inference/experiment.py` | Python orchestration with full matrix weight loading |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/16_catalytic_27b_inference/generate_gold_data.py` | Qwen oracle + catalytic verifier data collection |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/16_catalytic_27b_inference/_test_cavity_full.py` | Latent Phase Cavity (95% validated) |
| `THOUGHT/LAB/HOLO/4_holographic_brain/auto_feedback.py` | HOLO 4.5: Phase Adapters + Wave Attention — faster path |
| `THOUGHT/LAB/EIGEN_BUDDY/eigen_buddy_tokenizer.py` | EigenBuddy with complex SVD compression + --data flag |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/20_catalytic_eigen_shor/20_10_tiny_compress_phase/` | Moire decomposition, Phase Cavity, latent lattice — algorithm source |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/21_holographic_elliptic_sieve/` | Next level: elliptic curve factoring via phase resonance |
| `THOUGHT/LAB/CAT_CAS/3_physics_complexity/16_catalytic_27b_inference/ROADMAP.md` | Updated roadmap |
