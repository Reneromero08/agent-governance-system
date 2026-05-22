# HANDOFF — Experiment 16: Catalytic 27B Inference

## What This Is

A zero-RAM catalytic inference engine that runs Qwen 0.5B through a 256MB byte-level XOR fabric (the "tape"). Model weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape for compute, then re-scrambled after. Every token MUST restore the tape to its SHA-256 pre-computation state — zero bits erased. The ultimate target is coherent English text output from real model weights, at 50+ tok/s with warm-tape replay.

## Current State (2026-05-22 Agent Resume — Session 2)

**14 bugs fixed** (12 from previous agent, 2 from this session). **Warm cache correctly stores pre-uncompute hidden states.** **EigenBuddy trains to 100% on real catalytic data** but engine limited to 16 unique output tokens.

### What works:
- 100% tape restoration across all 48 layers
- Warm cache: stores correct pre-uncompute hidden state (COMPLEX_DIM = 7168 bytes), overwrite on collision
- Hidden state collection: 500 tokens collected, NaN-free, 16 unique engine output tokens
- EigenBuddy training on real data: 100% accuracy (normalized inputs, 8 classes / 40 samples)
- 2.94 tok/s, 82% warm-hit rate

### BLOCKER: 16-class output ceiling
The engine's output head (`lib.rs:1587-1591`) reads only `j in 0..64.min(HIDDEN_DIM)` f32 positions from the XOR'd tape. This caps unique output tokens at 64 theoretical / 16 observed. The hidden state is `initial_substrate XOR embedding XOR layer_outputs` — raw XOR'd bytes, not clean f32 values suitable for lm_head projection.

### Root cause of gibberish output
The compute uses element-wise `w[j] * x[j]` (diagonal application of weight vector to input vector) instead of full `W @ x` (each output dimension = dot product of one weight row against entire input). Every DeltaNet layer at `lib.rs:1546-1550` and every QKV projection at `lib.rs:1465-1469` applies weights diagonally. This means:
- `output[j] = w[j] * x[j]` — only self-interaction
- Should be: `output[i] = sum_j(W[i,j] * x[j])` — full cross-interaction

Without full dot products, the layers don't mix information across dimensions. The output is a dimension-wise scaling of a random XOR'd input — hence the limited token diversity.

## Paths Forward

| Path | Effort | Risk | Coherent output? |
|------|--------|------|-----------------|
| **A: Full W@x dot-product** | High (Rust-only) | Low (Q57-safe via multi-scale Feistel HDD tiling) | Yes — correct model output |
| **B: 16-class EigenBuddy** | Low (done) | High — limited to 16 tokens | No |
| **C: Expand output head** | Low (Rust 1-line) | Low — increases unique tokens to ~64 | Marginal improvement |

## Today's Changes

### Rust (`lib.rs`)
- Bug 13: Warm cache now saves `hidden_state_save` after forward (before uncompute), writes full COMPLEX_DIM bytes to cache after hash. Uses overwrite (`copy_from_slice`) not XOR for collision handling.
- Bug 14: `hidden_state` field returned to Python for direct consumption
- Removed unused `max_dim` variable

### Python
- `collect_hidden_states.py`: Uses Rust `hidden_state` field, NaN→0 normalization, engine token as target (not lm_head)
- `eigen_buddy_tokenizer.py`: `--data` flag for real catalytic training, input normalization (max-abs scaling), class remapping
- `ROADMAP.md`, `HANDOFF.md`: Updated with current state

## How to Run

```bash
cd "D:\CCC 2.0\AI\agent-governance-system"

# Build Rust
# (in THOUGHT\LAB\EIGEN_BUDDY\core\rust_ffi)
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" build --release
copy target\release\catalytic_ffi.dll target\release\catalytic_ffi.pyd

# Run experiment
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\experiment.py

# Collect hidden states
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\collect_hidden_states.py

# Train EigenBuddy on real data
.venv\Scripts\python.exe THOUGHT\LAB\EIGEN_BUDDY\eigen_buddy_tokenizer.py --data THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\collected_hidden_states\catalytic_hidden_states_500.pt

# Rust tests
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" test --release --lib
```
