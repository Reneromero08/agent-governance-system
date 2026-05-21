# HANDOFF — Experiment 16: Catalytic 27B Inference

## What This Is

A zero-RAM catalytic inference engine that runs Qwen 0.5B through a 256MB byte-level XOR fabric (the "tape"). Model weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape for compute, then re-scrambled after. Every token MUST restore the tape to its SHA-256 pre-computation state — zero bits erased. The ultimate target is coherent English text output from real model weights, at 50+ tok/s with warm-tape replay.

## Current State

**F32 tape engine compiles and runs** with real Qwen 0.5B BF16 weights. Forward pass is deterministic and correct. Uncompute pass fails tape restoration (0% restore rate). Output is real Qwen subword tokens (`'8'`, `'!'`, `']'`, etc.) — not coherent yet because restoration fails and weights accumulate damage across tokens.

Performance snapshot:
- 3.2 tok/s, 66% warm-hit rate, 42% on cold runs
- Real Qwen tokenizer and real embedding table (151,936 vocab × 896 dim, BF16→f32)
- HIDDEN_DIM = 896, byte-level XOR fabric with f32 values stored as 4-byte sequences
- Complex-plane memory: X channel (real activations) + Y channel (imaginary, currently a placeholder)

## Files That Matter

| File | Purpose |
|------|---------|
| `THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/src/lib.rs` | The entire inference engine: tape helpers, compute, attention, uncompute, SPN scramble |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/experiment.py` | Python orchestration: tokenizer, weight loading, embedding extraction, FFI calls |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/ROADMAP.md` | Full roadmap and current status |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/gemini_update/plan.md` | Complex-plane architecture plan from Gemini |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/gemini_update/qwen_0.5b/` | Actual Qwen 0.5B model files (0.9GB safetensors + tokenizer) |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/deprecated/` | Archived u8 quantization path (old engine) |

## How to Run

```bash
cd "D:\CCC 2.0\AI\agent-governance-system"

# Build Rust
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" build --release
cp THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/target/release/catalytic_ffi.dll THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/target/release/catalytic_ffi.pyd

# Run experiment
.venv\Scripts\python.exe THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/experiment.py

# Run Rust tests (single-layer DeltaNet f32 XOR restore passes)
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" test --release --lib
```

## The Critical Problem: Tape Restoration (Uncompute)

The single-layer f32 DeltaNet test PASSES (`test_delta_net_1layer_trace` in lib.rs). Forward: XOR input with gate, backward: reads gate from tape, XORs input → restores perfectly. The f32 XOR fabric works at the bit level.

The 48-layer engine fails restoration because:
1. **Multi-layer interaction**: layer N's uncompute reads from tape regions that layer N+1's forward modified. The per-layer `pre_gate` and `saved_output` buffers are supposed to isolate these, but something still leaks.
2. **Attention layers (3:1 stride)**: 12 of 48 layers are gated attention with Q·K† dot products on complex coordinates. These touch KV cache regions, RMS norm values, output projections — all of which need correct reversal.
3. **Weight save/restore**: The forward saves dirty weight substrate bytes, decatalyzes (SPN unscramble), computes, re-scrambles, restores. The uncompute does the same. These MUST produce identical weight values for the gate computations to match.

### What I Proved Works

- `tape_f32()` and `tape_f32_xor()` correctly XOR f32 values at the byte level (test passes)
- Single DeltaNet layer: forward gate + backward gate XOR restores input perfectly
- Forward pass is deterministic (same input → same token every time)
- Real Qwen weights produce real Qwen subword tokens
- 66% warm-hit rate proves the 256-slot FNV-1a cache is functional

### Debugging Approach

1. **Add SHA-256 checkpoints at each layer boundary**: Compute `Sha256::digest(&tape[0..scratch_base])` at the START of each layer's forward and the END of each layer's uncompute. The layer where they diverge is the broken one.
2. **Extend the Rust test harness**: Add a `test_delta_net_2layer_restore()` test to isolate multi-layer interaction. Add `test_attention_1layer_restore()` for attention.
3. **Compare forward vs uncompute per-layer values**: Dump gate values, Q projection values, and attention scores during forward and uncompute for a single layer. They MUST match exactly.
4. **Verify the SPN weight roundtrip**: `spn_unscramble` followed by `spn_scramble` should restore the original u8 bytes. The dirty substrate save/restore in the uncompute must capture the SAME bytes as the forward.

## Key Rust Architecture (lib.rs)

```rust
const HIDDEN_DIM: usize = 896;
const F32_BYTES: usize = 4;
const F32_DIM: usize = HIDDEN_DIM * 4;  // 3584 bytes
const COMPLEX_DIM: usize = F32_DIM * 2;   // 7168 bytes (XY channels)

fn tape_f32(tape: &[u8], base: usize, idx: usize) -> f32 { ... }
fn tape_f32_xor(tape: &mut [u8], base: usize, idx: usize, val: f32) { ... }
```

Tape layout:
```
input_offset = 0 (COMPLEX_DIM = 7168 bytes)
weight_offset = COMPLEX_DIM (HIDDEN_DIM bytes for u8 scrambled weights)
weight_f32     = COMPLEX_DIM (HIDDEN_DIM * 4 bytes for f32 weights — OVERLAPS with first 896 bytes of u8 region!)
scratch_base   = weight_offset + num_layers * HIDDEN_DIM
temp / pre_gate / saved_outputs / warm_tape_cache / kv_cache
```

**Critical**: `weight_offset` and `weight_f32` start at the SAME byte offset (7168) but have different sizes (896 vs 3584). The weight save/restore captures 3584 f32 bytes but the decatalysis writes to the first 896 bytes. Ensure the save captures ALL 3584 bytes BEFORE the decatalysis overwrites.

## Python Architecture (experiment.py)

- `HDDWeightStreamer`: Loads safetensors, extracts embedding table, scrambles weights via `catalytic_ffi.scramble_catalysis_weights()`
- `TokenizerBridge`: Real Qwen AutoTokenizer + hash-based fallback
- `CatalyticInferenceRuntime`: Orchestrates the FFI call per token, syncs working region, tracks warm hits, runs daemon
- `ThermodynamicDaemon`: Polar rotations at g=0.001 every 100 tokens

## What's Blocking Completion

1. **Uncompute restoration** (critical): The 48-layer engine must produce `tape_restored=True`. Single layer works. Multi-layer doesn't.
2. **Real imaginary channel**: The Y channel currently duplicates the X channel. The Gemini plan calls for phase curvature tracking in Y. This would make attention heads actually compute meaningful scores.
3. **Weight calibration**: Weights are BF16→f32 passed through unchanged. No quantization, which is correct. But the learned Qwen weight distribution may need scaling for the XOR fabric.

## Quick Wins (If Uncompute Is Fixed)

- Warm-tape replay already at 66% — just needs restoration to activate
- Real Qwen tokenizer already producing actual tokens from Qwen vocabulary
- lm_head projection already wired (hidden @ embed_tokens.T)
- 27B model path is configured at `G:/models/qwen3.6-27b-fp8-mtp.safetensors` — just needs the file
- Complex-plane memory already built

## Contact / Context

This is Experiment 16 in the CAT_CAS (Catalytic Space) lab at `D:\CCC 2.0\AI\agent-governance-system`. The broader project is "Agent Governance System" but the CAT_CAS lab specifically explores catalytic computing: computation on borrowed dirty substrate with zero erase, using black hole thermodynamics as the compute model. Other experiments in the lab include the Hawking Decompressor (#18), Temporal Bootstrap (#17), and Bekenstein Violator (#14).
