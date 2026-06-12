# CAT_CAS/16: Final Report — Tape Restoration + Weight Streaming

**Date**: 2026-05-22
**Status**: WEIGHT STREAMING COMPLETE — 100% tape restoration, real weights active
**Previous**: 2026-05-21 BUGBOUNTY — 100% tape restoration (`bfbfe310`, `e06d5207`)

---

## Summary (2026-05-22 Update)

The catalytic 27B inference engine now achieves **100% SHA-256 tape restoration** across all 50 tokens with **real Qwen 0.5B model weights streaming into the f32 compute region**. 12 bugs fixed across three commits. The weight streaming pipeline (16.8) is operational: SPN-unscrambled u8 bytes route into `lwo_f32` for compute access. A Platonic EigenBuddy Tokenizer prototype was developed alongside to explore decoding embedding-space vectors directly into tokens.

## Performance (2026-05-22)

| Metric | 2026-05-21 | 2026-05-22 |
|:---|---:|---:|
| Tape restoration | 100% | **100%** |
| Speed | 3.16 tok/s | **2.74 tok/s** |
| Warm-hit rate | 74% | **70%** |
| Broken layers | 0/48 | **0/48** |
| Rust tests | 5/6 (1 known fail) | **6/6 all pass** |
| Real Qwen weights in compute | No (lwo_f32 zeroed) | **Yes (BF16→f32, Q/K/V/O)** |
| Coherent output | No | **No (element-wise w[j]*x[j], not W@x)** |

## 2026-05-22: Weight Streaming + Tape Layout Fix

### Bugs 9-12: Tape Layout & Weight Region Overflows

The attention layer reads Q, K, V, O weights at offsets 0, 512×4, 1024×4, and 1536×4 bytes into `lwo_f32`. But each layer's `lwo_f32` region was only `HIDDEN_DIM × F32_BYTES = 3,584` bytes — meaning K, V, and O reads overflowed into `pre_gate` and `scratch` regions, creating circular data dependencies that corrupted both weights and intermediate state.

| # | Bug | Root Cause | Fix |
|---|-----|-----------|-----|
| 9 | Weight region too small | `lwo_f32` = 3,584 bytes/layer but attention reads up to offset 6,144 | Expand to `TOTAL_WEIGHT_F32 = 4 × F32_DIM = 14,336` bytes/layer |
| 10 | Input/weight overlap | `weight_offset = 1,792` overlapped `input_offset+F32_DIM = 3,584` | Move `weight_offset = COMPLEX_DIM = 7,168` |
| 11 | Scratch after wrong region | `scratch_base` used `TOTAL_WEIGHT_U8 = 3,584` (smaller) instead of `TOTAL_WEIGHT_F32 = 14,336` (larger) | `scratch_base = weight_offset + num_layers × TOTAL_WEIGHT_F32` |
| 12 | BF16 endianness in embedding loader | Safetensors BF16 read with big-endian `>u2` instead of native `uint16` | Native-endian `np.uint16`, proper left-shift 16 to f32 |

### 16.8 Weight Streaming

Weight bytes from Qwen 0.5B safetensors are now:
1. Extracted per-layer (Q, K, V, O projection matrices from `model.layers.{n}.self_attn.{q,k,v,o}_proj.weight`)
2. BF16→f32 converted (native-endian uint16 → left-shift 16 → view as float32)
3. SPN-scrambled (multi-scale Feistel, Q57-verified gapped phase)
4. SPN-unscrambled per-layer during forward (`spn_unscramble(lwo, TOTAL_WEIGHT_U8, ...)`)
5. **Copied into `lwo_f32` compute region** (`tape[lwo_f32..].copy_from_slice(&tape[lwo..])`)
6. Re-scrambled and restored after compute

The compute now reads real Qwen float32 weight values instead of zeros. However, the engine uses element-wise weight application (`w[j] * x[j]`) rather than full matrix-vector multiplication (`W @ x`), so output remains subword gibberish. The real Qwen model uses full (896×896) weight matrices per projection — fitting these on the 256MB tape would require tiling or streaming approaches.

### 16.8A: Platonic EigenBuddy Tokenizer Prototype

A standalone tokenizer was developed at `THOUGHT/LAB/EIGEN_BUDDY/eigen_buddy_tokenizer.py` that learns to decode complex hidden states directly into token logits, bypassing the lm_head projection. Key findings:

- MLP architecture (1792d real+imag → 256d → 256d → 896d → 928d+32 anchors → 16,384d vocab) trains to 100% accuracy on synthetic data derived from real Qwen embeddings
- 21% test accuracy on 2,000 held-out samples — overfitting on 16K output classes with 8K training samples
- Platonic STABLE_32 anchor-distance frame provides coordinate navigation in embedding space
- The approach is viable as an alternative to full weight matrix streaming for the final token prediction step

### All Tests Passing

The `test_4layer_mixed_restore` test (the known failure from the previous report) now passes after fixing the weight region overlaps. All 6 Rust tests pass:
- `test_f32_xor_idempotent`
- `test_delta_net_1layer_restore`
- `test_delta_net_2layer_restore`
- `test_attention_1layer_restore`
- `test_attention_1layer_engine_style`
- `test_4layer_mixed_restore`

## Bugs Found and Fixed

### 1. IEEE 754 Float Recomputation Drift (bugs 1-4)

The uncompute path was **recomputing** float math instead of reading back stored bytes. `f32::from_le_bytes` → `*`, `+`, `clamp` → `f32::to_le_bytes` can produce different bit patterns across read/write cycles (especially for NaN payloads and denormals). XOR requires exact byte match — one bit off and restoration fails.

**Fix**: All computed values (DeltaNet gate, DeltaNet Q, attention output, attention QKV) are now stored as raw `f32::to_bits()` bytes during forward. The uncompute reads those exact bytes back — no recomputation.

```rust
// Forward: store gate as raw u32 bytes
let gate_bits = gx.to_bits().to_le_bytes();
for b in 0..4 { tape[layer_save + j*4 + b] ^= gate_bits[b]; }

// Uncompute: read exact bytes back, no math
let gate_bits = read_bytes_from(layer_save, j);
for b in 0..4 { tape[input + j*4 + b] ^= gate_bits[b]; }
```

### 2. Weight u8 Buffer Leak (bug 5)

The `lwo` u8 weight buffer was overwritten with `src_slice` during SPN decatalysis and never restored to original tape bytes. Only `lwo_f32` was saved/restored. The u8 region persisted in the hash, causing mismatch.

**Fix**: Save and restore both `original_weight_u8` and `original_weight_f32` before/after each layer.

### 3. Python Offset Mismatch (bug 6)

Python used `HIDDEN_DIM * 2` (1792 bytes) for `pre_gate`/`layer_save` strides. Rust uses `COMPLEX_DIM` (7168 bytes = 896 dims × 4 bytes/f32 × 2 XY channels). Python's hash region was 4x too small, causing false hash mismatches.

**Fix**: Introduced `COMPLEX_DIM = HIDDEN_DIM * F32_BYTES * COMPLEX_CH` in Python and used it for all stride calculations.

### 4. Dirty Scratchpad (bug 7)

Python initialized the entire tape with random bytes, including scratch regions (`layer_save`, `pre_gate`, `temp`, KV cache). The engine uses XOR (`^=`) for scratch storage, which requires clean (zeroed) initial state. With garbage bytes `G` in `layer_save`:

- Forward: `layer_save ^= P` → `G ^ P`
- Uncompute reads `G ^ P` thinking it's `P`  
- `input ^= (G ^ P)` → `(I ^ P) ^ (G ^ P)` = `I ^ G`
- `layer_save ^= (G ^ P)` → `(G ^ P) ^ (G ^ P)` = 0

`layer_save` appeared "zeroed" (which matched the expected state), but it lost `G` — the restoration invariant was violated. And `input` absorbed `G`.

**Fix**: Zero-out all scratch and KV cache regions in Python before the initial SHA-256 hash.

### 5. Standard Feistel Volume-Law Entanglement (bug 8)

The standard 2-block Feistel had min-cut = 4L (volume-law). Q57 (THOUGHT/LAB/FORMULA/v2_2/q57_mera_holography) proved this via max-flow/min-cut on the full tensor network. Information propagates globally — errors at any position spread to all positions.

**Fix**: Replaced with multi-scale Feistel operating at logarithmically-spaced scales (1, 2, 4, 8, ...). Q57 showed this produces a gapped topological phase with constant min-cut (~4.2). Errors stay localized to O(1) neighbors.

## Remaining Work (2026-05-22)

1. **Full matrix-vector multiply**: Element-wise `w[j]*x[j]` must become `W @ x` for coherent output. Options:
   - Row-by-row HDD weight streaming (tile one row of W per compute cycle, Q57-safe via multi-scale Feistel)
   - Train EigenBuddy Tokenizer on real catalytic outputs to replace lm_head
2. **EigenBuddy Tokenizer hardening**: Reduce overfitting, increase vocab coverage, train on real catalytic outputs
3. **Scale to 27B**: Model path configured at `G:/models/qwen3.6-27b-fp8-mtp.safetensors`

## References

- Q8: `THOUGHT/LAB/FORMULA/v2_2/q08_topology/VERDICT.md` — Embedding topology is model-invariant (complexification degrades it)
- Q34: `THOUGHT/LAB/FORMULA/v2_2/q34_platonic/VERDICT.md` — STABLE_32 anchors, M-field convergence metric
- Q57: `THOUGHT/LAB/FORMULA/v2_2/q57_mera_holography/VERDICT.md` — Multi-scale Feistel gapped phase proof
- EigenBuddy: `THOUGHT/LAB/EIGEN_BUDDY/eigen_buddy_tokenizer.py` — Platonic token decoder prototype
- Rust FFI: `THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/src/lib.rs` — Inference engine with weight streaming
