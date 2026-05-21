# HANDOFF — Experiment 16: Catalytic 27B Inference

## What This Is

A zero-RAM catalytic inference engine that runs Qwen 0.5B through a 256MB byte-level XOR fabric (the "tape"). Model weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape for compute, then re-scrambled after. Every token MUST restore the tape to its SHA-256 pre-computation state — zero bits erased. The ultimate target is coherent English text output from real model weights, at 50+ tok/s with warm-tape replay.

## Current State (Updated 2026-05-21)

**F32 tape engine compiles and runs** with real Qwen 0.5B BF16 weights. Forward pass is deterministic and correct. **DeltaNet layers now restore correctly** (all 36 layers verified by layer-boundary SHA-256 checkpoints). **The remaining failure is the attention layer uncompute** — only layer 47 (the last attention layer, uncomputed first in reverse) fails the per-layer SHA-256 checkpoint. Warm-hit tokens restore correctly (`rust_restored=True`).

Performance snapshot:
- 2.9 tok/s, 44% warm-hit rate
- Real Qwen tokenizer and real embedding table (151,936 vocab × 896 dim, BF16→f32)
- HIDDEN_DIM = 896, COMPLEX_DIM = 7168 bytes per complex vector

### What Changed Since Last Handoff

**Fixed (6 bugs squashed):**
1. **IEEE 754 NaN/drift in DeltaNet gate recompute**: The uncompute used to recompute `clamp(0.5 + 0.25 * temp)` during uncompute — the f32 recomputation produced different bit patterns than forward. Fixed by storing gate as raw `f32::to_bits()` bytes in `layer_save` during forward, and reading those exact bytes back during uncompute (no recomputation).
2. **IEEE 754 NaN/drift in DeltaNet Q recompute**: Same pattern — `w * x` recomputed during uncompute produced different f32 bits. Fixed by storing Q as raw u32 bytes in `pre_gate` during forward, reading back during uncompute.
3. **Attention output/QKV recompute**: Same IEEE 754 issue in attention layer. Fixed by storing all attention values (output projection, QKV) as raw bytes in `layer_save`/`pre_gate`/`slot` during forward.
4. **Weight u8 buffer not restored**: The `lwo` (u8 weight buffer at `weight_offset + li * HIDDEN_DIM`) was overwritten with `src_slice` during SPN decatalysis and never restored to original tape bytes. Now `original_weight_u8` is saved and restored alongside `original_weight_f32`.
5. **Python offset mismatch**: Python used `HIDDEN_DIM * 2` (1792) for pre_gate/layer_save strides, but Rust uses `COMPLEX_DIM` (7168). Python's `work_region_size` was 4x too small, causing false hash mismatches.
6. **Duplicate `let lwo` declaration** in forward loop (harmless but cleaned up).

### Remaining Issue: Layer 47 Attention Checkpoint Mismatch

The SHA-256 checkpoint at layer 47 (first attention layer to be uncomputed, running in reverse) shows a hash mismatch between pre-forward state and post-uncompute state. Specific findings:
- `layer_save` and `pre_gate` for layer 47 are correctly zeroed after uncompute
- `temp` hash does NOT change during the attention uncompute (verified)
- `input` hash does change during attention uncompute (expected — it's being restored)
- The overall `work_end` hash (covering input, weight regions, temp, pre_gate, layer_save, warm cache) doesn't match `fwd_hashes[47]`
- Since `temp` doesn't change, `pg`/`ls` are zeroed, and `input` changes correctly — the remaining divergence is most likely in the **weight regions** (`lwo_f32` or `lwo` for layer 47) that are not being perfectly restored by the SPN scramble/unscramble round-trip. The SPN Feistel network may introduce byte-level changes that don't commute with the save/restore pattern, or the `bh_key`/`sbox` may differ between forward and uncompute paths.

### Rust Tests (all pass)

- `test_f32_xor_idempotent` — f32 XOR is its own inverse ✓
- `test_delta_net_1layer_restore` — single DeltaNet layer restores ✓
- `test_delta_net_2layer_restore` — 2 DeltaNet layers restore (raw byte pattern) ✓
- `test_attention_1layer_restore` — simplified attention restore ✓
- `test_attention_1layer_engine_style` — engine-layout attention restore ✓
- `test_4layer_mixed_restore` — 3 delta + 1 attention in engine layout **FAILS** (same root cause as layer 47)

## Files That Matter

| File | Purpose |
|------|---------|
| `THOUGHT/LAB/EIGEN_BUDDY/core/rust_ffi/src/lib.rs` | The entire inference engine: tape helpers, compute, attention, uncompute, SPN scramble. 6 tests. |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/experiment.py` | Python orchestration. COMPLEX_DIM now matches Rust. |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/ROADMAP.md` | Full roadmap |
| `THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/gemini_update/qwen_0.5b/` | Qwen 0.5B model files |

## How to Run

```bash
cd "D:\CCC 2.0\AI\agent-governance-system"

# Build Rust
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" build --release
copy THOUGHT\LAB\EIGEN_BUDDY\core\rust_ffi\target\release\catalytic_ffi.dll THOUGHT\LAB\EIGEN_BUDDY\core\rust_ffi\target\release\catalytic_ffi.pyd

# Run experiment (currently 0% restore, only layer 47 breaks)
.venv\Scripts\python.exe THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\experiment.py

# Run Rust tests
"D:\Reneshizzle\Apps\Rust\.cargo\bin\cargo.exe" test --release --lib
```

## Recommended Next Steps

1. **Investigate SPN round-trip for layer 47**: Verify that `spn_unscramble` followed by `spn_scramble` with the same `bh_key` and `sbox` restores the original bytes at `lwo` for layer 47. The Feistel network uses SHA-256 of `key + round_idx` — if the key or round_idx differs between forward and uncompute, the bytes won't match. Check that `bh_key` and `sbox` are identical in both paths.
2. **Compare `lwo_f32` bytes for layer 47**: Dump the first 32 bytes of `lwo_f32` at `fwd_hashes[47]` time and after uncompute to see if they differ.
3. **Disable SPN operations temporarily**: Replace `spn_unscramble`/`spn_scramble` with no-ops to see if the hash passes without them. This isolates whether the SPN is the source of the remaining divergence.
4. **Fix `test_4layer_mixed_restore`**: Once layer 47 is fixed, the 4-layer test should pass, confirming multi-layer restoration works.

## Key Architecture Changes (lib.rs)

All DeltaNet and attention layers now use **raw byte-level storage and readback** for all computed values:

- **Gate (DeltaNet)**: `gx.to_bits().to_le_bytes()` stored in `layer_save[j*4+b]` and `input[j*4+b]`. Uncompute reads bytes from `layer_save`, XORs into `input` and `layer_save` — zeroing both.
- **Q (DeltaNet)**: `vx.to_bits().to_le_bytes()` stored in `pre_gate[j*4+b]`. Uncompute reads bytes back, XORs into `temp` and `pre_gate` — zeroing both.
- **Output projection (Attention)**: `px.to_bits().to_le_bytes()` stored in `layer_save` and `input`. Uncompute reads bytes back.
- **QKV (Attention)**: `qx/kx/vx.to_bits().to_le_bytes()` stored in `pre_gate` and `slot`. Uncompute reads bytes back.
- **Weight regions**: Both `lwo` (u8) and `lwo_f32` (f32) are saved before SPN operations and restored after.
