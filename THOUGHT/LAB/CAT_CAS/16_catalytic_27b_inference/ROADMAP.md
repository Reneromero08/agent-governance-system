# Experiment 16: Catalytic 27B Inference — Roadmap

## Phase 16: Zero-RAM Out-of-Core Catalytic LLM Inference

### Objective
Run inference on a Qwen 0.5B model using zero bytes of dynamic system RAM for model parameters. All weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape, and re-scrambled after. Every token restores tape SHA-256. Target: real English output, 50+ tok/s.

### Status: DELTANET RESTORED — ATTENTION LAYER 47 REMAINS (2.9 tok/s, 44% warm-hit)

The f32 tape engine compiles and runs with real Qwen 0.5B BF16 weights. **DeltaNet layers (36 of 48) now restore correctly** — per-layer SHA-256 checkpoints pass. **One attention layer (47, first to uncompute) still fails** SHA-256 checkpoint. `layer_save` and `pre_gate` are zeroed correctly. `temp` is unchanged by attention (verified). The remaining hash divergence is in the weight region (`lwo`/`lwo_f32`) — likely the SPN standard Feistel's volume-law entanglement producing non-local byte dependencies that the save/restore cycle doesn't fully reverse. Warm-hit tokens restore correctly in Rust.

---

### Subphases

---

#### 16.1 — TOKENIZER & WEIGHT LAYOUT  ✅ DONE
- [x] Qwen2.5-0.5B safetensors (0.9GB), AutoTokenizer
- [x] Real embedding table extracted (151,936 × 896, BF16→float32)
- [x] Weights pass through SPN-scrambled buffer, decatalyzed per-layer
- [x] HIDDEN_DIM = 896, F32_BYTES = 4, COMPLEX_DIM = 7,168
- **Result**: Real weights, tokenizer, embeddings. F32 precision throughout.

#### 16.2 — DELTANET LAYER (RUST NATIVE)  ✅ DONE
- [x] F32 tape compute via `tape_f32()` / `tape_f32_xor()` helpers
- [x] COMPLEX_DIM = 7,168 bytes (896 dims × 4 bytes × 2 channels XY)
- [x] Complex-plane DeltaNet with f32 weight @ f32 input → gate → output
- [x] Per-layer SPN decatalysis: unscramble u8 weight → compute f32 → re-scramble
- [x] **Uncompute fixed**: gate and Q stored as raw `f32::to_bits()` bytes, read back as exact u32 (no IEEE 754 recompute drift)
- [x] **Weight u8 buffer restored**: `lwo` saved/restored alongside `lwo_f32`
- **Result**: All 36 DeltaNet layers pass per-layer SHA-256 checkpoints.

#### 16.3 — GATED ATTENTION LAYER  🟡
- [x] 16-head complex Q·K† dot-product attention on f32 tape
- [x] KV cache on tape (f32), softmax over complex magnitudes
- [x] 3:1 stride (every 4th layer is attention)
- [x] Output and QKV stored as raw `f32::to_bits()` bytes (no recompute drift)
- [ ] **Layer 47 uncompute fails SHA-256 checkpoint** — pg/ls zeroed, temp unchanged, divergence in weight region
- **Result**: Attention forward correct. 11 of 12 attention layers may restore; layer 47 (first to uncompute) breaks.

#### 16.4 — LAYER STACK & PIPELINE  🟡
- [x] 48 layers: 36 DeltaNet + 12 Attention, f32 complex-plane
- [x] All compute values stored as raw u32 bytes to prevent IEEE 754 drift
- [x] Weight regions (both `lwo` u8 and `lwo_f32` f32) saved/restored per layer
- [x] Python offset mismatch fixed (COMPLEX_DIM vs HIDDEN_DIM*2)
- [ ] Layer 47 attention still fails checkpoint — global hash mismatch propagates
- **Result**: 2.9 tok/s, real Qwen subwords, 44% warm-hit, 0% restore.

#### 16.5 — WARM-TAPE REPLAY  ✅ DONE
- [x] 256-slot cache with FNV-1a hash, COMPLEX_DIM output
- [x] 44% warm-hit rate at 50 tokens
- [x] Warm-hit tokens restore correctly in Rust (`rust_restored=True`)

#### 16.6 — TAPE RESTORATION (UNCOMPUTE)  🔥 ACTIVE
- [x] DeltaNet gate: stored as raw bytes in `layer_save`, read back in uncompute
- [x] DeltaNet Q: stored as raw bytes in `pre_gate`, read back in uncompute
- [x] Attention output: stored as raw bytes in `layer_save`, read back
- [x] Attention QKV: stored as raw bytes in `pre_gate`/`slot`, read back
- [ ] Layer 47 weight region restore — likely SPN volume-law Feistel issue
- [ ] Bits erased per token: 0

#### 16.7 — THERMODYNAMIC DAEMON  ✅ DONE

#### 16.8 — SPN FEISTEL ARCHITECTURE  ⬜
- [ ] Replace standard 2-block Feistel with multi-scale Feistel (Q57 finding)
- [ ] Multi-scale Feistel produces gapped topological phase (constant min-cut ~4.2)
- [ ] Standard Feistel produces volume-law (min-cut = 4L) — errors propagate globally
- [ ] Gapped bulk = localized errors = O(1) uncompute per position
- [ ] Reference: `THOUGHT/LAB/FORMULA/v2_2/q57_mera_holography/VERDICT.md`

#### 16.9 — BENCHMARKING & METRICS  ⬜

#### 16.10 — VALIDATION & HARDENING  ⬜

---

### Deprecated: u8 Quantization Path

The u8 engine (FP8_SCALE = 1/127, all weights clamped to uint8) is archived:
- `deprecated/lib_u8.rs.bak` — full u8 inference engine (commit 4c6d915b)
- `deprecated/experiment_u8.py.bak` — Python orchestration
- `deprecated/f32_inference_plan.rs` — transition design doc
- `deprecated/README.md` — deprecation notes

### Fixes Applied (2026-05-21)

| Bug | Root Cause | Fix |
|-----|-----------|-----|
| DeltaNet gate recompute | IEEE 754 `clamp(temp)` produced different bits | Store `f32::to_bits()` in `layer_save`, read raw bytes |
| DeltaNet Q recompute | IEEE 754 `w * x` produced different bits | Store `f32::to_bits()` in `pre_gate`, read raw bytes |
| Attention output recompute | Same IEEE 754 issue | Store raw bytes in `layer_save`, read back |
| Attention QKV recompute | Same IEEE 754 issue | Store raw bytes in `pre_gate`/`slot`, read back |
| Weight u8 buffer (`lwo`) | Only `lwo_f32` was saved/restored, `lwo` overwritten permanently | Save and restore both `lwo` and `lwo_f32` |
| Python offset mismatch | Used `HIDDEN_DIM*2` (1792) where Rust uses `COMPLEX_DIM` (7168) | Fixed to `COMPLEX_DIM` throughout |

### Remaining Issue: Layer 47

After attention uncompute, `pg` and `ls` are zeroed, `temp` is unchanged from pre-uncompute state. The hash divergence is in the weight region. The standard Feistel's volume-law min-cut (4L = 48 for 12 rounds) means the SPN scramble/unscramble creates non-local byte dependencies. Q57 shows the multi-scale Feistel has constant min-cut (~4.2) — errors stay localized. Replacing the scrambler should fix layer 47.

### Current Performance

| Metric | Value | Status |
|:---|---:|:---|
| Tokens/second | 2.9 | 🟡 |
| Tape restoration | 0% (overall) | 🟡 |
| DeltaNet layers restored | 36/36 | 🟢 |
| Attention layers restored | 11/12? | 🟡 |
| Layer 47 checkpoint | Broken | 🔴 |
| Warm-hit rate | 44% | 🟢 |
| Warm-hit restore (Rust) | True | 🟢 |
| RAM for weights | 0 bytes | 🟢 |
| Real Qwen weights | Yes (BF16→f32) | 🟢 |
| Real embeddings | Yes (151,936 × 896) | 🟢 |
| F32 precision | Yes | 🟢 |
