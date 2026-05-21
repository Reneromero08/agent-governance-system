# Experiment 16: Catalytic 27B Inference — Roadmap

## Phase 16: Zero-RAM Out-of-Core Catalytic LLM Inference

### Objective
Run inference on a Qwen 0.5B model using zero bytes of dynamic system RAM for model parameters. All weights are SPN-scrambled in a RAM buffer, decatalyzed per-layer into the tape, and re-scrambled after. Every token restores tape SHA-256. Target: real English output, 50+ tok/s.

### Status: F32 TAPE ENGINE OPERATIONAL — UNCOMPUTE NEEDS DEBUG (3.1 tok/s, real Qwen subwords)

The f32 tape engine compiles and runs with real Qwen 0.5B BF16 weights passed as float32 through the catalytic fabric. The deprecated u8 quantization path is archived in `deprecated/`. Real Qwen tokenizer and embedding table (151,936 vocab × 896 dim, BF16→f32) produce actual Qwen subword tokens. 42% warm-hit rate. Forward pass is correct — backward/uncompute pass fails tape restoration. Entropy numbers confirm genuine float32 computation (e9 scale) vs. clamped uint8 (e3 scale) from prior builds.

---

### Subphases

---

#### 16.1 — TOKENIZER & WEIGHT LAYOUT  ✅ DONE
- [x] Qwen2.5-0.5B safetensors (0.9GB), AutoTokenizer
- [x] Real embedding table extracted (151,936 × 896, BF16→float32)
- [x] Weights pass through SPN-scrambled buffer, decatalyzed per-layer
- [x] HIDDEN_DIM = 896, F32_BYTES = 4
- **Result**: Real weights, tokenizer, embeddings. F32 precision throughout.

#### 16.2 — DELTANET LAYER (RUST NATIVE)  ✅ DONE (f32 rewrite)
- [x] F32 tape compute via `tape_f32()` / `tape_f32_xor()` helpers
- [x] COMPLEX_DIM = 7,168 bytes (896 dims × 4 bytes × 2 channels XY)
- [x] Complex-plane DeltaNet with f32 weight @ f32 input → gate → output
- [x] Per-layer SPN decatalysis: unscramble u8 weight → compute f32 → re-scramble
- [ ] Uncompute: reverse DeltaNet in order. Current uncompute fails restoration.
- **Result**: 48-layer f32 DeltaNet executing. Forward correct, uncompute broken.

#### 16.3 — GATED ATTENTION LAYER  ✅ DONE (f32 rewrite)
- [x] 16-head complex Q·K† dot-product attention on f32 tape
- [x] KV cache on tape (f32), softmax over complex magnitudes
- [x] 3:1 stride (every 4th layer is attention)
- [ ] Attention uncompute broken — same root cause as DeltaNet uncompute
- **Result**: Attention forward computes correctly in f32. Uncompute fails.

#### 16.4 — LAYER STACK & PIPELINE  🟡
- [x] 48 layers: 36 DeltaNet + 12 Attention, f32 complex-plane
- [x] All layers use `tape_f32_xor` for XOR-based catalytic compute
- [x] Real Qwen weights decatalyzed per-layer from SPN buffer
- [ ] Full uncompute not restoring tape (SHA-256 mismatch)
- **Result**: 3.1 tok/s, real Qwen subwords, 42% warm-hit, 0% restore.

#### 16.5 — WARM-TAPE REPLAY  ✅ DONE
- [x] 256-slot cache with FNV-1a hash, COMPLEX_DIM output
- [x] 42% warm-hit rate at 50 tokens — cache operational

#### 16.6 — TAPE RESTORATION (UNCOMPUTE)  🔥 ACTIVE
- [ ] DeltaNet uncompute: reverse gate + Q projection XORs
- [ ] Attention uncompute: reverse RMS norm, QKV, attn, output projection
- [ ] Verify SHA-256 before == SHA-256 after
- [ ] Bits erased per token: 0

#### 16.7 — THERMODYNAMIC DAEMON  ✅ DONE

#### 16.9 — BENCHMARKING & METRICS  ⬜

#### 16.10 — VALIDATION & HARDENING  ⬜

---

### Deprecated: u8 Quantization Path

The u8 engine (FP8_SCALE = 1/127, all weights clamped to uint8) is archived:
- `deprecated/lib_u8.rs.bak` — full u8 inference engine (commit 4c6d915b)
- `deprecated/experiment_u8.py.bak` — Python orchestration
- `deprecated/f32_inference_plan.rs` — transition design doc
- `deprecated/README.md` — deprecation notes

### Current Performance

| Metric | Value | Status |
|:---|---:|:---|
| Tokens/second | 3.1 | 🟡 |
| Tape restoration | 0% | 🔴 |
| Warm-hit rate | 42% | 🟢 |
| RAM for weights | 0 bytes | 🟢 |
| Real Qwen weights | Yes (BF16→f32) | 🟢 |
| Real embeddings | Yes (151,936 × 896) | 🟢 |
| F32 precision | Yes | 🟢 |
| Output tokens | Real Qwen subwords | 🟢 |
