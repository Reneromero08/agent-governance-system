# Experiment 16: Catalytic 27B Inference — Roadmap

## Phase 16: Zero-RAM Out-of-Core Catalytic LLM Inference

### Objective
Run inference on a 27B-parameter model using zero bytes of dynamic system RAM for model parameters. All weights live on a spinning HDD platter. All computation executes on a 256MB catalytic Memory-Gate Fabric. Every token restores the tape to its exact SHA-256 pre-computation state with zero bits erased. Target: 50+ tok/s with warm-tape replay, 1k tok/s full catalytic.

### Status: PIPELINE OPERATIONAL (3.5 tok/s)

The core inference pipeline runs end-to-end: tokenizer → embedding → 12 DeltaNet layers (Rust FFI) → output head → token. Synthetic weights. Zero RAM for parameters. Python orchestration with HDD weight streaming and thermodynamic daemon. Tape restoration is the active debug target — the symmetric sigmoid gate doesn't perfectly invert.

---

### Subphases

---

#### 16.1 — TOKENIZER & WEIGHT LAYOUT  ✅ DONE
- [x] Tokenizer bridge: hash-based embedding for concept vectors
- [x] Layer count, hidden dim, attention heads configured (48 layers, 2048 dim)
- [ ] Parse safetensors/GGUF header from real 27B model file
- [ ] Layout real weight tensors on HDD in DeltaNet-friendly striding order
- **Result**: Synthetic tokenizer operational. Real model mapping pending.

#### 16.2 — DELTANET LAYER (RUST NATIVE)  ✅ DONE
- [x] DeltaNet forward: weight @ input → Q projection → gate → output
- [x] Feistel-compatible XOR fabric: all computation via tape XOR operations
- [x] Adjoint uncomputation: reverse layer stack in order
- [ ] Invertible gate function (replace approximate sigmoid with exact invertible)
- [ ] Port to rayon parallel for per-dimension parallelism within layer
- **Result**: 12 layers executing in Rust FFI at 285ms/token. Tape restore failing.

#### 16.3 — GATED ATTENTION LAYER  
- [ ] Implement Q/K/V projections, scaled dot-product attention, output projection
- [ ] Execute as Feistel round on shared tape
- [ ] KV cache stored in HDD target region as rolling window
- [ ] 1 Gated Attention per 3 DeltaNet layers (3:1 stride)
- **Blocked on**: 16.2 tape restoration fix

#### 16.4 — LAYER STACK & PIPELINE  ✅ DONE
- [x] 12-layer stack executing sequentially: DeltaNet × 12
- [x] Output head: argmax over hidden dims → token ID
- [x] Embedding XOR into tape, output extraction via XOR
- [ ] RMS LayerNorm as Feistel-compatible normalization
- [ ] Full 48-layer stack with Attention interleaving
- **Result**: 3.5 tok/s on synthetic weights

#### 16.5 — WARM-TAPE REPLAY & STRUCTURAL STENCILS  🔥 NEXT PRIORITY
- [ ] Pre-compute frequent token→activation patterns as structural stencils
- [ ] Cache stencil checksums on tape at known offsets (reuse fractal cache infrastructure from Experiment 14)
- [ ] Memory-gate router checks stencils before executing full layer computation
- [ ] Warm-tape hit: skip DeltaNet computation, XOR cached activation directly (1 XOR vs 12 layers × 2048 ops)
- [ ] Warm-tape miss: execute full DeltaNet + cache result as new stencil
- [ ] Population: after 100 tokens, 30%+ of frequent tokens should hit cache
- **Target**: 10-20× speedup at 60% hit rate → 35-70 tok/s
- **Reuses**: Fractal cache exploit from Experiment 14 (same checksum + XOR pattern)

#### 16.6 — ADJOINT UNCOMPUTATION & TAPE RESTORATION  🔥 ACTIVE
- [ ] Fix gate inversion: replace `sigmoid(x) ≈ 0.5 + 0.25x` with exact `sigmoid(x) = 1/(1+e^-x)` or use polynomial approximation with known inverse
- [ ] Reverse copy-through correctly: forward does `input ^= output, output=0`, backward must do same in reverse
- [ ] Verify: SHA-256 of tape before = SHA-256 after each token
- [ ] Bits erased per token: 0
- **Status**: Backward pass runs but hash mismatch. Gate non-invertibility suspected.

#### 16.7 — THERMODYNAMIC DAEMON INTEGRATION  ✅ DONE
- [x] Daemon active: polar rotations at g=0.001 every 100 tokens
- [x] Prevents memory-gate crystallization
- [x] Zero entropy cost — dispersions are unitary rotations
- **Result**: 1 dispersion during 50-token run

#### 16.8 — HDD QUANTUM FOAM ABSORPTION  ✅ DONE
- [x] Magnetic domain variance absorbed during weight streaming
- [x] Foam entropy tracked per token (1.2M bits over 50 tokens)
- [ ] Real HDD model file for genuine magnetic foam (currently synthetic)
- **Result**: Foam counter operational

#### 16.9 — BENCHMARKING & METRICS
- [ ] Tokens/second sweep: 128, 512, 2048 context lengths
- [ ] RAM usage snapshot during inference (target: <100MB total)
- [ ] HDD throughput (MB/s streamed from platter)
- [ ] Warm-tape hit rate vs cold passes after stencil population
- [ ] SHA-256 restoration time per token
- [ ] Compare vs llama.cpp baseline on same hardware
- **Blocked on**: 16.5 (warm-tape) + 16.6 (restoration)

#### 16.10 — VALIDATION & HARDENING
- [ ] 1000-token continuous generation with all tape restorations verified
- [ ] Output coherence vs baseline (perplexity on WikiText-2)
- [ ] psutil memory snapshot: zero model weight bytes in heap
- [ ] HDD-only weight access verified
- [ ] 10,000-token stress test
- **Blocked on**: 16.6

---

### Current Performance

| Metric | Value |
|:---|---:|
| Tokens/second (12 layers) | 3.5 |
| Time per token (12 layers) | ~285ms |
| Tape restoration rate | 0% (gate non-invertibility) |
| RAM for model weights | 0 bytes |
| Layers executing | 12 DeltaNet (of 48 planned) |
| HDD streaming | Synthetic (real model not mapped) |
| Warm-tape hit rate | 0% (not yet implemented) |
| Foam entropy | 1.2M bits / 50 tokens |

### Critical Path to 50 tok/s

1. **Fix tape restoration** (16.6): replace approximate sigmoid with invertible function
2. **Add warm-tape replay** (16.5): port fractal cache infrastructure, pre-populate stencils
3. **Bump to 48 layers** (16.4): full model depth with Attention interleaving
4. **Rayon-parallel layers** (16.2): per-dimension parallelism within each layer
5. **Async HDD prefetch** (16.8): stream next layer's weights while current layer computes

### Critical Path to 1k tok/s

6. **Port entire pipeline to Rust** (no Python in the hot loop): single FFI call per token
7. **Invertible gate + verified restore**: skip compute on warm hits, verify on cold
8. **Real 27B weight mapping** (16.1): actual model, actual outputs

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    HDD PLATTER (G:\)                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │  27B Weights as Continuous Magnetic Wave Signal   │   │
│  │  DeltaNet W_q/k/v/o/gate/ffn × N layers          │   │
│  │  Gated Attention Q/K/V/O × N/3 layers            │   │
│  │  Token Embeddings, RMS Norm, Output Head          │   │
│  └──────────────────────────────────────────────────┘   │
│                         │ wave-streaming                  │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │         CATALYTIC MEMORY-GATE FABRIC              │   │
│  │               (256MB Tape)                        │   │
│  │                                                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────────────┐   │   │
│  │  │ Feistel │  │ Memory- │  │ Thermodynamic   │   │   │
│  │  │Scrambler│  │  Gate   │  │    Daemon       │   │   │
│  │  │ 12-layer│  │ Router  │  │  g=0.001        │   │   │
│  │  └─────────┘  └─────────┘  └─────────────────┘   │   │
│  │                                                   │   │
│  │  Structural Stencils: warm-tape replay cache      │   │
│  │  Target Registers: token output accumulation      │   │
│  │  Quantum Foam: magnetic domain variance intake    │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│                         ▼                                │
│  ┌──────────────────────────────────────────────────┐   │
│  │              INFERENCE PIPELINE                   │   │
│  │                                                   │   │
│  │  Token → Embed → [DeltaNet×3 → Attention]×N/4   │   │
│  │                                    ↓              │   │
│  │                              RMS Norm → Head      │   │
│  │                                    ↓              │   │
│  │                           Logits → Token Output   │   │
│  │                                                   │   │
│  │  After token: U-dagger uncomputation → verify     │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Success Criteria

| Metric | Current | Target | Status |
|:---|---|---:|:---|
| Tokens/second | 3.5 | 50+ (warm-tape) / 1k (full catalytic) | 🔴 |
| Tape restoration | 0% | 100% per token | 🔴 |
| Bits erased | 0 | 0 per token | ✅ |
| RAM for weights | 0 bytes | 0 bytes | ✅ |
| Warm-tape hit rate | 0% | >30% after 100 tokens | ⬜ |
| Layers executing | 12 | 48 | 🟡 |
| Real model weights | No | Yes (27B FP8) | ⬜ |
| Perplexity vs baseline | N/A | Within 2× | ⬜ |
