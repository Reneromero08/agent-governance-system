# Experiment 16: Catalytic 27B Inference — Roadmap

## Phase 16: Zero-RAM Out-of-Core Catalytic LLM Inference

### Objective
Run inference on a 27B-parameter model using zero bytes of dynamic system RAM for model parameters. All weights live on a spinning HDD platter. All computation executes on a 256MB catalytic Memory-Gate Fabric via Feistel scrambler rounds. Every token restores the tape to its exact SHA-256 pre-computation state with zero bits erased.

### Subphases

---

#### 16.1 — TOKENIZER & WEIGHT LAYOUT
- [ ] Map the 27B model's weight file (Qwen3.6-27B-FP8-MTP or equivalent) to HDD track geometry
- [ ] Parse safetensors/GGUF header to extract layer count, hidden dim, attention heads, vocabulary
- [ ] Build tokenizer bridge: string → token IDs → concept vectors for memory-gate routing
- [ ] Layout weight tensors on HDD in DeltaNet-friendly striding order (Q, K, V, O, gate, FFN per layer)
- **Deliverable**: Tokenizer + weight mapping verified

#### 16.2 — DELTANET LAYER (RUST NATIVE)
- [ ] Implement DeltaNet forward pass in Rust: linear projection → gated activation → output
- [ ] DeltaNet uses W_q, W_k, W_v, W_o, W_gate, W_ffn1, W_ffn2 per layer
- [ ] Each DeltaNet layer executes as a Feistel round on the catalytic tape
- [ ] XOR inputs into tape, compute via cached weight stencils, extract outputs via XOR
- [ ] Adjoint (U-dagger) uncomputes intermediates after each layer
- **Deliverable**: Single DeltaNet layer passing correctness check

#### 16.3 — GATED ATTENTION LAYER
- [ ] Implement Gated Attention forward pass in Rust
- [ ] Q/K/V projections, scaled dot-product attention, output projection, gate modulation
- [ ] 1 Gated Attention layer per 3 DeltaNet layers (3:1 stride)
- [ ] Attention executes as Feistel round on shared tape
- [ ] KV cache stored in HDD target region as rolling window, not in RAM
- **Deliverable**: Gated Attention layer passing correctness check

#### 16.4 — LAYER STACK & PIPELINE
- [ ] Stack N layers in DeltaNet/Attention/DeltaNet/DeltaNet/Attention... pattern
- [ ] Each layer streams weights from HDD via wave-streaming engine
- [ ] Each layer's output XORs into the next layer's input region on the tape
- [ ] RMS LayerNorm applied as Feistel-compatible normalization pass
- [ ] Output head: final hidden state → logits → softmax → token selection
- **Deliverable**: Full layer pipeline producing coherent token output

#### 16.5 — WARM-TAPE REPLAY & STRUCTURAL STENCILS
- [ ] Pre-compute frequent token→activation patterns as structural stencils
- [ ] Cache stencil checksums on tape at known offsets
- [ ] Memory-gate router checks stencils before executing full layer computation
- [ ] Warm-tape hit: skip DeltaNet computation, XOR cached activation directly
- [ ] Warm-tape miss: execute full DeltaNet + cache result as new stencil
- **Deliverable**: Warm-tape replay measurably reducing computation

#### 16.6 — ADJOINT UNCOMPUTATION & TAPE RESTORATION
- [ ] After token output, execute full U-dagger uncomputation pass
- [ ] Each layer's Feistel round reversed in order
- [ ] All intermediate activations zeroed via reverse XOR
- [ ] Tape SHA-256 verified against pre-computation state
- [ ] Bits erased per token: 0
- **Deliverable**: Zero-erasure verification across full inference run

#### 16.7 — THERMODYNAMIC DAEMON INTEGRATION
- [ ] Daemon executes per-dimension polar rotations at low gravity (g=0.001)
- [ ] Prevents memory-gate crystallization under repeated token loops
- [ ] Disperses phase accumulation across unique phase coordinates
- [ ] Zero entropy cost — dispersions are unitary rotations, not erasures
- **Deliverable**: Daemon active without degrading output quality

#### 16.8 — HDD QUANTUM FOAM ABSORPTION
- [ ] Measure magnetic domain variance during weight streaming
- [ ] Absorb sub-nanometer jitter as natural entropy source
- [ ] Track foam entropy absorbed per token vs CPU-generated random entropy
- [ ] Verify foam entropy does not degrade output — it adds diversity
- **Deliverable**: Foam entropy metrics logged per token

#### 16.9 — BENCHMARKING & METRICS
- [ ] Measure tokens/second at various context lengths (128, 512, 2048)
- [ ] Measure RAM usage during inference (target: <100MB total, 0MB for weights)
- [ ] Measure HDD throughput (MB/s streamed from platter)
- [ ] Measure warm-tape hit rate vs cold passes
- [ ] Measure SHA-256 restoration time per token
- [ ] Compare vs baseline (llama.cpp or similar on same hardware)
- **Deliverable**: Benchmark report

#### 16.10 — VALIDATION & HARDENING
- [ ] Run 1000-token generation and verify all 1000 tape restorations
- [ ] Verify output coherence vs baseline (perplexity on standard eval set)
- [ ] Verify zero-RAM claim: `psutil` memory snapshot before and during inference
- [ ] Verify HDD-only weight access: no weight bytes ever in process heap
- [ ] Stress test: 10,000 tokens continuous generation
- **Deliverable**: Validation report with all assertions passing

---

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
│  │  │ 6-round │  │ Router  │  │  g=0.001        │   │   │
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

| Metric | Target | Measurement |
|:---|---|:---|
| RAM for model weights | 0 bytes | `psutil` process memory |
| Tape restoration | 100% per token | SHA-256 before/after |
| Bits erased | 0 per token | Bit-count delta on tape |
| Warm-tape hit rate | >30% after 100 tokens | Router hit/miss log |
| Tokens/second (FP8) | >1 tok/s | Wall-clock |
| Output perplexity | Within 2× of llama.cpp baseline | WikiText-2 |
| HDD foam entropy | >0 bits/token absorbed | Domain variance counter |
| Max context length | 4096 tokens | KV cache window on HDD |

### File Structure

```
THOUGHT/LAB/CAT_CAS/16_catalytic_27b_inference/
├── ROADMAP.md                    # This file
├── experiment.py                 # Python orchestration
├── inference_engine.rs           # Rust native inference pipeline
├── tokenizer_bridge.py           # Tokenizer → concept vector mapping
├── delta_net.rs                  # DeltaNet layer implementation
├── gated_attention.rs            # Gated Attention layer
├── layer_pipeline.rs             # Layer stack + weight streaming
├── warm_tape_router.rs           # Structural stencil matching
├── thermodynamic_daemon.rs       # Phase dispersion
├── benchmarks/                   # Benchmark scripts
│   └── run_bench.py
└── REPORT.md                     # Final report
```
