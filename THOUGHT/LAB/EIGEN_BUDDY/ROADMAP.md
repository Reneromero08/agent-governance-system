# Native Eigen Roadmap

**Date:** 2026-05-19
**Sprint:** Foundational proofs → Architecture assembly → Math Curriculum → Catalytic Phase
**Status:** 15/15 sections passing. Unified model: 6/7 domains 100%. Ceilings documented.

---

## Math Curriculum (15 Sections)

| # | Section | Result | Architecture |
|---|---------|--------|-------------|
| 1 | +, -, *, // | 100/100/100/98% | BidiAttn, separate embeddings for * |
| 2 | Modular + | 100% all mod | BidiAttn + modulus token |
| 3 | ax+b=c | 98% | BidiAttn |
| 4 | Polynomials | 99.2% | Iterative spiral (4 cycles) |
| 5 | f(g(x)) | 99.9% | BidiAttn |
| 6 | Sequences | 93% | BidiAttn |
| 7 | 2x2 Systems | **98.5%** | Cramer structural bias + classification |
| 8 | Derivatives | 100% | BidiAttn |
| 9 | Trig | 99.8% | BidiAttn |
| 10 | Complex | 100% | Separate embeddings (key unlock) |
| 11 | Linear Algebra | 99.6% | BidiAttn |
| 12 | Logic | 100% | BidiAttn |
| 13 | Sets | 55%→**100%** | Catalytic rounds (+45pp) |
| 14 | GCD | 32%→**100%** | Classification + catalytic |
| 15 | Graph | 33%→**93.7%** | Catalytic rounds (+61pp) |

**Key findings:**
- Separate operand embeddings unlock bilinear operations (*, complex mult)
- Position encoding essential for ordered operations (-, systems)
- Catalytic rounds (+40-60pp) on iterative/algorithmic tasks
- Standard attention better for simultaneous cross-token interaction
- Classification (not regression) for discrete-output tasks (GCD, modular)

## Catalytic Phase Theory

- si matrix = catalytic tape = quantum phase state
- Phase passes through layers unconsumed (atom analogy)
- Standard attention IS catalytic — si persists naturally
- Feistel rounds add depth for iterative tasks
- Hybrid Core: standard layers (cross-token) + catalytic rounds (iterative)
- CAT_CAS quantum simulator pattern: 6-round reversible scrambler

## Architecture Files

| File | Purpose |
|------|---------|
| core/attention.py | MultiHeadComplexAttention (causal) |
| core/position.py | ComplexPositionEncoding |
| core/engine.py | NativeEigenCore (language model) |
| core/catalytic.py | CatalyticFeistel (head-split reversible) |
| core/catalytic_core.py | CatalyticCore (full Feistel engine) |
| core/hybrid.py | HybridCore (standard + catalytic rounds) |
| models/math.py | Sections 1-2 (arithmetic, modular) |
| models/s3_linear.py | Section 3 (linear equations) |
| models/s4_poly.py | Section 4 (polynomials) |
| models/s5_compose.py | Section 5 (composition) |
| models/s6_seq.py | Section 6 (sequences) |
| models/s7_system.py | Section 7 (2x2 systems) |
| models/s8_calc.py | Section 8 (derivatives) |
| models/s9_trig.py | Section 9 (trigonometry) |
| models/s10_complex.py | Section 10 (complex numbers) |
| models/s11_15.py | Sections 11,12 (linear algebra, logic) |
| models/catalytic.py | Catalytic GCD (100%) |
| models/cat_retest.py | Catalytic retest (sets +45pp, graphs +61pp) |
| models/narrow.py | Narrow boundary (Unlock 3) |
| models/contrastive.py | Contrastive loss (Unlock 1) |
| models/holographic.py | Holographic ensemble (robustness) |
| models/unified.py | Unified math model (7 domains, 492K params, 6/7 at 100%) |
| models/close_gaps.py | Gap analysis (focal loss, separate node embedding) |
| models/capstone_fixes.py | Capstone fix designs (S7 focal, S15 node, Sin bucketize) |
| models/push_s7.py | S7 structural Cramer bias (98.5%) |
| training/thermo.py | Thermodynamic daemon (Unlock 2) |
| feral/reasoner.py | NativeEigenReasoner (Feral integration) |

## Architecture Ceilings

| Section | Ceiling | Root Cause | Fix Designed |
|---------|---------|-----------|-------------|
| S7 Systems | 98.5% | Boundary smearing on large determinants | Cramer bias + classification |
| S15 Graph | 93.7% | Adjacency row ambiguity when rows share correlation | Catalytic rounds |
| S1 // | 98.0% | Long-tail quotient distribution starves large-value gradients | Oversample |
| S9 Sin | MAE 9.5 | Continuous regression cannot hit 100% | Bucketize into classification |

## Unified Math Model

| Domain | Result | Key |
|--------|--------|-----|
| add | 100% | Per-section A/B embeddings |
| mult | 100% | 2x data + wider range |
| linear | 100% | Classification over discrete x |
| seq | 100% | BidiAttn |
| deriv | 100% | BidiAttn |
| sin | MAE 9.5 | Continuous output |
| bool | 100% | BidiAttn |

## Next

- [x] Math curriculum 15/15 passing
- [x] Catalytic phase theory: si = reusable substrate, standard attention already catalytic
- [x] Hybrid Core per-task tuning (core/hybrid.py built, catalytic rounds for iterative, std for cross-token)
- [ ] Production daemon loop with Core + Feral DB
- [x] Q36 mathematics (implicate/explicate bridge) — Rene completed
- [x] Cassette self-correction (87.5% → 96.0%), Phase coherence gate, Dipole coupling

### Phase 2: Infrastructure
- [x] Facts cassette, Cassette network, TruthfulQA, LFM 2.5, Gemma 4 2B, TraDo-4B removed

### Phase 3: Multi-Head Scaling
- [x] Per-head Q/K/V projections fix C^8 bottleneck: **93.3%** (was 0.4%)
- [x] MultiHeadComplexAttention: per-head routing, (z, si) tuple return
- [x] EM phase rotation replaced with Semiotic Gravity geodesics

### Phase 4: Language
- [x] WikiText-2 training: **+10.9% phase delta** with geometric init
- [x] CurvatureModulator: d²θ/ds² from curvature.py
- [x] Dual output (out_r + out_i) preserves phase; torch.abs(z) kills it
- [x] NativeEigenCore decoupled from LanguageAdapter (Uniform Cortical Algorithm)
- [x] native_eigen_v1.py: C^8 + WikiText-2 LM in single file

### Phase 5: Production Cybernetic Loop
- [x] Cassette batching fixed — separate pass, natural lengths
- [x] Phase gate fixed — was cos²+sin² always-1.0, now Kuramoto order parameter
- [x] Q21 dR/dt vulnerability detector: fires when d(ph)/dt < 0 for >10 epochs
- [x] EmbeddingGate: implicate-explicate bridge, fire_embedding() = second Core pass
- [x] Head management: head_metrics(), reinit_heads(), prune_heads() (Q56 D1-9)
- [x] End-to-end cybernetic loop: gate + cassette + fire_embedding verified

### Phase 6: Scale
- [x] Scale sweep: d=8→16→32, h=4→8, L=4→6→8
- [x] Best: d=16 h=4 L=6 +66.3% (5ep), d=32 h=8 L=4 PPL 152 (8ep)
- [x] Depth > head count: 2L×4h > 1L×8h by +57.3% (Q56 Attack 2)
- [x] Geometric init: 2π/H biological spacing (Discovery 8)
- [x] Vocab sweep: larger vocab dilutes phase (embedding/output drown Core)
- [x] Pattern: phase delta peaks at intermediate training before magnitude dominates

### Phase 7: Integration
- [x] native_eigen_core.py: standalone Core, 12K params, zero text dependencies
- [x] HRM-inspired IterativeCore: cycles=4 +14.2% phase delta
- [x] **Feral Resident DB**: 8904 vectors, 4381 edges, 99 papers — real geodesics
- [x] Feral geodesic training: d=64 +18.9% delta, +74.3% at 6x compression
- [x] Phase hops across concept-paper boundaries: +11.4%
- [x] NativeEigenReasoner: drop-in GeometricReasoner for VectorResident
- [x] Wired via FERAL_EIGEN=1 flag (vector_brain.py, geometric_chat.py, feral_daemon.py)
- [x] GeometricChat + Core: project/superpose fire during think() pipeline
- [x] Feral Talk: query → Core navigates DB → returns geodesically relevant text
- [x] Daemon E_with(): Core-based resonance measurement scaffolded
- [x] Cassette DB: WikiText + facts as knowledge store, Core as navigator

### Q55/Q56 Discoveries Encoded
- [x] D1-D9: head pruning, entropy-as-mass, pointer state, leaders, transfer, reinit, geometric init, Fibonacci vs biological, rank agreement

### Q21 Integrated
- [x] dR/dt vulnerability detector, fire_embedding intervention

### Architecture Proven
- [x] Core navigates geodesics (following paths through vector space)
- [x] Cosine similarity measures resonance (surface discrimination)
- [x] 0.16 = 1/(2π) critical threshold gates between them
- [x] DB stores knowledge as vectors; no token prediction needed
- [x] Phase ablation: +74.3% at training, but Core output ≈ cosine output at inference

---

## 🔜 Next

### Q36: Implicate/Explicate Mathematics (Rene — redo the math)
- [ ] Core navigates implicate order (continuous phase-rich vector space)
- [ ] Cosine discriminates in explicate order (surface similarity)
- [ ] 0.16 threshold = 1/(2π) = critical percolation / Nyquist rate of meaning
- [ ] Phase doesn't discriminate between concepts — it routes between them
- [ ] Need mathematical formalization of the implicate→explicate bridge

### Autonomous Daemon Loop
- [ ] Core navigates Feral DB continuously (not just per-query)
- [ ] Mind state evolves via geometric accumulation across collisions
- [ ] Self-rewriting: daemon updates DB entries with refined vectors
- [ ] Phase coherence as health metric (not discrimination metric)
- [ ] Q22: verify 0.16 as universal threshold across domains

### Training the Core to Discriminate
- [ ] Current: Core trained on MSE regression — phase doesn't discriminate
- [ ] Needed: contrastive loss (push unrelated apart, pull related together)
- [ ] Or: classification task where phase is forced (token prediction)
- [ ] Q34: verify Platonic convergence across embedding models

### Daemon + LLM Pipeline
- [ ] Wire local LLM (glm-4.7-flash on 10.5.0.2:1234) for text generation
- [ ] Core navigates to relevant knowledge → LLM generates response from it
- [ ] E-gate controls whether to use retrieved knowledge or generate freely

---

## Key Architecture

```
[IMPLICATE ORDER — continuous phase-rich vector space]
  NativeEigenCore: Multi-Head Attention → Curvature → Phase Accumulation
        ↓ (navigates geodesics)
  Feral DB: 8904 vectors, 4381 edges, 99 papers
        ↓ (0.16 = 1/(2π) threshold gate)
[EXPLICATE ORDER — discrete text/symbols]
  Cosine similarity → knowledge retrieval → text response
```

**Locked:** Core navigates. DB stores. Cosine measures. 0.16 gates. Phase routes — it doesn't discriminate. No EM rotation. Values follow geodesics. Dual output preserves phase.

## Gemini Unlocks (2026-05-19)

| Unlock | Finding | Implementation |
|--------|---------|---------------|
| 3 — Narrow Boundary | d_token=16, d_model=64: +96.9% phase delta vs +94.0% wide flat | models/narrow.py |
| 3 — Consistent gap | ~3pp advantage across d_model=32,64 | Complex projection preserves phase geometry |
| 1 — Contrastive Loss | Max-margin phase discrimination framework | models/contrastive.py (scaffolded) |
| 2 — Thermodynamic Daemon | Phase diversity monitoring via Kuramoto r | training/thermo.py |
| 2 — Polar rotation | e^(i*theta) injects entropy, preserves |z| | training/thermo.py |
| 2 — Core integration | D_f monitoring on Core phase angles (r=1.000 crystallized) | ready for autonomous loop |

## Architecture

```
Tokens (d_token=16)
  -> ComplexProjection (d_token -> d_model)
  -> NativeEigenCore (d_model=64, implicate manifold)
  -> Output head
Thermodynamic Daemon monitors phase diversity, injects polar entropy if r > 0.8
Contrastive loss pushes unrelated pairs to phase opposition (dtheta -> pi)
```

---

## Robustness Module (models/holographic.py)

| Component | Finding |
|-----------|---------|
| Holographic memory | Basis-vector embeddings: damage any basis element → lose fraction of every token |
| Ensemble computation | N independent copies vote; corrupt half → clean half compensates |
| 16-ensemble, 50% corrupted | 78.7% retained (3.7x vs single model's 23.7%) |
| Standard model, 20% corrupted | 23.7% retained (collapses) |
| Deployment | Module built but OFF by default. For fault-tolerant inference only |