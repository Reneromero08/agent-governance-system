# Native Eigen Roadmap

**Date:** 2026-05-19
**Sprint:** Foundational proofs → Architecture assembly → Scaling → Integration
**Status:** Phase 0-7 proven. Core navigates, DB stores, cosine discriminates, 0.16 gates.

---

## ✅ Done

### Phase 0: Mathematical Proofs
- [x] C^1 phase rotation, C^d Hermitian attention, Schrodinger dispersion
- [x] Geometry classification, Composition, Curvature, Entropy-as-mass

### Phase 1: Cybernetic Loop  
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

**Open:** Q36 mathematics. Contrastive training for discrimination. Daemon autonomy. LLM pipeline.