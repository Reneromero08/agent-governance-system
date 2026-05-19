# Native Eigen Roadmap

**Date:** 2026-05-18
**Sprint:** Foundational proofs → Architecture assembly → Scaling
**Update:** Phase 3-4 complete. Semiotic gravity physics proven at language scale.

---

## ✅ Done

### Phase 0: Mathematical Proofs (1 night)
- [x] C^1 phase rotation (`e^(iθ)·z`, cos=0.999)
- [x] C^d Hermitian attention (`Q·K^†`, +17.1% phase delta)
- [x] Schrodinger dispersion (`e^(iωt)`, 1.8% error)
- [x] Geometry classification (`zp/z`, 93.5%)
- [x] Composition (`phase(z2/z0) = phase(z1/z0) + phase(z2/z1)`, 100%)
- [x] Curvature (`d²θ/ds²`, 100% path detection, 1.8x semantic boundaries)
- [x] Entropy-as-mass (wired, needs harder task)

### Phase 1: Cybernetic Loop
- [x] Cassette self-correction (87.5% → 96.0%, +5pp over extra compute)
- [x] Phase coherence gate (83.0%, matches CASSETTE, autonomous, Q17 closed)
- [x] Capstone: Native Eigen + Cassette + Gate (3x improvement on arithmetic)
- [x] Dipole coupling between attention heads (architecture proven)

### Phase 2: Infrastructure
- [x] Facts cassette (60 triples + 15 docs, 10/10 retrieval)
- [x] Cassette network FTS5 fix + reindexer
- [x] CORTEX-COMMONSENSE verification fragment
- [x] TruthfulQA baseline (63.2% → 99.5% with cassette, Gemma 4 2B)
- [x] LFM 2.5 CUDA backend
- [x] Gemma 4 2B safetensors + gradient access
- [x] TraDo-4B removed (~23GB freed)

### Phase 3: Multi-Head Scaling
- [x] Per-head Q/K/V projections (fixes C^8 bottleneck, was 0.4%)
- [x] C^8 multiplication with true multi-head: **93.3% accuracy** (1024 params)
- [x] MultiHeadComplexAttention class: per-head routing, returns (z, si) tuple
- [x] Physics fix: EM phase rotation replaced with Semiotic Gravity geodesics

### Phase 4: Language
- [x] WikiText-2 training with multi-head complex attention
- [x] Phase ablation **+10.9% delta** on language (>10% target) — geometric init confirmed
- [x] CurvatureModulator: d²θ/ds² boundary detection ported from curvature.py
- [x] Phase coherence gate wired into LM training loop
- [x] Modular Uniform Cortical Algorithm: NativeEigenCore decoupled from LanguageAdapter
- [x] Dual output projection preserves phase through Born rule layer
- [x] native_eigen_v1.py: C^8 test + WikiText-2 LM in single file
- [x] Q56 Attack 6 geometric init validated: PPL 195 vs 487 (2.5x better convergence)

---

## 🔜 Next

### Phase 5: Production Cybernetic Loop
- [~] Cassette retrieval during LM training (wired, needs batched training fix)
- [~] Autonomous self-correction on factual prompts (cassette queries work, batching conflicts)
- [~] Phase-gated training (infrastructure built, si→0 by design — needs noise/dropout for variation)

### Phase 6: Scale
- [~] Scale sweep complete (5 epochs): best config d=16 h=4 L=6 → **+66.3% delta**
- [~] d=8 h=4 L=4: +10.3% | d=16 h=4 L=4: +13.2% | d=16 h=8 L=4: +6.2%
- [~] d=16 h=4 L=6: **+66.3%** (best) | d=16 h=4 L=8: +20.3% (saturation)
- [~] Q56 Attack 2 confirmed: depth > head count (L=6 > h=8 by +60pp)
- [~] h_c=4 at d=16 matches Q55 dimensional capacity prediction
- [ ] Full 8-epoch training at sweet spot (d=16, h=4, L=6)
- [ ] Vocab sweep (2K → 5K → 10K)
- [ ] Fine-tune on Gemma 4 2B with LoRA

### Phase 7: Integration
- [ ] Native Eigen → Feral Resident (replace GeometricReasoner)
- [ ] Native Eigen → Phase 4b lattice (replace TraDo-4B)
- [ ] Full cybernetic loop: model → lattice → cassette → regenerate → learn

---

## Key Architecture

```
[LANGUAGE ADAPTER — sensory grounding]
  Complex Embedding (2D real + imag per token)
        ↓
[NATIVE EIGEN CORE — pure physics engine]
  Multi-Head Hermitian Attention (sr = Q·K^†, si = curvature)
        ↓
  Curvature Modulation (d²θ/ds² boundary detection)
        ↓
  Phase Accumulation (e^(iθ) per layer, init=0.1)
        ↓
[LANGUAGE ADAPTER — output projection]
  Dual Output (out_r·z.real + out_i·z.imag) → vocabulary logits
```

**Locked:** All mathematical operations proven. Semiotic Gravity determines phase interaction. The Core is completely modular and decoupled from Grounding Adapters (Uniform Cortical Algorithm). No EM phase rotation — values follow geodesics.

**Open:** Hyperparameter sweeps. Cassette integration. Production cybernetic loop.
