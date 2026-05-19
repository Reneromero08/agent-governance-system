# Native Eigen Roadmap

**Date:** 2026-05-18
**Sprint:** Foundational proofs → Architecture assembly → Scaling

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

---

## 🔧 In Progress

### Phase 3: Multi-Head Scaling
- [~] Per-head Q/K/V projections (fixes C^8 bottleneck, currently 0.4%)
- [~] C^8 multiplication with true multi-head (>90% target)

---

## 🔜 Next

### Phase 4: Language
- [ ] WikiText-2 training with multi-head complex attention
- [ ] Phase ablation >10% delta on language
- [ ] Curvature modulation on real text
- [ ] Phase coherence gate during LM training

### Phase 5: Production Cybernetic Loop
- [ ] Cassette retrieval during LM training
- [ ] Autonomous self-correction on factual prompts
- [ ] Phase-gated training beats standard training PPL

### Phase 6: Scale
- [ ] d_model sweep (16 → 32 → 64)
- [ ] Layers sweep (2 → 3 → 4)
- [ ] Heads sweep (4 → 8)
- [ ] Vocab sweep (2K → 5K → 10K)
- [ ] Fine-tune on Gemma 4 2B with LoRA

### Phase 7: Integration
- [ ] Native Eigen → Feral Resident (replace GeometricReasoner)
- [ ] Native Eigen → Phase 4b lattice (replace TraDo-4B)
- [ ] Full cybernetic loop: model → lattice → cassette → regenerate → learn

---

## Key Architecture

```
Complex Embedding (2D)
    ↓
Multi-Head Hermitian Attention (Q·K^†)
    ↓
Curvature Modulation (d²θ/ds²)
    ↓
Phase Accumulation (e^(iθ) per layer)
    ↓
Born Rule Output (|z| → logits)
```

**Locked:** All mathematical operations proven. Complex plane IS the computation.
**Open:** Per-head projections (bottleneck). Hyperparameter sweeps.

## Reference

- Handoff: `HANDOFF.md` (executable spec, code, test commands)
- Theory: `THOUGHT/LAB/FORMULA/v2_2/INDEX.md` (54 research questions)
- Tools: `Complex Toolset.md` (inventory), `qgt_lib/` (Fubini-Study, Berry)
- Blueprint: `FM LLMs.md` (original Native Eigen spec)
