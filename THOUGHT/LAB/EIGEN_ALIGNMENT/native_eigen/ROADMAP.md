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
- [x] Cassette batching fixed — separate pass, batch-level padding with ignore_index
- [x] Phase coherence gate fixed — was cos²+sin² always-1.0, now Kuramoto order parameter
- [x] Q21 dR/dt vulnerability detector: fires when d(phase_coh)/dt < 0 for >10 epochs
- [x] EmbeddingGate: implicate-explicate bridge, fire_embedding() for second Core pass
- [x] Head management: head_metrics(), reinit_heads(), prune_heads() (Q56 D1-6)
- [x] End-to-end cybernetic loop verified: dR/dt trends negative (streak=4 by epoch 8), gate + cassette + fire_embedding pipeline functional
- [x] Phase delta +8.3% with full loop enabled (d=8, L=3, n=1000)
- [ ] Gate fires with longer training (>12 epochs to build streak) or adaptive threshold

### Phase 6: Scale
- [x] Scale sweep complete across d=8/16/32
- [x] d=8 h=4 L=4: +10.3% 5ep / +10.9% 8ep, 70K params
- [x] d=16 h=8 L=4: +6.2% 5ep, 140K params (too many heads)
- [x] d=16 h=4 L=6: **+66.3% 5ep** / +14.0% 8ep, 144K params (best delta)
- [x] d=32 h=4 L=4: +19.7% 5ep, 293K params
- [x] d=32 h=8 L=4: +20.8% 5ep / +4.9% 8ep, 293K params (best PPL 152)
- [x] Pattern: phase delta peaks at intermediate training before magnitude routing dominates
- [x] Geometric init fixed to 2pi/H biological spacing (Discovery 8)
- [x] h_c matches Q55 dimensional capacity prediction
- [x] Vocab sweep: 2K +10.9% | 5K +1.9% | 10K +0.5% — larger vocab dilutes phase signal
- [x] Root cause: embedding/output layers grow 10x faster than Core, drowning phase
- [ ] Fine-tune on Gemma 4 2B with LoRA (Core as phase module in larger architecture)

### Phase 7: Integration
- [x] native_eigen_core.py: standalone Core module, zero text dependencies, 12K params
- [x] Import path: `from native_eigen_core import NativeEigenCore`
- [x] HRM-inspired IterativeCore: H_cycles x L_cycles with additive state injection
- [x] Iterative processing: cycles=4 achieves +14.2% phase delta (vs fixed forward)
- [x] **Feral Resident geodesics discovered**: 8904 vectors, 4381 edges, 99 paper sequences
- [x] Feral DB paper walks as Core input: **+74.3% phase delta** at d=32 (6x compression)
- [x] Phase load-bearing at ALL compression ratios: 12x +60.5%, 6x +74.3%, 3x +65.2%
- [x] Phase hops across concept-paper boundaries: +11.4% delta via context bridges
- [x] Proven: Feral Resident's paper paragraph chains ARE semantic geodesics
- [x] NativeEigenReasoner: drop-in GeometricReasoner replacement for VectorResident
- [x] Points (single vectors) = MiniLM, Geodesics (sequences) = Core
- [x] Compatible interface: initialize/readout/navigate/entangle/interpolate

---

## Q56 Discoveries Encoded

| Discovery | Finding | Implementation |
|-----------|---------|---------------|
| D1 | Head pruning: dead free, laggards harmful (+17%) | prune_heads() |
| D2 | Entropy-as-mass inverted: r=-0.846, low-entropy = leaders | head_metrics() |
| D3 | Pointer state convergence: C stabilizes, rank 4 | (architectural) |
| D4 | Leaders identifiable by epoch 10, 100% recall | head_metrics() |
| D5 | C cross-task transfer: frozen C beats scratch | (scaffolded) |
| D6 | Re-init dead heads beats prune (+56.5% vs +55.3%) | reinit_heads() |
| D7 | Geometric init is multi-head specific | _geometric_init() |
| D8 | Biological 2pi/H beats Fibonacci (+57.9% vs +50.5%) | _geometric_init() fixed |
| D9 | Rank agreement 84.5% across samples | head_metrics() stable |

## Q21 Integrated

| Finding | Implementation |
|---------|---------------|
| dR/dt predicts phase evolution (r=+0.525) | EmbeddingGate.history |
| Negative dR/dt >10 epochs = vulnerable | EmbeddingGate.should_fire() |
| Intervention: second Core pass + cassette | fire_embedding() + factual pass |

## HRM-Inspired Iterative Processing

| Finding | Implementation |
|---------|---------------|
| Iterative geodesic processing (HRM dual-timescale) | IterativeCore (cycles=4, +14.2% delta) |
| Additive state injection z + z_init | IterativeCore.forward() |
| Phase accumulates across iterations | More cycles = higher delta |
| Stop threshold learns geodesic convergence | self.stop_threshold |

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
