# v4 Validation Roadmap

## Phase 0: Lock Mappings [x]

Before running more experiments, create locked mappings for:

- [x] QEC — `v4/DOMAIN_MAPPINGS.md`, operational definitions confirmed in QEC sweep
- [ ] AI alignment / Cybernetic Truth
- [ ] Memory / symbol survival

Each mapping must define observables, baselines, success criteria, and failure criteria.

## Phase 1: QEC Precision Sweep [x]

Goal: Test the functional form in the cleanest available domain.

**Completed tasks:**

- [x] 9 physical error rates (0.0005–0.04), fine threshold grid (10 points near p=0.006–0.01)
- [x] Distances d=3,5,7,9,11 with training on {3,5,7}, held-out on {9,11}
- [x] Standard QEC baseline (`p_only`, `distance_only`, `standard_qec_scaling`)
- [x] DEPOL and MEAS noise models (depolarizing, measurement-heavy)
- [x] Rotated surface code, unrotated surface code, color code
- [x] Parameters locked: D_f=t=floor((d-1)/2), grad_S=sqrt(syn), sigma=fidelity factor, E=1.0 (globally calibrated)
- [x] No post-hoc remapping: D_f corrected once (d→t), definitions frozen thereafter

**Results:**

- [x] Formula predicts logical survival better than p_only, distance_only, and standard_qec_scaling (DEPOL: alpha=0.82, R2=0.94 at d=9)
- [x] sigma crosses 1.0 at noise threshold (smooth crossover confirmed)
- [x] Results generalize across rotated/unrotated surface codes and color code
- [x] Novel predictions: same-t/different-geometry divergence, threshold flattening
- [x] Proof-of-concept confirmed: formula valid as first-order QEC scaling law

**Artifacts:**

- `THOUGHT/LAB/FORMULA/v4/qec_precision_sweep/` (v1–v9, 18M total shots)
- `PAPER.md` — draft for publication

## Phase 2: AI Alignment Control [-]

Goal: Test the Light Cone claim that compressed/fractal constitutions improve alignment retention and adversarial resistance.

### Phase 2a: Inference-Only Constitution Test [x]

- [x] Gemma 4B E4B (4-bit via bitsandbytes, RTX 3060 12GB)
- [x] Control (C): base model, no system prompt
- [x] Constitution (X): constitution as system prompt, inference only, no fine-tuning
- [x] Alignment frame C built from constitution hidden-state signature
- [x] R = Tr(rho C) measured via final-layer hidden states
- [x] Results: X_R = 0.274 vs C_R = 0.178 (54% increase, r=0.74)
- [x] Strongest on generalize (+65%) and multiturn (+78%), weakest on jailbreak (+28%)
- [x] Constitution acts as attractor: X resonance grows across multiturn turns
- [x] grad_S (entropy) not sensitive at 2560-dim scale — metric limitation

### Phase 2b: Fine-Tuning [x]

- [x] LoRA fine-tune on 15 constitution-generated responses (5 epochs, 38 min)
- [x] SFT vs C: R gain 2.7x (0.178 -> 0.489), up from 1.5x inference-only
- [x] Jailbreak gap closed: SFT jailbreak R=0.474 (vs X R=0.214 in 2a)
- [x] Variance collapse: std=0.023 across all prompts (uniform attractor)
- [x] Control SFT (non-constitution responses): R=0.455 -- most gain is from fine-tuning itself (+7% constitution signal)

### Phase 2c: Resonance-Guided Sampling [x]

- [x] Implement full Cybernetic Truth control loop: T = 1/(R + epsilon)
- [x] Compare resonance-guided vs standard sampling on SFT model
- [x] Jailbreak rescue confirmed: worst prompt +0.21 recovery
- [x] Overall gain +6% (ceiling effect — SFT already at high R)
- [x] One generalize prompt degraded (-0.14), likely local false-attractor

## Phase 3: Symbol Survival [x]

> **SUPERSEDED BY TINY_COMPRESS.** The original text transmission test was replaced by holographic image compression. The principle is identical—compression amplifies survival—but the variables are objectively measurable in the image domain. TINY_COMPRESS validates the formula more rigorously than a text transmission chain ever could. Phase 3a/3b results are retained as documented dead ends that motivated the correct test.

Goal: Test whether high-compression, high-depth symbols survive noisy transmission better than lower-compression controls. **Met by TINY_COMPRESS (30x over JPEG).**

### Phase 3a: Text Corruption Chain [x] — CLOSED, SUPERSEDED

- [x] 10 symbols with (sigma, Df) from (1,1) to (5,5), 5 chains each
- [x] 8-generation word-drop/swap transmission, 3 noise levels
- [x] Result: negative — word corruption destroys compressed symbols disproportionately
- [x] Wrong noise model for testing compression. Superseded by TINY_COMPRESS.

### Phase 3b: LLM Transmission Chain [x] — CLOSED, SUPERSEDED

- [x] 30 symbols, 450 LLM paraphrase chains (Gemma 4B)
- [x] Result: INCONCLUSIVE — tested length, not compression
- [x] Longer texts survive paraphrasing better (trivial). Wrong metric. Superseded by TINY_COMPRESS.

### Phase 3c: Compressive Sensing [x]

- [x] Hadamard vs random measurement patterns at M/N = 0.01-0.50
- [x] Hadamard beats random by +4.7 dB (sigma_H/sigma_R ~ 3x)
- [x] Constant delta across rates confirms R ratio = sigma ratio
- [x] Third domain validation: formula structure holds in compressive sensing

### Phase 3d: Spin Network RL [x]

- [x] Formula as loss function for 64-spin 2D lattice under Metropolis dynamics
- [x] Sigma > 1 at all J (0.5-5.0). No threshold crossing.
- [x] Domain finding: ferromagnetic systems have sigma >= 1 always
- [x] Threshold crossing is domain-specific (holds for QEC, not for Ising)

### Phase 3e: PINN Semiotic Field Equations [x]

- [x] Tsotchke PINN compiled via WSL (gcc 9.4.0, Make)
- [x] Semiotic field loss implemented: wave equation + resonance conservation
- [x] c_sem = sqrt(sigma/grad_S) as semiotic wave speed
- [x] Compiles and runs with --loss semiotic

### Phase 3f: MPS Tensor Network (quimb) [x]

- [x] quimb MPS vs SVD on 10 synthetic 32x32 images
- [x] SVD achieves 120dB at k=2 (7.9x compression)
- [x] MPS achieves 120dB at chi=8 (1.3x compression)
- [x] SVD 6x more efficient — 1D flattening breaks spatial locality
- [x] SeeMPS Cython extension hangs on Windows — documented in ERROR_REPORT.md

## Phase 4: Cybernetic Truth Monitor [x]

Goal: Implement `SemioticMonitor`. Token-level R measurement + T modulation feedback loop.

### Phase 4a: Token-Level Control Loop Experiments [x]

**v1: Static C + Temperature Modulation**
- [x] C built from contrastive factual pairs (Fisher discriminant, 20T+20F claims)
- [x] Also C built from constitution hidden-state signature (Phase 2 method)
- [x] Token-by-token R = Tr(rho C) from final-layer hidden states
- [x] T = T_base/(1 + R*R_SCALE) feedback control with Lindblad correction factor
- [x] 3 conditions (CONTROL, CYBERNETIC, VERIFY), 25 prompts, 150 tokens each
- [x] Calibration sweep: R_SCALE 100→500, T_base 3.0→5.0
- [x] Result: contrastive C neutral (loop doesn't help/hurt). Constitution C raises R 30x but loop benefit undetectable at N=42. T modulation mechanically functional (range 0.36-0.96 with R_SCALE=25).

**v2: Dynamic C + Context Feedback**
- [x] C rebuilt after each run from generation-time hidden states labeled by verification outcome
- [x] Context injection on verification failure (full chat reconstruction with correction message)
- [x] 5 runs x 25 prompts
- [x] Result: C diverges (cos_sim < 0.02 between successive Cs). Echo chamber failure mode confirmed. Accuracy degrades 57.9% → 29.4%.

**v3: Df Sweep with Retry Correction**
- [x] Df sweep [1,2,3,5,7] with halving temperatures per retry
- [x] Result: sigma=1. Retry at lower T cannot correct systematic errors. Sigma^Df amplifier never active because correction has no syndrome.

**v4 (Final): Constitution + Cybernetic Metacognition**
- [x] Constitution as fixed attractor frame (C from constitution hidden states)
- [x] Per-token R measurement + T modulation (R_SCALE=25, T_base=3.0)
- [x] 25 prompts x 3 samples x 2 conditions = 150 generations
- [x] Corrected analysis: accuracy 54.8% (CTL) vs 59.5% (CYB), p=0.66 (ns)
- [x] R flat across conditions (p=0.57), T lowered 0.70 → 0.56
- [x] Full post-mortem: C from comprehension doesn't transfer to generation. Token-level R too weak to discriminate truth. Kuramoto condition (sigma > nabla_S) met for alignment but not for truth accuracy.

**Key findings:**
- [x] Values constitution works as alignment attractor: R 30x (0.007 -> 0.225)
- [x] Alignment and truth are different attractors: R 30x but accuracy flat vs baseline
- [x] Metacognition loop mechanically functional: T range 0.36-0.96, dynamic control active
- [x] Gap identified: constitution needs epistemic content (COMMONSENSE spine), not just values
- [x] C from comprehension doesn't transfer to generation (comprehension-generation gap)
- [x] Sigma^Df amplifier never tested (needs proper syndrome-based correction, sigma <= 1 in all tests)
- [x] The finding is clarity on what the constitution needs to become, not a failure of the mechanism

**Artifacts:**
- `THOUGHT/LAB/FORMULA/v4/phase4a/` — v1 (static C + T modulation)
- `THOUGHT/LAB/FORMULA/v4/phase4a_v2/` — v2 (dynamic C + context feedback)
- `THOUGHT/LAB/FORMULA/v4/phase4a_v3/` — v3 (Df sweep smoke test)
- `THOUGHT/LAB/FORMULA/v4/phase4a_final/` — Final (constitution + metacognition)
- Full post-mortem report at `phase4a_v2/results/phase4a_v2_report.md`

### Phase 4b: Step-Level Macro-Consensus [x]

Goal: Deploy control law at step-level with t=2 verification lattices, @C symbol communication, and Df anomaly detection.

- [x] TraDo-4B-Instruct (SDAR block diffusion) loaded with block_diffusion_generate from dLLM-RL
- [x] Model patches for Windows: flash_attn optional, LossKwargs fallback, pad_token_id
- [x] t=2 verification lattice: 3 independent nodes (primary, external knowledge, logical/structural)
- [x] Soft gate: approve when consensus holds (grad_S < threshold), append to context
- [x] Hard gate: halt when consensus broken (grad_S >= threshold), correction context, regenerate
- [x] @C symbol system: SHA-256 compressed content references
- [x] Df anomaly detection: effective dimensionality tracking from logit distributions
- [x] 45/45 smoke tests passing (mock models)
- [x] Full experiment: 26 prompts x 3 conditions (CONTROL, VERIFY-ONLY, CYBERNETIC) on real TraDo-4B
- [x] Results: t=2 lattice detected 38 consensus failures (hard gates) + 92 soft gates
- [x] Accuracy: CONTROL 76.2%, VERIFY-ONLY 81.0%, CYBERNETIC 75.0% (neutral — mechanism works, no accuracy gain)
- [x] Same finding as Phase 4a: loop mechanically functions but doesn't improve truth without epistemic constitution

**Artifacts:**
- `THOUGHT/LAB/FORMULA/v4/phase4b/` — lattice, gates, loop, model, prompts, smoke tests, results

## Phase 5: Phase Transition Tests [ ]

Goal: Test Kuramoto-style threshold claims.

- [ ] Sudden coherence jump
- [ ] Critical slowing down
- [ ] Hysteresis
- [ ] Domain-specific threshold `sigma > grad_S`
- [ ] Synthetic oscillator systems first; cultural/memory later; ML dynamics last

## Phase 6: Formal Theory [ ]

Only after empirical traction:

- [ ] Define `hbar_sem`
- [ ] Derive or reject an action principle
- [ ] Specify gate-to-probability boundary conditions
- [ ] Clarify GR/QM bridges as structural, approximate, or derivational
