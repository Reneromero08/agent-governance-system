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

### Phase 2c: Resonance-Guided Sampling [ ]

- [ ] Implement full Cybernetic Truth control loop
- [ ] Temperature modulation: T = 1/(R + epsilon)
- [ ] Compare resonance-guided vs standard sampling

## Phase 3: Symbol Survival [ ]

Goal: Test whether high-compression, high-depth symbols survive noisy transmission better than lower-compression controls.

- [ ] Controlled transmission chain with variable noise
- [ ] Compare high-sigma/high-D_f vs low-sigma/low-D_f symbols
- [ ] Measure recall, paraphrase fidelity, persistence over generations

## Phase 4: Cybernetic Truth Monitor [ ]

Goal: Implement `SemioticMonitor`.

- [ ] State `rho` from hidden states/logits
- [ ] Alignment frame `C`
- [ ] Resonance `R = Tr(rho C)` or justified approximation
- [ ] Feedback control over temperature or candidate selection
- [ ] External verification coupling
- [ ] Test on hallucination-prone or ambiguity-heavy tasks

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
