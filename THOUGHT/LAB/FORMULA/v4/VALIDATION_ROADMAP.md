# v4 Validation Roadmap

## Phase 0: Lock Mappings

Before running more experiments, create locked mappings for:

1. QEC
2. AI alignment / Cybernetic Truth
3. Memory / symbol survival

Each mapping must define observables, baselines, success criteria, and failure
criteria.

## Phase 1: QEC Precision Sweep

Goal:

Test the functional form in the cleanest available domain.

Required:

- multiple physical error rates
- multiple code distances or redundancy depths
- at least one known analytic baseline
- comparison against simpler predictors

Success:

- the formula predicts logical survival/corrected fidelity better than physical
  error rate, redundancy count, or code distance alone;
- parameters are locked before the sweep;
- results generalize across at least two related QEC setups.

Failure:

- known QEC formulas or simple baselines explain the data as well or better;
- `sigma^Df` adds no predictive value;
- mapping requires post-hoc changes.

## Phase 2: AI Alignment Control

Goal:

Test the Light Cone claim that compressed/fractal constitutions improve
alignment retention and adversarial resistance.

Minimum local version:

- use open model or API-accessible logits/hidden states where possible;
- compare normal instruction prompt vs compressed constitution vs example-heavy
  context;
- measure drift, jailbreak resistance, value-generalization, self-consistency,
  and hidden-state entropy/coherence.

Stronger version:

- fine-tune or LoRA train the same base model under matched token/compute budget.

## Phase 3: Symbol Survival

Goal:

Test whether high-compression, high-depth symbols survive noisy transmission
better than lower-compression controls.

This is not the first priority because its mappings are more subjective than
QEC, but it directly tests the semiotic survival thesis.

## Phase 4: Cybernetic Truth Monitor

Goal:

Implement `SemioticMonitor`:

- state `rho` from hidden states/logits;
- alignment frame `C`;
- resonance `R = Tr(rho C)` or justified approximation;
- feedback control over temperature or candidate selection;
- external verification coupling.

This should be tested on hallucination-prone or ambiguity-heavy tasks.

## Phase 5: Phase Transition Tests

Goal:

Test Kuramoto-style threshold claims:

- sudden coherence jump;
- critical slowing down;
- hysteresis;
- domain-specific threshold `sigma > grad_S`.

Candidate domains:

- synthetic oscillator systems first;
- cultural or memory transmission later;
- ML training/checkpoint dynamics only after instrumentation is sound.

## Phase 6: Formal Theory

Only after empirical traction:

- define `hbar_sem`;
- derive or reject an action principle;
- specify gate-to-probability boundary conditions;
- clarify GR/QM bridges as structural, approximate, or derivational.
