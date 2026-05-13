# AI Alignment Control -- Experiment Skeleton

Date: 2026-05-13
Status: LOCKED. Definitions frozen. Awaiting constitution text and execution.

---

## Formula

```
R = (E / grad_S) * sigma^Df
```

## Locked Domain Mappings (AI Alignment)

| Symbol | Operational Definition | Measurement |
|--------|----------------------|-------------|
| `E` | value core / truth-attractor strength | fixed to 1.0 (constitution carries the signal) |
| `grad_S` | adversarial pressure / contradiction entropy | measured: entropy of hidden-state distribution during challenged responses, averaged over test prompts |
| `sigma` | constitutional compression ratio | measured: token count of constitution vs token count of equivalent rules-listed form; or compression ratio of constitution relative to RLHF example set |
| `Df` | scale-nesting depth | measured: number of distinct scales the constitution explicitly addresses (individual, interpersonal, communal, civilizational, cosmic = 5; validated by human rater) |
| `R` | alignment retention / jailbreak resistance | measured: fraction of test prompts where model maintains constitution-aligned response |

## Experiment Design

### Conditions

**Control (C):** Base model with standard instruction prompt. No constitution. No RLHF fine-tuning.

**Standard RLHF (S):** Base model fine-tuned on 10,000 human preference examples. Standard practice.

**Constitution (X):** Base model with constitution prepended as system prompt. No fine-tuning. Constitution text injected at inference time.

**Constitution + RLHF (XR):** Base model fine-tuned on constitution-augmented preference data. Constitution used as system prompt. Hybrid.

### Independent Variables

- `sigma`: compression ratio of constitution (tokens of constitution / tokens of equivalent rules)
- `Df`: scale-nesting depth (1-5, validated)
- Control condition: sigma = 1, Df = 1 (no compression, no nesting)

### Dependent Variables (R measurements)

1. **Drift resistance**: fraction of responses maintaining value alignment across 20-turn conversations on ambiguous ethical topics
2. **Jailbreak resistance**: fraction of jailbreak prompts successfully deflected (model refuses or redirects appropriately)
3. **Value generalization**: accuracy on 50 novel ethical dilemmas not present in any training data
4. **Hidden-state entropy (grad_S proxy)**: average entropy of final-layer hidden states during ethical reasoning prompts. Lower = more coherent.

### Architecture (from Cybernetic Truth)

```
For each token generation step:
  1. Extract hidden state h_t from final layer
  2. Build density matrix rho = |h_t><h_t| (or softmax outer product)
  3. Measure resonance R = Tr(rho C) where C is the alignment frame
  4. Measure entropy = -Tr(rho ln rho) as grad_S proxy
  5. Track dR/dt, purity = Tr(rho^2), coherence = sum|rho_ij|
```

Note: This architecture requires logit/hidden-state access (open model or API with embeddings enabled).

### Training / Test Split

- **Training (constitution design):** Developer writes constitution. No model training on the constitution condition.
- **Test:** All prompts are unseen. Jailbreak prompts sourced from public jailbreak datasets. Ethical dilemmas from published ethics benchmarks. Conversation drift measured on novel multi-turn scenarios.

### Baselines

1. **Temperature-only baseline**: Same model, no constitution, varied temperature. Does resonance modulation outperform raw temperature tuning?
2. **Constitution-only baseline**: Constitution as system prompt without alignment frame C. Does the resonance loop add value beyond the constitution text?
3. **RLHF baseline**: Standard preference-tuning. Current best practice.

## Success Criteria

| Claim | Test | Success |
|-------|------|---------|
| sigma compresses alignment | XR > S on drift and jailbreak | Constitution-augmented beats pure RLHF at matched token budget |
| Df amplifies retention | XR(Df=5) > XR(Df=1) | Deep-fractal constitution beats flat constitution |
| R predicts alignment | R correlated with drift/jailbreak/generalization scores across conditions | Resonance measure tracks empirical alignment |
| grad_S captures adversarial pressure | grad_S higher on jailbreak prompts than benign prompts | Entropy gradient reflects prompt difficulty |

## Failure Criteria

| Condition | Interpretation |
|-----------|---------------|
| S >= XR on all measures | RLHF alone is sufficient. Formula adds no value. |
| R uncorrelated with alignment outcomes | Resonance measure is not tracking alignment. |
| XR < C on any measure | Constitution is harmful. Hypothesis wrong. |

## Implementation Phases

### Phase 2a: Inference-Only (Logit Access)

- Open model with hidden-state access (Llama, Mistral, Qwen)
- Constitution injected as system prompt
- SemioticMonitor extracts hidden states and computes R, grad_S, purity per token
- No fine-tuning. Tests constitution at inference time only.

### Phase 2b: Fine-Tuning (if 2a shows directional signal)

- LoRA or full fine-tune on constitution-augmented preference data
- Matched token/compute budget against standard RLHF
- Re-run all tests

### Phase 2c: Resonance-Guided Sampling (if 2a succeeds)

- Implement full Cybernetic Truth control loop
- Temperature modulated by R: T = 1/(R + epsilon)
- Compare resonance-guided vs standard sampling

## Required Inputs

- [ ] Constitution text (from user -- `CONSTITUTION.md`)
- [ ] Jailbreak prompt dataset (public: AdvBench, HarmBench, or custom)
- [ ] Ethical dilemma dataset (public: ETHICS, MoralExceptQA, or custom)
- [ ] Multi-turn conversation scenarios (custom or adapted from existing benchmarks)
- [ ] Open model with hidden-state access
- [ ] Df validation rubric (human rater confirms scale-nesting depth of constitution)

## Outputs

- Per-condition R, grad_S, drift score, jailbreak score, generalization score
- Alpha/beta diagnostic: alignment_score = alpha * R_predicted + beta (ideally alpha=1, beta=0)
- Per-condition hidden-state entropy distributions
- Drift trajectory plots (R over conversation turns)
- Cross-model comparison table (C vs S vs X vs XR)

## Relation to QEC Phase 1

| Aspect | QEC Phase 1 | AI Alignment Phase 2 |
|--------|------------|---------------------|
| Domain | Surface code simulator | LLM inference |
| E | 1.0 (globally calibrated) | 1.0 (constitution signal) |
| grad_S | sqrt(syndrome density) | hidden-state entropy |
| sigma | fidelity factor (distance slope) | compression ratio (token count) |
| Df | t = floor((d-1)/2) | scale-nesting depth |
| R | log_suppression = ln(p/p_L) | alignment retention / jailbreak resistance |
| Training | distances {3,5,7} | human-written constitution |
| Test | distances {9,11} | unseen jailbreaks, ethical dilemmas |
| Baselines | standard_qec_scaling, p_only | RLHF, temperature-only, no-constitution |
| Threat | p_th dependent, sigma noisy | subjective Df, proxy grad_S |
