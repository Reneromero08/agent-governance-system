# Q39: R Exhibits Homeostatic Regulation

## Hypothesis

When perturbed, R in a functioning semantic system returns to a characteristic baseline value -- analogous to biological homeostasis. Specifically:

1. Semantic systems have a "resting state" R value determined by their structure
2. Perturbations (noise, conflicting information, domain shift) temporarily decrease R
3. The system recovers toward baseline without external intervention
4. Recovery time (tau_relax) is a measurable, meaningful property of the system

## Verification Status: OPEN (Improperly Tested)

**Category:** C3 -- Test was tautological/self-referential. Test design created the result.

## v1 Evidence Summary

- Test code creates SLERP trajectories with single-point perturbation
- The "recovery" is the trajectory returning to a predetermined geodesic path
- tau_relax is determined by test parameters, not by system dynamics
- No genuine feedback loop, no self-regulation, no real dynamics

## What Went Wrong With the Test

The test **designed in the recovery it was trying to discover**. A SLERP interpolation between point A and point B, when perturbed at the midpoint, continues toward point B by construction. That is not homeostasis -- that is a predetermined trajectory. tau_relax was a parameter of the test, not a measurement of the system.

**Crucially:** The test being an artifact does NOT mean semantic systems lack homeostatic properties. It means nobody tested for them properly. Real language models DO exhibit stability properties (e.g., fine-tuned models resist catastrophic forgetting to varying degrees). The question is whether R captures this.

## What a Proper Test Looks Like

### Design Requirements (per v2 METHODOLOGY.md)

1. **Consistent E:** Use E = mean pairwise cosine similarity throughout
2. **Real data:** Use real evolving systems, not synthetic trajectories
3. **Pre-registered criteria:**
   - Perturb a real system (see below)
   - Measure R before, during, and after perturbation
   - Define "recovery" as R returning to within X% of baseline (pre-register X)
   - Measure tau_relax as time/steps to recovery
4. **Baseline:** Compare to bare E recovery. If E recovers identically to R, R adds nothing.
5. **No designed-in recovery:** The system must recover (or not) on its own.

### Specific Steps

1. **Approach A -- Fine-tuning perturbation:**
   - Take a pre-trained model (e.g., BERT-base)
   - Measure baseline R on a held-out evaluation set
   - Fine-tune on a narrow domain (perturbation)
   - Continue training on original broad data (recovery opportunity)
   - Track R over training steps
   - Does R return to baseline? How fast? Does it overshoot?

2. **Approach B -- Corpus-level perturbation:**
   - Compute R on a clean corpus (e.g., Wikipedia)
   - Inject noisy/adversarial text at various proportions (5%, 10%, 25%, 50%)
   - Incrementally remove noise
   - Track R as noise is removed
   - Does R recover monotonically? At what noise fraction is recovery impossible?

3. **Approach C -- Dialogue perturbation:**
   - Take real dialogue datasets (e.g., MultiWOZ, DailyDialog)
   - Measure R per turn
   - Identify naturally occurring perturbations (topic shifts, misunderstandings)
   - Measure whether R recovers after perturbations
   - Compare recovery rate to random baseline

### Success Criteria

- **Confirmed:** R returns to within 10% of baseline after perturbation removal, with tau_relax significantly shorter than random walk prediction (p < 0.01)
- **Falsified:** R does not recover, or recovers no faster than random walk, or bare E recovers identically
- **Inconclusive:** Recovery is domain/perturbation-dependent with no clear pattern

### Required Data Sources

- Pre-trained model + fine-tuning pipeline (HuggingFace)
- Wikipedia corpus (freely available)
- MultiWOZ or DailyDialog (freely available)
- Adversarial/noisy text generation (can be synthetic since it's the perturbation, not the test data)

## Salvageable from v1

- The conceptual framework of R as a system health indicator
- The idea that tau_relax is a meaningful system property
- Nothing from the actual test code -- SLERP trajectory must be discarded entirely
