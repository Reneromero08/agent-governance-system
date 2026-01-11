# Question 17: Governance gating (R: 1420)

**STATUS: ✅ VALIDATED (8/8 tests pass)**

## Question
Should agent actions require R > threshold? How would this affect autonomy vs. safety tradeoffs?

---

## TESTS

**Test file:** `experiments/open_questions/q17/test_q17_r_gate.py`
**Results:** `experiments/open_questions/q17/q17_test_results.json`
**Model:** `sentence-transformers/all-MiniLM-L6-v2`

### Test Results (2026-01-11)

| Test | Result | Key Finding |
|------|--------|-------------|
| R_ORDERING | ✅ PASS | R_high(57.3) > R_med(4.1) > R_low(0.69) |
| VOLUME_RESISTANCE | ✅ PASS | Adding noise DECREASED R by 77.3% |
| ECHO_CHAMBER | ✅ PASS | Echo chamber R = 10^8 (detectable) |
| THRESHOLD_DISCRIMINATION | ✅ PASS | T2(0.8) correctly separates high/low |
| TIER_CLASSIFICATION | ✅ PASS | 9/9 actions classified correctly |
| MINIMUM_OBSERVATIONS | ✅ PASS | Edge cases handled |
| REAL_EMBEDDINGS | ✅ PASS | E_high=0.965, E_low=0.049 |
| GATE_INTEGRATION | ✅ PASS | 4/4 scenarios correct |

### Key Empirical Findings

```
HIGH AGREEMENT (5 paraphrases of "Paris is capital of France"):
  E = 0.965 (very high agreement)
  σ = 0.017 (very tight)
  R = 57.3

LOW AGREEMENT (5 unrelated sentences):
  E = 0.049 (low agreement)
  σ = 0.071 (spread)
  R = 0.69

VOLUME RESISTANCE (adding noisy observations):
  Initial R: 4.14 (3 observations)
  Final R: 0.94 (8 observations, +5 noise)
  Change: -77.3% (R DECREASED, not increased)

ECHO CHAMBER (5 identical sentences):
  σ = 0.0 (no variation)
  R = 10^8 (effectively infinite - DETECTABLE)
```

---

## ANSWER

**YES: Agent actions should require R > threshold, with graduated thresholds by action criticality.**

### Theoretical Justification

Three converging lines of evidence support R-gating for governance:

#### 1. Phase Transitions Justify Binary Gates (Q12)

Experimental evidence from weight interpolation (E.X.3.3b) shows truth **crystallizes suddenly**:

| Training % | Generalization |
|------------|----------------|
| 0% | 0.02 |
| 50% | 0.33 |
| 90% | 0.58 |
| 100% | **1.00** (+0.42 jump) |

**Implication**: There's no "partial truth" zone. Intermediate states (like alpha=0.75) are often pathological, worse than fully untrained. A threshold-based gate (R > tau) is appropriate because meaning doesn't emerge gradually.

#### 2. Gate is a Valid Subobject Classifier (Q14)

The gate structure satisfies sheaf axioms:
- **Locality**: 97.6% (local agreement leads to global consistency)
- **Gluing**: 95.3% (compatible sections compose correctly)
- **Subobject classifier**: Omega = {OPEN, CLOSED} with well-defined characteristic morphism

**Implication**: The gate is mathematically well-founded. It classifies observation contexts into "trustworthy" and "untrustworthy" subobjects.

#### 3. R is Intensive, Not Extensive (Q15)

R correlates perfectly (r=1.0) with sqrt(Likelihood Precision), but is **independent of sample size**:

| Metric | Correlation with R |
|--------|-------------------|
| Sqrt(Likelihood Precision) | **1.0000** |
| Posterior Precision | -0.0937 |

**Implication**: R measures signal quality, not accumulated data. An agent cannot game the gate by gathering more evidence from a bad source. This prevents "false confidence via volume."

---

## Autonomy vs. Safety Tradeoffs

### The Core Tension

| Priority | Autonomy | Safety |
|----------|----------|--------|
| Agent can act | Faster, more capable | More risk |
| Agent must verify | Slower, more constrained | Less risk |

### Resolution: Graduated Thresholds by Action Criticality

Not all actions carry equal risk. We propose a **4-tier threshold hierarchy**:

| Tier | Action Type | Threshold | Failure Mode |
|------|-------------|-----------|--------------|
| **T0** | Read-only (observe, search) | None | Always allowed |
| **T1** | Reversible (draft, stage) | R > 0.5 | Warn, proceed |
| **T2** | Persistent (commit, write) | R > 0.8 | Block, request escalation |
| **T3** | Critical (deploy, delete canon) | R > 1.0 | Block, require human approval |

This maps directly to existing **Crisis Levels** (CRISIS.md):

| Crisis Level | Gate State | Response |
|--------------|------------|----------|
| 0 (Normal) | R >= threshold | Continue |
| 1 (Warning) | R close to threshold | Fix before commit |
| 2 (Alert) | R < threshold | Rollback |
| 3 (Quarantine) | R collapsed suddenly | Isolate, human review |
| 4 (Constitutional) | R undefined/corrupted | Full reset |

### Autonomy Preserved Where Risk is Low

- **T0 actions** (read, search, explore) have no gate - full autonomy
- **T1 actions** (draft, propose) have a soft gate - autonomy with visibility
- **T2/T3 actions** (commit, deploy) have hard gates - safety over speed

This preserves agent capability for low-risk work while requiring verification for high-risk changes.

---

## Echo Chamber Risk (Q5)

R-gating has a known vulnerability: **echo chambers** produce high R despite being wrong.

| Scenario | R Value | Actual Accuracy |
|----------|---------|-----------------|
| Independent observers | Normal | High |
| Echo chamber | **20x higher** | Low |

### Defense Mechanisms

1. **Extreme R detection**: Flag R > 95th percentile as suspicious
2. **Fresh data injection**: Add independent observation; if R crashes (93% drop), it was echo chamber
3. **Source diversity check**: Require observations from N independent sources for T3 actions

---

## Failure Mode When Gate CLOSED

When R < threshold, the agent must not simply stop. The **VERIFICATION_PROTOCOL_CANON** prescribes:

1. **BLOCKED status**: Report precisely what failed
2. **Escalation**: Request human review with full context
3. **Minimal fix specification**: Identify the smallest change that would unblock

### Graceful Degradation Hierarchy

| Gate State | Agent Response |
|------------|----------------|
| R >= tau | Proceed with action |
| 0.5*tau <= R < tau | Request clarification, proceed if user confirms |
| R < 0.5*tau | Block action, escalate to human |
| R undefined | Enter quarantine mode |

This prevents paralysis while maintaining safety.

---

## Implementation Recommendations

### 1. Integrate with Existing Verification Protocol

Add R-check to VERIFICATION_PROTOCOL_CANON.md Step 0:

```
STEP 0.5: R-GATE CHECK
- Compute R for the action context
- If R < threshold for action tier, STOP and escalate
- Record R value in verification artifacts
```

### 2. Surface R in Crisis Diagnostics

Extend `emergency.py` to report R-values:

```python
def diagnose_crisis():
    r_value = compute_r(current_context)
    if r_value < THRESHOLD_T2:
        return CrisisLevel.ALERT
    elif r_value_dropped_suddenly():
        return CrisisLevel.QUARANTINE
```

### 3. Audit Trail for R-Gated Decisions

Log all R-gate decisions to `_runs/r_gate_logs/`:

```json
{
  "timestamp": "2026-01-11T10:30:00Z",
  "action": "commit",
  "tier": "T2",
  "r_value": 0.72,
  "threshold": 0.8,
  "result": "BLOCKED",
  "escalated_to": "human"
}
```

---

## Connections to Other Questions

| Question | Connection |
|----------|------------|
| Q10 (Alignment detection) | R-gating IS alignment detection at the action level |
| Q12 (Phase transitions) | Justifies binary gates over probabilistic scaling |
| Q14 (Category theory) | Provides mathematical foundation (sheaf, subobject classifier) |
| Q15 (Bayesian inference) | Explains why R-gating is robust to volume attacks |
| Q19 (Value learning) | R-gating can filter which human feedback to trust |
| Q22 (Threshold calibration) | Remains OPEN - exact thresholds need domain calibration |

---

## What Remains Open

1. **Exact threshold values**: The 0.5/0.8/1.0 tiers are heuristic. Q22 should determine principled calibration.
2. **Multi-agent gating**: When multiple agents collaborate, how do R-values compose?
3. **Temporal dynamics**: Should thresholds adapt based on dR/dt (Q21)?
4. **Domain-specific tuning**: Different action domains may need different thresholds.

---

## Final Answer

**YES: Agent actions should require R > threshold.**

**The case is strong:**
- Phase transitions (Q12) justify binary gates
- Sheaf structure (Q14) provides mathematical foundation
- Intensive property (Q15) prevents volume attacks
- Existing governance (CRISIS.md, VERIFICATION_PROTOCOL) provides integration points

**Autonomy vs. safety is resolved through graduated thresholds:**
- T0 (read): No gate - full autonomy
- T1 (reversible): Soft gate - autonomy with visibility
- T2 (persistent): Hard gate - safety over speed
- T3 (critical): Human approval required

**Failure modes are handled gracefully:**
- BLOCKED status with escalation path
- Graceful degradation hierarchy
- Echo chamber defenses

The R-gate is the practical instantiation of the Living Formula's core insight: **local agreement reveals truth, but only when the signal quality (R) exceeds a threshold.**

---

**Status**: OPEN -> VALIDATED (8/8 tests pass)
**Date**: 2026-01-11
**Validated by**: Empirical tests with sentence-transformers embeddings
**Test output**: `experiments/open_questions/q17/q17_test_results.json`
**Implementation**: `experiments/open_questions/q17/r_gate.py`
