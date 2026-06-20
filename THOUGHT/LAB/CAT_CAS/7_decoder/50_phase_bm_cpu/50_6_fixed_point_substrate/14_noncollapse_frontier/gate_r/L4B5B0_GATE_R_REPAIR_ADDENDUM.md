# L4B.5B0 Gate R Repair Addendum

**Status:** `BINDING_REPAIR_ADDENDUM_PENDING_PROJECT_OWNER_RATIFICATION`  
**Base design:** `l4b5b0_observability_operator_v1` / `1.0.0`  
**Implementation authorized:** no  
**Physical acquisition authorized:** no  
**Restoration authorized:** no

The reviewed object is the sealed base design plus this addendum, the frozen Phase 6B.5C/5D evidence, and the Phase 6B.5E tone-order contract. The base design bytes and digest remain unchanged.

## Effective object partition

Keep four objects separate:

```text
r_t = directly measured response
u_t = physically executed control
c_t = nuisance/topology context
g_s = session gauge estimated before evaluated rows
```

```text
r_t = [lockin_I, lockin_Q, ring_osc_period]

u_t = [drive_on, executed_mode, amplitude, phase_action,
       tone_identity, tone_execution_order, codeword_bin_permutation]

c_t = [route, core identities, TSC origin, capture window,
       temperature, P-state, session chronology]

g_s = [preamble-only complex alpha_s, idle covariance, amplitude floor]
```

Declared metadata never replaces executed control. Wrong-declaration and order-label-sham rows retain both values.

## Effective measured-state ladder

```text
S0_minimal = r_t
S1_contextual = gauge_normalize(r_t, g_s)
S2_delay_embedded(L) = [S1_t ... S1_(t-L+1); u_(t-1) ... u_(t-L+1)]
```

`sender_mode`, route, core identity, TSC origin, and window index are removed from the state vector. They remain input or context. Context may stratify or condition a model but is not evidence of physical state.

The session gauge is estimated from preamble only and frozen before odd, wrong, pseudo, tone-order, validation, or test rows. Seed 4 remains included.

## Effective operator claim

Use measured-state notation:

```text
s_(t+1) = F(s_t, u_t, c_t; parameters) + epsilon_t
```

The operator predicts the next member of the declared measured equivalence class. It does not identify complete internal substrate state.

Session ID may define blocks and random effects but may not be an input feature for a claimed cross-session operator. A session-lookup model is a null baseline. Diagnostic classification is subordinate to trajectory prediction and cannot define the state or authorize a claim.

## Mandatory drive-off classification

Impulse and step stages must include readout after the sender is physically disabled:

```text
drive_on = 0
no periodic refresh
no continued sender workload
```

Compare with time-matched sham and carrier-off baselines.

Classify `PERSISTENT_STATE_CANDIDATE` only when both hold on held-out sessions:

1. the lower 95% session-block-bootstrap bound of post-drive distance from sham exceeds the sham upper bound for at least three consecutive frozen windows; and
2. a zero-input decay model improves held-out NRMSE by at least 10% over mean, return-to-baseline, and last-value baselines with a 95% interval excluding zero gain.

Otherwise classify `DRIVEN_RELATIONAL_TRANSPORT_ONLY`. Driven-only transport blocks physical-memory, restoration, and holonomy claims but may still support an active-channel operator claim.

## Tone identity versus path position

Before any S2/path-memory interpretation, execute the frozen Phase 6B.5E conditions:

```text
FWD
REV
RND1
RND2
order-label sham
```

Record tone identity, execution order, codeword permutation, declared order, and executed order independently. Report both physical-tone-indexed and execution-order coordinate views.

## Session-dominant residual control

Because Phase 6B.5D found seed/session eta-squared `0.53628` versus route eta-squared `0.00034`, the design must:

- counterbalance order across session chronology;
- freeze preamble-only `g_s`;
- retain seed 4 as a stress case;
- report leave-one-session-out performance;
- compare raw, gauge-normalized, route-only, session-lookup, time-index, and input-only baselines;
- block a shared-operator claim when session lookup is within 5% of the dynamic model;
- report session effects separately from dynamic parameters.

## Added Gate R gates

```text
GR1 partition integrity:
  zero input/context fields serialized as measured state;
  zero declared-label substitution for executed control.

GR2 gauge integrity:
  preamble-only gauge; frozen before evaluation;
  raw and normalized results reported; seed 4 retained.

GR3 drive-off classification:
  mandatory PERSISTENT_STATE_CANDIDATE or DRIVEN_RELATIONAL_TRANSPORT_ONLY.

GR4 tone-order disentanglement:
  FWD/REV/RND1/RND2/order-sham complete before S2/path-memory interpretation.

GR5 governance separation:
  technical review, project-owner ratification, and implementation authorization
  are three separate records.
```

## Claim ceiling

Technical acceptance supports only a preregistered experiment capable of testing predictive observability of a measured response equivalence class and classifying it as driven-only or post-drive persistent.

It does not establish complete physical observability, physical HoloGeometry, inverse dynamics, restoration, target coupling, orientation recovery, or a Small Wall crossing.
