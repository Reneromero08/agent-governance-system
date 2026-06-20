# Phase 6B.5E Reversed/Randomized Tone-Order Control

**Status:** `PREREGISTERED_NOT_AUTHORIZED`  
**Prerequisite:** Gate R external review and project-owner authorization  
**New acquisition:** forbidden until authorization  
**Purpose:** separate spectral tone identity from within-symbol path position

## 1. Unique question

The retained T48 campaign executes twelve tone bins in one fixed ascending order. Tone identity and within-symbol elapsed position are therefore confounded.

The control asks:

```text
Does the transported relation attach primarily to tone identity,
ordered path position, or an equivariant combination of both?
```

It does not repeat T48 merely to seek a better pass rate.

## 2. Frozen hypotheses

### H-spectral

After reindexing measurements by physical tone identity, forward, reverse, and randomized schedules preserve the same transferred mode/phase/permutation relations.

### H-path

Changing order produces a reproducible transformation that follows path position or path composition rather than tone identity alone.

### H-drift-artifact

Randomized/counterbalanced order removes the prior bin-position association and degrades relational recovery toward scramble/null behavior.

### H-mixed

Tone identity remains primary but a smaller, reproducible order-dependent residual survives.

## 3. Schedule conditions

For each authorized route and seed, use balanced conditions:

```text
FWD  = original ascending tone order
REV  = exact reversed tone order
RND1 = deterministic random permutation
RND2 = independent deterministic random permutation
```

The permutations are frozen before acquisition and bound to the campaign manifest.

Tone order is separate from pseudo codeword-bin permutation. The metadata must record both independently:

```text
tone_execution_order
codeword_bin_permutation
```

No hidden label, result-dependent permutation, or post-hoc order selection is allowed.

## 4. Preserved factors

Hold constant except for declared order:

```text
routes
sender/victim cores
P-state and frequency policy
slot duration and gap
read cadence
physical tones
codebook
phase levels
family schedule
trials per family
seed family
silent and scramble controls
raw writer and reconstruction path
```

Counterbalance order conditions across session chronology so temperature/time cannot masquerade as order.

## 5. Required controls

```text
silent carrier-off for each order block
scramble unshared-schedule for each order block
wrong execution-over-declaration
pseudo exact executed permutation
order-label sham: declared order differs from executed order
cross-order chart transfer
```

The receiver analysis may use executed order only where the contract explicitly permits it. Declared-versus-executed order must be separated in sham rows.

## 6. Frozen analysis

Use the Phase 6B.5C complex scalar chart as the first chart family. Chart selection remains calibration-only.

Report two coordinate views:

```text
A. reassembled by physical tone identity
B. retained in execution/path order
```

Primary tests:

1. held-out mode and wrong execution-over-declaration margins;
2. phase equivariance;
3. exact pseudo permutation covariance;
4. cross-order chart transfer;
5. residual dependence on tone identity versus path position;
6. order-sham actual-over-declared behavior;
7. silent/scramble null preservation.

No candidate ranking or AUC-first adjudication is permitted.

## 7. Primary outcomes

Choose one:

```text
TONE_IDENTITY_EQUIVARIANCE_SUPPORTED
ORDER_PATH_COVARIANCE_SUPPORTED
MIXED_TONE_PATH_GEOMETRY_SUPPORTED
FIXED_ORDER_ARTIFACT_SUPPORTED
NO_ORDER_RESOLUTION
```

## 8. Advancement rule

This campaign may be authorized only after Gate R confirms:

- the question is not already answered by the retained packet;
- order conditions are sufficiently counterbalanced;
- no-smuggle controls are complete;
- the claim ceiling remains channel-level;
- raw provenance and reconstruction remain mandatory.

A positive order result does not establish physical restoration, target coupling, orientation, or a Small Wall crossing.
