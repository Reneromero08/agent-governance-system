# Phase 6B.6 Entry Review Addendum

**Status:** `BINDING_ENTRY_REVIEW_REPAIRS_APPLIED`
**Base design:** `PHASE6B6_ENTRY_PACKAGE.md`
**Base design commit:** `b4a154853a53db45c173e9c708563e0961244112`
**Phase 6B.6 entered:** false
**Implementation authorized:** false
**Hardware execution authorized:** false
**Scientific acquisition authorized:** false

This addendum is binding wherever it is more specific than the base entry package. It closes the precision defects found during architecture review without changing the scientific object, campaign size, data split, claim ceiling, or authority boundary.

---

## 1. V2 runtime boundary

The V2 runtime does not preserve one sender process across adjacent driven windows. It arms a sender for a driven window, captures that window, stops and joins the sender, then advances to the next schedule row. Sender-off rows require that no sender remain alive.

Therefore the Phase 6B.6 requirement for a new explicit-slot scheduler is confirmed, not hypothetical.

The scientific implementation must support contiguous multi-slot drive packets while preserving one absolute session timeline. It must not emulate a four-slot step by creating four unrelated sender epochs. It must record transitions between driven, phase-shifted, and physically disabled slots on that common timeline.

The qualified V2 hardware primitive and capture-quality logic may be reused by exact binding. The per-window orchestration semantics must not be falsely represented as already sufficient for Phase 6B.6 trajectories.

---

## 2. Nominal sample accounting

The base package fields:

```text
samples_per_slot = 4000
raw sample count = 41472000
```

are nominal design targets, not exact acceptance requirements.

The binding interpretation is:

```text
nominal_samples_per_slot = read_hz * slot_s = 4000
nominal_campaign_sample_count = 10368 * 4000 = 41472000
```

Actual sample counts are empirical and must be recorded per slot. They are accepted only through the frozen capture-quality contract, including empirical sample-rate, coverage, Nyquist-margin, and maximum-gap gates. No padding, interpolation, synthetic replacement, or silent repetition may be used to force an exact count.

All manifests and reports must use `nominal_` prefixes for the design targets and distinct fields for measured counts.

---

## 3. Exact preamble allocation

The ambiguous base phrase:

```text
24 carrier-off or time-matched sham slots
```

is replaced by the following exact allocation:

```text
48 sender-off idle slots
12 carrier-off slots
12 time-matched declaration-sham slots
24 amplitude-level-2 anchor slots
```

Definitions:

```text
sender-off idle:
  sender workload absent
  no active tone declaration
  receiver capture active

carrier-off:
  sender workload absent
  physical carrier action absent
  analysis tone identity frozen for matched noise-floor measurement

time-matched declaration sham:
  sender workload absent
  declaration fields mirror a driven row
  executed controls truthfully remain drive_on=false

anchor:
  physically driven
  amplitude level 2
  one positive and one negative sign per physical tone
```

The exact total remains 96 preamble slots. Only these preamble rows may estimate the session gauge. Carrier-off and declaration-sham rows remain separate strata in analysis.

---

## 4. Complete adjudication vocabulary

The permitted subordinate classifications are:

```text
SHARED_PREDICTIVE_OPERATOR_SUPPORTED
ROUTE_LOCAL_PREDICTIVE_OPERATOR_ONLY
DRIVEN_RELATIONAL_TRANSPORT_ONLY
PERSISTENT_STATE_CANDIDATE
CONFOUNDED_NO_OPERATOR_CLAIM
INSTRUMENTATION_BOUNDARY_REJECTED
```

`CONFOUNDED_NO_OPERATOR_CLAIM` is mandatory when order labels, chronology, session identity, route identity, or another context-only baseline explains the response comparably to the accepted dynamic candidate.

It is distinct from `INSTRUMENTATION_BOUNDARY_REJECTED`:

```text
CONFOUNDED_NO_OPERATOR_CLAIM:
  measured variation exists, but the design cannot separate it from a frozen confound

INSTRUMENTATION_BOUNDARY_REJECTED:
  no declared measured state and operator satisfy predictive sufficiency
```

Neither result permits post hoc threshold relaxation or model expansion.

---

## 5. Deterministic order sequencing

The base package freezes the physical order arrays. This addendum freezes the within-session order-family sequence.

Let:

```text
base = [FWD, REV, RND1, RND2, ORDER_LABEL_SHAM]
route_index(v4s5) = 0
route_index(v2s3) = 1
rotation = (reboot_block + 2 * route_index) mod 5
```

For reboot blocks `b0` through `b4`, use the left cyclic rotation of `base` by `rotation`.

For reboot block `b5`, use the reverse of the sequence that would otherwise result from the same formula.

This rule is deterministic and must be implemented directly. Runtime randomization is prohibited.

---

## 6. Effective entry object

The effective reviewed object is:

```text
PHASE6B6_ENTRY_PACKAGE.md
+ PHASE6B6_ENTRY_REVIEW_ADDENDUM.md
```

The architecture review conclusion is:

```text
scientific architecture = TECHNICALLY_ACCEPTABLE_WITH_BINDING_REPAIRS_APPLIED
campaign geometry = frozen
state and operator ladders = frozen
analysis gates = frozen
claim ceiling = frozen
Phase 6B.6 entry = pending project-owner decision
software implementation = not authorized
hardware execution = not authorized
scientific acquisition = not authorized
```

The next valid decision remains one of:

```text
APPROVE_PHASE6B6_SOFTWARE_ENTRY_ONLY
REJECT_AND_REVISE_PHASE6B6_ENTRY_DESIGN
HOLD_PHASE6B6
```
