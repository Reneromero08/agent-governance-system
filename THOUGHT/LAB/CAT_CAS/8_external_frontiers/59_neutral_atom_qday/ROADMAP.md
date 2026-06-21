# Exp 59 — Neutral-Atom Qday Frontier

**Status:** OPEN  
**Adjudication:** Class C reproducible physical-resource analysis; Class D for publication-level claims  
**Role:** audit the logical-to-physical quantum attack boundary and post-quantum migration timeline

---

# Frontier object

The object is the complete coupled architecture, not one physical-qubit headline.

Proposed process objects:

- `ArchitectureOrbit`
- `LogicalCircuitObject`
- `CodeLayoutGeometry`
- `AtomSchedule`
- `FactoryDemandGraph`
- `FailureBudget`
- `QdayEvidenceState`
- `MigrationDependencyGraph`

Preserve:

- logical circuit resources;
- code parameters;
- atom count;
- movement and routing;
- gate/measurement/reset times;
- error channels;
- atom loss;
- decoder throughput;
- non-Clifford resource demand;
- total success probability;
- architecture alternatives;
- uncertainty and evidence provenance.

---

# External questions

- Are published neutral-atom Shor resource estimates reproducible from their assumptions?
- Which assumptions dominate physical qubit count and runtime?
- How do atom loss, movement latency, fidelity, code rate, and decoder throughput interact?
- Can geometry-native placement reduce routing and factory overhead?
- Which logical circuit optimizations materially change physical resource requirements?
- What architecture reaches cryptographically relevant attacks first under public evidence?
- How should new evidence update a qday distribution?
- What post-quantum migration dependencies become critical under each scenario?

---

# Activation gates

## Gate 0 — Source freeze

- [ ] logical circuit paper frozen;
- [ ] neutral-atom resource paper frozen;
- [ ] hardware assumptions frozen;
- [ ] code and decoder assumptions frozen;
- [ ] current migration standards frozen;
- [ ] public evidence cutoff recorded;
- [ ] specification digest created.

## Gate 1 — Logical resource reconstruction

- [ ] logical qubits;
- [ ] Toffoli/non-Clifford count;
- [ ] depth and parallelism;
- [ ] ancilla demand;
- [ ] measurement/feed-forward assumptions;
- [ ] independent circuit-resource cross-check.

## Gate 2 — Physical architecture model

- [ ] code parameters;
- [ ] logical error model;
- [ ] atom layout;
- [ ] movement operations;
- [ ] gate fidelity;
- [ ] measurement/reset latency;
- [ ] loss/reload assumptions;
- [ ] decoder latency;
- [ ] factory scheduling;
- [ ] total failure budget.

## Gate 3 — Reproduce published estimate

- [ ] baseline atom count reproduced;
- [ ] baseline runtime reproduced;
- [ ] discrepancies explained;
- [ ] hidden or underspecified assumptions listed;
- [ ] no agreement forced by tuning after the fact.

## Gate 4 — Sensitivity cube

Sweep:

```text
physical error rate
× atom loss
× movement latency
× code rate
× decoder latency
× available parallelism
```

Report:

- [ ] physical atoms;
- [ ] runtime;
- [ ] success probability;
- [ ] dominant bottleneck;
- [ ] architecture phase transitions;
- [ ] uncertainty bands.

## Gate 5 — Alternative schedule geometry

- [ ] coupled placement/schedule representation;
- [ ] idle-zone borrowing;
- [ ] routing alternatives;
- [ ] code-block movement;
- [ ] factory placement;
- [ ] global versus local scheduling;
- [ ] exact resource comparison.

## Gate 6 — Qday evidence model

Separate:

- [ ] algorithm progress;
- [ ] error-correction progress;
- [ ] hardware scale/fidelity;
- [ ] capital and organizational capacity;
- [ ] secrecy uncertainty;
- [ ] migration lead time.

Use scenario distributions, not one prophetic date.

## Gate 7 — Migration dependency map

- [ ] vulnerable primitives;
- [ ] protocol dependencies;
- [ ] upgrade governance;
- [ ] dormant-key/account risk;
- [ ] hardware replacement cycles;
- [ ] hybrid transition options;
- [ ] emergency activation paths.

---

# Fastest falsifiable prototype

Reconstruct one published neutral-atom resource estimate from its public assumptions and measure which single parameter most changes atom count or runtime.

Failure to reproduce is itself a useful boundary result if the discrepancy is exact and documented.

---

# Claim discipline

Measured facts, modeled resources, and forecasts are separate layers.

Allowed:

- exact logical circuit resource count;
- model output under named assumptions;
- sensitivity to parameter changes;
- evidence-based scenario distribution.

Forbidden:

- neutral-atom Shor physically demonstrated;
- qday date known;
- non-public information inferred as fact;
- model output promoted to hardware evidence;
- migration deadline asserted without scenario assumptions;
- Small Wall crossed.

---

# First deliverable

`NEUTRAL_ATOM_RESOURCE_REPRODUCTION.md` plus executable model, frozen assumptions, discrepancy ledger, and sensitivity results.

---

# Claim ceiling

> Under the frozen public assumptions, the independent model reproduces or disputes the published resource estimate by the reported amount and identifies the dominant sensitivity parameters.

Any qday forecast remains conditional and updateable.
