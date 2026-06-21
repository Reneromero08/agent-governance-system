# Exp 51 — Poseidon Cryptanalysis

**Status:** OPEN  
**Adjudication:** Class A exact witnesses; Class D for research-paper rewards  
**Role:** first exact prize-bearing algebraic frontier for Track 8

---

# Frontier object

Poseidon is treated as an exact algebraic dynamical system over a finite field:

```text
state
→ nonlinear S-box layer
→ linear mixing layer
→ repeated round geometry
→ constrained output boundary
```

The internal primitive is the complete algebraic constraint fiber, not a scalar input candidate.

Proposed process objects:

- `PoseidonOrbit`
- `ConstraintFiber`
- `CollisionOrbit`
- `ZeroClosureObject`
- `DensityConfinementObject`

---

# External targets

## CICO

Fixed input/output boundary with internal degrees of freedom.

Primary mechanism families:

- split-round intersection;
- resultants;
- Gröbner bases;
- Jacobian rank loss;
- MDS basis changes;
- partial-round recurrence.

## Zero-test

Polynomial coefficients generate a hash-derived field element that must be a root of the same polynomial.

Primary mechanism families:

- factor-first closure;
- root-fiber parameterization;
- fixed-point continuation;
- extension-field conjugation;
- singular-component search.

## Density

Two input defects must constrain the entire output to two multiplicative classes.

Primary mechanism families:

- output-first inversion;
- class-pattern symmetry quotient;
- defect propagation;
- split-round meet-in-the-middle;
- coset confinement invariants.

## Partial collision

Two distinct states must share an output prefix.

Primary mechanism families:

- paired orbit;
- differential/rebound structure;
- diagonal-saturated collision ideal;
- invariant subspaces;
- structural entropy reduction before distributed search.

---

# Activation gates

## Gate 0 — Source freeze

- [ ] official rules archived;
- [ ] official implementation commit frozen;
- [ ] constants and test vectors frozen;
- [ ] prize/deadline snapshot recorded;
- [ ] specification digest created.

## Gate 1 — Independent exact engine

- [ ] base-field arithmetic implemented;
- [ ] extension-field arithmetic implemented where required;
- [ ] full and partial rounds implemented;
- [ ] MDS layer independently implemented;
- [ ] official samples reproduced;
- [ ] independent outputs match official implementation.

## Gate 2 — Constraint compiler

- [ ] round equations compiled exactly;
- [ ] variable graph generated;
- [ ] degree and sparsity tracked;
- [ ] elimination order represented;
- [ ] process object serializes and reloads;
- [ ] forbidden target fields absent.

## Gate 3 — CICO calibration

- [ ] reproduce relaxed sample;
- [ ] reproduce a lower-round solved structure where public;
- [ ] benchmark resultants;
- [ ] benchmark Gröbner methods;
- [ ] define RP10 mechanism and stop conditions.

## Gate 4 — Zero-test native route

- [ ] direct formulation;
- [ ] factor-first formulation;
- [ ] linear root-fiber formulation;
- [ ] reduced-round continuation;
- [ ] exact witness replay.

## Gate 5 — Density route

- [ ] inverse solver;
- [ ] class-orbit quotient;
- [ ] exact confinement verifier;
- [ ] reduced-round ladder.

## Gate 6 — Collision escalation

- [ ] enter only after named structural reduction;
- [ ] paired orbit live;
- [ ] trivial diagonal removed;
- [ ] generic-search cost separated from structural gain.

---

# No-smuggle model

Forbidden:

- known witness in prompts, fixtures, or retrieval caches;
- challenge labels controlling algebraic choices;
- relaxed verifier promoted as exact success;
- omitted state coordinates or resource costs;
- post-hoc round/metric selection;
- using production secrets or live user targets.

All work is limited to official challenge instances, test keys, and authorized research targets.

---

# First deliverable

A single exact engine that:

1. matches official test vectors;
2. serializes a complete constraint fiber;
3. runs official and independent verification;
4. packages a replayable witness;
5. records the transfer mechanism separately from the prize result.

---

# Claim ceiling

Before official acceptance:

> An exact independent Poseidon frontier engine exists and reproduces the frozen official examples.

After an accepted witness:

> The specified public reduced-round challenge instance was solved and independently replayed.

Forbidden without further evidence:

- full Poseidon broken;
- production SNARK systems compromised;
- Small Wall crossed;
- Big Wall broken.
