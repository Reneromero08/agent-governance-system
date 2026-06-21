# Exp 52 — Proximity Prize

**Status:** OPEN  
**Adjudication:** Class D proof or counterexample  
**Role:** external theorem frontier for critical Reed–Solomon agreement boundaries

---

# Frontier object

The primary object is the full incidence geometry between:

- evaluation domain;
- Reed–Solomon code;
- received/correlated word families;
- affine combinations;
- agreement sets;
- error supports;
- syndromes;
- nearby codeword fibers;
- list-size and failure-probability boundaries.

Do not reduce the experiment to selecting one decoded polynomial.

Proposed process objects:

- `AgreementOrbit`
- `CodeFiber`
- `SyndromeGeometry`
- `SupportIntersectionLattice`
- `ProximityBoundaryAtlas`

---

# External questions

## Mutual correlated agreement

Locate the largest proximity radius for which bad correlated-agreement configurations remain below the required probability threshold.

## List decoding / list recovery

Locate the largest proximity radius for which the relevant correlated/interleaved list remains within the required bound.

The official companion paper and exact quantifiers control the experiment. Homepage summaries are not sufficient.

---

# Activation gates

## Gate 0 — Statement freeze

- [ ] complete companion paper obtained;
- [ ] every quantifier transcribed;
- [ ] rate/domain/field assumptions frozen;
- [ ] probability and list-size thresholds frozen;
- [ ] organizer ambiguities recorded;
- [ ] current award conditions confirmed;
- [ ] specification digest created.

## Gate 1 — Exact finite laboratory

- [ ] finite-field arithmetic;
- [ ] Reed–Solomon encoder;
- [ ] evaluation-domain constructor;
- [ ] correlated/interleaved family constructor;
- [ ] Hamming/agreement geometry;
- [ ] syndrome computation;
- [ ] exact small list enumeration;
- [ ] official predicate checker where available.

## Gate 2 — Symmetry quotient

- [ ] affine/domain symmetries identified;
- [ ] code automorphisms identified;
- [ ] canonical representatives implemented;
- [ ] orbit sizes verified;
- [ ] exhaustive search avoids duplicate classes.

## Gate 3 — Boundary atlas

For small parameters:

- [ ] exact threshold transitions;
- [ ] extremal families;
- [ ] support-intersection patterns;
- [ ] syndrome ranks;
- [ ] list-size growth;
- [ ] counterexamples to naive lemmas;
- [ ] complete witness serialization.

## Gate 4 — Invariant discovery

Search:

- [ ] syndrome-space rank;
- [ ] common-zero polynomial degree;
- [ ] gcd structure of difference polynomials;
- [ ] support-intersection lattice invariants;
- [ ] incidence-matrix rank;
- [ ] matroid circuits;
- [ ] affine-subspace dimension;
- [ ] higher-order agreement tensor rank;
- [ ] subspace-design parameters;
- [ ] orbit stabilizers.

## Gate 5 — Theorem extraction

- [ ] choose one recurrent finite mechanism;
- [ ] state a parameterized lemma;
- [ ] test the lemma adversarially on finite cases;
- [ ] derive a proof or counterexample family;
- [ ] separate computational evidence from universal proof;
- [ ] obtain independent mathematical review;
- [ ] formalize critical lemmas where practical.

---

# Fastest falsifiable prototypes

1. Find a finite counterexample to an overstrong interpretation.
2. Reproduce known small-parameter bounds exactly.
3. Discover a new extremal support family.
4. Prove a special-rate or special-domain lemma.
5. Improve one side of a known bound.

A full optimal-radius theorem is not the only scientifically useful first result.

---

# No-smuggle model

Forbidden:

- using the target theorem as an assumed lemma;
- treating finite enumeration as asymptotic proof;
- omitting field-size or smooth-domain assumptions;
- selecting a favorable definition among unresolved variants after results appear;
- importing a literature result without provenance;
- hiding failed parameter regimes.

---

# First deliverable

`PROXIMITY_BOUNDARY_ATLAS.md` plus exact replay code for the smallest faithful parameter families.

The atlas must state:

- exact definition version;
- parameter domain;
- symmetry quotient;
- extremal witnesses;
- known versus newly observed boundaries;
- candidate invariants;
- theorem gaps.

---

# Claim ceiling

Finite success licenses:

> The exact finite laboratory maps these parameter cases and produces independently checkable extremal witnesses.

Only a complete argument licenses:

> The stated bound or conjecture is proved or disproved under the frozen assumptions.

Forbidden before proof:

- Proximity Prize solved;
- asymptotic threshold established;
- general Wall broken.
