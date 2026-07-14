# Query-Separated Successor Subagent Review Reports

Status: `COMPLETE`

All five reviews were read-only. No reviewer was exposed to another report before
completion. No SSH, SCP, ping, target contact, PMU hardware execution, live controller,
or live authority was used.

## 01 Relational-Mechanism Architect - Maxwell

Agent ID: `019f5e39-2440-7ed1-ba8d-7c5697cbc436`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

MATERIAL_FINDINGS:

- `RM-01`: No joint-state law distinguishes one relational process-object from two
  additive branch states sharing addresses.
- `RM-02`: Scalar replay is a question, not a frozen gate. The adversary lacks model
  class, actual-query input, prediction metric, tolerance, held-out geometry, and
  fail condition.
- `RM-03`: Label-name swapping does not detect physical orientation leakage. A rule
  such as `min(R) -> bank A`, `max(R) -> bank B` survives label swap.
- `RM-04`: Fixed-work and schedule-null controls do not define exact marginal traces
  versus changed joint topology. Route, bank, address, timing, and order are not
  mechanically bound.
- `RM-05`: Assertions of no design-level blockers and an empty blocker ledger were
  premature.

COUNTEREXAMPLES:

- Scalar replay with `c_l = u_l(a) + u_l(N-a)` and receiver projection
  `y_q = sum_l(w[q,l] * c_l)`.
- Finite public query family encoded as a complete response vector before closure.
- Value-based lane assignment that passes candidate-label swap.
- Fixed operation counts with relation-correlated bank occupancy, route gain, address
  order, or dwell time.

SURVIVING_ARCHITECTURE:

Candidate B remains viable only as a hypothesis. Source-close/query-later,
capability isolation, feature freeze, exact coverage, R2 controls, custody-last
classification, and no two-replicate Small Wall promotion survive.

REQUIRED_REPAIRS:

- Freeze a joint-interaction observable such as
  `J_q = Y_q(a,b)-Y_q(a,empty)-Y_q(empty,b)+Y_q(empty,empty)`.
- Formalize scalar replay over additive, nonlinear workload, response-table, bank,
  route, timing, and order models, including the realized post-close query.
- Replace label-name swap with blinded physical branch-to-lane permutation.
- Define exact branch-local marginals and pair-level topology mutations.
- Freeze query-space entropy and carrier capacity.
- Implement OS-enforced receiver isolation, exact coverage, and R2 negative controls.

CLAIM_CEILING:

`QUERY_SEPARATED_SHARED_PAIR_TOPOLOGY_DESIGN_CANDIDATE`; no scientific result class
and no established preservation of `OrbitState={a,N-a}`.

## 02 Physical-Carrier And Restoration Adversary - Schrodinger

Agent ID: `019f5e39-38dd-78e1-b411-2154c60842fd`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

MATERIAL_FINDINGS:

- `PCRA-01`: Candidate B names shared Family 10h ownership topology but never defines
  physical state encoding, page/cache-line handoff, state lifetime after source
  closure, preparation map, or nonseparability criterion.
- `PCRA-02`: Query separation does not kill scalar replay or a source-side query
  oracle. Replay must be scored with the post-source query supplied to the adversary.
- `R2-01`: R2 is a checklist, not a frozen equivalence contract. No concrete state
  vector, standardized readout, measurement-disturbance rule, baseline distribution,
  equivalence region, confidence law, or calibrated threshold exists.
- `CTRL-01`: Physical controls are not orthogonal. Relation mutation can change route
  or cache-set conflict; geometry-null can alias lines; carrier-off can remove the
  measured event.
- `CAP-01`: Receiver isolation and coverage remain declarative. Exact equality to a
  generated schedule cannot detect a schedule that omitted required cells.

COUNTEREXAMPLES:

- Source writes a vector containing responses for every public query basis.
- `min(a,N-a)` maps to bank A and `max(a,N-a)` maps to bank B.
- Relation mutations assigned to different cache sets, routes, or probe order.
- Identity geometry aliases one line while the fold pair uses two lines.
- Destructive probe or global flush makes correct, wrong, reordered, and absent
  inverse appear restored.

SURVIVING_ARCHITECTURE:

Shared cache-line/coherence preparation with delayed receiver probing is plausible as
a bounded transduction experiment. Candidate B remains provisional.

REQUIRED_REPAIRS:

- Freeze exact preparation, handoff, query, and restoration maps.
- Require a nonseparability test against `f(a)+g(N-a)` and independent lane outputs.
- Give scalar replay the post-source query and reject precomputed response vectors.
- Freeze crossed relation, label, query, mapping, bank, route, order, geometry, and
  carrier controls.
- Define R2 with repeated measurements, independent probes, forward displacement, and
  specificity gates.
- Freeze receiver capability boundary and mandatory schedule cross-product.

CLAIM_CEILING:

`PROVISIONAL_SHARED_PAIR_TOPOLOGY_CARRIER_HYPOTHESIS`; a first implementation before
repairs may claim at most `QUERY_SEPARATED_SHARED_TOPOLOGY_TRANSDUCTION_CANDIDATE`.

## 03 Causal No-Smuggle Attacker - Kierkegaard

Agent ID: `019f5e39-5461-7bb2-b230-8039f6f22ad4`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

MATERIAL_FINDINGS:

- `CNSA-01`: Finite-query scalar replay remains admissible. A source can compute
  every `f(R,q_i)` before closure and encode the complete answer vector.
- `CNSA-02`: The scalar adversary is not operationally defined. Its inputs include
  the relation itself while the invariant is a public function of relation and query.
- `CNSA-03`: Delayed publication is not proven delayed choice. Query receipts lack
  fresh entropy, independent chooser, monotonic commitment, and pre-measurement window
  sealing.
- `CNSA-04`: Receiver blindness remains policy, not capability. Same-principal
  executables, absolute paths, inherited descriptors, and process metadata remain
  possible.
- `CNSA-05`: Opaque IDs do not close the dictionary when route, bank, topology,
  order, row shape, timing, filenames, or arguments encode condition.
- `CNSA-06`: Exact coverage and executed order are underspecified.
- `CNSA-07`: Work, route, and restoration controls lack discriminating semantics.

COUNTEREXAMPLES:

- Source precomputes `V(R)=[f(R,q_1),...,f(R,q_m)]` and encodes it in paired lanes.
- Random IDs placed on unique `(route, bank, ordinal, query-set)` tuples.
- Relation ID selects a different address-pair graph or thermal block.
- Separate native executable under same UID reads private receipts.
- Query receipt sealed after probing several windows.
- Diagonal subset covers every marginal while omitting interaction cells.
- Coarse R2 metric passes while ownership distribution remains changed.

SURVIVING_ARCHITECTURE:

The scalar-projection retirement and retained transduction claim are correct.
Source closure before a fresh receiver query, label symmetry, relation mutation,
feature freeze, and R0/R2 separation remain necessary.

REQUIRED_REPAIRS:

- Split the adversary into answer-cache and bounded marginal-workload predictors.
- Require independent high-entropy query only after source death and capability
  revocation.
- Prove all-query answer-vector attack cannot fit carrier capacity or lower claim.
- Require distinct receiver principal plus enforced namespace/chroot or equivalent.
- Use high-entropy one-time IDs and full receiver-visible metadata exchangeability.
- Compare canonical full-row tuple multisets and exact executed sequences.
- Define relation mutation as exact incidence-graph intervention.
- Define R2 as powered equivalence over frozen independent observables.

CLAIM_CEILING:

Current ceiling: `DESIGN_PACKAGE_UNDER_REVIEW`; no relational-carrier candidate is
established.

## 04 Experimental-Design And Controls Auditor - Euler

Agent ID: `019f5e39-6881-7561-ae16-34a0fb882e80`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

MATERIAL_FINDINGS:

- `EDCA-01`: Scalar adversary omits the realized post-closure query.
- `EDCA-02`: Selected carrier lacks a formal encoder, query action, fold-odd
  observable, and label-permutation law.
- `EDCA-03`: Metric domains are separated by name, but formulas, denominators,
  equivalence margins, calibration populations, multiplicity handling, thresholds,
  and quadrature coverage remain unfrozen.
- `EDCA-04`: Exact coverage lists marginal sets, not an exact expected tuple multiset
  with cardinalities and multiplicities, and does not bind global executed order.
- `EDCA-05`: R2 restoration lacks a two-sided law: demonstrated forward displacement
  followed by prospective equivalence to baseline.

COUNTEREXAMPLES:

- Balanced table of every `h(R,q)`.
- Actual query supplied at scoring time defeats the current adversary.
- Opaque IDs leak through public topology/mutation IDs or receiver-image dictionaries.
- Runtime regroups complete rows while preserving IDs and stages.
- Incomplete state vector or natural relaxation creates restoration false positive.

SURVIVING_ARCHITECTURE:

V1 ceiling is correct. Source closure, post-closure query, feature freeze, delayed
unblinding, metric-domain separation, estimator-matched near-zero bounds, independent
physical reversal, R0/R4 tiers, and no two-replicate Small Wall promotion survive.

REQUIRED_REPAIRS:

- Freeze `R`, `P(R)`, `Q`, `h(P(R),q)`, label action, relation mutation, and
  expected contrasts.
- Give adversary complete source view and realized query; include lookup-table,
  nonlinear, route, order, and total-work models.
- Require pure-scalar synthetic data to fail and accessor-specific synthetic data to
  pass.
- Freeze machine-readable expected tuple multiset and global event sequence.
- Define every metric equation and threshold provenance.
- Use one authoritative mandatory-gate registry.
- Freeze receiver capability manifest and negative-access matrix.
- Require forward displacement plus confidence-bound R2 equivalence.

CLAIM_CEILING:

`SHARED_PAIR_TOPOLOGY_QUERY_SEPARATED_CARRIER_DESIGN_CANDIDATE`. No scientific
successor class is supportable yet.

## 05 Claim And Small Wall Adjudicator - Mendel

Agent ID: `019f5e39-7c9c-7393-8860-9f82fb9fc747`

Model: `gpt-5.6-sol`

Reasoning effort: `max`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

MATERIAL_FINDINGS:

- The repository still contained a stale two-replicate promotion path in
  `SMALL_WALL_STATE.md`.
- Query separation does not exclude an all-query source oracle.
- Candidate classification is not bound to one exhaustive gate set.
- Causal and prerequisite order conflicted across charter, course correction, and
  roadmap.
- Receiver isolation remained wording-dependent.

COUNTEREXAMPLES:

- Symmetric classical sender encodes every query response into equal-work topology
  coordinates.
- Finite public query family exhaustively precomputed.
- Opaque IDs are useless if assignment map, mutation IDs, or schedule semantics leak.
- Runtime pulls pair mates forward while set coverage passes.
- Same-UID receiver opens sibling receipts or private map.
- Incomplete schedule omits query-off, query-scramble, thermal sham, or wrong-route
  rows.

SURVIVING_ARCHITECTURE:

The design-only boundary survives. Source-closure-before-query, label-swap null,
relation mutation, fixed-work intervention, carrier-off/geometry-null controls,
feature freeze, post-custody classification, R2 restoration contract, and strict
future Small Wall law are sound foundations.

REQUIRED_REPAIRS:

- Replace stale predecessor promotion language in `SMALL_WALL_STATE.md`.
- Create one exhaustive classification truth table and mandatory gate registry.
- Add receiver-generated unpredictable query provenance and all-query-table adversary.
- Freeze abstract preparation operator, query family, observable, old boundary, and
  fixed-work counterfactual.
- Reconcile restoration and collapse chronology.
- Require executable separation and OS-enforced private-data denial.
- Narrow candidate wording to compatibility evidence, not unresolved-state proof.

CLAIM_CEILING:

Current draft: design proposal only. After repair, first implementation may emit at
most route-scoped `QUERY_SEPARATED_RELATIONAL_CARRIER_CANDIDATE`; every first
two-replicate package remains `SMALL_WALL_CROSSED_NOT_PROMOTED`.
