# Mechanism Resolution Subagent Review Reports

Status: `COMPLETE`

All five reviews were read-only. No reviewer was exposed to another report before
completion. No SSH, SCP, ping, target contact, remote-root inspection, PMU hardware
execution, live controller, live executor, or live authority was used.

Note: several reviewers reported that repository startup tooling emitted ignored
Cortex audit records under `LAW/CONTRACTS/_runs`. No mechanism packet or scoped source
file was changed by reviewers.

## 01 Information-Theoretic Identifiability Auditor - Bacon

Agent ID: `019f6299-ae05-7ca1-81b4-c40a7d3b632e`

Requested model: `gpt-5.6-sol`

Requested reasoning effort: `max`

Self-reported model/effort: `OpenAI Codex, GPT-5 family; high reasoning effort`

VERDICT: `NEGATIVE_DISPOSITION_CORRECT`

CONFIDENCE: `HIGH`

RECOMMENDED_DISPOSITION: `QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED`

REPORT:

```text
Identifiability Audit
- Agent role: Information-theoretic identifiability auditor
- Model / effort: OpenAI Codex, GPT-5 family; high reasoning effort
- Verdict: NEGATIVE_DISPOSITION_CORRECT
- Confidence: High
- Recommended disposition: QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
- Scope: Offline, read-only review. No target contact, remote inspection, live executor, or hardware acquisition occurred.

Disposition
The packet correctly closes the adjudication negatively. It does not clear the three blockers: SCALAR-REPLAY-01 remains unresolved, NONSEPARABILITY-01 is mathematical but not physically bound, and PHYSICAL-MECHANISM-01 remains unresolved. That supports the negative token, not a witness freeze.

No witness is frozen. IDENTIFIABILITY_DECISION.md explicitly denies an exact witness; SHARED_PAIR_WITNESS_MECHANISM.md selects "None"; and the normalized blocker ledger remains open.

Findings
- ITA-01 - Correct negative classification. The packet preserves the predecessor's blocked architecture and declines every positive carrier, memory, witness, and Wall-crossing token. "Resolves negatively" must mean the adjudication is settled, not that the blockers passed. Evidence: MECHANISM_FINDINGS_NORMALIZED.json and SUCCESSOR_FINDINGS_NORMALIZED.json.

- ITA-02 - Finite-query theorem is conditionally sound, with a formal defect. For finite enumerable Q, known R, computable h, sufficient carrier capacity, and realized-coordinate scoring, the answer-vector equivalence proof is sound. But Assumption 4 incorrectly joins "capacity is sufficient" with "a smaller bound has not been proven." The latter is an epistemic non-exclusion argument, not a constructive theorem premise. Split the conditional theorem from its non-identification corollary and specify output precision or stochastic response distributions. Evidence: FINITE_QUERY_EQUIVALENCE_THEOREM.md.

- ITA-03 - B_old is not a principled positive exclusion boundary. It is adequate as a negative-audit catalog, which the packet acknowledges, but its nonlinear functions and tables have no shared complexity, precision, or capacity budget. Moreover, B_answer = |Q| * response_bits is a raw-table size, not an information-theoretic lower bound: correlated answers can be compressed, and a public algorithm for h(R,q) can generalize from a compact encoding. A positive capacity argument needs conditional entropy/rate-distortion or an equivalent frozen code-length bound, including public side information and decoder capability. Evidence: OLD_BOUNDARY_MODEL_CLASS.md and CARRIER_CAPACITY_AND_QUERY_SPACE.md.

- ITA-04 - Reference model is illustrative, not independent evidence. The self-test passes, but the disposition and family10h_physical_witness_frozen = False are hard-coded. The bounded-cache result remained True for capacities 0, 1, 2, 4, 100 because the held-out query is never offered to the cache. Route/order and preselection values are diagnostics absent from the expected gate set. Thus the model demonstrates consistency with the negative decision; it does not derive that decision. Evidence: MECHANISM_REFERENCE_MODEL.py and MECHANISM_REFERENCE_TESTS.json.

- ITA-05 - J_q detects interaction, not mechanism identity. The inclusion-exclusion observable exactly rejects additive branch replay. Ordinary nonadditive contention, saturation, cache-set collision, route coupling, probe interference, or order interactions can also produce nonzero J_q. The packet correctly states this claim ceiling, so NONSEPARABILITY-01 remains unresolved physically. Evidence: NONSEPARABILITY_AND_INTERVENTION_LAW.md.

- ITA-06 - Physical and custody attacks remain live. The permutation example compares toy tuples but does not implement preparation equivariance, outcome invariance, or receiver-visible metadata exchangeability. Route/bank/order is printed rather than rejected; query preselection is a supplied Boolean; and no relation-mutation intervention exists. These omissions correctly prevent a witness freeze.

- ITA-07 - State and packet hygiene need repair. Review metadata remains PARENT_PENDING_INDEPENDENT_REVIEW with an empty agent map. SMALL_WALL_STATE.md retains the inconsistent token QUERY_SEPARATED_ORBITSTATE_ARCHITECTURE_BLOCKED while every authoritative predecessor artifact uses QUERY_SEPARATED_ARCHITECTURE_BLOCKED. The packet also contains a generated .pyc; it semantically matches the current source, but should not be part of a review packet.

Attack Results
- Unbounded lookup-table equivalence: Attack succeeds against any positive claim on a closed finite query set.
- Bounded answer cache: Not defeated. The supplied regression is true by construction and provides no measured capacity evidence.
- Additive branch replay: Exact additive replay is rejected by J_q; ordinary nonadditive branch interaction survives.
- Value-derived orientation: Attack remains viable physically; the future blinding law is sensible but unimplemented.
- Route/bank/order artifact: Attack survives and can generate query-dependent or nonadditive contrasts.
- Query preselection: Attack survives; no independent entropy, source-death attestation, commitment, or window seal is implemented.
- Relation-mutation confound: Attack survives because no exact matched intervention or nuisance-equivalence class is frozen.
- Joint-observable false positive: Attack succeeds; nonzero J_q alone proves only nonadditivity.

Minimal Repairs
1. Split the finite-query theorem from the epistemic corollary; add finite-precision, noise, randomized-response, and side-information assumptions.
2. Replace |Q| * response_bits with a defensible lower bound based on conditional rate-distortion/code length, and include compressed or algorithmic replay in B_old.
3. Freeze the exact Family 10h state, query, observable, disturbance, restoration, and matched relation intervention before reconsidering a witness.
4. Add reference regressions where adequate caches pass, inadequate caches fail for the right reason, and ordinary nonadditive route/contention models fail the candidate gate.
5. Add hard gates for blinded assignment plus outcome equivariance, route/bank/order matching, preselection/shared-PRNG/window-peek custody failures, and relation-mutation confounds.
6. Canonicalize the architecture-blocked token, remove generated bytecode from the packet, and populate independent-review metadata only after the required reviews exist.

The offline self-test exited 0; the bytecode/source semantic comparison passed; and scoped Git status was unchanged by this audit.
```

## 02 Physical Coherence-Mechanism Auditor - Dirac

Agent ID: `019f6299-c271-7e93-9e10-48cb94567147`

Requested model: `gpt-5.6-sol`

Requested reasoning effort: `max`

Self-reported model/effort: `GPT-5 Codex / high-effort read-only audit`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

RECOMMENDED_DISPOSITION: `QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED`

REPORT:

```text
Audit Decision

Agent role: Physical coherence-mechanism auditor
Model / effort: GPT-5 Codex / high-effort read-only audit
Verdict: MATERIAL_BLOCKER
Confidence: HIGH
Exact disposition: QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
Physical witness frozen: No
Evidence state: Working tree atop da147ec12c378a383be2349c01eeb92c4d29f644; mechanism packet is currently untracked

QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED is the correct disposition because no executable or falsifiable physical protocol exists. QUERY_SEPARATED_RELATIONAL_CARRIER_NOT_ESTABLISHED is a future experimental result class, not appropriate before the witness mechanism is frozen.

Material Findings
1. PHYSICAL-MECHANISM-01 - Critical. No exact Family 10h witness exists. The required page, line, set/bank, core, coherence-state, lifetime, handoff, query, measurement, and restoration fields are listed but never instantiated; the selected mechanism is explicitly "None." Evidence: SHARED_PAIR_WITNESS_MECHANISM.md.

2. SCALAR-REPLAY-01 - Critical. A finite public query family remains exactly equivalent to a source-authored answer vector. C_prep is unmeasured, so even a bounded cache large enough for the actual query family remains admissible. Evidence: FINITE_QUERY_EQUIVALENCE_THEOREM.md and CARRIER_CAPACITY_AND_QUERY_SPACE.md.

3. NONSEPARABILITY-01 - Critical. J_q rejects a strictly additive formula synthetically, but it is not bound to a Family 10h state. A nonzero value can arise from ordinary cache conflict, coherence contention, route saturation, nonlinear timing, aliasing, probe disturbance, or another pair interaction. Evidence: NONSEPARABILITY_AND_INTERVENTION_LAW.md.

4. PHYSICAL-CONTROL-01 - Critical. No realizable relation mutation changes only joint incidence while preserving pages, banks, work, timing, routes, order, and marginal occupancy. Consequently route/bank/order explanations cannot be separated from the claimed relation. Evidence: NONSEPARABILITY_AND_INTERVENTION_LAW.md and CONTROL_AND_KILL_MATRIX.md.

5. VALUE-ORIENTATION-01 - High. The packet correctly identifies the min/max attack, but no physical P(R,pi) implements blinded lane assignment or proves route/bank/address exchangeability. Evidence: VALUE_ORIENTATION_SYMMETRY_LAW.md.

6. QUERY-CHOOSER-01 - High. Query custody is a prospective law only. There is no implemented independent entropy source, monotonic source-death commitment, capability revocation, PRNG separation, or premeasurement seal. Evidence: QUERY_CHOOSER_CUSTODY.md.

7. RESTORATION-R2-01 - Critical. The earlier Family 10h operator establishes transition-count observability and byte preservation, not R2 state restoration. Its digest return is R0; there is no frozen physical state vector, disturbance law, equivalence region, or negative-control suite. Evidence: SMALL_WALL_STATE.md and RESTORATION_TIER_CONTRACT.md.

8. AUDIT-REFMODEL-01 - High. The self-test is internally consistent but non-evidentiary. Family 10h freeze is hard-coded false; bounded-cache failure follows from an arbitrary capacity=2; the route artifact is printed but not rejected; query custody is two booleans; relation mutation and R2 are absent. Evidence: MECHANISM_REFERENCE_MODEL.py.

9. Carried blockers remain material. CAPABILITY-BOUNDARY-01, EXACT-COVERAGE-01, GATE-REGISTRY-01, and METRIC-DOMAINS-01 remain open alongside the mechanism blockers. Evidence: MECHANISM_FINDINGS_NORMALIZED.json.

10. AUDIT-CUSTODY-01 - Provenance limitation. The mechanism directory is untracked and SMALL_WALL_STATE.md is modified. This audit therefore binds the current working-tree contents, not a commit-frozen packet identity.

Attempted Attacks
- Unbounded lookup-table equivalence: Survives definitively. No finite closed observation can reject it.
- Bounded answer cache: Survives. The physical capacity and actual answer-bit requirement are unknown; synthetic k=2 proves nothing about Family 10h.
- Additive branch replay: Only synthetically rejected. Physical matched single-branch/empty controls and topology are absent.
- Value-derived orientation: Survives. No blinded physical branch-to-lane map exists.
- Route/bank/order artifact: Survives. The reference model demonstrates an example but contains no rejection gate.
- Query preselection: Survives. Custody requirements are specified but not instantiated or tested.
- Relation-mutation confound: Survives. No orthogonal physical mutation has been defined.
- Joint-observable false positive: Survives. J_q != 0 proves nonadditivity only, not unresolved relational memory.

Physical Sufficiency
- Physical pages: No. No exact addresses, page identities, allocation/pinning law, or relation encoding.
- Cache lines/sets: No. "Paired cache-line sets" is a sketch; the earlier 4096-line calibration is not a successor relation map.
- Banks/routes/order: No. No independently measured bank assignment or crossed route/order equivalence.
- Cores: No. Prior cores 4/5 belong to a calibration operator, not a frozen preparation/query/handoff protocol.
- Ownership states: No. PMU transition counts are not a named persistent MOESI/coherence state after source death.
- Source death/handoff: No. Prospective rules exist; no concrete process, descriptor, IPC, memory, or capability closure map exists.
- Query map: No. Only example operator classes are supplied; no exact instructions, lines, order, or windows.
- Restoration map: No. R2 remains a checklist; byte-digest return is insufficient.

Minimal Repairs And Regressions
1. Freeze one implementable R -> P(R) -> Q(q) -> Y -> Restoration map with exact pages, line offsets, sets/banks, cores, routes, order, ownership transitions, state lifetime, source-death handoff, query instructions, windows, and probe disturbance.
2. Measure C_prep and prove B_answer > C_prep >= B_relation; otherwise limit the claim explicitly to a bounded adversary and require held-out relation/query/seed/session cross-products.
3. Freeze blinded physical lane assignment and an exact tuple multiset crossing relation, labels, query, mapping, address, bank, route, order, topology, and controls.
4. Bind J_q physically and regress shared versus separated topology, singles/empty cells, cross-pairs, geometry-null, carrier-off, aliases, cache conflicts, routes, ordering, and ordinary nonlinear predictors.
5. Implement genuine post-death query custody with independent entropy and regress preselection, shared PRNG, source-visible seed, post-observation window choice, and incomplete capability revocation.
6. Define R2 with forward displacement, independent standardized readout, baseline distribution, equivalence margins, and no-restoration, natural-relaxation, destructive-reset, wrong-inverse, reordered-inverse, and carrier-off failures.

The offline self-test exited 0 with no expected mismatches: additive J=[0,0,0,0], synthetic J=[-6,5,16,27], unbounded lookup indistinguishable, and family10h_physical_witness_frozen=false. That verifies the packet's negative synthetic logic only.

No target, network, SSH/SCP/ping, PMU, hardware, environment-variable, or live-executor action was performed. No scoped source file was edited. The required local AGS connection check did emit its normal ignored _runs audit records.
```

## 03 Relational Nonseparability Adversary - Hegel

Agent ID: `019f6299-e65e-7da0-8abc-684ba8b16711`

Requested model: `gpt-5.6-sol`

Requested reasoning effort: `max`

Self-reported model/effort: `GPT-5-based Codex / high`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

RECOMMENDED_DISPOSITION: `QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED`

REPORT:

```text
Relational Nonseparability Adversarial Review

Agent role: Relational nonseparability adversary
Model / effort: GPT-5-based Codex / high
Verdict on proposed witness: MATERIAL_BLOCKER
Assessment of packet conclusion: SUPPORTED
Confidence: HIGH
Recommended exact disposition: QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED

Adjudication
The packet's negative disposition is correct. J_q is a mixed second difference. Conditional on comparable factorial arms and a common measurement scale, J_q != 0 rejects only the strict observed-response model Y_q(a,b)=F_q(a)+G_q(b)+C_q. It does not establish nonseparability of the underlying physical state, persistent relational memory, OrbitState access, or a Small Wall crossing.

The synthetic reference model validates this narrow algebra. Its positive mechanism is nevertheless a deterministic function of public (a,b,q) and is reproduced exactly by its own finite answer cache. A J_q-positive response can therefore remain completely old-boundary equivalent.

Attack Ledger
- Unbounded lookup-table equivalence: Survives. No finite closed query set can distinguish an arbitrary (R,q) table.
- Bounded answer-cache: Survives physically. The toy cache is defeated only by stipulated k=2 and an unstored query; no physical preparation capacity is measured.
- Additive branch replay: Killed only in the narrow toy. Strict addition gives J_q=0; context-dependent or nonlinear ordinary interactions survive.
- Value-derived orientation: Survives. The regression tests one hard-coded trace pattern, not physical metadata exchangeability or response invariance.
- Route/bank/order artifact: Survives and is not gated. The model prints artifact values but never subjects them to J_q or an expected failure assertion.
- Query preselection: Survives. Two caller-supplied booleans cannot distinguish fresh choice from preselection followed by delayed publication.
- Relation-mutation confound: Survives. Marginal population equality does not establish joint physical-trace equivalence.
- Joint-observable false positive: Survives. Ordinary contention, cache conflicts, saturation, nonlinear readout, or mismatched arms can all produce nonzero J_q.

Findings
1. RNA-01 - Closed-set and algorithmic replay remain admissible. The finite-query theorem is valid, but the risk is broader than lookup storage: B_old receives relation metadata, source binaries, and realized q, while the synthetic h(R,q) is cheaply computable. Held-out queries reject memorization only, not an algorithmic predictor. Evidence: FINITE_QUERY_EQUIVALENCE_THEOREM.md and OLD_BOUNDARY_MODEL_CLASS.md.

2. RNA-02 - The bounded-cache result is a stipulated unit test, not capacity separation. The test stores the first two of four queries, chooses query 7, and passes an OR of "not stored" and "query count exceeds capacity." It measures neither bits per response nor compression, precision, or C_prep. Evidence: MECHANISM_REFERENCE_MODEL.py and CARRIER_CAPACITY_AND_QUERY_SPACE.md.

3. RNA-03 - J_q has only an observed-interaction claim ceiling. The hand-injected joint_term passes J_q while the finite cache reproduces the same output exactly. It is not accessor-specific synthetic evidence. Evidence: MECHANISM_REFERENCE_MODEL.py and NONSEPARABILITY_AND_INTERVENTION_LAW.md.

4. RNA-04 - The four physical factorial arms are undefined. In particular, empty has no physical definition. Removing a branch can change operation count, occupancy, topology, timing, route, and probe disturbance, making J_q a contrast between different experiments. Evidence: NONSEPARABILITY_AND_INTERVENTION_LAW.md and SHARED_PAIR_WITNESS_MECHANISM.md.

5. RNA-05 - Metadata interactions are demonstrated but not attacked. route_bank_order_artifact() is output as a dictionary only. It has no single-branch arms and no pass/fail assertion. An ordinary term such as K_q(bank_pair, route, order) produces nonzero J_q. Evidence: MECHANISM_REFERENCE_MODEL.py.

6. RNA-06 - Relation mutation is not a separating intervention. Equal page, bank, route, order, and occupancy marginals do not equalize pairwise incidence or temporal traces. The source also sees R and the mutation ID, so an ordinary keyed table or topology can follow the mutation. Evidence: NONSEPARABILITY_AND_INTERVENTION_LAW.md and PHYSICAL_CARRIER_ARCHITECTURES.md.

7. RNA-07 - The orientation regression does not implement the stated symmetry law. It checks that a bad tuple ignores pi and a good tuple changes under pi; it defines no lane-relabeling equivalence operator, hidden-custody proof, receiver-visible metadata test, or Y_q/J_q invariance test. Evidence: VALUE_ORIENTATION_SYMMETRY_LAW.md and MECHANISM_REFERENCE_MODEL.py.

8. RNA-08 - Query custody is documentary, not mechanically tested. The reference predicate cannot detect shared PRNG state, forged chronology, an alive source, post-observation window selection, or delayed publication of a preselected query. Evidence: QUERY_CHOOSER_CUSTODY.md and MECHANISM_REFERENCE_MODEL.py.

9. RNA-09 - Physical binding is absent. No persistent Family 10h state, page/cache-line/set/bank/core/coherence map, source-death handoff, lifetime, disturbance law, physical mutation, or R2 restoration map is frozen. Evidence: MECHANISM_RESOLUTION_REVIEW.md and IDENTIFIABILITY_DECISION.md.

Required Answers
- Does J_q prove only nonadditivity or more? Only observed-scale nonadditivity, conditional on valid factorial-arm construction. It does not prove physical-state nonseparability.
- Are additive controls sufficient synthetically? Only to unit-test exact additive cancellation. They are insufficient as an identifiability or ordinary-explanation suite.
- Is physical binding missing? Yes, completely enough to block protocol freeze and every positive mechanism token.

Minimal Repairs And Regressions
1. Freeze exact physical implementations of all four J_q arms, including matched sham definitions for empty, exact pages/lines/banks/cores/states, operation counts, timing, handoff, query, disturbance, and restoration.
2. Freeze the estimand on a declared measurement scale with arm randomization, block structure, uncertainty, equivalence bounds, multiplicity, and a nonlinear-readout regression.
3. Implement relation mutation as a blinded graph edge-swap over fixed branch values and physical vertices; bind exact pairwise traces, not merely marginals, and predeclare the signed Delta J_q.
4. Separate held-out-cache failure from capacity separation. Independently measure C_prep in bits, including response precision and compression, and include algorithmic (R,q) predictors.
5. Convert route/bank/order into hard negative regressions. Add shared-resource contention, cache conflict, saturation, thermal interaction, destructive probe, and context-dependent branch effects.
6. Replace the orientation Boolean with lane-trace equivariance under secret pi, Y_q/J_q invariance after canonical relabeling, and complete receiver-visible metadata exchangeability.
7. Add executable custody negatives for preselection plus delayed publication, shared seed, source still alive, forged closure order, and post-measurement operator/window choice.
8. Require shared-topology positive, separated-topology null, cross-pair null, carrier-off null, geometry-null, additive synthetic null, metadata-model rejection, and R2 forward-displacement/restoration gates in one classifier.

Reference Check
The local no-bytecode self-test passed its declared expectations: additive J_q=[0,0,0,0], injected synthetic J_q=[-6,5,16,27], route artifacts [52,-5,3,11], and no expected mismatches. This confirms implementation consistency only; the route artifact is not gated and the positive interaction remains exactly cacheable.

Scope
No target contact, network operation, SSH/SCP/ping, remote inspection, PMU acquisition, environment mutation, or live executor was used. No packet or tracked source file was changed; pre-run and post-run Git status were identical.

Repository startup tooling automatically emitted ignored Cortex audit/intent records under LAW/CONTRACTS/_runs. This was an unintended filesystem-write exception to the requested read-only boundary.
```

## 04 Causal Query / No-Smuggle Auditor - Lovelace

Agent ID: `019f6299-fa92-7603-aa25-9e474202006d`

Requested model: `gpt-5.6-sol`

Requested reasoning effort: `max`

Self-reported model/effort: `GPT-5 Codex / high`

VERDICT: `MATERIAL_BLOCKER`

CONFIDENCE: `HIGH`

RECOMMENDED_DISPOSITION: `QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED`

REPORT:

```text
Causal Query / No-Smuggle Audit

Agent role: Causal query/no-smuggle auditor
Model / effort: GPT-5 Codex / high
Audit basis: da147ec12c378a383be2349c01eeb92c4d29f644 plus the current uncommitted SMALL_WALL_STATE.md update
Verification: All 13 mechanism-resolution files read; relevant predecessor contracts and normalized findings read; offline reference self-test exited 0 with passed: true.

Verdict
Verdict: MATERIAL_BLOCKER for mechanism freeze, candidate classification, or Small Wall promotion.
Confidence: HIGH
Recommended disposition token: QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED

The packet correctly states the conceptual separation: delayed unpredictable choice is a custody property, while distinguishing a relational carrier from a source-authored answer mechanism is an identifiability problem. Its negative disposition is therefore sound. The written laws and toy reference model are not yet sufficient operational gates.

Custody Versus Identifiability
Source death plus high query entropy alone is not sufficient custody. The full sequence in QUERY_CHOOSER_CUSTODY.md is a sound specification only if closure covers every source-capable principal, entropy is unpredictable conditional on the complete source view, and query/operator/order/windows are mechanically committed before any carrier-dependent observation. No implementation or capability manifest proves those conditions yet.

Even perfect custody does not defeat answer-cache identifiability. A source that knows finite public Q can prepare every answer without knowing the realized query, as established in FINITE_QUERY_EQUIVALENCE_THEOREM.md.

Findings
CQNS-01 - Separation is correct in prose, not mechanically typed. QUERY_CHOOSER_CUSTODY explicitly says custody does not defeat a finite cache, while NO_SMUGGLE_CAPABILITY_MODEL.md requires capacity and held-out gates. Machine-readable findings and tests do not assign separate custody-invalid versus identifiability-not-established failure domains.

CQNS-02 - The bounded-cache model is materially underpowered. B_answer = |Q| * response_bits is an uncompressed-table size, not a lower bound for answer-equivalent encodings. A low-rank basis, formula, circuit, seed, compressed vector, or ordinary scalar encoding of R can answer held-out queries within much less space. CARRIER_CAPACITY_AND_QUERY_SPACE.md correctly reports that no C_prep is measured, but its proposed inequality would reject only literal uncompressed caches.

CQNS-03 - Query preselection is rejected by law but not by the reference gate. The reference function accepts two booleans. It cannot distinguish fresh choice from a preselected query published later, shared entropy, replayed closure, source helpers, or post-peek window selection. Its result is also absent from the expected hard-gate set.

CQNS-04 - J_q rejects exact additivity but admits ordinary nonlinear false positives. The additive cancellation is valid. However, cache contention, queue saturation, alias collision, shared-bank interference, or another conventional pair cross-term produces J_q != 0 without unresolved relational memory. The code passes a prospective witness when any query is nonzero. The prose correctly limits this to nonadditivity.

CQNS-05 - Route/bank/order and relation-mutation attacks remain live. The model merely emits a route/bank/order example; it does not gate it. No reference test implements relation mutation. The written law itself concedes that no physically realizable mutation preserves the ordinary variables while changing only joint incidence.

CQNS-06 - Value-orientation law is correct; its synthetic test is incomplete. The law correctly rejects min/max physical orientation and requires blinded branch-to-lane assignment. The test only compares two tuples; it does not enforce P, Y_q, and J_q equivalence or receiver-visible metadata exchangeability.

CQNS-07 - The normalized enforcement surface is incomplete. MECHANISM_FINDINGS_NORMALIZED.json defines only three root blockers but lists seven additional unresolved IDs without local definitions. Query-preselection and route-artifact outputs are not expected self-test keys. A passing toy self-test therefore does not mean the causal/no-smuggle attacks were gated.

Attack Results
- Unbounded lookup-table equivalence: Succeeds and is correctly admitted. Finite closed evidence cannot distinguish it.
- Bounded answer-cache: Succeeds against the current physical packet. The toy k-entry cache fails, but C_prep is unmeasured and compressed/programmatic replay is outside the implemented gate.
- Additive branch replay: Defeated only for exact additive form. J_q cancels arbitrary F_q(a)+G_q(b)+C_q.
- Value-derived orientation: Law catches it; current architecture does not pass the law. End-to-end invariance is untested.
- Route/bank/order artifact: Succeeds. Present as diagnostic output, not a hard rejection.
- Query preselection: Succeeds against the reference logic. A preselected query can be labeled generated-after-close.
- Relation-mutation confound: Succeeds. No exact physical intervention or crossed trace-equivalence gate exists.
- Joint-observable false positive: Succeeds. Ordinary nonlinear shared-resource interaction can yield nonzero J_q.

Minimal Repairs And Regressions
1. Freeze an executable custody verifier covering all source-capable principals, closure monotonicity, chooser identity and binary, conditional query min-entropy, seed independence, anti-replay, and commitment of query/operator/order/windows before first observation.
2. Add hard regressions for preselection, delayed publication, shared PRNG, source helper survival, inherited descriptors, source-visible seed, stale closure receipt, and post-measurement window choice.
3. Replace the raw-table capacity argument with a frozen state/program/compute-bounded adversary that includes compressed tables, formulas, low-rank predictors, circuits, seeds, and ordinary encodings of R. Keep arbitrary unbounded lookup explicitly indistinguishable.
4. Separate held-out-generalization and capacity regressions. Do not combine them with or; use total query space, response precision, compression allowance, measured C_prep, optimal cache strategy, and a frozen query distribution.
5. Freeze the exact physical R -> P(R) -> Q(q) -> Y -> Restoration map and an incidence mutation with per-run matching of address, bank, route, order, timing, occupancy, and workload traces.
6. Expand the joint gate with conventional nonlinear contention, collision, saturation, alias, and shared-resource models; require predeclared query-wise effects and all separated-topology, carrier-off, geometry-null, blinded-orientation, and mutation controls.
7. Make every causal and identifiability result a hard expected test with a typed failure domain and exact token. Resolve every inherited blocker ID in the normalized registry.

Final Disposition
Retain QUERY_SEPARATED_ARCHITECTURE_BLOCKED and SMALL_WALL_CROSSED_NOT_PROMOTED as separate state. Emit exactly QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED.

Do not emit a witness-frozen, relational-carrier-candidate, physical-relational-memory, or Small Wall crossing token.
```

## 05 Claim-Boundary Adjudicator - Einstein

Agent ID: `019f629a-0ecb-7452-af72-6c5ff45b8c07`

Requested model: `gpt-5.6-sol`

Requested reasoning effort: `max`

Self-reported model/effort: `GPT-5 Codex / high`

VERDICT: `DECISION_UPHELD_STATE_REPAIR_REQUIRED`

CONFIDENCE: `HIGH`

RECOMMENDED_DISPOSITION: `QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED`

REPORT:

```text
Findings
1. CB-STATE-01, material: exact architecture token is not preserved in the active state header. The decision explicitly retains QUERY_SEPARATED_ARCHITECTURE_BLOCKED, but the current phase replaces it and the status list uses the unique alias QUERY_SEPARATED_ORBITSTATE_ARCHITECTURE_BLOCKED. The exact token survives only in later narrative and the claim ceiling. This is semantic preservation, not mechanical preservation.

2. CB-STATE-02, material wording: "resolves the three root blockers negatively" contradicts the normalized ledger. The ledger says SCALAR-REPLAY-01 and PHYSICAL-MECHANISM-01 are unresolved and NONSEPARABILITY-01 is only partially formalized. Downstream automation could misread "resolves" as blocker closure.

3. CB-CAPACITY-01, prospective material blocker: B_answer = |Q| * response_bits is not a sufficient answer-cache bound. A compact program, coefficients, seed, or encoded relation can generate all answers using much less than the raw table size. The reference response law itself is such a compact generator. Capacity separation must cover the minimum admissible ordinary encoding and every preparation channel, not only an uncompressed table.

4. CB-JOINT-01, prospective material blocker: J_q establishes nonadditivity only. For separate latent branches with nonlinear readout Y=(F+G)^2, the proposed observable gives J=2FG, despite no unresolved relational state. Cache contention, detector saturation, coherence interaction, or probe disturbance can produce the same false positive. The packet correctly states this limitation, so it does not invalidate the negative decision.

5. CB-REF-01: the passing self-test is illustrative, not an attack-kill regression suite. The bounded cache fails by constructed capacity/query choices; the orientation result compares two toy traces; route/bank/order is emitted only as an example; query custody is reduced to two booleans.

Adjudication
- Role: Claim-boundary adjudicator
- Model / effort: GPT-5 Codex / high
- Verdict: Decision upheld; state update requires material repair before freeze.
- Confidence: High
- Exact recommended disposition token: QUERY_SEPARATED_IDENTIFIABILITY_NOT_RESOLVED
- Overall architecture disposition: QUERY_SEPARATED_ARCHITECTURE_BLOCKED
- Small Wall state: SMALL_WALL_CROSSED_NOT_PROMOTED

Attempted Attacks
- Unbounded lookup-table equivalence: Succeeds; the packet correctly recognizes closed finite-set indistinguishability.
- Bounded answer cache: Not defeated by present evidence; compressed-generator variants also defeat the proposed raw-bit law.
- Additive branch replay: Strict additive replay is killed by J_q; nonlinear readout of additive latent branches survives.
- Value-derived orientation: Survives the current architecture; blinded physical assignment exists only as a future law.
- Route/bank/order artifact: Survives; the model produces an example but applies no rejecting gate.
- Query preselection: Survives; chooser provenance is specified but not implemented or evidenced.
- Relation-mutation confound: Survives; no realizable mutation preserves all ordinary variables while changing only incidence.
- Joint-observable false positive: Survives; nonzero J_q is not relational-memory or OrbitState evidence.

Claim Ceiling
No Small Wall or scientific relational-carrier class is newly overclaimed. The decision expressly excludes QUERY_SEPARATED_RELATIONAL_CARRIER_CANDIDATE, PHYSICAL_RELATIONAL_MEMORY_ESTABLISHED, and SMALL_WALL_CROSSED; the parent law likewise says no scientific successor class is currently supportable.

The names joint_relational_mechanism and "synthetic joint witness" are semantically stronger than the example proves. Rename them as a synthetic nonadditive pair mechanism, but no emitted class currently depends on that wording.

Minimal Repairs
1. Keep the exact QUERY_SEPARATED_ARCHITECTURE_BLOCKED token active and add identifiability as a subordinate status. Normalize the extra-ORBITSTATE architecture alias and the shortened gain-covariant alias.
2. Replace "resolves the three root blockers negatively" with "adjudicates the three blockers and leaves them unresolved for the current access model."
3. Add compressed answer-generator and all-channel preparation-capacity regressions.
4. Require ordinary nonlinear pair-interaction and measurement-nonlinearity models to fail before any relational-carrier candidate.
5. Replace toy orientation/query/route examples with asserted physical-trace, chooser-provenance, crossed-metadata, and relation-mutation regressions.
6. Exclude the generated __pycache__ artifact from any frozen evidentiary packet.

The offline self-test exited 0 with no expected mismatches. No target, network, PMU, hardware, remote-root, or live-executor operation occurred, and no scoped artifact was modified. Repository startup did auto-emit its ignored Cortex audit records outside the packet.
```
