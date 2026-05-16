# Meta-Logic Spine: Common Sense Engine for a Local Agent

The common sense of a thing is not what it knows but what it defaults to
when it does not know enough to be certain. This document defines the
meta-inference layer that produces those defaults. It is not domain
content (no physics, no social rules). It is the engine that *generates*
plausible defaults, revises them under surprise, decides which patterns
are worth generalising, and chooses among competing explanations.

---

## 0. Core Distinction: Common Sense vs. Logic vs. Spine

### 0.1 What Common Sense Is (Operationally)

Common sense is **defeasible induction over a projectibility-constrained
predicate language, guided by simplicity, graded by uncertainty,
stabilized by belief revision, and evaluated by global coherence**.

Each term is load-bearing:

| Term | What it does operationally | Failure mode if absent |
|------|---------------------------|----------------------|
| **Defeasible** | Conclusions can be retracted when new information arrives | Brittleness: every exception breaks the system |
| **Induction** | Patterns observed in the past are assumed to hold for unseen cases | No generalization: every situation is novel |
| **Projectibility-constrained** | Not every predicate is eligible to appear in generalizations | "Grue" paradox: spurious patterns treated as laws |
| **Guided by simplicity** | Among equally-fitting generalizations, prefer the shortest | Overfitting: memorizing noise as signal |
| **Graded by uncertainty** | Every belief carries a confidence score, not just true/false | Binary collapse: can't distinguish "likely" from "certain" |
| **Stabilized by belief revision** | When a contradiction appears, the system repairs itself surgically | Catastrophic collapse: one surprise invalidates everything |
| **Evaluated by global coherence** | Beliefs must fit together; local plausibility is not enough | Fragmentation: contradictory beliefs in different contexts |

Common sense is what you use when deduction runs out of premises and
statistics runs out of samples. It is the **default policy** for an
agent that must act under partial information.

### 0.2 What Logic Is (Operationally)

Logic is **validity-preserving consequence**. Given premises P and a
conclusion C, logic answers: "Does P force C in every model where P holds?"

Logic provides:
- **Deduction**: P entails C. If premises are true, conclusion must be true.
- **Model theory**: truth-in-a-structure semantics.
- **Proof theory**: syntactic derivation rules.
- **Metalogic**: soundness, completeness, compactness, decidability.

Logic is **monotonic**: adding premises never removes conclusions. This
is its strength (predictability) and its weakness (inability to handle
surprise). Logic does not tell you which premises to hold; it only tells
you what follows from the premises you already hold.

### 0.3 Why Logic Alone Fails for Common Sense

| Scenario | What logic does | What common sense does |
|----------|----------------|----------------------|
| "Birds fly" | Cannot conclude anything about Tweety without an explicit premise about Tweety | Defaults to "Tweety flies" until given reason to retract |
| "Tweety is a penguin" | Now has two premises: bird→flies and penguin→not-flies. Contradiction. | Retracts the general default because the specific case overrides |
| "Penguins are birds that swim" | Must be explicitly encoded as a new axiom | Generalizes from a few examples if the predicate passes projectibility checks |
| "Why did the glass break?" | Cannot generate candidate explanations | Abduces: impact, thermal shock, manufacturing defect; scores them by coherence |

Logic is the **safety net** under common sense, not its replacement.
Common sense proposes; logic checks for consistency. When they conflict,
belief revision repairs the common sense, not the logic.

### 0.4 What the Spine Is (and Is Not)

The spine is the **meta-inference layer** that sits between raw observations
and the agent's actions. It does not store facts about the world. It stores:

- **Rules about rules**: how to create, modify, and retire defaults.
- **Operators that consume beliefs and produce revised beliefs**.
- **Constraints on predicate formation**: which predicates are eligible for induction.
- **Scoring functions**: which explanations, defaults, and generalizations are preferred.

It answers questions like:
- *Why* is "birds fly" a reasonable belief? (Projectibility + defeasibility)
- *When* should I stop believing it? (Non-monotonicity + belief revision)
- *How* do I choose between two plausible stories? (Explanatory coherence)

The spine is not Bayes' theorem written as prose. Bayes is one scoring
policy among several. The spine defines the **interface** that any
scoring policy must implement, and the **conditions** under which one
policy is preferred over another.

---

## 1. The Master Graph: 11 Nodes, 3 Load-Bearing Loops

### 1.1 Node Definitions (Expanded)

Each node is a function from belief-state to belief-state, with defined
inputs, outputs, and boundary conditions.

#### A. Representation & Predicate Governance

**What it decides**: Which predicates are admissible in the language.
**Input**: Raw feature observations, existing predicate vocabulary.
**Output**: A typed predicate language with admissibility constraints.
**Why it matters**: The choice of predicates determines what can be
thought. If "grue" is an admissible predicate, the system will
overfit. If "electron" is inadmissible, the system cannot learn physics.
**Key tension**: Expressiveness vs. projectibility. More predicates =
more things you can say, but fewer things you should generalize.
**Dependencies**: Feeds E (induction eligibility). Consumes from G
(simplicity constrains predicate complexity). Consumes from J
(bounded agents cannot entertain all possible predicates).

#### B. Inference Operators

**What it decides**: How to move from premises to conclusions.
**Input**: Belief set K, query Q.
**Output**: Deductive closure, inductive generalization, abductive hypothesis,
or analogical mapping.
**Operators defined**:
- **Deduction**: K |- Q. Monotonic, truth-preserving.
- **Induction**: From P(a1), Q(a1), P(a2), Q(a2), ... infer "All P are Q"
  subject to projectibility gate (node E).
- **Abduction**: From surprising observation O, generate candidate causes
  H such that H would make O unsurprising. Feeds node I.
- **Analogy**: From "A is to B as C is to ?", infer D by structural
  alignment over the predicate graph.
- **Causal inference**: From interventions and counterfactuals, infer
  causal direction. Requires a causal graph (stored as `hard_constraint`
  entries with `type: causal`).
**Dependencies**: Consumes from A (must use admissible predicates).
Feeds C (defaults are a form of induction). Feeds I (abduction seeds hypotheses).

#### C. Defeasible Reasoning (Defaults)

**What it decides**: What to believe "normally, unless."
**Input**: Fact-set F, default rule base D.
**Output**: Default closure C(F, D) — the maximal consistent extension
of F by applicable defaults.
**Core mechanism**: Specificity ordering (Section 4A, D1) + priority
lattice (D4). When defaults conflict, the more specific antecedent wins.
When equally specific, priority and confidence break the tie. When all
tiebreakers are exhausted, skeptical semantics: believe neither.
**Why "normally/unless"?** Because the world is full of regularities
that have exceptions not yet enumerated. You want the regularity to fire
*now* and retract *later* when the exception materializes.
**Dependencies**: Consumes from B (defaults are induced or authored).
Feeds D (non-monotonicity is the formal property; defeasibility is the
implementation). Feeds H (when a default is retracted, belief revision
must clean up its consequences).

#### D. Non-Monotonicity

**What it decides**: That adding information can remove conclusions.
**Input**: Belief set K, new fact f.
**Output**: K' = K U {f} with all conclusions that depended on assumptions
now contradicted by f removed.
**Formal property**: A logic is non-monotonic iff there exist K, f, p
such that K |- p but K U {f} |-/- p.
**Implementation**: Truth maintenance. Every derived conclusion records
its justifications (the defaults and facts it depends on). When a
justification is retracted, the conclusion is retracted unless it has
alternative justification.
**Circumscription** (McCarthy): Minimize the extension of "abnormal"
predicates. "Birds fly" becomes "Birds fly unless they are abnormal,
and assume as few things are abnormal as necessary to maintain consistency."
**Dependencies**: Consumes from C (defaults create non-monotonic commitments).
Feeds H (retraction triggers belief revision).

#### E. Induction & Projectibility

**What it decides**: Which patterns are worth generalizing.
**Input**: Observed co-occurrences, predicate vocabulary from A.
**Output**: Candidate default rules, each with a projectibility score.
**Core problem**: For any finite set of observations, infinitely many
predicates fit the data equally well. "Green" and "grue" both fit all
observed emeralds. The projectibility gate (Section 4C, C3) filters out
gerrymandered predicates before they become defaults.
**Key insight**: Induction is not a logical operation. It is a **biased
search** over predicate space, guided by entrenchment (good predicates
have worked before), simplicity (short predicates over natural kinds),
and causal structure (predicates that participate in causal laws).
**Dependencies**: Consumes from A (predicate admissibility). Feeds G
(simplicity selects among competing inductions). Feeds C (successful
inductions become defaults). Closes the Projectibility Loop.

#### F. Uncertainty & Degrees of Belief

**What it decides**: How confident the system is in each belief.
**Input**: Belief set K with supporting evidence.
**Output**: Credence c(p) in [0,1] for each belief p in K.
**Representations available**:
- **Point-valued probability**: c(p) = 0.7. Simple, but hides precision.
- **Interval probability**: c(p) = [0.6, 0.8]. Distinguishes uncertainty
  from ignorance.
- **Ranking theory** (Spohn): k(p) = integer rank of disbelief. 0 = believed,
  1 = disbelieved with strength 1, etc.
- **Dempster-Shafer**: belief(p) and plausibility(p) as lower/upper bounds.
**Which representation when?** The spine does not commit to one. It stores
a `confidence` field (Section 2) and leaves the interpretation to the
scoring policy. The resolver uses confidence as a tiebreak (Section 4A, D4),
which works for any representation that provides a total order.
**Calibration**: Confidence should track empirical frequency. If the system
assigns c=0.8 to 100 predictions, roughly 80 should be correct. Miscalibration
is detected by comparing predicted vs. observed error rates.
**Dependencies**: Consumes from B (inference produces beliefs with evidence
weights). Consumes from H (revision adjusts confidence down for contradicted
beliefs). Feeds G (simplicity can be traded against confidence).

#### G. Preference, Simplicity & Compression

**What it decides**: Among equally-fitting beliefs, which is preferred.
**Input**: Set of candidate beliefs/hypotheses, each with fit-to-data score.
**Output**: Total ordering by preference.
**Simplicity measures**:
- **Occam's razor**: Prefer fewer entities. Operationalized as predicate count.
- **Minimum Description Length** (MDL): Prefer the hypothesis H that minimizes
  L(H) + L(D|H), where L is description length in bits. MDL is equivalent to
  Bayesian inference with a Solomonoff prior (K-complexity).
- **Structural simplicity**: Prefer hypotheses with fewer free parameters,
  fewer ad-hoc clauses, shallower nesting of exceptions.
**When simplicity overrides fit**: A hypothesis that fits perfectly (memorizes
every data point) but requires as many bits as the data itself is *no
compression*. The spine rejects such hypotheses as non-explanatory.
**Dependencies**: Consumes from B (candidate inductions). Consumes from E
(projectibility provides a simplicity bias over predicates). Consumes from I
(explanatory coherence includes simplicity as a virtue). Feeds A (complex
predicates are penalized). Feeds B (simpler inductions are preferred).

#### H. Belief Revision & Consistency Maintenance

**What it decides**: What to do when beliefs contradict.
**Input**: Belief set K, new information P that contradicts K.
**Output**: Revised belief set K * P.
**Core operations** (detailed in Section 4B):
- **Contraction** K / P: Remove P while losing as little else as possible.
- **Revision** K * P: Add P while maintaining consistency.
  Implemented via Levi Identity: K * P = (K / NOT P) + P.
- **Entrenchment ordering**: Which beliefs are hardest to give up.
  Hierarchy: hard constraints > canon > tested defaults > untested defaults >
  derived conclusions > single observations.
**Iterated revision**: After K is revised to K * P, the entrenchment ordering
itself may shift. A belief that survived contradiction gains entrenchment.
A domain where contradictions are frequent loses entrenchment globally.
**Truth maintenance**: A lighter-weight alternative to full AGM revision.
When a justification is removed, the derived belief is retracted. Useful
for local consistency without global re-computation.
**Dependencies**: Consumes from C (defaults create revisable commitments).
Consumes from D (non-monotonic retraction triggers revision). Feeds F
(revised beliefs get updated confidence). Closes the Default-Revision Loop
with C and D.

#### I. Coherence & Explanation

**What it decides**: Which hypothesis best explains the observations.
**Input**: Surprising observation O, candidate hypotheses H1...Hn.
**Output**: Acceptance/rejection for each hypothesis, with scores.
**Core mechanism** (detailed in Section 4D):
- **Inference to the Best Explanation** (IBE): Accept the hypothesis
  that would, if true, make O least surprising.
- **Explanatory virtues**: Consilience (breadth), simplicity, analogy,
  conservatism, causal depth, falsifiability. Weighted sum produces a
  composite score.
- **Coherence network** (Thagard): Accept/reject hypotheses by maximizing
  constraint satisfaction in a graph where edges connect hypotheses to
  evidence (positive: explains) and hypotheses to competitors (negative:
  contradicts).
**When is an observation "surprising"?** An observation O is surprising
relative to belief set K if K does not entail O and K assigns O low
probability. Surprise triggers abduction (node B), which seeds hypotheses,
which compete via coherence (node I).
**Dependencies**: Consumes from B (abduction generates hypotheses).
Consumes from G (simplicity weights hypothesis scores). Feeds H (accepted
hypotheses may force revision of existing beliefs). Closes the
Explanation-Selection Loop with Abduction and G.

#### J. Normativity & Bounded Rationality

**What it decides**: When to stop thinking and start acting.
**Input**: Computational budget, decision stakes, current belief quality.
**Output**: A stopping rule and an action policy.
**Key concepts**:
- **Satisficing** (Simon): Find a "good enough" solution, not the optimal one.
  Operationalized as: stop when marginal improvement in belief quality
  falls below threshold tau per unit of computation.
- **Metareasoning**: Reason about reasoning. Is it worth spending another
  second to improve this belief? Model the value of information.
- **Epistemic virtues**: Accuracy, simplicity, consistency, scope, fertility.
  Different contexts weight these differently. An emergency room weights
  speed over precision; a mathematical proof weights precision over speed.
- **Heuristics**: Fast, frugal rules that work well in typical environments.
  "Take the best" (Gigerenzer): use the single most valid cue and ignore
  the rest. "Recognition heuristic": if one option is recognized and the
  other is not, choose the recognized one.
**Dependencies**: Consumes from F (uncertainty determines when more
computation is warranted). Consumes from G (simpler computations are
cheaper). Feeds all nodes (every operator has a cost; bounded agents
must budget).

### 1.2 The Three Loops (Data-Flow Detail)

These are not just conceptual groupings. Each loop is a **feedback
cycle** that, if severed, produces a specific failure mode.

#### Loop 1: Projectibility Loop (A → E → G → A)

```
Representation (A) defines predicate admissibility
  → Induction (E) generalizes using admissible predicates only
    → Simplicity (G) prefers shorter predicates over natural kinds
      → Representation (A) prunes complex/gerrymandered predicates

What it produces: A vocabulary that makes generalization safe.
Failure mode if severed: "Grue"-type predicates proliferate.
  The system induces rules that fit all past data but fail on
  the very next observation. Without A→E, any predicate can be
  induced. Without E→G, all inductions are equally good. Without
  G→A, complex predicates accumulate and slow everything.
```

#### Loop 2: Default-Revision Loop (C → D → H → C)

```
Defaults (C) produce "normally" conclusions
  → Non-Monotonicity (D) retracts when exceptions appear
    → Belief Revision (H) surgically removes consequences
      → Defaults (C) are updated with new exception lists

What it produces: Stability under surprise.
Failure mode if severed: Either brittleness (C→D severed: exceptions
  cause contradiction, not retraction) or amnesia (D→H severed:
  retraction removes too much) or stagnation (H→C severed: defaults
  never learn from their mistakes).
```

#### Loop 3: Explanation-Selection Loop (Abduction → I → G → Abduction)

```
Abduction (B) generates candidate explanations for surprising observations
  → Coherence (I) scores them by explanatory virtues
    → Simplicity (G) penalizes overcomplicated explanations
      → Abduction (B) uses simpler accepted explanations as templates

What it produces: A preference ordering over possible stories.
Failure mode if severed: Either undergeneration (B→I severed:
  no hypotheses to score), or no selection (I→G severed: all hypotheses
  accepted, no pruning), or template collapse (G→B severed: each new
  observation spawns entirely novel explanations with no reuse).
```

### 1.3 Loop Composition: How the Three Interact

The loops are not independent. They share nodes and influence each other:

- **Projectibility constrains defaults**: A predicate that fails the
  projectibility gate (Loop 1) cannot appear in a default rule (Loop 2).
  The Default-Revision Loop never sees "grue."

- **Revision validates projection**: When a default is retracted by the
  Default-Revision Loop (Loop 2), the predicate's entrenchment score
  drops (Loop 1). Predicates whose defaults are frequently retracted
  lose projectibility.

- **Coherence selects among defaults**: When two defaults could explain
  the same observation, the Explanation-Selection Loop (Loop 3) scores
  them; the winner may cause the loser to be contracted via Loop 2.

- **Simplicity is the shared currency**: All three loops use node G
  (simplicity) as a tiebreaker. Simpler predicates (Loop 1), simpler
  default structures (Loop 2), and simpler explanations (Loop 3) are
  preferred, ceteris paribus.

### 1.4 Node Dependency Matrix

| Node | Depends on | Feeds into | Loop membership |
|------|-----------|-----------|----------------|
| A | G, J | B, E | Loop 1 |
| B | A | C, E, I, F | All three (abduction feeds Loop 3) |
| C | B, H | D, H | Loop 2 |
| D | C | H | Loop 2 |
| E | A, F | C, G | Loop 1 |
| F | B, H | E, J | — |
| G | B, E, I | A, B, C, I | All three |
| H | C, D, I | C, F | Loop 2 |
| I | B, G | G, H | Loop 3 |
| J | F, G | All (budget) | — |

---

## 2. Minimal Database Primitives (Full Schema)

Bayes' theorem and MDL are not stored as prose. They are **scoring
policies implemented by the engine**. The database stores the structured
objects that the engine scores. Each primitive below maps to a table or
a YAML frontmatter block. Each is consumed by at least one operator
(Section 4) and produced by at least one inference node (Section 1).

### 2A. Predicate Schema

What it governs: Which predicates are admissible, what they mean, and
whether they are eligible to appear in inductive generalizations.

```yaml
predicate_schema:
  id: string                # e.g., "bird", "flies", "green"
  arg_types: list[string]   # e.g., ["entity"] or ["entity", "time"]
  meaning: string           # natural-language gloss (for audit, not execution)
  canonical_form: string    # e.g., "bird(x)", "flies(x)"
  is_projectible: boolean   # passes the natural kind test (Section 4C, C3)
  entrenchment_score: float # from predicate_entrenchment table (C2)
  super_predicates: list    # e.g., "bird" might list "animal"
  disjoint_from: list       # e.g., "bird" disjoint from "fish"
  source: string            # how this predicate entered the vocabulary
```

**Resolver usage**: Node A (Representation) reads `is_projectible` to
decide whether a rule using this predicate in `if_all` position is an
inductive generalization or a mere observation summary. Node E reads
`entrenchment_score` to prefer entrenched predicates in new inductions.

**Why it exists**: Without explicit predicate governance, any string can
appear in a rule, and the system has no way to distinguish "green" from
"grue" except by authorial fiat. The predicate schema makes projectibility
a computed property, not a stored assumption.

### 2B. Default Rule

What it governs: A defeasible conditional — "if these conditions hold,
then conclude these effects, unless any of these exceptions apply."

```yaml
default_rule:
  id: string                # stable identifier
  if_all: list[string]      # ALL must be present in fact-set for rule to fire
  if_any: list[string]      # (optional) AT LEAST ONE must be present
  then:                     # effects to apply when rule fires
    - op: "set" | "unset" | "add" | "remove" | "emit"
      path: string          # e.g., "facts", "events[]"
      value: any
      note: string          # audit trail
  unless: list[string]      # ANY of these present → rule is blocked
  unless_exhaustive: bool   # false = open default (more exceptions may exist)
  scope:
    applies_when: list      # ALL must hold for rule to be considered
    not_when: list          # ANY present → rule is not considered (veto)
  priority: int             # [-1000, 1000], higher wins tiebreaks
  weight: float             # confidence in this default specifically (not the
                            #   predicate; a high-confidence predicate can have
                            #   low-weight specific rules)
  specificity: int          # computed at runtime (antecedent cardinality);
                            #   NOT stored — see Section 4A, D1
  entrenchment: float       # [0.0, 1.0], default = weight; used by revision
  status: "active" | "retracted" | "superseded"
  retraction_history:       # audit: when was this rule retracted, and why
    - timestamp: string
      reason: string
      by_rule: string       # which rule/observation caused retraction
```

**Resolver usage**: Node C (Defeasible Reasoning) scans all `default_rule`
entries whose `scope.applies_when` matches the current context and whose
`status` is "active". Conflicting conclusions are resolved by the
specificity → priority → weight chain (Section 4A, D4). When a rule is
retracted (via non-monotonicity or revision), its `status` flips to
"retracted" and the `retraction_history` is appended.

**Distinction from `hard_constraint`**: A hard constraint is a `default_rule`
with `entrenchment: 1.0` and `unless_exhaustive: true` and no `unless`
list. The engine may choose to optimize these separately. The schema
unifies them for simplicity.

### 2C. Hard Constraint

What it governs: Invariants. Must-never-be-violated conditions.

```yaml
hard_constraint:
  id: string
  condition: string         # predicate that must never be true
  type: "must" | "must_not"
  domain: string            # e.g., "governance", "physics", "logic"
  entrenchment: 1.0         # immutable
  violation_response: "halt" | "contract" | "quarantine"
  violation_history:        # audit of every violation
    - timestamp: string
      facts_present: list   # snapshot of fact-set that caused violation
      resolution: string    # what was done
```

**Resolver usage**: After every inference step, the engine checks all
`hard_constraint` entries against the current fact-set + derived facts.
If a `must_not` constraint's condition is satisfied, or a `must`
constraint's condition is unsatisfied, the `violation_response` fires:
- `halt`: stop execution, emit error.
- `contract`: trigger belief revision (Section 4B, B4) to remove the
  beliefs that caused the violation.
- `quarantine`: isolate the violating facts into a separate context;
  continue execution in the main context without them.

**Why it exists**: Some things are not negotiable. In a governance system,
"no unsigned reports" is a hard constraint. In a physical reasoning
system, "no object in two places at once" is a hard constraint. Hard
constraints are the boundary between common sense (which can be wrong)
and logic (which cannot).

### 2D. Evidence Update

What it governs: How a new observation changes the system's beliefs.

```yaml
evidence_update:
  id: string
  observation: string       # the new fact or fact-pattern
  affects: list[string]     # IDs of rules/beliefs affected
  effect: "strengthen" | "weaken" | "contradict" | "confirm" | "trigger_revision"
  magnitude: float          # [-1.0, 1.0]; negative = weaken/contradict
  condition: string         # (optional) only apply if this predicate holds
  source_reliability: float # [0.0, 1.0]; how much to trust this observation
                             #   (sensor noise, reporter credibility, etc.)
```

**Resolver usage**: When a new fact enters the system, the engine scans
`evidence_update` entries to determine which beliefs are affected and
how. A `contradict` with `magnitude: -1.0` on a belief with high
entrenchment triggers the full AGM revision pipeline (Section 4B).
A `weaken` with `magnitude: -0.3` merely reduces the weight/confidence
of the affected rule without retracting it.

**Bayesian interpretation**: `magnitude` is the log-likelihood ratio
contributed by this observation. `source_reliability` is the discount
factor. The engine may choose to implement full Bayesian updating or a
lighter-weight heuristic. The schema does not prescribe the math; it
prescribes the interface.

### 2E. Revision Policy

What it governs: Which revision strategy to use when consistency is
threatened.

```yaml
revision_policy:
  id: string
  strategy: "AGM_partial_meet" | "AGM_entrenchment" | "prioritized_removal" |
            "minimal_change" | "explanation_first"
  entrenchment_source: "stored" | "derived_from_confidence" | "derived_from_usage"
  iterated: boolean         # apply Darwiche-Pearl postulates?
  max_retraction_depth: int # max number of beliefs to retract in one revision
  preserve: list[string]    # belief IDs that must never be contracted
  recovery_policy: "full_recovery" | "no_recovery" | "partial"
```

**Resolver usage**: Node H (Belief Revision) reads the active `revision_policy`
to decide which contraction function to use. The default policy for a
new COMMONSENSE instance: `AGM_entrenchment` with entrenchment derived
from confidence, iterated revision on (`true`), max depth 10, and
hard constraints in `preserve`.

**Why multiple strategies?** Different domains need different revision
behavior. A theorem prover wants minimal change (preserve as many
lemmas as possible). A sensor-fusion system wants explanation-first
(prefer to reject the sensor reading over the world model). The policy
object lets the system switch strategies by context.

### 2F. Projectibility Constraints

What it governs: The induction bias — which predicates can generalize.

```yaml
projectibility_constraints:
  id: string
  allowed_forms: list       # e.g., ["natural_kind", "causal_property",
                            #        "quantitative_magnitude"]
  blocked_forms: list       # e.g., ["temporal_indexical", "spatial_indexical",
                            #        "grue_variant", "disjunctive_kind"]
  similarity_metric: "cosine" | "euclidean" | "jaccard" | "learned"
  entrenchment_threshold: float  # min entrenchment to project (default 0.3)
  sample_variety_required: int   # min distinct conditions before induction
  hierarchy_bias: boolean   # prefer predicates higher in the taxonomy
  causal_requirement: "required" | "preferred" | "none"
```

**Resolver usage**: Node E (Induction) consults the active
`projectibility_constraints` before creating a new `default_rule` from
observed co-occurrences. If the subject predicate's form matches a
`blocked_forms` entry, the induction is rejected regardless of evidence.
If `causal_requirement` is "required" and the predicate lacks causal
grounding, the induction is rejected. The `entrenchment_threshold` and
`sample_variety_required` together form the induction gate (Section 4C, C5).

### 2G. Coherence Scoring Policy

What it governs: How competing hypotheses are compared.

```yaml
coherence_scoring:
  id: string
  method: "weighted_sum" | "thagard_network" | "bayesian_model_selection"
  virtues:                  # which virtues to include and their weights
    consilience: {enabled: true, weight: 0.25, cap: 10}
    simplicity:  {enabled: true, weight: 0.20}
    analogy:     {enabled: true, weight: 0.15}
    conservatism:{enabled: true, weight: 0.20}
    causal_depth:{enabled: true, weight: 0.10}
    falsifiability:{enabled: true, weight: 0.10}
  acceptance_threshold: float      # min composite score to accept (default 0.3)
  competitor_margin: float         # min score gap to reject competitor (default 0.05)
  max_hypotheses: int              # max hypotheses to score per observation
  network_params:                  # only if method = thagard_network
    decay: 0.05
    max_iterations: 100
    convergence_epsilon: 0.001
```

**Resolver usage**: Node I (Explanation) reads the active `coherence_scoring`
policy to score candidate hypotheses generated by abduction (node B).
The same policy can be used for different domains by adjusting weights:
scientific reasoning weights falsifiability higher; legal reasoning
weights analogy and conservatism higher; engineering weights simplicity
and causal depth higher.

### 2H. Explanation Template

What it governs: Patterns that generate candidate hypotheses from
observation types.

```yaml
explanation_template:
  id: string
  observation_pattern: string    # e.g., "unexpected_absence(X)",
                                 #       "coincidence(A, B)",
                                 #       "failure(component, context)"
  candidate_hypotheses: list     # structured templates, not free text
    - type: "causal_chain"
      template: "{observation} caused by {cause} via {mechanism}"
      required_predicates: list
      forbidden_predicates: list
    - type: "analogy"
      template: "{observation} similar to {known_case} in domain {domain}"
      similarity_threshold: float
    - type: "default_override"
      template: "Default rule {rule_id} has hidden exception {exception}"
  priority: int                  # try these templates first/second/last
  source_domains: list           # domains this template is valid for
```

**Resolver usage**: Node B (Abduction) matches a surprising observation
against `explanation_template` entries by `observation_pattern`. For each
matching template, it instantiates the `candidate_hypotheses` into
concrete hypotheses (replacing template variables with actual predicate
instances from the fact-set). These hypotheses are then passed to node I
for coherence scoring.

### 2I. Entrenchment Registry

What it governs: A separate table tracking predicate entrenchment over time.

```yaml
entrenchment_registry:
  predicate_id: string           # references predicate_schema.id
  projections_attempted: int     # times this predicate appeared in a default rule
  projections_succeeded: int     # times the default rule survived N observations
  projections_failed: int        # times the default rule was retracted
  entrenchment_score: float      # succeeded / (attempted + 1); cached, recomputed
  last_projection: timestamp
  last_failure: timestamp
  domains_used_in: list          # e.g., ["physics", "social", "governance"]
```

**Resolver usage**: Node E recomputes `entrenchment_score` after each
projection attempt (success or failure). Node A reads it to decide
`is_projectible` for borderline predicates (those that pass the natural
kind test but have low entrenchment). Node G reads it as a prior for
MDL-based model selection (entrenched predicates are "cheaper" to use
in explanations).

**Why a separate table?** Predicate entrenchment is global (across all
rules using that predicate), while rule entrenchment is local (per rule).
A predicate like "bird" can be highly entrenched (many successful
projections) even if a specific rule about birds (e.g., "birds migrate
south") has low confidence.

### 2J. Primitive-to-Node Mapping

| Primitive | Produced by node | Consumed by node | Operator section |
|-----------|-----------------|------------------|-----------------|
| `predicate_schema` | A | B, E, G | 4C (Projectibility) |
| `default_rule` | B, E | C, D, H | 4A (Defeasibility) |
| `hard_constraint` | A, J | H | 4B (Revision) |
| `evidence_update` | B, F | F, H | 4B (Revision) |
| `revision_policy` | J | H | 4B (Revision) |
| `projectibility_constraints` | J | E, G | 4C (Projectibility) |
| `coherence_scoring` | J | I | 4D (Coherence) |
| `explanation_template` | B, I | B, I | 4D (Coherence) |
| `entrenchment_registry` | E | A, E, G | 4C (Projectibility) |

**Bayes and MDL are not in this table.** They are implemented in the
engine's scoring functions (node F, node G, node I) and configured via
the policy primitives above (which weights to use, which virtue to
prioritize). The database stores what to score; the engine stores how.

## 3. Canonical Outline for the Spine Vault (11 Sections, Expanded)

Use these as folder/note headers inside the Obsidian vault. No domain
content (physics, social, etc.) — only meta-operators. Each section
below specifies what the note must answer, what formal machinery it
must define, and where the canonical sources are.

### 1. Problem Frame

**What it must answer**:
- What is the difference between having a fact and having common sense?
- Why can't deduction alone handle everyday reasoning?
- What are the canonical failure modes when a system has facts but no defaults?
  - Brittleness: one exception breaks every rule
  - Paralysis: cannot act without complete information
  - Fragmentation: contradictory beliefs in different contexts with no repair mechanism
  - Overfitting: treats every observed coincidence as a law
- What is the Turing-test equivalent for common sense? (The "penguin test":
  given "birds fly" and "Tweety is a bird," conclude Tweety flies; given
  also "Tweety is a penguin," retract and conclude Tweety does not fly.)

**Key distinction to nail down**: Common sense is not a collection of
facts about the world. It is a collection of **policies for forming,
revising, and selecting beliefs under partial information**.

**Canonical sources**: McCarthy "Programs with Common Sense" (1959),
Minsky "Framework for Representing Knowledge" (1974), Davis & Marcus
"Commonsense Reasoning and Commonsense Knowledge in AI" (2015).

### 2. Representation Substrate

**What it must answer**:
- What is the predicate language? First-order? Typed? Higher-order?
- How are predicates typed? What are the primitive types? (Entity, Event,
  Time, Location, Quantity, Proposition)
- What makes a predicate admissible? (Section 4C, natural kind test)
- How does the choice of representation language bias induction?
  - A language with only "green" and "blue" cannot express "grue."
  - A language with "grue" but without projectibility constraints will
    overfit.
  - Ontology IS bias. The representation substrate must make this bias
    explicit and auditable.
- Compositionality: how do complex predicates inherit projectibility from
  their constituents? E.g., if "green" and "ball" are projectible, is
  "green ball" projectible? (Yes, by compositional inheritance, but with
  lower confidence than either constituent alone.)

**Formal machinery**: Predicate type system, admissibility function
`admissible(p) -> bool`, compositionality rules, semantic features for
similarity-space computation (required by node E, natural kind test).

**Canonical sources**: Goodman "Fact, Fiction, and Forecast" (1954/1983),
Brachman & Levesque "Knowledge Representation and Reasoning" (2004),
Levesque "Foundations of a Functional Approach to Knowledge Representation"
(1984), Rosch "Principles of Categorization" (1978), Fagin, Halpern,
Moses, Vardi "Reasoning about Knowledge" (1995), Gardenfors
"Conceptual Spaces" (2000).

### 3. Inference Operators

**What it must answer**:
- Deduction: what proof system? (Natural deduction, resolution, or a
  lightweight forward-chaining engine.)
- Induction: under what conditions does "All observed P are Q" warrant
  "All P are Q"? (Gate: Section 4C, C5.)
- Abduction: how are candidate hypotheses generated from a surprising
  observation? (Templates: Section 2H.)
- Analogy: given a source domain S and a target domain T, and a mapping
  M between them, what transfers? (Structural alignment over the
  predicate graph, constrained by causal roles.)
- Causal inference: how is causal direction determined? (Intervention
  semantics, counterfactual test: "If I wiggle X, does Y change?")

**Each operator must specify**:
1. Preconditions (when is this operator applicable?)
2. Input types (what does it consume from the belief set?)
3. Output type (what does it produce?)
4. Monotonicity (is the conclusion retractable?)
5. Computational complexity (bounded by J)

**Canonical sources**: Pearl "Causality" (2000/2009), Gentner "Structure-
Mapping" (1983), Peirce on abduction (collected papers).

### 4. Defeasibility & Non-Monotonicity

**What it must answer**:
- How do defaults interact? Specificity ordering (Section 4A, D1).
- What happens when a default's conclusion is contradicted? Retraction
  via truth maintenance, not global inconsistency.
- Closed-world assumption: when is "not P" inferred from absence of "P"?
  (Circumscription: minimize abnormal predicates.)
- How does argumentation fit? (Dung's abstract argumentation: defaults
  are arguments; specificity determines attack/defeat relations;
  preferred extensions are consistent default closures.)

**Formal machinery**: Default logic (Reiter 1980), circumscription
(McCarthy 1980), autoepistemic logic (Moore 1985), argumentation
frameworks (Dung 1995).

**Key test case**: The Nixon Diamond. Nixon is both a Quaker (Quakers
are pacifists) and a Republican (Republicans are not pacifists). Neither
default is more specific. Skeptical semantics: conclude neither. Credulous
semantics: pick one arbitrarily. The spine must state which it uses and why.

**Canonical sources**: Reiter "A Logic for Default Reasoning" (1980),
McCarthy "Circumscription — A Form of Non-Monotonic Reasoning" (1980),
Dung "On the Acceptability of Arguments" (1995).

### 5. Induction, Confirmation & Projectibility

**What it must answer**:
- Why does "all observed ravens are black" confirm "all ravens are black,"
  but "all observed non-black things are non-ravens" does not?
  (Hempel's paradox. Resolution: projectibility. "Raven" is a natural
  kind; "non-black thing" is not.)
- What makes a predicate "lawlike"? Counterfactual support, causal
  grounding, similarity clustering, non-gerrymandering (Section 4C, C3).
- How does sample variety affect confirmation? Ten ravens from ten
  continents confirm more strongly than a hundred ravens from one tree.
- How does the system avoid overfitting? Simplicity penalty (node G)
  plus projectibility gate (node E).

**Formal machinery**: Goodman's entrenchment theory, Hempel's confirmation
paradox and its resolutions, Carnap's continuum of inductive methods,
similarity spaces (Gardenfors, conceptual spaces).

**Canonical sources**: Hempel "Studies in the Logic of Confirmation"
(1945), Goodman "The New Riddle of Induction" (1954), Quine "Natural
Kinds" (1969), Gardenfors "Conceptual Spaces" (2000).

### 6. Uncertainty & Degrees of Belief

**What it must answer**:
- What does `confidence: 0.8` mean? It MUST mean: "in 100 predictions
  with confidence 0.8, approximately 80 should be correct." If the
  system cannot provide this calibration guarantee, it must not report
  numeric confidence.
- What representation? Point probability, interval, ranking, or DS?
  The spine must pick ONE default and explain why.
- How is confidence updated? Bayes' rule is the normative standard.
  The engine may approximate it. The approximation error must be bounded
  and auditable.
- How are confidence and entrenchment related? Confidence is "how likely
  true." Entrenchment is "how reluctant to give up." A belief can be
  low-confidence but high-entrenchment (e.g., a working assumption that
  we know is approximate but rely on heavily).

**Formal machinery**: Probability calculus, ranking theory (Spohn),
Dempster-Shafer theory, calibration plots, Brier score.

**Canonical sources**: Jaynes "Probability Theory: The Logic of Science"
(2003), Spohn "The Laws of Belief" (2012), Shafer "A Mathematical Theory
of Evidence" (1976).

### 7. Preference, Selection & Compression

**What it must answer**:
- Why prefer simpler hypotheses? Because they compress better. MDL
  formalizes this: the best hypothesis minimizes L(H) + L(D|H).
- Is MDL equivalent to Bayesian inference? Yes, under the Solomonoff
  prior (universal distribution over computable hypotheses). But
  Solomonoff is uncomputable. The spine needs a bounded approximation.
- What is the relationship between simplicity and projectibility?
  Simpler predicates tend to be more projectible (shorter definitions,
  fewer gerrymandered disjunctions). This is not a coincidence — it's
  why Occam's razor works.
- How are structural simplicity and Kolmogorov complexity related?
  Structural simplicity (fewer free parameters, shallower exception
  nesting) is a proxy for K-complexity that is computable in practice.

**Formal machinery**: MDL principle (Rissanen, Grunwald), Solomonoff
induction, structural risk minimization (Vapnik), two-part codes.

**Canonical sources**: Grunwald "The Minimum Description Length Principle"
(2007), Li & Vitanyi "An Introduction to Kolmogorov Complexity" (2008),
Solomonoff "A Formal Theory of Inductive Inference" (1964).

### 8. Belief Change & Consistency Maintenance

**What it must answer**:
- When a new observation contradicts existing beliefs, what is the
  algorithm for deciding what to give up? (Section 4B: AGM revision
  with entrenchment ordering.)
- What are the AGM postulates and does the implementation satisfy them?
  (Postulates listed in Section 4B, B1-B2.)
- Is iterated revision coherent? The Darwiche-Pearl postulates constrain
  how revision operators should behave across multiple revisions.
  The spine must decide whether to enforce them.
- Truth maintenance vs. full belief revision: when is it sufficient to
  retract consequences by removing justifications (TMS), and when is
  global re-computation necessary (AGM)? The spine should use TMS for
  routine non-monotonic retraction and AGM for genuine contradictions.

**Formal machinery**: AGM (Alchourron, Gardenfors, Makinson 1985),
Darwiche-Pearl postulates (1997), JTMS/ATMS (Doyle 1979, de Kleer 1986),
kernel contraction (Hansson 1994).

**Canonical sources**: Gardenfors "Knowledge in Flux" (1988), Hansson
"A Textbook of Belief Dynamics" (1999), de Kleer "An Assumption-Based TMS"
(1986).

### 9. Explanation & Coherence

**What it must answer**:
- What makes one explanation better than another? The six explanatory
  virtues (Section 4D, D2) with weighted scoring.
- Is Inference to the Best Explanation (IBE) rational? Lipton's defense:
  IBE is not a rival to Bayes but a description of how we generate and
  select priors.
- How does the coherence network (Thagard) work? Constraint satisfaction
  with parallel activation update (Section 4D, D3).
- What is the relationship between explanation and causation? Causal
  explanations are preferred because they support counterfactuals and
  interventions. Statistical explanations ("smoking correlates with
  cancer") are weaker than causal explanations ("smoking causes cancer").

**Formal machinery**: Lipton's IBE (2004), Thagard's explanatory coherence
(2000), Pearl's causal hierarchy (association → intervention →
counterfactual), Woodward's manipulationist account (2003).

**Canonical sources**: Lipton "Inference to the Best Explanation" (2004),
Thagard "Coherence in Thought and Action" (2000), Halpern & Pearl "Causes
and Explanations: A Structural-Model Approach" (2005), Woodward "Making
Things Happen" (2003).

### 10. Normativity & Bounded Rationality

**What it must answer**:
- What is the stopping rule? When has the system "thought enough"?
  (Satisficing: stop when marginal improvement < threshold.)
- What are the epistemic virtues and how are they traded off?
  Accuracy, simplicity, consistency, scope, fertility, speed.
- How does the system decide which inference operator to apply next?
  (Metareasoning: expected value of computation vs. cost.)
- What heuristics are acceptable as approximations? "Take the best,"
  recognition heuristic, tallying. The spine must list which heuristics
  it uses and under what conditions.

**Bounded rationality is not a concession. It is a design principle.**
An agent with unlimited computation does not need common sense; it can
compute the optimal action from first principles. Common sense exists
BECAUSE computation is bounded.

**Formal machinery**: Simon's satisficing, Gigerenzer's adaptive toolbox,
Russell & Wefald's metareasoning, Horvitz's bounded optimality.

**Canonical sources**: Simon "The Sciences of the Artificial" (1969/1996),
Gigerenzer et al. "Simple Heuristics That Make Us Smart" (1999), Russell
& Wefald "Do the Right Thing" (1991).

### 11. Minimal Operators (One-Line List, Expanded)

Each operator is a function the engine must implement. Each is tested by
at least one fixture in `/TESTS/`.

| # | Operator | Signature | What it computes |
|---|----------|-----------|-----------------|
| 1 | **admit** | `predicate → bool` | Is this predicate admissible in the language? (A) |
| 2 | **induce** | `observations → default_rule` | Does this pattern warrant a new default? (E) |
| 3 | **deduce** | `belief_set → belief_set` | What follows logically from current beliefs? (B) |
| 4 | **abduce** | `observation → list[hypothesis]` | What could explain this surprising observation? (B) |
| 5 | **analogize** | `(source, target, mapping) → list[belief]` | What transfers from source to target? (B) |
| 6 | **close_defaults** | `(facts, defaults) → belief_set` | What is the maximal consistent default extension? (C) |
| 7 | **detect_conflict** | `belief_set → list[contradiction]` | Where are the inconsistencies? (D) |
| 8 | **retract** | `(belief, justification_graph) → belief_set` | Remove belief and all beliefs that depended on it. (D) |
| 9 | **contract** | `(belief_set, proposition) → belief_set` | Remove proposition while preserving as much as possible. (H) |
| 10 | **revise** | `(belief_set, proposition) → belief_set` | Add proposition while maintaining consistency. (H) |
| 11 | **entrench** | `belief → float` | How hard is this belief to give up? (H) |
| 12 | **score_hypothesis** | `(hypothesis, evidence) → float` | How good is this explanation? (I) |
| 13 | **select_best** | `list[hypothesis] → hypothesis` | Which hypothesis best explains the evidence? (I) |
| 14 | **evaluate_coherence** | `belief_set → float` | How well do all beliefs hang together? (I) |
| 15 | **update_confidence** | `(belief, evidence) → belief` | Revise credence based on new evidence. (F) |
| 16 | **calibrate** | `predictions × outcomes → calibration_score` | Do reported confidences match empirical frequencies? (F) |
| 17 | **compress** | `belief_set → bit_length` | What is the description length of this belief set? (G) |
| 18 | **prefer** | `(belief_A, belief_B) → ordering` | Which belief is preferred, and why? (G) |
| 19 | **stop** | `(budget, belief_quality) → bool` | Should the system stop thinking and act? (J) |
| 20 | **metareason** | `(candidate_actions, state) → action` | Which action has the highest expected value of computation? (J) |

---

### Central Spine Questions (The Four Pillars)

These four questions are the navigation hubs of the vault. Every note,
every operator, every fixture should be traceable to at least one of them.
If a piece of content answers none of these, it does not belong in the spine.

| Question | Node(s) | What it governs | Failure mode if wrong |
|----------|---------|-----------------|----------------------|
| **Projectibility**: Why are some predicates lawlike and others not? | A, E, G | Which patterns get promoted to rules | Overfitting to noise ("grue") |
| **Defeasibility**: How do "usually" conclusions remain rational under exceptions? | C, D | How defaults coexist with exceptions | Brittle collapse on first counterexample |
| **Revision**: How should beliefs change under surprise without collapsing? | H, D | Surgical repair of inconsistency | Either no repair (contradictions accumulate) or too much repair (amnesia) |
| **Coherence**: When multiple explanations are possible, which one wins? | I, G | Hypothesis selection under competition | No selection (all hypotheses accepted) or wrong selection (simpler, better explanation ignored) |

---

### Gap Analysis: Current COMMONSENSE vs. Full Spine

| Capability | Current COMMONSENSE | Full Spine Requires | Priority |
|-----------|-------------------|-------------------|----------|
| Default reasoning | Flat `if_all`/`unless` with no specificity | Specificity ordering + priority lattice + Reiter-style extensions | P0 |
| Belief revision | No contraction or revision operators | AGM contraction/revision with entrenchment ordering | P0 |
| Projectibility | Hard-coded rules by author | Natural kind test + predicate entrenchment + grue filter | P0 |
| Explanatory coherence | All applicable rules fire additively | Scoring by six virtues + Thagard network + acceptance thresholds | P0 |
| Non-monotonicity | None (monotonic resolution) | Truth maintenance with justification tracking | P1 |
| Uncertainty | `confidence` float field, no calibration | Calibrated credences with bounded approximation error | P1 |
| Simplicity | None (all rules equal) | MDL-based preference with computable simplicity proxies | P1 |
| Bounded rationality | None (exhaustive resolution) | Satisficing stopping rules + metareasoning budget | P2 |
| Predicate governance | Ad-hoc string matching | Typed predicates with admissibility functions | P2 |

---

## 4. Concrete Operators (Formal Semantics)

The nodes in Section 1 name concepts. The primitives in Section 2 are storage shapes.
This section defines **what the engine actually computes** for the four load-bearing
concepts that COMMONSENSE names but does not yet mechanise. Each subsection provides:
(1) the formal operator, (2) the storage fields it consumes, and (3) pseudo-code for
the resolver's decision procedure.

### 4A. Defeasible Reasoning (Defaults)

**Problem**: COMMONSENSE has `if_all`, `unless`, `priority`, and `not_when` as flat
lists. Two defaults can conflict (e.g., "birds fly" vs "penguins don't fly") but
there is no principle for deciding which wins; `priority` is a static integer with no
structural relationship to the rules' content.

#### D1. Specificity Ordering (lex specialis)

When two defaults D1 and D2 both apply but their conclusions conflict, the default
with the **more specific antecedent** overrides the more general one.

- **Antecedent(A)** of a rule = the conjunction of `scope.applies_when` + `rule.if_all`.
- D2 is **more specific** than D1 iff Antecedent(D1) is a proper subset of Antecedent(D2).
  (Every fact that satisfies D2 also satisfies D1, but not vice versa.)
- Example: `{bird} => flies` vs `{bird, penguin} => NOT flies`. The penguin rule's
  antecedent contains the bird rule's antecedent as a proper subset; the penguin rule
  wins for any fact-set containing `penguin`.

**Resolver integration**: Before the `candidate_key` tiebreak (priority, confidence, id),
apply **defeasible narrowing**:

```
for each pair (D1, D2) where both apply and conclusions conflict:
    if Antecedent(D1) is proper subset of Antecedent(D2):
        suppress D1 (the general default)
    elif Antecedent(D2) is proper subset of Antecedent(D1):
        suppress D2
    else:
        fall through to priority/confidence tiebreak (skeptical if still tied)
```

**Schema delta**: The `priority` field remains the fallback. Specificity is computed
from rule structure at runtime; it does not need a new stored field.

#### D2. Default Logic Semantics (Reiter 1980)

A default rule D has the form `A : B / C` where:
- **A** = prerequisite (the conjunction from `if_all` + `applies_when`)
- **B** = justification (what must be *consistent* with current beliefs; the negation
  of each `unless` predicate)
- **C** = consequent (the `then` effects)

A default is **applicable** when A is satisfied and B is consistent with the current
belief set. The **extension** of a default theory is the fixed-point of applying all
applicable defaults until no new conclusions can be added without violating consistency.

**Current gap**: The `unless` list is checked as simple fact non-membership, not as
consistency with the full derived-fact closure. The engine should consider a default
inapplicable if its conclusion would contradict any fact in `facts | derived | hard_constraints`.

#### D3. Open vs. Closed Defaults

- **Open default**: `unless` is a non-exhaustive sample of exceptions. The default
  remains warranted even when an unforeseen exception appears; the exception is added
  to `unless` and the default fires for all other cases.
- **Closed default**: `unless` is exhaustive. If an exception not in the list occurs,
  the default is falsified (not just blocked).

**Schema delta**: Add optional boolean field `unless_exhaustive` (default: `false`)
to the rule schema.

#### D4. Priority Lattice

Defaults form a partial order. The tiebreak chain is:

1. **Specificity** (structural, computed at runtime)
2. **Priority** (stored integer in entry)
3. **Confidence** (stored float in entry)
4. **Skeptical semantics**: if still unresolved, *neither* conclusion is warranted
   (no arbitrary tiebreak by ID; IDs are for reference, not truth)

**Variable-priority extension (Horty 2012)**: The flat `priority: int` field
is insufficient when priority depends on context. Horty's prioritized default
logic extends Reiter with explicit priority relations: `D1 < D2` means "D2
defeats D1 when both apply." Priority relations can themselves be defeasible
(default priorities that can be overridden by higher-priority meta-defaults).

**Schema delta**: Add optional `priority_over` field to `default_rule`:
```yaml
priority_over: list[string]   # IDs of rules this rule defeats regardless of specificity
priority_defeated_by: list[string]  # IDs of rules that defeat this rule
```
When both `specificity` and `priority_over`/`priority_defeated_by` exist,
Horty-priority takes precedence over specificity. The resolution algorithm:
```
for each conflicting pair (D1, D2):
    if D1.id in D2.priority_over: suppress D2
    elif D2.id in D1.priority_over: suppress D1
    elif specificity(D1, D2) resolves: use specificity
    else: fall through to priority/confidence tiebreak
```

#### D5. Undercutting vs. Rebutting Defeat (Pollock's OSCAR)

Pollock (1991, 2000) distinguishes two kinds of defeat that the current
flat `unless` list conflates:

- **Rebutting defeat**: attack the *conclusion*. "Birds fly, but Tweety is a
  penguin, so Tweety does not fly." This is what `unless` currently handles.
- **Undercutting defeat**: attack the *inference rule itself*. "That looks red,
  but the lighting is red, so the 'looks red → is red' rule is unreliable in
  this context." The conclusion is not contradicted; the *warrant* for the
  conclusion is removed.

**Operator**: For each default rule D firing on facts F, check `defeaters`
before accepting D's conclusion:
```yaml
defeaters:
  - type: "undercut" | "rebut"
    target: string          # rule ID (undercut) or predicate (rebut)
    condition: list[string] # facts that must hold for this defeater to fire
    effect: "block" | "weaken"  # block = prevent conclusion; weaken = reduce confidence
    priority: int           # defeaters themselves can be defeated
```
- An undercutting defeater removes the rule from consideration entirely
  (as if the rule never fired) for the current fact-set.
- A rebutting defeater adds a contradictory conclusion; specificity and
  priority resolve which conclusion is accepted.
- Defeaters can themselves be defeated (meta-defeat), creating a defeat-status
  graph that OSCAR computes via recursive labelling.

**Resolver integration**: Before `candidate_key` sorting, filter rules whose
undercutting defeaters are active. After sorting, for each firing rule, check
rebutting defeaters and resolve via the priority lattice (D4).

#### D6. Causal Production Semantics (Bochman 2004)

Bochman unifies default logic, argumentation, and abduction under a causal
production framework. A rule `A ⇒ B` is read as "A is a cause (explanation)
for B," not as a material conditional.

**Operator**: Default rules gain an optional `causal_direction` field:
```yaml
causal_direction: "forward" | "backward" | "bidirectional"
```
- `forward`: A causes B. If A holds, infer B. (Standard default.)
- `backward`: B is explained by A. If B is observed and needs explanation,
  abduce A. (Feeds Section 4D, abductive hypothesis generation.)
- `bidirectional`: A ⇔ B. Both forward inference and backward abduction.

**Resolver integration**: When a surprising observation O matches the `then`
consequent of a backward or bidirectional rule, the rule's `if_all` predicates
are proposed as candidate explanations (feeding the abduction step in node B
and the hypothesis scoring in Section 4D). This connects the default rule base
directly to explanatory coherence without requiring separate `explanation_template`
entries for every causal rule.

#### D7. KLM Preferential Consequence Relations

Kraus, Lehmann, and Magidor (1990) define the foundational semantic framework
for nonmonotonic consequence relations via preferential models. A consequence
relation `|~` is:

- **Cumulative**: If `A |~ B` then `A |~ C` iff `A ∧ B |~ C`. (Learning B
  does not change what follows from A.)
- **Rational**: Cumulative + if `A |~ C` and `A |~/- ¬B` then `A ∧ B |~ C`.
  (If C is a default consequence of A and B is consistent with A, then C is
  still a default consequence of A ∧ B.)

**Engine requirement**: The default closure operator (Section 4A, D2) must
satisfy the cumulative and rational postulates. This constrains the
implementation:
- **Cumulativity**: After deriving B from A via defaults, the set of defaults
  applicable to A ∧ B must be a superset of those applicable to A (no default
  is lost by learning a default consequence).
- **Rationality**: If C is in the default closure of A, and B is consistent
  with that closure, then C must remain in the closure of A ∧ B.

**Verification**: The Nixon Diamond and other test fixtures must validate that
the closure operator satisfies these postulates. A closure that fails
cumulativity is not a valid nonmonotonic consequence relation.

#### D8. Autoepistemic Logic & Modal Non-Monotonicity

The current spine uses object-level defaults (Reiter). Several major
formalisms reason at the meta-level about what the agent *believes*:

- **McDermott & Doyle (1980)**: Nonmonotonic modal logic using a consistency
  operator M: "If A is true and M B is consistent, conclude B." Generalizes
  Reiter's defaults by internalizing consistency checking.
- **Moore (1985)**: Autoepistemic logic. A proposition is believed iff it is
  true in all epistemic alternatives consistent with what is believed.
  Stable expansions = fixed points of `T(E) = { φ : E |= φ }`.
- **Lin & Shoham (1992)**: GK logic. Synthesizes autoepistemic logic with
  Shoham's preferential semantics: knowledge + justified assumptions.

**Engine integration**: These formalisms are most relevant when COMMONSENSE
must reason about its own beliefs (node J, metareasoning) or about other
agents' beliefs (multi-agent COMMONSENSE). The immediate integration point
is in node F (Uncertainty): when the system asks "what do I currently
believe?", it is performing autoepistemic reasoning. The answer is the
default closure of the current fact-set relative to the active rule base.

---

### 4B. Belief Revision (AGM Framework)

**Problem**: COMMONSENSE has no mechanism for what happens when a new observation
contradicts an existing belief. The `not_when` and `unless` fields can *block* a
rule from firing, but they cannot answer: "I believed P; I now observe NOT P; what
do I give up to restore consistency?"

#### B1. Contraction K / P (remove P while preserving as much as possible)

**AGM Postulates** (Alchourron, Gardenfors, Makinson 1985):

| # | Name | Meaning |
|---|------|---------|
| /1 | Closure | K / P is a belief set (closed under consequence) |
| /2 | Inclusion | K / P is a subset of K |
| /3 | Vacuity | If P is not in K, then K / P = K |
| /4 | Success | If P is not a tautology, then P is not in K / P |
| /5 | Recovery | K is a subset of (K / P) + P |
| /6 | Extensionality | If P and Q are logically equivalent, K / P = K / Q |

**Engine procedure** (partial meet contraction):
1. Identify all maximal subsets of K that do not entail P.
2. Select among them using the **entrenchment ordering** (B3 below).
3. The selected subset is K / P.

#### B2. Revision K * P (add P while maintaining consistency)

**Levi Identity**: `K * P = (K / NOT P) + P`
(First contract to remove NOT P, then simply add P.)

**AGM Postulates**:

| # | Name | Meaning |
|---|------|---------|
| *1 | Closure | K * P is a belief set |
| *2 | Success | P is in K * P |
| *3 | Inclusion | K * P is a subset of K + P |
| *4 | Vacuity | If NOT P is not in K, then K + P is a subset of K * P |
| *5 | Consistency | K * P is inconsistent only if P is inconsistent |
| *6 | Extensionality | If P and Q are logically equivalent, K * P = K * Q |

**Engine procedure**:
1. If P is already in K: return K (no change).
2. If P is consistent with K: return K U {P} closed under consequence.
3. If P is inconsistent with K: compute K / NOT P (contraction), then add P.

#### B3. Entrenchment Ordering (what to preserve)

Not all beliefs are equally defensible. An **entrenchment pre-order** `A <= B`
("B is at least as entrenched as A") guides contraction: remove the *least*
entrenched beliefs first.

**Entrenchment hierarchy** (most → least entrenched):
1. **Hard constraints** (INVARIANTS, `hard_constraint` entries) — never contracted
2. **Canon rules** (LAW/CANON/*) — contracted only under explicit governance change
3. **Explicit defaults with confirmed `unless` exceptions** — the default survived a test
4. **Explicit defaults without tests** — plausible but unverified
5. **Derived conclusions** — inferred, not asserted
6. **Single observations** — could be noise

**Schema delta**: Add optional `entrenchment` field (0.0–1.0, default = confidence).
For backward compatibility, `entrenchment` defaults to `confidence` when absent.
The `hard_constraint` entries get entrenchment = 1.0 (immutable).

**Iterated revision** (Darwiche-Pearl): After revision, the entrenchment ordering itself
may need adjustment. If we revise by P and later by NOT P, the second revision should
not simply restore the pre-first-revision state — the fact that P was accepted and then
contradicted provides evidence that the domain is unstable, meriting lower confidence
in all beliefs in that domain.

#### B4. Contraction Trigger Detection

When does contraction run? Two triggers:
1. **Explicit observation** of NOT P where P is in K (user-supplied fact that contradicts).
2. **Hard constraint violation** in the derived closure (derived facts contradict an invariant).

In both cases, the engine identifies the **minimal conflicting set** (the smallest
subset of beliefs that together entail the contradiction) and contracts by removing
the least-entrenched member of that set.

#### B5. Iterated Revision: Darwiche-Pearl (DP) Postulates

Single-step AGM revision (B1-B2) does not constrain what happens across
*multiple* revisions. The DP postulates (Darwiche & Pearl 1997) fill this gap:

| # | Name | Meaning |
|---|------|---------|
| DP1 | Success | If P |= Q, then (K * P) * Q = K * Q |
| DP2 | Consistency | If P |= ¬Q, then (K * P) * Q = K * Q |
| DP3 | Disjunction | If Q |= P, then (K * P) * Q = (K * Q) * P |
| DP4 | Irrelevance | If Q is consistent with K * P, then (K * P) * Q = K * (P ∧ Q) |

**Engine requirement**: The revision operator `K * P` must satisfy DP1-DP4
for all P, Q. This constrains the entrenchment ordering update: after
revising by P, the relative entrenchment of P and beliefs that supported P
must be adjusted so that a subsequent revision by a related proposition Q
produces the DP-consistent result.

**DP1-DP2 intuition**: If P entails Q (DP1) or P entails ¬Q (DP2), then
revising first by P and then by Q is equivalent to revising by Q alone.
The intermediate revision by P adds no information beyond what Q already
determines.

**DP3-DP4 intuition**: If Q entails P (DP3), the order of revision doesn't
matter — both P and Q end up accepted. If Q is consistent with K * P (DP4),
revising by P then Q is equivalent to revising by P∧Q at once (no order
effects when the second revision doesn't contradict the first).

#### B6. Conditional Belief Revision (Boutilier 1993, 1994)

Boutilier unifies default reasoning and belief revision in a modal framework.
A conditional belief `B ⇒ C` is read as "in the most normal worlds where B
holds, C holds." Revision by P is modelled as shifting the normality ordering
so that P-worlds become the new most-normal worlds.

**Operator**: The belief set K is represented as a total pre-order ≤ over
possible worlds (w1 ≤ w2 means w1 is at least as normal/plausible as w2).
- **Default inference**: `B |~ C` iff C holds in all ≤-minimal B-worlds.
- **Revision K * P**: Restrict ≤ to P-worlds; minimal P-worlds define K * P.
- **Natural revision**: If the observation P is consistent with the current
  most-normal worlds, revision is just conditioning on P (no re-ordering).
  If P contradicts the most-normal worlds, the system shifts to the
  most-normal P-worlds — a minimal change in the plausibility ordering.

**Engine integration**: Boutilier's framework provides a single mechanism
for both defeasible inference (Section 4A) and belief revision (Section 4B),
replacing the current dual-stack approach (Reiter defaults + AGM revision)
with a unified possible-worlds semantics. Implementation: maintain a
plausibility ordering over fact-sets, where the current fact-set is the
"actual world" and alternative fact-sets are ranked by their distance
(entrenchment-weighted Hamming distance) from the actual world.

#### B7. Contraction Taxonomy (Hansson 1999)

The current spine uses simplified partial-meet contraction. Hansson provides
the full taxonomy of contraction operators, each with distinct properties:

| Operator | Mechanism | Recovery postulate? | When to use |
|----------|-----------|-------------------|-------------|
| **Partial-meet** | Select among maximal P-avoiding subsets via selection function γ | Yes (/5) | Default; balances minimal loss with computational tractability |
| **Kernel** | For each minimal P-entailing subset, remove at least one element via incision function σ | No (fails /5) | When recovery is undesirable (e.g., removing a single observation shouldn't allow re-deriving it) |
| **Safe** | Remove beliefs according to a safety hierarchy (beliefs that "rest on" P are less safe) | Yes (/5) | When beliefs have dependency structure (justification graphs) |
| **Severe withdrawal** | Remove P and everything that entails P (maximally radical) | No | When P is discovered to be fundamentally unreliable (fraud, sensor failure) |

**Engine integration**: The `revision_policy` primitive (Section 2E) gains a
`contraction_type` field selecting among these four. Default: `partial_meet`.
For observations from unreliable sources, switch to `kernel` (no recovery).
For catastrophic sensor failure, switch to `severe_withdrawal`.

---

### 4C. Projectibility (Induction Theory)

**Problem**: The `if_all → then` rule structure can express arbitrary patterns, but
the engine has no criterion for distinguishing patterns that *should* generalize
("all emeralds are green") from patterns that *should not* ("all emeralds are grue").
The `db.example.json` simply hard-codes rules written by the author.

#### C1. Goodman's "Grue" and the New Riddle of Induction

- **"Grue"** = an object is grue iff it is first observed before time t and is green,
  OR it is first observed after time t and is blue.
- All emeralds observed so far (before t) are both green and grue.
- "All emeralds are green" and "All emeralds are grue" are equally supported by the
  evidence, but only the former is projectible.
- **The riddle**: what principled criterion selects "green" over "grue"?

**Goodman's answer**: **entrenchment**. "Green" is a more entrenched predicate than
"grue" because "green" has been used in more successful past projections. Entrenchment
is a *historical* property of the predicate in the language, not a logical property
of the evidence.

#### C2. Predicate Entrenchment Score

For each predicate P in the COMMONSENSE vocabulary, maintain:

```
entrenchment(P) = (number of successful projections using P) /
                  (number of attempted projections using P + 1)
```

- A **projection** is a rule of the form `if_all: [..., P(x), ...] then: [Q(x)]`
  where P is the subject predicate.
- A projection is **successful** if, after N observations, no counterexample has been
  found (i.e., no fact-set contains `P(x)` and `NOT Q(x)` simultaneously, or the
  `unless` list has absorbed all exceptions without invalidating the default).
- A projection is **failed** if a *projectible* counterexample appears (see C4).

**Schema delta**: Add a `predicate_entrenchment` table (not per-entry, but per-predicate):
```
predicate_id: string (e.g., "bird", "green", "grue")
projections_attempted: int
projections_succeeded: int
entrenchment_score: float (computed, cached)
last_updated: timestamp
```

#### C3. Natural Kind Test (What Makes a Predicate Projectible)

A predicate P passes the natural kind test iff:

| Criterion | Operational test |
|-----------|-----------------|
| **Counterfactual support** | The rule `if_all: [P(x)] then: [Q(x)]` also supports "If this *were* a P, it *would be* Q" — the rule appears in at least one accepted default that has survived exception introduction |
| **Similarity clustering** | Instances of P are more similar to each other (by feature vector cosine distance) than to non-instances of P |
| **Causal unification** | P appears in at least one causal rule (an entry tagged `causal`) or is decomposable into predicates that do |
| **Non-gerrymandering** | P's definition does not contain a disjunction of natural kinds, a specific time constant, a specific location constant, or an individual constant |

A predicate that fails the natural kind test is **non-projectible**. Rules that use a
non-projectible predicate in `if_all` position are treated as mere summaries of
observation, not as inductive generalizations.

#### C4. Grue Filter (Operational)

For each predicate P defined in the system:
1. Parse its definition. If it contains:
   - A temporal indexical ("before t", "after January 1 2030", etc.)
   - A spatial indexical ("in this room", "north of the equator")
   - An individual constant ("belonging to Alice")
   - A disjunction of predicates from different similarity clusters
   → **Flag as grue-suspect.**
2. If grue-suspect AND `entrenchment(P) < 0.5`: **block projection**. The predicate
   can be used in observations but not in the `if_all` position of a new default rule.
3. If grue-suspect AND `entrenchment(P) >= 0.5`: **warn but allow** (the predicate
   has earned entrenchment through successful use, e.g., predicates defined relative
   to "noon" or "sea level" that have proven stable).

#### C5. Induction Gate

When the engine considers creating a new default rule `P(x) => Q(x)` from observed
co-occurrences:

1. **Projectibility gate**: P must pass the natural kind test.
2. **Sample variety gate**: observations must span sufficiently diverse conditions
   (not all from the same time, location, or initial condition).
3. **Defeater gate**: no accepted default currently asserts `P(x) => NOT Q(x)`.
4. **Entrenchment threshold**: if `entrenchment(P)` is low, require more observations
   before generalizing (Bayesian-like: low prior → high evidence threshold).

#### C6. Hempel's Confirmation Theory (Raven Paradox)

Hempel (1945) identified a structural problem in naive induction that the
projectibility gate must handle:

**The paradox**: "All ravens are black" is logically equivalent to "All
non-black things are non-ravens." By Nicod's criterion (a universal
generalization is confirmed by its positive instances), a green apple
(which is a non-black non-raven) confirms "All non-black things are
non-ravens" and therefore confirms "All ravens are black." This is
counterintuitive: observing a green apple should not increase confidence
that ravens are black.

**Resolution via projectibility**: Nicod's criterion fails when applied to
non-projectible predicates. "Non-black thing" and "non-raven" are
gerrymandered predicates — they fail the natural kind test (Section 4C, C3).
The projectibility gate blocks induction over these predicates, so the
green apple does NOT confirm the raven hypothesis through the back door.

**Engine requirement**: When the induction engine evaluates a candidate rule
`P(x) => Q(x)`, it must also check whether `P` and `Q` pass the natural kind
test. If either is gerrymandered (disjunctive, temporal-indexical, defined
by negation over an unnatural category), the rule is rejected regardless of
how many positive instances have been observed. This blocks both the
direct induction of spurious rules (Goodman's grue) and the indirect
confirmation of legitimate rules through their logically equivalent
gerrymandered forms (Hempel's ravens).

**Hempel's positive contribution**: The paradox also reveals a legitimate
source of weak confirmation when predicates ARE natural kinds. Observing
a green emerald DOES confirm "all emeralds are green" and also confirms
"all non-green non-emeralds are non-emeralds," but only because "emerald"
and "green" are both entrenched projectible predicates. The engine accepts
this weak confirmation as a Bayesian evidence update (small weight) rather
than as a binary rejection.

---

### 4D. Explanatory Coherence (Hypothesis Scoring)

**Problem**: When multiple rules could explain an observation, COMMONSENSE has no
way to prefer one over another. The resolver selects *all* applicable rules and
applies their effects additively; there is no competition between alternative
explanations, no coherence scoring, and no pruning of inferior hypotheses.

#### D1. Inference to the Best Explanation (IBE)

Given an observation O (a surprising fact or fact-set that triggers explanation-seeking)
and a set of candidate hypotheses H1...Hn (each being a rule or a set of facts that
entails or probabilifies O), select the hypothesis that **best explains** O.

#### D2. Explanatory Virtues (Scoring Dimensions)

For each hypothesis H explaining observation O:

| Virtue | Symbol | Scoring function | Range | High when |
|--------|--------|-----------------|-------|-----------|
| **Consilience** | c(H) | count of distinct observations H explains | [0, N] | H unifies diverse evidence |
| **Simplicity** | s(H) | 1 / (1 + free_params + ad_hoc_clauses) | (0, 1] | H has few moving parts |
| **Analogy** | a(H) | max cosine-sim to accepted explanations in sibling domains | [0, 1] | H resembles what already works |
| **Conservatism** | v(H) | 1 / (1 + count of accepted beliefs H contradicts) | (0, 1] | H fits existing knowledge |
| **Causal depth** | d(H) | min(chain_length / max_depth, 1.0) | [0, 1] | H cites mechanisms, not surfaces |
| **Falsifiability** | f(H) | count of testable predictions / (1 + count) | [0, 1) | H sticks its neck out |

**Composite score**: `S(H) = w1*c_norm + w2*s + w3*a + w4*v + w5*d + w6*f`

Default weights (empirically calibrated): w1=0.25, w2=0.20, w3=0.15, w4=0.20, w5=0.10, w6=0.10.

For consilience, `c_norm = min(c(H) / c_max, 1.0)` where `c_max` is a reasonable cap
(e.g., 10 observations) to prevent one dimension from dominating.

#### D3. Coherence Maximization (Thagard's Model)

Explanatory coherence is a **constraint satisfaction problem**:

- **Nodes** = propositions (observations, hypotheses, accepted beliefs).
- **Positive constraints** (weight +1): H explains E, H implies E, H is analogous to H'.
- **Negative constraints** (weight -1): H contradicts H', H and H' compete to explain the same E.

**Activation update rule** (parallel):
```
a_i(t+1) = a_i(t) * (1 - decay) + sum_over_j( w_ij * a_j(t) ) * (1 - a_i(t))   if net > 0
a_i(t+1) = a_i(t) * (1 - decay) + sum_over_j( w_ij * a_j(t) ) * a_i(t)         if net < 0
```

Run until convergence (max delta < epsilon). Nodes with activation > 0 are accepted;
nodes with activation <= 0 are rejected.

This is NP-hard in the general case. For bounded problem sizes (<= 32 hypotheses and
<= 64 observations, matching the schema's `maxItems` limits), a greedy algorithm with
local search is practical.

#### D4. Acceptance Threshold

A hypothesis H is **accepted** iff:
- `S(H) > tau` (the composite score exceeds the acceptance threshold), AND
- `a(H) > 0` in the Thagard network (the coherence network accepts it), AND
- No accepted competitor H' has `S(H') > S(H) + delta` (H is at least delta-close to
  the best hypothesis; delta prevents thrashing between near-ties).

Default `tau = 0.3`, `delta = 0.05`.

#### D5. Explanation-Then-Revise Cycle

When a hypothesis is accepted:
1. **Identify contradictions**: find all currently accepted beliefs that conflict
   with the hypothesis or its consequences.
2. **Contract**: remove the least-entrenched member of each minimal conflicting set
   (per Section 4B, B4).
3. **Expand**: add the hypothesis and all its consequences to the belief set.
4. **Re-score**: re-run explanatory coherence on any hypotheses that shared premises
   with contracted beliefs (they may now be stronger or weaker).

#### D6. Structural Causal Models for Explanation (Halpern & Pearl 2005)

The current explanatory virtues (D2) include `causal_depth` as a scalar but
do not define how causation is determined. Halpern & Pearl provide the
structural-model framework that grounds causal explanation:

**Structural equation model** M = (U, V, F):
- **U**: exogenous variables (background conditions, not modeled).
- **V**: endogenous variables (outcomes the model explains).
- **F**: structural equations, one per V_i: V_i = f_i(PA_i, U_i), where PA_i
  are the parents of V_i in the causal graph.

**Actual causation**: X = x is an *actual cause* of Y = y in context u iff:
1. (X = x) ∧ (Y = y) holds in the actual world under u.
2. There exists a counterfactual setting X = x' and a (possibly empty) set
   W of other variables such that, holding W fixed at their actual values,
   changing X to x' changes Y to some y' ≠ y.
3. X = x is minimal (no proper subset of X satisfies 1-2).

**Explanatory depth via causal chains**: An explanation citing an actual
cause is deeper than one citing a mere correlation. The `causal_depth`
virtue (D2) is now computed as:
```
causal_depth(H) = length of the longest causal chain from H's root causes
                  to the observation, capped at max_depth
```
Chains are recovered from the structural equations' parent relations
transitively. A hypothesis that cites a root cause (e.g., "the glass broke
because it was struck") has higher causal depth than one citing an
intermediate effect ("the glass broke because it was fragile"), even if
both are true.

**Engine integration**: COMMONSENSE maintains a causal graph as part of its
`hard_constraint` entries (type: `causal`). Each edge A → B is a structural
equation assertion. When scoring hypotheses for explanatory coherence,
hypotheses that trace a causal path to the observation receive higher
`causal_depth` scores. Hypotheses that merely assert a correlation without
a causal path are penalized.

**Connection to Bochman (Section 4A, D6)**: The `causal_direction` field on
default rules provides the structural equations. A rule with
`causal_direction: forward` asserting `struck(x) ⇒ broken(x)` is a causal
production; the Halpern-Pearl framework verifies whether it constitutes an
actual cause in a given context.

#### D7. Schema Delta

Add optional fields to the `commonsense_entry.schema.json`:

```json
{
  "hypothesis": {
    "explains": {"type": "array", "items": {"type": "string"},
      "description": "Observation IDs this hypothesis claims to explain"},
    "competitors": {"type": "array", "items": {"type": "string"},
      "description": "IDs of competing hypotheses"},
    "score": {"type": "object",
      "properties": {
        "consilience": {"type": "number"},
        "simplicity": {"type": "number"},
        "analogy": {"type": "number"},
        "conservatism": {"type": "number"},
        "causal_depth": {"type": "number"},
        "falsifiability": {"type": "number"},
        "composite": {"type": "number"}
    }},
    "acceptance_status": {"enum": ["proposed", "accepted", "rejected", "superseded"]}
  }
}
```

---

## 5. Research Plan: Autonomous Construction Pipeline

The goal is not to write the spine by hand. It is to build a pipeline
that retrieves canonical sources, extracts formal content, verifies it
against the source, and writes it into the Obsidian vault. A human
reviews the output; the agent does the mechanical work.

### Phase A: Retrieval (Source Canon)

**Objective**: Collect the ~40 core papers and SEP articles that define
the formal landscape for all 11 spine sections.

**Primary sources** (Stanford Encyclopedia of Philosophy):
- Non-Monotonic Logic (SEP entry by Gabbay, et al.)
- Logic of Belief Revision (SEP entry by Hansson)
- Inductive Logic (SEP entry by Hawthorne)
- Logic and Artificial Intelligence (SEP entry by Thomason)
- Defeasible Reasoning (SEP entry by Koons)
- Abduction (SEP entry by Douven)
- The OSCAR Project (SEP entry on Pollock's defeasible reasoner)

**Secondary sources** (original papers):
| Author(s) | Year | Paper | Spine section |
|-----------|------|-------|---------------|
| Reiter | 1980 | A Logic for Default Reasoning | 4A (Default logic) |
| McCarthy | 1980 | Circumscription — A Form of Non-Monotonic Reasoning | 4A (Circumscription) |
| McDermott & Doyle | 1980 | Non-Monotonic Logic I | 4A (Modal nonmonotonicity) |
| Moore | 1985 | Semantical Considerations on Nonmonotonic Logic | 4A (Autoepistemic logic) |
| Shoham | 1987 | A Semantical Approach to Nonmonotonic Logics | 4A (Preferential semantics) |
| Kraus, Lehmann, Magidor | 1990 | Nonmonotonic Reasoning, Preferential Models and Cumulative Logics | 4A (KLM systems) |
| Lin & Shoham | 1992 | A Logic of Knowledge and Justified Assumptions | 4A (GK logic) |
| Dung | 1995 | On the Acceptability of Arguments | 4A (Argumentation) |
| Pollock | 1991 | A Theory of Defeasible Reasoning | 4A (OSCAR, undercutting/rebutting) |
| Pollock | 2000 | Defeasible Reasoning in OSCAR | 4A (Defeat-status algorithm) |
| Horty | 2007 | Defaults with Priorities | 4A (Prioritized defaults) |
| Horty | 2012 | Reasons as Defaults | 4A (Variable-priority defaults) |
| Bochman | 2004 | A Causal Approach to Nonmonotonic Reasoning | 4A (Causal production) |
| Bochman | 2005 | Explanatory Nonmonotonic Reasoning | 4A (Biconsequence relations) |
| Alchourron, Gardenfors, Makinson | 1985 | On the Logic of Theory Change | 4B (AGM revision) |
| Darwiche & Pearl | 1997 | On the Logic of Iterated Belief Revision | 4B (DP postulates) |
| Boutilier | 1993 | Revision by Conditional Beliefs | 4B (Natural revision) |
| Boutilier | 1994 | Unifying Default Reasoning and Belief Revision in a Modal Framework | 4B (Conditional logic) |
| Hansson | 1999 | A Textbook of Belief Dynamics | 4B (Contraction taxonomy) |
| Goodman | 1954/1983 | Fact, Fiction, and Forecast (Ch. 3: The New Riddle of Induction) | 4C (Grue, projectibility) |
| Hempel | 1945 | Studies in the Logic of Confirmation | 4C (Raven paradox, confirmation) |
| Shafer | 1976 | A Mathematical Theory of Evidence | 3.6 (Dempster-Shafer) |
| Spohn | 2012 | The Laws of Belief | 3.6 (Ranking theory) |
| Thagard | 2000 | Coherence in Thought and Action | 4D (Explanatory coherence) |
| Lipton | 2004 | Inference to the Best Explanation | 4D (IBE) |
| Halpern & Pearl | 2005 | Causes and Explanations: A Structural-Model Approach | 4D (Actual causation) |
| Pearl | 2000/2009 | Causality (Ch. 1-3) | 4D (Structural causal models) |
| Grunwald | 2007 | The Minimum Description Length Principle | 3.7 (MDL) |
| Solomonoff | 1964 | A Formal Theory of Inductive Inference | 3.7 (Algorithmic probability) |
| Levesque | 1984 | Foundations of a Functional Approach to Knowledge Representation | 2 (TELL/ASK semantics) |
| Fagin, Halpern, Moses, Vardi | 1995 | Reasoning about Knowledge | 2 (Epistemic logic) |
| Gardenfors | 2000 | Conceptual Spaces | 2 (Similarity spaces) |
| Rosch | 1978 | Principles of Categorization | 2, 4C (Typicality) |
| Simon | 1969/1996 | The Sciences of the Artificial (Ch. 3, 5) | 3.10 (Bounded rationality) |
| Gabbay, Hogger, Robinson (eds.) | 1994 | Handbook of Logic in AI, Vol. 3: Nonmonotonic and Uncertain Reasoning | Cross-cutting survey |
| Makinson | 1994 | General Patterns in Nonmonotonic Reasoning (in Handbook, Vol. 3) | Cross-cutting taxonomy |

**Less obvious but high-yield**:
- 1980s KR literature: Brachman & Levesque (1985), Hayes "Naive Physics Manifesto" (1978)
- Old Cyc papers: Lenat & Guha "Building Large Knowledge-Based Systems" (1990)
- Cognitive science: Tversky & Kahneman "Judgment under Uncertainty" (1974, heuristics); Gigerenzer "Simple Heuristics That Make Us Smart" (1999)
- Philosophy of science: Kuhn "The Structure of Scientific Revolutions" (1962, paradigm shifts as belief revision); Lakatos "Falsification and the Methodology of Scientific Research Programmes" (1970, hard core vs protective belt as entrenchment)

**Output of Phase A**: A directory `SOURCES/` containing clean text + metadata
for each source. File naming convention: `Author_Year_Title.txt`.

### Phase B: Extraction Pipeline (Tiny Models + Deterministic Verification)

**Pipeline architecture** (each stage is independently testable):

```
SOURCE.txt → [Scraper: clean text] → CHUNKS.jsonl
                                    → [Triage: 1B-3B model] → RELEVANT.jsonl
                                                              → [Extraction: 3B-8B model] → EXTRACTS.jsonl
                                                                                           → [Verifier: regex + source check] → VERIFIED.jsonl
                                                                                                                                     → [Formatter: Obsidian writer] → VAULT/*.md
```

**Stage 1: Scraper (no model)**
- Input: `SOURCES/Author_Year_Title.txt` (already downloaded, HTML stripped)
- Processing: Split by section headings, number paragraphs, assign chunk IDs
- Chunk ID format: `{SOURCE_ID}/sec{section}/para{paragraph}`
- Output: `CHUNKS.jsonl` — one JSON object per chunk with fields:
  `{chunk_id, source_id, section_heading, paragraph_number, text}`

**Stage 2: Triage (1B–3B model)**
- Input: `CHUNKS.jsonl`
- Task: For each chunk, classify into one or more spine sections (1-11)
  AND tag with `canonical` (contains formal definitions/postulates) or
  `supplementary` (commentary, examples, motivation)
- Prompt discipline: The model MUST output structured JSON with fields
  `{chunk_id, sections: [int], is_canonical: bool, confidence: float}`.
  No freeform text.
- Verification: If `confidence < 0.7`, flag for human review.
- Output: `RELEVANT.jsonl` — chunks annotated with section tags

**Stage 3: Extraction (3B–8B model)**
- Input: `RELEVANT.jsonl` filtered to canonical chunks only
- Task: Extract structured definitions, distinctions, postulates, examples,
  and cross-links in schema-constrained JSON
- Schema: Each extract must have fields:
  ```json
  {
    "extract_id": "...",
    "chunk_id": "...",
    "type": "definition | postulate | distinction | example | theorem",
    "content": { ... schema depends on type ... },
    "formal_statement": "...",  // verbatim quote from source
    "quote_span": "para 3, lines 2-5",
    "natural_language_gloss": "...",  // allowed to be model-generated
    "cross_links": ["[[Other Concept]]", ...]
  }
  ```
- Critical rule: `formal_statement` MUST be a verbatim quote from the source.
  The model can paraphrase in `natural_language_gloss`. If the model cannot
  locate a verbatim formal statement, it MUST NOT invent one.

**Stage 4: Verification (no model, deterministic)**
- Input: `EXTRACTS.jsonl` + `SOURCES/`
- For each extract:
  1. Verify that `chunk_id` exists in `CHUNKS.jsonl`
  2. Verify that `source_id` exists in `SOURCES/`
  3. Verify that `formal_statement` appears verbatim in the source text at
     the claimed `quote_span`
  4. Verify that `cross_links` reference either existing vault notes or
     canonical source IDs (no dangling links)
- Extracts that fail any check are flagged `verification: failed` and
  routed to human review with the failure reason
- Output: `VERIFIED.jsonl` — extracts that passed all checks

**Stage 5: Format (no model, deterministic template)**
- Input: `VERIFIED.jsonl`
- Output: One `.md` file per extract, written to the appropriate vault
  folder (matching the 11-section outline)
- Template:
  ```markdown
  ---
  title: "{type}: {concise summary}"
  tags: [{section_tags}, "canonical"]
  source: "{source_id}"
  chunk_id: "{chunk_id}"
  extract_id: "{extract_id}"
  verified: {timestamp}
  ---

  ## Formal Statement
  > {formal_statement}  <!-- verbatim quote -->

  ## Gloss
  {natural_language_gloss}

  ## Cross-References
  {cross_links as wikilinks}

  ## Source
  - **Paper**: {source citation}
  - **Location**: {quote_span}
  ```

### Phase C: Vault Organization (Obsidian-Compatible)

**Directory structure** (flat within each category, wikilinks connect them):

```
VAULT/
  /SPINE/
    01_Problem_Frame.md
    02_Representation_Substrate.md
    ...
    11_Minimal_Operators.md
  /Representation/          # detailed notes extracted from sources
    predicate_typing.md
    admissibility_function.md
    natural_kinds.md
    similarity_spaces.md
    ...
  /Inference/
    deduction_operators.md
    induction_operators.md
    abduction_operators.md
    analogy_mapping.md
    causal_inference.md
    ...
  /Defaults/
    specificity_ordering.md
    default_logic_reiter.md
    circumscription.md
    argumentation_frameworks.md
    ...
  /Revision/
    AGM_postulates.md
    entrenchment_ordering.md
    iterated_revision.md
    truth_maintenance.md
    ...
  /Induction/
    goodman_grue.md
    projectibility_constraints.md
    hempel_paradox.md
    sample_variety.md
    ...
  /Coherence/
    explanatory_virtues.md
    IBE_lipton.md
    thagard_network.md
    causal_explanation.md
    ...
  /Uncertainty/
    bayesian_updating.md
    ranking_theory.md
    calibration.md
    ...
  /Simplicity/
    MDL_principle.md
    solomonoff_induction.md
    occam_formalized.md
    ...
  /Bounded/
    satisficing.md
    metareasoning.md
    epistemic_virtues.md
    heuristics_toolbox.md
    ...
  /TESTS/
    competing_defaults.json
    exception_introduction.json
    belief_revision_contradiction.json
    grue_rejection.json
    abduction_scoring_revision.json
    nixon_diamond.json
    ...
  /META/                    # this document and governance
    META_LOGIC.md
    CODEBOOK.md
    CHANGELOG.md
    ...
```

**Frontmatter standard** (every .md file):

```yaml
---
title: string         # e.g., "Specificity Ordering in Default Logic"
tags: [list]          # e.g., ["defaults", "reiter", "canonical"]
categories: [list]    # e.g., ["/Defaults/", "/SPINE/"]
source: string        # SOURCE_ID from Phase A
date_extracted: ISO   # when the pipeline produced this note
verified: bool        # passed Stage 4 verification?
confidence: float     # model's extraction confidence (for non-canonical content)
aliases: [list]       # alternative names for this concept
---
```

### Phase D: Export to Queryable Form

**SQLite schema**:

```sql
CREATE TABLE notes (
    id TEXT PRIMARY KEY,          -- extract_id
    title TEXT NOT NULL,
    content TEXT NOT NULL,        -- full markdown body
    tags TEXT,                    -- JSON array
    categories TEXT,              -- JSON array
    source_id TEXT,
    chunk_id TEXT,
    verified INTEGER DEFAULT 0,
    extracted_at TEXT,
    embedding BLOB                -- if using sqlite-vec
);

CREATE TABLE cross_links (
    source_note_id TEXT,
    target_note_id TEXT,
    link_type TEXT,               -- "definition", "postulate", "example", etc.
    PRIMARY KEY (source_note_id, target_note_id)
);

CREATE TABLE predicate_entrenchment (
    predicate_id TEXT PRIMARY KEY,
    projections_attempted INTEGER DEFAULT 0,
    projections_succeeded INTEGER DEFAULT 0,
    projections_failed INTEGER DEFAULT 0,
    entrenchment_score REAL,
    last_updated TEXT
);

CREATE VIRTUAL TABLE notes_fts USING fts5(title, content, tags, categories);
```

**Vector index** (via sqlite-vec or Chroma):
- Embed chunk content using `all-MiniLM-L6-v2` (384-dim, runs on CPU,
  good semantic retrieval for short technical text)
- Alternative: `nomic-embed-text-v1.5` (768-dim, better for longer
  paragraphs, requires more RAM)
- Hybrid search: combine SQL WHERE clauses (filter by section, tag,
  verified status) with vector similarity (ORDER BY embedding_distance)

**Query examples the agent should support**:
- "What is the AGM postulate for vacuity?" → semantic search "vacuity"
  filtered by category `/Revision/`
- "Show me all postulates related to belief change" → SQL filter:
  `tags LIKE '%postulate%' AND categories LIKE '%Revision%'`
- "How does projectibility relate to induction?" → hybrid: semantic query
  "projectibility induction" + filter categories `['/Induction/']`


## 6. Tiny Model Strategy (Graded Capability)

### 6.1 Model Size vs. Task Mapping (Expanded)

| Model size | Examples | Good for | Good for (spine-specific) | Avoid | Failure rate on logic tasks |
|------------|----------|----------|--------------------------|-------|---------------------------|
| < 1B | Qwen2.5-0.5B, SmolLM2-360M | Routing, tagging, light classification | Identifying whether a chunk contains a numbered postulate (binary) | Any extraction or synthesis | ~40% (random on formal content) |
| 1B–3B | Qwen2.5-1.5B, Llama-3.2-1B, SmolLM2-1.7B | Triage, coarse topic bucketing, schema-constrained extraction with verification | Classifying chunks into spine sections (1-11), tagging as canonical/supplementary | Logic-heavy formalism, long context, multi-step reasoning | ~20% on section classification |
| 3B–8B | Qwen2.5-7B, Llama-3.1-8B, Phi-4-mini | Structured extraction with hard schema, definitions, postulates, examples | Extracting AGM postulates from text, identifying formal definitions, mapping concepts to cross-links | Cross-source synthesis, symbol-level precision, generating new formal content | ~10% on extraction with verification |
| > 8B (local) | Qwen2.5-14B, Llama-3.1-70B (quantized), DeepSeek-V3 | Synthesis across sources, narrative polishing, complex cross-domain analogy | Writing SPINE/*.md summary notes from extracted canonical content | Unverified formal claims (still hallucinates) | ~5% on synthesis from extracted notes |
| Online (API) | Claude, GPT-4, Gemini | Draft narrative summaries, teaching versions, QA over vault content | "Explain AGM revision to a newcomer" — but input must be already-extracted canonical notes | New formal claims, "discovering" new postulates | ~2% on constrained summarization |

### 6.2 Why Tiny Models Can Work (With the Right Constraints)

The core insight of this strategy: **tiny models are bad at logic but
acceptable at pattern matching within a constrained schema**. The pipeline
design exploits this by never asking a tiny model to *reason*. It only
asks it to *locate and reformat*.

**What a 3B model can do reliably on formal logic text**:
- Identify that "If K is a belief set and P is a proposition..." is a
  definition (pattern: "If X is a Y and Z is a W..." introduces a
  formal definition in analytic philosophy prose)
- Extract the antecedent and consequent of an "if-then" postulate
  (pattern: the sentence structure is formulaic)
- Distinguish a postulate from an example (pattern: postulates use
  abstract variables; examples use concrete instances)
- Map a named concept ("Levi Identity") to its definition sentence
  (pattern: "The Levi Identity states that K * P = (K / ~P) + P")

**What a 3B model CANNOT do reliably on formal logic text**:
- Verify that a set of postulates is consistent
- Generate a proof that one postulate follows from another
- Detect subtle errors in formal notation (e.g., a missing negation sign)
- Synthesize a novel postulate from examples
- Transcribe formal notation without hallucination (replaces symbols,
  drops quantifiers, swaps connectives)

**The verifier is the equalizer**: Every extraction that contains a
`formal_statement` is checked character-by-character against the source
text. If the model hallucinates "K * P = (K / ~P) + P" as "K * P = K / ~P"
(dropping the "+ P"), the verifier catches it because the character
strings don't match. This is why Stage 4 (verification) is deterministic
and has no model.

### 6.3 Chunking Strategy (How to Feed Tiny Models)

**Rule 1**: One section at a time. A 10-page section on AGM revision is
chunked by subsection heading, not by fixed token count. Each chunk
contains exactly one conceptual unit (one postulate, one definition, one
theorem statement, one example).

**Rule 2**: Context window for each chunk includes the section heading and
the immediately preceding/succeeding paragraphs (for context), but the
model is asked to extract ONLY from the target paragraph.

**Rule 3**: Formal statements are NEVER split across chunks. If a postulate
statement spans 3 sentences, those 3 sentences form one chunk, even if
that chunk is longer than the target token count.

**Rule 4**: Examples are extracted separately from postulates. The model
gets the postulate chunk first, extracts the formal statement. Then it
gets the example chunk and maps it to the postulate it already extracted.

### 6.4 Failure Modes and Mitigations

| Failure mode | Likely cause | Mitigation |
|-------------|-------------|-----------|
| Extracted postulate is missing a quantifier | Model dropped "for all" or "there exists" | Verifier catches non-verbatim extraction; flagged for human |
| Two postulates are swapped (postulate 5 content under postulate 3 label) | Model confusion from adjacent chunk context | Verifier cross-references chunk_id; if the extracted text doesn't appear in the claimed chunk, it's flagged |
| Model invents a plausible but false cross-link | Hallucination in cross-link generation | Verifier checks that linked note exists; if dangling, flagged |
| Model misclassifies a counterexample as a supporting example | Failure to understand negation scope | Extracts both the example and its classification; human review for `confidence < 0.7` |
| Model removes informal commentary that clarifies the formal statement | Over-aggressive "formal only" filtering | Extraction prompt explicitly asks to preserve gloss paragraphs; they go into `natural_language_gloss` |


## 7. Compression & Offloading (Hybrid Local/Online Strategy)

### 7.1 Local-First Architecture

The pipeline runs fully locally for extraction, verification, and database
population. Online models are used ONLY for optional post-processing steps
that do not introduce new formal claims.

**Local pipeline** (runs on a laptop with 16GB RAM):
```
SOURCES/ → [chunk] → [triage: 1B] → [extract: 7B quantized] → [verify: deterministic] → VAULT/
```

**Online pass** (optional, requires explicit user approval, MCP/web tool):
```
VAULT/ → [online model: narrative compression] → VAULT/SUMMARIES/ (teaching versions)
                                              → VAULT/GLOSSARY.md (cross-linked concept index)
```

### 7.2 What Local Models Can Do (Safe Operations)

- **Extractive compression**: Given a chunk of source text, identify the
  key sentences that contain definitions and postulates. Store them
  verbatim. This is classification + transcription, not generation.
- **Structural compression**: Given a chunk's extracted formal statements,
  organize them into a hierarchical outline (e.g., "Postulate 1 (Closure)
  → Postulate 2 (Success) → ..."). The model reorders but does not
  rewrite the formal statements.
- **Paraphrasing plain prose**: Given a paragraph of philosophical
  motivation (not formal content), rewrite it more concisely. The model
  may introduce errors here, but they are in the gloss, not in the
  formal record.
- **Citation binding**: Given an extracted formal statement, locate the
  exact character span in the source text. Deterministic regex + fuzzy
  matching; no model needed.
- **Link graph construction**: Given two extracted notes, determine if
  one concept is used in the definition of another. The model classifies
  the relationship type (defines, uses, contrasts, generalizes). Errors
  here produce bad cross-links, not bad formal content.

### 7.3 What Local Models Cannot Do (Unsafe Operations)

- **Formal compression**: "Summarize the 6 AGM postulates into 2." The
  model WILL hallucinate. The formal statements are already minimal.
- **Cross-source synthesis**: "Synthesize Reiter's default logic and AGM
  revision into a unified framework." The model does not understand the
  math; it will produce plausible-sounding nonsense.
- **Theorem proving**: "Does postulate 5 follow from postulates 1-4?"
  Even large models fail at multi-step formal reasoning.
- **Notation consistency checking**: "Does this paper use K for belief
  set and this other paper use B?" The model can flag this as a question
  but should not resolve it.

### 7.4 The Online Safety Protocol

When online models ARE used (with user approval):

1. **Input gate**: The input to the online model MUST be an already-extracted
   canonical note from the vault (formatted as Markdown). Never feed raw
   source text to an online model for extraction.
2. **Output label**: All online-generated content MUST be prefixed with:
   > `[ONLINE DRAFT — NOT VERIFIED]`
3. **No new claims**: The online model may reorganize, paraphrase, and
   simplify, but it must not introduce concepts, postulates, or citations
   not already present in the input.
4. **Citation preservation**: Every claim in an online-generated summary
   must preserve its source citation. If the model drops a citation, the
   summary is rejected.
5. **Human-in-the-loop**: Online-generated content goes to `VAULT/_DRAFTS/`
   (not `VAULT/SPINE/`). A human must move it to the main vault.

### 7.5 Token Budget (Per Source)

For a typical 20-page paper (~10,000 words of original text):

| Stage | Model | Token input | Token output | Cost (local) |
|-------|-------|------------|-------------|-------------|
| Chunking | None | — | — | Free |
| Triage | 1.5B | ~8,000 | ~2,000 | ~1 sec |
| Extraction | 7B q4 | ~16,000 | ~4,000 | ~10 sec |
| Verification | None | — | — | Free |
| Formatting | None | — | — | Free |
| **Total per paper** | | | | **~11 sec** |

For 20 papers: ~4 minutes of local compute. The bottleneck is human
review, not model throughput.


## 8. Success Exit Condition (Formal Acceptance Test)

### 8.1 What "Done" Means

The spine vault is complete when ALL of the following conditions are
satisfied, verified deterministically:

#### Gate 1: Coverage

- [ ] Vault contains notes for all 11 spine sections
- [ ] Each section has at least one canonical note (formal content verified
  against a primary or secondary source)
- [ ] Each section has a `/SPINE/` summary note (the outline entry,
  synthesizing the canonical extracts in that section)
- [ ] No domain content (physics, social, etc.) — verified by tag audit:
  zero notes tagged `physics`, `social`, `domain:*`

#### Gate 2: Source Provenance

- [ ] Every canonical note has a `source` field referencing a specific
  paper/SEP entry from Phase A
- [ ] Every `formal_statement` in every canonical note appears verbatim
  in the referenced source (verified by the deterministic verifier)
- [ ] Zero dangling cross-links: every `[[wikilink]]` resolves to an
  existing note in the vault

#### Gate 3: Operator Coverage

- [ ] Each of the 20 minimal operators (Section 3, item 11) is defined
  in at least one vault note with a canonical source
- [ ] Each operator has a corresponding entry in the SQLite database
  with its formal signature and source reference

#### Gate 4: Test Suite

The `/TESTS/` directory contains executable fixtures that exercise:

| Test case | What it verifies | Section reference |
|-----------|-----------------|-------------------|
| **Competing defaults** | Specificity ordering resolves penguin/bird correctly | 4A, D1 |
| **Exception introduction** | Adding "Tweety is a penguin" retracts "Tweety flies" | 4A, D2 |
| **Belief revision after contradiction** | Observing NOT P when P is believed triggers contraction | 4B, B4 |
| **Grue-like predicate rejection** | A rule using a time-indexical predicate is blocked from induction | 4C, C3-C4 |
| **Abduction → scoring → revision** | Surprising observation → candidate hypotheses generated → scored → best accepted → beliefs revised | 4D, D1-D5 |
| **Nixon Diamond** | Conflicting defaults with equal specificity → skeptical semantics applied | 4A, D4 |
| **Iterated revision** | P → NOT P → P: the second revision does not restore the pre-first-revision state | 4B, B3 |
| **Sample variety gate** | Induction from homogeneous sample blocked; induction from varied sample accepted | 4C, C5 |
| **Coherence network convergence** | Thagard network with 3 hypotheses and 5 observations converges to stable activation within 100 iterations | 4D, D3 |
| **Entrenchment ordering** | Contraction removes observation before default, default before hard constraint | 4B, B3 |

Each fixture is a JSON file with:
```json
{
  "fixture_id": "string",
  "description": "string",
  "input": { "facts": [...], "db": [...], "policies": {...} },
  "expected": { "selected_ids": [...], "derived_facts": [...], "emits": [...],
                "contracted": [...], "accepted_hypotheses": [...] },
  "tolerance": { "fields": {"confidence": 0.05}, "ordering": true }
}
```

#### Gate 5: Query Competence

A tiny agent (1B-3B model) connected to the SQLite + vector index can
answer these questions without hallucination (answers must cite vault notes):

| Query | Expected answer source |
|-------|----------------------|
| "What is specificity in default logic?" | Note: specificity_ordering.md, source: Reiter 1980 |
| "How does AGM revision work?" | Note: AGM_postulates.md, source: AGM 1985 |
| "Why is 'grue' not projectible?" | Note: goodman_grue.md, source: Goodman 1954 |
| "What are the explanatory virtues?" | Note: explanatory_virtues.md, source: Thagard 2000 |
| "What's the relationship between MDL and Bayes?" | Note: MDL_principle.md, source: Grunwald 2007 |

The agent is considered competent if it answers 5/5 correctly, where
"correctly" means: (1) the answer is factually consistent with the vault
note, (2) the answer cites the correct vault note, and (3) the answer
does not add claims not present in the vault.

#### Gate 6: Self-Hosting

The COMMONSENSE resolver (the engine that runs the operators in Section 4)
can load the vault's SQLite database as its `db.json` equivalent and
resolve queries against it. In other words: **the spine describes an
engine that can consume the spine's own output.**

This is the final and hardest gate. If the extracted formal content
(definitions, postulates, constraints) is sufficiently structured, the
resolver should be able to reason about the spine itself — answering
meta-questions like "Is there a postulate in the AGM framework that
contradicts Reiter's default logic?" by loading both sets of formal
statements and running conflict detection.

### 8.2 One-Line Compression (For the Agent's Own Use)

> **Common sense = defeasible induction over a projectibility-constrained
> language, guided by simplicity + coherence, graded by uncertainty,
> stabilized by AGM belief revision, and selecting hypotheses via
> explanatory coherence. The spine is the meta-inference layer that
> implements these operators as deterministic functions over structured
> primitives, enabling a local agent to reason under partial information
> without hallucination.**

### 8.3 What Comes After

**That is the spine.** Build that, and any domain (physics, social, legal,
governance) can later be plugged in as content — as `default_rule`
entries, `hard_constraint` entries, `explanation_template` entries —
and the spine will reason about them with the same operators. The spine
does not know about electrons or contracts or social norms. It knows
about defaults, exceptions, contradictions, generalizations, and
explanations. Those are enough.