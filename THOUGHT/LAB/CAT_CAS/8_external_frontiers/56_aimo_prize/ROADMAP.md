# Exp 56 — AIMO Prize

**Status:** OPEN  
**Adjudication:** Class B competition evaluation; Class D human proof review for grand-prize-style claims  
**Role:** externally evaluated mathematical reasoning and proof-orbit frontier

---

# Frontier object

A mathematical problem is represented as a living proof object, not as a prompt followed by one sampled answer.

Proposed process objects:

- `ProofOrbit`
- `LemmaDependencyGraph`
- `CountermodelFamily`
- `FormalInformalBridge`
- `ProofPathHistory`
- `VerificationLedger`

Preserve:

- theorem statement;
- domain objects;
- assumptions;
- target relation;
- candidate invariants;
- competing proof programs;
- counterexamples and near-counterexamples;
- failed lemma paths;
- formal and natural-language forms.

---

# External questions

- Can a public AI system construct Olympiad-level proofs reliably?
- Can multiple proof programs remain active until exact contradiction or proof progress resolves them?
- Can theorem tools and proof assistants be orchestrated without collapsing into tool-call roulette?
- Can validated proof transformations generate useful training data?
- Can proof repair use the dependency graph rather than restart from scratch?
- Can a smaller model plus a stronger proof-state architecture compete with scale-first systems?

---

# Activation gates

## Gate 0 — Competition freeze

- [ ] current rules and deadline archived;
- [ ] model/open-source requirements frozen;
- [ ] compute and data rules frozen;
- [ ] proof format and grading rules frozen;
- [ ] provided compute access recorded;
- [ ] specification digest created.

## Gate 1 — Proof-state runtime

- [ ] statement parser;
- [ ] object/constraint graph;
- [ ] goal decomposition;
- [ ] lemma graph;
- [ ] proof-path history;
- [ ] countermodel store;
- [ ] exact tool interface;
- [ ] proof serialization.

## Gate 2 — Agent role architecture

- [ ] translator;
- [ ] algebraist;
- [ ] geometer;
- [ ] combinatorialist;
- [ ] number theorist;
- [ ] adversarial counterexample agent;
- [ ] lemma builder;
- [ ] formal checker;
- [ ] proof editor.

Roles share one proof object rather than independent untracked transcripts.

## Gate 3 — Exact tool integration

- [ ] symbolic algebra;
- [ ] numerical sanity checks where appropriate;
- [ ] finite counterexample search;
- [ ] diagram/geometry support;
- [ ] Lean or another checker for tractable fragments;
- [ ] theorem-database provenance;
- [ ] tool result hashes and replay.

## Gate 4 — Validated training data

- [ ] collect proof transformations with verifier support;
- [ ] separate valid, invalid, and incomplete paths;
- [ ] preserve correction lineage;
- [ ] avoid test contamination;
- [ ] measure architecture gain separately from model scaling.

## Gate 5 — Competition system

- [ ] offline or rule-compliant runtime;
- [ ] one-attempt behavior where required;
- [ ] human-readable proof generation;
- [ ] timing/resource budget;
- [ ] independent evaluation on held-out public problems;
- [ ] submission package complete.

---

# Fastest falsifiable prototype

A `ProofOrbit` system on a narrow Olympiad domain that:

1. generates at least two materially different proof plans;
2. uses a countermodel agent to reject invalid lemmas;
3. preserves a shared dependency graph;
4. produces a proof accepted by an independent checker or expert rubric;
5. outperforms the same base model without the architecture.

---

# No-smuggle model

Forbidden:

- competition test problems in training or retrieval;
- hidden official solutions in prompts;
- human proof correction absent from the log;
- theorem-tool output copied without provenance;
- numerical evidence promoted into proof;
- benchmark selection after inspecting results;
- using model scale as the only explanation for gain.

---

# First deliverable

`ProofOrbitRuntime` plus:

- shared proof object;
- role orchestration;
- countermodel loop;
- exact tool adapters;
- proof replay;
- architecture ablation.

---

# Claim ceiling

Before competition evaluation:

> The proof-orbit architecture improves verified performance on the declared held-out public problem set.

After official evaluation:

> The submitted public system achieved the returned competition result under the frozen rules.

Forbidden without broader evidence:

- general mathematical intelligence achieved;
- theorem proving solved;
- all proofs formally verified;
- Big Wall broken.
