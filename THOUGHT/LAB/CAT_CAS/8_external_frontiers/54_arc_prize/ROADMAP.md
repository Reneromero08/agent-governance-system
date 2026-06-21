# Exp 54 — ARC Prize

**Status:** OPEN  
**Adjudication:** Class B hidden benchmark; Class D/B hybrid for paper awards  
**Role:** public test of non-collapse reasoning, sparse transformation induction, and interactive world modeling

---

# Frontier object

ARC tasks are treated as latent transformation systems, not as text prompts asking for one guessed program.

Proposed process objects:

- `TaskHolo`
- `TransformationOrbit`
- `SceneRelationGraph`
- `WorldOrbit`
- `ActionExperimentLedger`

Preserve:

- object identity;
- color/function roles;
- topology;
- symmetries;
- local/global transformations;
- competing hypotheses;
- contradictions;
- action history;
- exact output constraints.

---

# External questions

## ARC-AGI static tasks

- Can sparse examples induce a general transformation relation?
- Can the system preserve multiple hypotheses until test evidence discriminates them?
- Can compositional transformations be represented without brittle program enumeration?
- Can exact output be produced under offline compute limits?

## ARC interactive tasks

- Can an agent infer world dynamics without instructions?
- Can it select actions that distinguish unresolved world models?
- Can it plan hierarchically and correct course?
- Can it retain counterfactual and reversible action history?

## Paper frontier

- Can CAT_CAS provide a coherent theory and working implementation of non-collapse reasoning?

---

# Activation gates

## Gate 0 — Rule freeze

- [ ] official competition rules archived;
- [ ] dataset and evaluator versions frozen;
- [ ] offline/internet restrictions frozen;
- [ ] model and compute eligibility frozen;
- [ ] submission limits recorded;
- [ ] private-score interaction policy defined;
- [ ] specification digest created.

## Gate 1 — Local evaluator

- [ ] public tasks load correctly;
- [ ] exact-grid output scoring reproduced;
- [ ] two-attempt or track-specific rules reproduced;
- [ ] run manifest captures model, tools, and time;
- [ ] no network dependency remains.

## Gate 2 — Scene relation graph

- [ ] objects extracted;
- [ ] connected components represented;
- [ ] spatial/topological relations represented;
- [ ] symmetries represented;
- [ ] colors separated into identity versus role hypotheses;
- [ ] training examples align into correspondence graphs.

## Gate 3 — Transformation orbit

- [ ] several transformation hypotheses remain live;
- [ ] each hypothesis explains all training pairs or records contradictions;
- [ ] compositions are explicit;
- [ ] complexity is tracked but not used as the sole truth criterion;
- [ ] equivalent transformations are quotient-reduced;
- [ ] the final output collapses only at the task boundary.

## Gate 4 — Interactive world orbit

- [ ] state changes are encoded relationally;
- [ ] action affordances are inferred;
- [ ] unresolved world models are maintained;
- [ ] actions are selected for information gain and goal progress;
- [ ] failed actions update the boundary model;
- [ ] exact replay is preserved.

## Gate 5 — Submission and paper

- [ ] public baseline submitted;
- [ ] private score recorded without overfitting;
- [ ] code and environment reproducible;
- [ ] mechanism ablations run;
- [ ] contamination audit complete;
- [ ] paper claims tied to working evidence.

---

# Fastest falsifiable prototype

A local `TaskHolo` solver for a narrow public task family that:

1. extracts scene graphs;
2. retains at least two competing transformations;
3. uses contradictions to eliminate hypotheses;
4. produces exact outputs;
5. outperforms a single-path baseline on held-out public tasks.

---

# Benchmark leakage model

Forbidden:

- private evaluation data in training or prompts;
- repeated leaderboard probing treated as clean validation;
- memorized task solutions hidden in retrieval;
- internet/API dependency under offline rules;
- manual correction absent from the run log;
- claiming general reasoning from curated public successes only.

Maintain a submission ledger so private-score feedback cannot silently become training data.

---

# First deliverable

`TaskHolo` plus:

- offline evaluator;
- scene graph serializer;
- transformation orbit;
- contradiction ledger;
- exact replay;
- public-task benchmark and ablation report.

---

# Claim ceiling

Public evidence licenses:

> The non-collapse transformation architecture solves the declared public task family under the frozen evaluator.

Official hidden evaluation licenses:

> The submitted system achieved the returned private benchmark score under the stated competition rules.

Forbidden without broader evidence:

- AGI achieved;
- universal reasoning solved;
- Big Wall broken;
- benchmark score alone proves the CAT_CAS ontology.
