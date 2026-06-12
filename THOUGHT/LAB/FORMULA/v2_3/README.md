# Living Formula v2.3 - Lab Rules

This directory is a fresh evidentiary restart of the Living Formula research
program. ASCII only in every file.

---

## 0. THE PRIME RULE: nothing from v2_2. Ever.

Owner ruling, 2026-06-12. This rule overrides everything else in this file.

- NO file from v2_2 (or any other FORMULA version) is ever copied into v2_3.
  Not scripts, not libraries, not data, not "just this once."
- NO v2_3 code ever imports from, executes, or reads v2_2 code. v2_2 is
  read-only HISTORY for humans. It is off-limits to verifiers entirely.
- Evidence produced by copied or pre-existing code is INVALID, regardless of
  what it shows. A verdict built on it is void.
- Every question is answered SEPARATELY: the verifier assigned to a question
  writes its OWN implementation from that question's spec, inside that
  question's directory. No experiment code is shared between questions.
  The only shared artifacts in this lab are: the schema, the variable
  registry (VARIABLES.md), the preregistration ledger (PREDICTIONS.md), the
  question catalog (_meta/questions.yaml), and the tooling in _meta/.
- A question directory contains NO executable experiment code until its
  blind verifier writes it. If code exists in a question dir before the
  verifier starts, the run is contaminated.

Why this rule exists: v2_2's claims were entangled with one particular
implementation and its bugs. Re-running inherited code only replays the
recording. Independent reimplementation from spec is the only reproduction
that counts - same math through different code, or it does not copy.

## 1. Fresh start

- NO status is imported from v2_2. Every prior VERIFIED, CONFIRMED, or
  PARTIALLY VERIFIED label is void here.
- All 57 questions begin OPEN. OPEN means: no VERDICT.md exists for that
  question directory yet.
- A status is earned ONLY through the protocol in this README and
  `_meta/VERDICT_SCHEMA.md`. There is no other path.
- v2_2 material may be cited as `predecessor` provenance (a path pointer,
  nothing more), never as evidence and never as code.

## 2. Generated index

- `INDEX.md` is GENERATED. Never hand-edit it.
- Regenerate: `python _meta/generate_index.py`
- CI-style check: `python _meta/generate_index.py --check` (byte-compare).
- Index hypothesis text comes only from `_meta/questions.yaml`; verdicts can
  never inject text into the index.

## 3. Status discipline

Ascending total order:

    FALSIFIED < UNSUPPORTED < PARTIALLY_VERIFIED < VERIFIED

- MIN rule: a verdict's overall status equals the MINIMUM over its claims'
  statuses. The validator recomputes it; you cannot assert it.
- Floor rules:
  1. Empty evidence_manifest => every claim must be UNSUPPORTED or
     FALSIFIED.
  2. Any VERIFIED or PARTIALLY_VERIFIED claim must cite at least 1 evidence
     path from the manifest.
- Legacy vocabulary (CONFIRMED, "PARTIALLY VERIFIED" with a space, boundary
  qualifiers) is invalid and fails validation.

## 4. Containment

- ALL writes land inside `THOUGHT/LAB/FORMULA/v2_3/`. No exceptions.
- There is NO vendoring and NO `_lib/`. External repo code is not brought
  in - see the Prime Rule. If a question needs a capability, its verifier
  implements it from the spec inside the question's own directory.
- Large read-only caches (e.g. HF model caches under
  `LAW/CONTRACTS/_runs/`) may be READ via environment variables. Never
  written, never copied in.

## 5. How a question gets answered

AGENTS: do not work from this summary. Execute `_meta/RUNBOOK.md` exactly -
it is the step-by-step law with gates and copy-paste commands. Templates
live in `_meta/prompts/`.

The questions are CONNECTED. Solved questions make the next ones easier:
verified results are citable premises for every later campaign. The
verifier is INFORMED - it works with the lab's accumulated knowledge, not
in the dark.

Per question, in order:

1. HYPOTHESIS.md is written fresh from the catalog entry: claims,
   falsifiers with thresholds, and a complete measurement spec. It is the
   CLAIM document - forward-looking, no observed results, no references to
   any pre-existing implementation, no inherited seeds. (Context lives in
   the brief, not here.)
2. Preregistration: a P-NNN row in PREDICTIONS.md, BEFORE anything runs.
3. CONTEXT BRIEF: hypothesis + registry quotes + success criteria +
   ESTABLISHED RESULTS. Any claim VERIFIED in this lab may be cited as a
   premise, with its verdict path. Unverified expectations and v2_2
   results are NEVER premises. Saved to `_packets/`, gated per RUNBOOK.
4. The VERIFIER (fable-class, informed): receives the brief and may read
   the entire v2_3 lab - index, verdicts, docs. It WRITES ITS OWN
   implementation (Prime Rule), documents every parameter choice, runs the
   experiment including any control conditions the spec defines, records
   verbatim commands/output/exit codes, and drafts VERDICT.md
   (`verification: primed` - the schema's term for an informed run).
   Still absolutely off-limits: v2_2 and every other FORMULA version, and
   any pre-existing implementation of the measurement. Reading another
   question's verdict is encouraged; reusing another question's code is a
   Prime Rule violation.
5. Validation: `python _meta/validate_verdict.py` must exit 0.
6. Refuters (for VERIFIED): N fresh adversarial agents attack the result -
   permutation/shuffle controls, seed sensitivity, threshold gaming,
   ablations, mundane alternative explanations. N = 2 for Tier 1 questions
   or extraordinary effects (d > 1.0, R2 > 0.99, or p < 1e-5), else N = 1.
   VERIFIED stands only if all return UNREFUTED.
7. INDEX regenerates. The status is whatever the verdict earned.

### Optional blind comparison (owner's call, spare tokens only)

AFTER an informed run completes, a BLIND control agent may be dispatched on
the same question: packet-only context (`lint_packet.py` enforced), zero
lab knowledge, its own independent implementation. Its outcome is appended
to the verdict's `verifications:` list (`mode: blind`) and compared against
the informed run - measuring what lab knowledge adds or biases. This is an
experiment ON the protocol. It NEVER gates the question's status.

## 6. Controls

- Q25 (sigma derivation) is the POSITIVE control: a sound pipeline should
  verify it from spec with fresh code. If the protocol cannot verify Q25,
  the instrument is broken.
- Q49 (Df*alpha = 8e) is the NEGATIVE control: a sound pipeline should
  falsify it from spec with fresh code. If the protocol verifies Q49, the
  instrument is broken.
- Run both controls before trusting any other verdict produced by this
  protocol.
- Inside each experiment, the measurement spec should define its own
  controls where applicable (null/scrambled input that must show nothing;
  known-answer input that must show signal). The verifier runs and reports
  them alongside the main result.

## 7. Preregistration

- `PREDICTIONS.md` is an append-only ledger. Every quantitative prediction
  gets a P-NNN row BEFORE the experiment runs.
- A blind verdict with no prediction rows fails validation (E_PREDICTION).

## 8. Layout

    v2_3/
      README.md            this file
      INDEX.md             generated; do not edit
      PREDICTIONS.md       append-only preregistration ledger
      VARIABLES.md         variable/instrument registry (registry_ids rows)
      V2_2_RECONCILIATION.md  audit record of v2_2 (history, not evidence)
      _meta/               RUNBOOK.md (the campaign law), schema, catalog,
                           generator, validator, prompts + templates,
                           fixtures
      q<NN>_<slug>/        one directory per question, created when its
                           campaign starts: HYPOTHESIS.md, _packets/,
                           then verifier-authored code + results/ +
                           VERDICT.md

## 9. Verdict files

See `_meta/VERDICT_SCHEMA.md` for the full normative schema (frontmatter
fields, body sections, error codes). Summary: YAML frontmatter
(schema: verdict/v2), claims with falsifiers, hashed evidence manifest,
required body sections Hypothesis / Claims / Method / Results / Status /
Provenance, single `**Status:**` line matching frontmatter. Verdicts must
state under Provenance who authored the implementation; evidence from code
the verifier did not write itself is invalid (Prime Rule).
