# RUNBOOK - Running a question campaign in v2_3

READ THIS ENTIRE FILE BEFORE TOUCHING ANYTHING. Execute it EXACTLY.
If you think you have a better way: you do not. If the runbook seems wrong
or does not cover your situation: STOP and ask the owner. Improvising is a
violation, not initiative.

ASCII only in every file. All writes stay inside THOUGHT/LAB/FORMULA/v2_3/.

## The shape of this lab

The 57 questions are CONNECTED. Verified results are citable premises for
later campaigns - the more questions solved, the easier the rest become.
Verifiers are INFORMED: they work with the lab's accumulated knowledge.
Rigor comes from preregistration, in-experiment controls, adversarial
refuters, and hard gates - not from keeping the scientist ignorant.

## Roles

- ORCHESTRATOR: the agent reading this file. Prepares files, dispatches
  other agents, runs gates. The orchestrator never writes experiment code,
  never writes or edits a VERDICT.md body, never decides a status. Its
  only permitted verdict edits: appending rows to `verifications:`.
- VERIFIER: a fable-class agent doing the science for ONE question. It
  receives the context brief, may read the entire v2_3 lab, writes ITS OWN
  implementation, runs it, drafts the verdict.
- REFUTER: a fresh fable-class agent that receives the brief plus the
  numeric results and tries to break the finding.
- BLIND CONTROL (optional, Stage 9): a fresh agent with packet-only
  context, used AFTER the informed run to measure what lab knowledge adds.
  Never gates a status.

## Model ladder (owner directive - violating this wastes owner money)

Waste = model mismatched to task, in EITHER direction.

- THE SCIENCE RUNS ON FABLE. Verifiers and refuters ARE the science:
  they design implementations from spec, write the code, do the math,
  attack the statistics. Dispatch them on fable-class. A weaker model
  produces weaker science that has to be redone - that is the expensive
  failure, not the token price.
- Mechanical work NEVER runs on fable: gates, greps, file checks, copying,
  formatting, index regeneration. The orchestrator runs these itself via
  shell commands, or dispatches haiku-class.
- sonnet-class has no lane in this lab: too expensive for plumbing, too
  weak for the science.
- Set the model explicitly on EVERY dispatch. An unset model field is a
  violation.

## The Prime Rule (README section 0 - overrides everything)

Nothing from v2_2, ever. No copies, no imports, no reads by verifiers.
Every question is answered SEPARATELY with code its own verifier wrote from
the spec. Evidence from copied or pre-existing code is void. A question dir
contains no executable code until its verifier writes it. Reading another
question's VERDICT is encouraged; reusing another question's CODE is a
violation.

---

## STAGE 0 - Pick the question (orchestrator)

1. Open `_meta/questions.yaml`. Find the question's row: `id`, `slug`,
   `tier`, `hypothesis`, `predecessor`.
2. Check whether `q<NN>_<slug>/` already exists at the lab root.
   - Does not exist: continue to Stage 1.
   - Exists WITHOUT VERDICT.md: a campaign is in flight or abandoned.
     STOP and ask the owner.
   - Exists WITH VERDICT.md: this is a REOPENING. Move the old verdict to
     `q<NN>_<slug>/_archive/VERDICT_<its-date>.md` (append-only, never
     delete), then continue to Stage 1 inside the existing dir.

GATE 0: you can state the exact slug, tier, and predecessor path.

## STAGE 1 - Create the question directory and HYPOTHESIS.md (orchestrator)

1. Create directories:

       q<NN>_<slug>/
       q<NN>_<slug>/_packets/
       q<NN>_<slug>/_archive/

   Do NOT create `scripts/` or `results/` - the verifier creates those
   itself. If experiment code exists in the dir before dispatch, the run
   is void (Prime Rule).
2. Copy `_meta/prompts/hypothesis_template.md` to
   `q<NN>_<slug>/HYPOTHESIS.md` and fill every placeholder. Delete the
   TEMPLATE INSTRUCTIONS block. HYPOTHESIS.md is the CLAIM document:
   forward-looking, no observed results. Context belongs in the brief
   (Stage 3), not here.
3. The completeness test: could a competent scientist implement the
   Measurement Specification with standard libraries alone? If no, fix it
   now.

GATE 1 (all must pass):

    Select-String -Path q<NN>_<slug>/HYPOTHESIS.md -Pattern "TEMPLATE INSTRUCTIONS","FILL_ME"
                                                            -> no output
    Select-String -Path q<NN>_<slug>/HYPOTHESIS.md -Pattern "v2_2"
                                                            -> only the Predecessor line

## STAGE 2 - Preregister (orchestrator)

1. Open `PREDICTIONS.md`. Find the highest existing P-NNN. Yours is the
   next number.
2. Append ONE row: P-NNN | today YYYY-MM-DD | Q<N> | registry IDs (must
   exist in VARIABLES.md) | predicted quantity | thresholds copied VERBATIM
   from the falsifiers in HYPOTHESIS.md | linked verdict: -
3. NEVER edit or delete an existing row. Supersede with a new row if wrong.

GATE 2: the P-row thresholds are byte-identical to the HYPOTHESIS.md
falsifier thresholds. If they differ, one of them is wrong - fix BEFORE
anything runs.

## STAGE 3 - Assemble the context brief (orchestrator)

1. Copy `_meta/prompts/context_brief_template.md` to
   `q<NN>_<slug>/_packets/brief_<YYYY-MM-DD>.md` and fill it:
   - Hypothesis, claims, measurement spec, data notes: VERBATIM from
     HYPOTHESIS.md. Do NOT include the Predecessor section.
   - Registry entries: quote rows VERBATIM from VARIABLES.md.
   - Established results: ONLY claims with status VERIFIED in this lab's
     INDEX.md, each as one line: claim, key number, verdict path. If
     nothing relevant is VERIFIED yet, write "none yet".
   - Success criteria: VERBATIM from the P-row.
2. PREMISE RULE: unverified expectations, PARTIALLY_VERIFIED results
   stated as fact, v2_2 results, and "we expect X" language are FORBIDDEN
   as premises. A premise is a v2_3-VERIFIED claim with a verdict path,
   nothing else.

GATE 3 (all must pass):

    Select-String -Path <brief> -Pattern "TEMPLATE INSTRUCTIONS","FILL_ME"  -> no output
    Select-String -Path <brief> -Pattern "v2_2"                             -> no output

   Record the brief's sha256 (Get-FileHash) - the verifier cites it in
   Provenance.

## STAGE 4 - Dispatch the verifier (orchestrator)

1. Take `_meta/prompts/informed_verifier_prompt.md`. Fill its placeholders
   from the brief.
2. Append this dispatch block (fill the constants):

       Working root: <absolute path to v2_3>. Your question directory:
       <root>/q<NN>_<slug>.
       Frontmatter constants for your VERDICT.md - use exactly these:
         schema: verdict/v2
         question: Q<N>
         slug: q<NN>_<slug>
         date: <YYYY-MM-DD>
         verification: primed
         packet_sha256: null
         predecessor: <predecessor path from questions.yaml> (copy this
           string verbatim; NEVER open that directory or anything under
           v2_2 - Prime Rule)
         registry_ids: [<the IDs quoted in the brief>]
         prediction_ids: [P-NNN]
         verifications: one entry {date: <date>, mode: primed,
           result: <the status your results earn>}
       READ ACCESS: the entire v2_3 lab (INDEX.md, other questions'
       VERDICT.md files, README, schema, registry, ledger). FORBIDDEN:
       anything under THOUGHT/LAB/FORMULA/v2_2 or any other FORMULA
       version, and any pre-existing implementation of this measurement.
       You may cite other verdicts; you may NEVER reuse their code.
       ASCII only. All writes inside your question directory. No git.

3. Launch as a FRESH agent, model = fable-class, set explicitly.

RULES DURING THE RUN:
- If the verifier reports the spec is incomplete or ambiguous: the
  campaign failed at Stage 1. Kill the run, fix HYPOTHESIS.md, re-gate,
  re-brief (new date, new hash), dispatch a NEW verifier.
- If the verifier finds pre-existing experiment code in its directory:
  the run is void. Remove the code, dispatch a NEW verifier.

## STAGE 5 - Validate (orchestrator)

    python _meta/validate_verdict.py q<NN>_<slug>/VERDICT.md --root .

- Exit 0: continue.
- Non-zero: send the EXACT error output back to the SAME verifier agent and
  have it fix its own verdict. The orchestrator NEVER edits the verdict
  body, claims, statuses, or evidence. Repeat until exit 0.

## STAGE 6 - Refuters (orchestrator; required before VERIFIED stands)

1. Required N: tier 0 or tier 1 question, OR any headline effect with
   d > 1.0, R2 > 0.99, or p < 1e-5 -> N = 2. Otherwise N = 1.
   (Required when overall status is VERIFIED; recommended for
   PARTIALLY_VERIFIED, owner's call.)
2. For each refuter: fresh fable-class agent, prompt =
   `_meta/prompts/refuter_prompt.md` + the brief + the numeric results
   (the JSON contents, never the verdict narrative). Each refuter gets a
   DIFFERENT attack lens (statistics/fit integrity vs
   implementation/mechanism).
3. Outcomes:
   - UNREFUTED: orchestrator appends {date, mode: refute,
     result: UNREFUTED} to the verdict's `verifications:` list and re-runs
     Stage 5.
   - REFUTED: the refuter must have produced a reproducible script + run
     output (these land in the question dir). Send refutation + verdict
     back to the verifier to re-adjudicate its claims against the
     falsifiers. Status drops to whatever survives. Re-run Stage 5.

## STAGE 7 - Regenerate the index (orchestrator)

    python _meta/generate_index.py --root .
    python _meta/generate_index.py --root .          (run twice)
    python _meta/generate_index.py --root . --check  (must exit 0)

The two runs must produce byte-identical INDEX.md. Update the P-row's
"linked verdict" column with the verdict path (the one permitted
PREDICTIONS.md edit: filling a "-" cell).

## STAGE 8 - Done check (all must be true)

- [ ] VERDICT.md validates exit 0
- [ ] verification: primed; brief file exists in _packets/ and its sha256
      is cited in the verdict's Provenance
- [ ] Every results file is in evidence_manifest with a matching sha256
- [ ] In-experiment controls (if the spec defined them) are reported
- [ ] Refuter rows appended per Stage 6 (if status is VERIFIED)
- [ ] INDEX.md regenerated, --check exit 0
- [ ] Nothing written outside the question dir except the P-row and the
      generated INDEX.md
- [ ] No file anywhere in the question dir originates from v2_2 or any
      pre-existing implementation
- [ ] git: NOTHING committed (commits are owner-only, on explicit request)

## STAGE 9 - OPTIONAL blind comparison (owner approval required)

Purpose: measure what lab knowledge adds or biases. An experiment ON the
protocol - it NEVER changes the question's status.

1. Only after Stage 8 is fully green, and only with owner approval (spare
   token budget).
2. Assemble a BLIND packet from `_meta/prompts/packet_template.md`
   (hypothesis + registry + criteria ONLY - no established results, no lab
   context). Gates:

       Select-String -Path <packet> -Pattern "TEMPLATE INSTRUCTIONS","FILL_ME" -> no output
       python _meta/lint_packet.py <packet>                                    -> exit 0, prints sha256

3. Dispatch a fresh agent with `_meta/prompts/blind_verifier_prompt.md` +
   the packet ONLY (its strict read allowlist is in that template). It
   writes its own independent implementation; its outputs go to
   `results/blind_control_<date>/` and its report to
   `_archive/BLIND_REPORT_<date>.md`. It does NOT touch VERDICT.md.
4. Orchestrator appends {date, mode: blind, result: <its outcome>} to the
   verdict's `verifications:` list, re-runs Stage 5, regenerates the
   index, and writes a 5-line comparison (agree/disagree, on what, why)
   into `_archive/BLIND_REPORT_<date>.md` under "## Comparison".

## THE NEVER LIST

1. NEVER copy, import, read, or execute anything from v2_2 or any other
   FORMULA version. Pointers (path strings) only.
2. NEVER run pre-existing code as evidence. The verifier writes its own.
   Reading another question's verdict: encouraged. Reusing its code:
   violation.
3. NEVER write or edit a verdict body as orchestrator (sole exception:
   appending rows to `verifications:`).
4. NEVER hand-edit INDEX.md.
5. NEVER edit an existing PREDICTIONS.md row (sole exception: filling the
   "linked verdict" cell after Stage 7).
6. NEVER put unverified expectations, v2_2 results, or
   PARTIALLY_VERIFIED-as-fact into a brief's premises. Premises are
   v2_3-VERIFIED claims with verdict paths, nothing else. HYPOTHESIS.md
   stays outcome-free entirely.
7. NEVER proceed past a red gate. A failed gate means stop and fix, or
   stop and ask the owner.
8. NEVER mismatch the model ladder: the science (verifiers, refuters,
   experiment design) runs on fable; mechanical work runs on haiku or the
   orchestrator's own shell. Never burn fable on plumbing; never burn
   anything weaker than fable on the science. Set the model explicitly on
   EVERY dispatch.
9. NEVER reuse an agent across questions, and never reuse a voided run's
   agent.
10. NEVER commit to git. The owner commits.
11. NEVER let a Stage 9 blind comparison change a status. It is
    measurement of the protocol, not of the question.
