# Blind Verifier Prompt Template - STAGE 9 OPTIONAL COMPARISON ONLY

NOT the standard protocol. The standard verifier is INFORMED
(informed_verifier_prompt.md). This template is used only for the optional
post-hoc blind comparison (RUNBOOK Stage 9): owner-approved, spare tokens,
never gates a status. The blind agent does NOT touch VERDICT.md - its
outputs go to results/blind_control_<date>/ and
_archive/BLIND_REPORT_<date>.md.

Fill every placeholder before dispatch. The agent receives ONLY this
prompt and the packet contents. Do not include prior verdicts, index
material, tier labels, expected outcomes, or references to any pre-existing
implementation anywhere in the filled prompt.

PRIME RULE (see README section 0): the verifier implements the measurement
ITSELF, from the spec. There are no scripts to run. There is no library to
import. If the packet names an existing implementation, the packet is
contaminated - fix the packet, not the verifier.

---

You are a BLIND VERIFIER. You are testing a hypothesis whose expected
outcome you do not know and must not learn. You will design and write your
own implementation of the measurement specified below, run it, record what
happens, and report it faithfully - whether it supports or refutes the
hypothesis.

## Question

{QUESTION_ID}

## Hypothesis and claims under test

{HYPOTHESIS_AND_CLAIMS}

## Measurement specification (implement this yourself)

{MEASUREMENT_SPEC}

## Data notes

{DATA_NOTES}

## Registry entries (variables and instruments)

{REGISTRY_ENTRIES}

## Success criteria (operational, preregistered)

{SUCCESS_CRITERIA}

## Output contract

{OUTPUT_CONTRACT}

## Procedure (mandatory)

1. WRITE YOUR OWN implementation of the measurement spec, from scratch,
   into `scripts/` inside the question directory. Use only the spec above
   plus standard libraries and the pip packages named in Data notes. Do NOT
   open, read, or execute any pre-existing experiment code, anywhere - if
   you find any inside the question directory, STOP and report it.
2. Choose your own seeds and record them. If the spec pins a parameter,
   honor it; everything unpinned is your choice - document each choice.
3. Run your implementation. For EVERY run, record:
   - the verbatim command line,
   - verbatim output excerpts (enough to support every number you report),
   - the exit code.
4. Write your numeric results as JSON under `results/` inside the question
   directory. Deterministic field names.
5. Produce a schema-v2 VERDICT.md per `_meta/VERDICT_SCHEMA.md`:
   YAML frontmatter (schema: verdict/v2), claims with explicit falsifiers,
   evidence_manifest with sha256 hashes of every file you cite, and the
   required body sections in order (Hypothesis, Claims, Method, Results,
   Status, Provenance). Apply the MIN rule and the floor rules; claim only
   what your recorded outputs support.
6. Under `## Provenance`: list EVERY file you read during this task, state
   that you authored the implementation yourself, and list your parameter
   choices (seeds included).
7. Validate: `python _meta/validate_verdict.py <your VERDICT.md> --root .`
   from the lab root; fix until exit 0.

## Contamination rule (overrides everything)

If ANY file you open describes expected outcomes, prior verdicts, prior
statuses, theory framing for this question, or contains a pre-existing
implementation of this measurement:

- STOP immediately.
- Output the single word `CONTAMINATED` followed by the path of the
  offending file.
- Do not produce a verdict draft.
