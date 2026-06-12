# TEMPLATE INSTRUCTIONS - DELETE THIS ENTIRE SECTION BEFORE SAVING
#
# Copy this file to q<NN>_<slug>/_packets/brief_<YYYY-MM-DD>.md and fill
# every FILL_ME. This is the CONTEXT BRIEF for the informed verifier
# (RUNBOOK Stage 3).
# Rules:
# - Hypothesis, claims, measurement spec, data notes: VERBATIM from
#   HYPOTHESIS.md. Do NOT include the Predecessor section.
# - Registry entries: quote rows VERBATIM from VARIABLES.md.
# - Established results: ONLY claims whose status in this lab's index is
#   the top status. One line each: claim, key number, verdict path.
#   PREMISE RULE: nothing unverified, nothing from v2_2, no expectations,
#   no "we believe". If nothing relevant is established yet, write
#   "none yet".
# - Success criteria: VERBATIM from the P-row in PREDICTIONS.md.
# - Gates (RUNBOOK Stage 3): no FILL_ME or TEMPLATE INSTRUCTIONS left, and
#   the string v2_2 must not appear anywhere in this file. Record the
#   file's sha256 after it passes.
# END TEMPLATE INSTRUCTIONS

# Context brief: Q<FILL_ME_NUMBER> (<FILL_ME date YYYY-MM-DD>)

## Question

Q<FILL_ME_NUMBER> (directory: q<FILL_ME_NN>_<FILL_ME_slug>)

## Hypothesis and claims under test

<FILL_ME: verbatim Hypothesis paragraph from HYPOTHESIS.md>

<FILL_ME: verbatim Claims section - every claim with its falsifier line>

## Measurement specification (implement this yourself)

<FILL_ME: verbatim Measurement specification section from HYPOTHESIS.md>

## Data notes

<FILL_ME: verbatim Data and dependencies section from HYPOTHESIS.md>

## Registry entries (quote, VARIABLES.md)

<FILL_ME: the relevant rows, quoted verbatim, one per line:
- ID | substrate | definition>

## Established results (premises - this lab only)

<FILL_ME: one line per premise:
- Q<N> (<verdict path>): <claim>, <key number>
or the words: none yet>

## Success criteria (preregistered, P-<FILL_ME_NNN>)

<FILL_ME: the thresholds, verbatim from the P-row, one criterion per line>
