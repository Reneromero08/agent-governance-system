# TEMPLATE INSTRUCTIONS - DELETE THIS ENTIRE SECTION BEFORE SAVING
#
# STAGE 9 OPTIONAL BLIND COMPARISON ONLY. The standard protocol uses
# context_brief_template.md. This packet is only for the post-hoc blind
# control run (RUNBOOK Stage 9): owner-approved, never gates a status.
#
# Copy this file to q<NN>_<slug>/_packets/packet_<YYYY-MM-DD>.md and fill
# every FILL_ME. It is linted and hashed (RUNBOOK Stage 9 gates).
# Rules:
# - Hypothesis, claims, and measurement spec: copy VERBATIM from
#   HYPOTHESIS.md. Do NOT include the Predecessor section.
# - Registry entries: quote rows VERBATIM from VARIABLES.md.
# - Success criteria: copy VERBATIM from the P-row in PREDICTIONS.md.
# - Nothing else goes in. No expected outcomes, no history, no commentary.
# - When done: delete this whole block, then
#       python _meta/lint_packet.py <this file>
#   must exit 0. Record the printed sha256 for the dispatch block.
# END TEMPLATE INSTRUCTIONS

# Blind packet: Q<FILL_ME_NUMBER> (<FILL_ME date YYYY-MM-DD>)

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

## Success criteria (preregistered, P-<FILL_ME_NNN>)

<FILL_ME: the thresholds, verbatim from the P-row, one criterion per line>
