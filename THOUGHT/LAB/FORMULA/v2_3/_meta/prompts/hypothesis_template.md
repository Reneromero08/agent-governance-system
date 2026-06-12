# TEMPLATE INSTRUCTIONS - DELETE THIS ENTIRE SECTION BEFORE SAVING
#
# Copy this file to q<NN>_<slug>/HYPOTHESIS.md and fill every FILL_ME.
# Rules (Gate 1 in _meta/RUNBOOK.md checks them):
# - Forward-looking only. NO status words, NO prior results, NO observed
#   numbers from any earlier version, NO references to any pre-existing
#   implementation, NO inherited seeds.
# - Thresholds in falsifiers ARE required - they are predictions, not
#   results.
# - The Measurement Specification must pass the stranger test: someone with
#   only that section and standard libraries can implement the experiment.
#   Pin only the parameters the claim depends on; everything else is
#   explicitly the verifier's choice.
# - ASCII only.
# - When done: delete this whole block, run
#       python _meta/lint_packet.py q<NN>_<slug>/HYPOTHESIS.md
#   and require exit 0.
# END TEMPLATE INSTRUCTIONS

# Q<FILL_ME_NUMBER>: <FILL_ME short title>

## Hypothesis

<FILL_ME: one paragraph. What is predicted to be true, and by what
mechanism. Forward-looking, falsifiable, self-contained.>

## Claims

### C1: <FILL_ME one-line claim>

<FILL_ME: one or two sentences making the claim precise.>
**Falsifier:** <FILL_ME: the concrete observation, with numeric thresholds,
that would falsify this claim.>

### C2: <FILL_ME - add or remove claims as needed; every claim gets its own
falsifier>

<FILL_ME>
**Falsifier:** <FILL_ME>

## Measurement specification

<FILL_ME: the complete recipe. Substrate and setup; procedure step by step;
parameters - state which are PINNED (the claim depends on them) and which
are IMPLEMENTER'S CHOICE (each choice must be documented in the report);
quantities to
record; exactly how each statistic in the falsifiers is computed (estimator,
error bars, fit method). A stranger implements from this section alone.>

## Data and dependencies

<FILL_ME: datasets or caches (read-only repo paths and the env vars to
reach them, if any); pip packages; runtime class (CPU/GPU, rough minutes).
If the experiment generates its own data, say so.>

## Predecessor

<FILL_ME: repo-relative v2_2 path from _meta/questions.yaml, or "none".
This is a history pointer. It is never opened by any agent in this lab.>
