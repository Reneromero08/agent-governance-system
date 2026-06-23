# Phase 6 V2 Final Qualification Work Package

**Scope:** local source repair, deterministic regeneration, exact-head Linux verification, and final evidence packaging only.

**Forbidden:** hardware calibration, scientific acquisition, restoration execution, target coupling, Small Wall execution, V1 reinterpretation, branch deletion, or merge.

## Source repair

Complete one coherent source-repair commit:

1. Replace `atoi`, `atol`, and `atof` CLI handling with full-consumption checked parsing. Reject suffixes, prefixes, overflow, empty values, and non-finite values.
2. Replace the C authorization string search with complete fail-closed JSON validation. Reject malformed JSON, duplicate or unknown fields, trailing content, wrong types, non-singleton sessions, incorrect route/core pairs, and source-bundle mismatch.
3. Mechanically bind C capture-quality constants to the frozen plan, either through a generated header or an exact source-identity test.
4. Add direct C regression tests for capture coverage, empirical sample-rate bounds, Nyquist margin, and maximum timestamp gap. Do not skip the quality logic under mock testing.
5. For sender-off rows, derive the Nyquist requirement from `sender_off_control_for_tone_index`.
6. In the analyzer, hash the same bytes that are parsed, validate exact nested schemas, enforce source-commit agreement, reject directories and symlinks in run roots, and read every capture-quality threshold from the plan.
7. Add negative tests for every new rejection path.

Do not add reversed/randomized tone order to the current ascending-order calibration object.

## Generated contracts

After source tests pass, create one generated-contract commit. Regenerate the plan, four sessions, session manifests, source-bundle manifest, and checksum sidecars from the exact source-repair commit. Do not hand-edit generated files.

Require:

- four sessions;
- 672 windows per session;
- 1,344 windows per route;
- 2,688 windows total;
- eight sender-off controls per tone;
- exact source-repair commit in every required binding;
- deterministic bytes and matching SHA-256 sidecars.

## Qualification

At the generated-contract head, require:

- updated GitHub Phase 6 workflow success;
- strict C compile with `-Wall -Wextra -Werror`;
- V2 runner tests;
- C/Python waveform equivalence;
- Slot2 primitive identity;
- ASan and UBSan;
- V2 Python contract and analyzer tests;
- full no-write repository gate;
- the same strict and sanitizer lanes on the Linux target.

No hardware mode may run.

## Evidence package

Create one evidence-only commit after all qualification passes. Bind the exact tested final head, generated-contract commit, plan digest, source-bundle digest, GitHub workflow run, local tests, Linux strict tests, sanitizer tests, and full repository gate.

Record explicitly:

```text
hardware_ran=false
authorization_artifact_created=false
acquisition_authorized=false
restoration_authorized=false
target_coupling_authorized=false
small_wall_authorized=false
```

Update the architecture review, roadmaps, navigation, and PR body with final hashes. Keep PR #21 draft and stop for independent review.
