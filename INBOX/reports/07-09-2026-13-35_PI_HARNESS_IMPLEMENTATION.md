---
uuid: "4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452"
title: "Pi Harness Implementation Report"
section: "report"
bucket: "implementation/agent_harness"
author: "GPT-5 Codex"
priority: "Medium"
created: "2026-07-09 13:35"
modified: "2026-07-09 13:35"
status: "Complete"
summary: "Implementation and verification of a persistent local Pi agent worker harness."
tags: [pi, harness, agents, sessions, background]
---
<!-- CONTENT_HASH: 882a96cbfb54a8ef6ad48814f6da08e89a58ea2ea44aa9eaadaa665a4a4abfdb -->
# Pi Harness Implementation Report

**Date:** 2026-07-09

**Status:** VERIFIED COMPLETE

**Agent:** GPT-5 Codex@4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452 | 2026-07-09

## Executive Summary

Added an ADR-017-compliant `pi-harness` skill modeled on the persistent-worker
workflow of `hermes-harness`. It runs the installed local Pi coding agent as a
headless background worker, assigns a stable Pi session UUID, records task state
under the approved run-artifact root, and can submit later prompts to the same
session after an earlier turn finishes.

## What Was Built

- Skill instructions and UI metadata for discovery and correct invocation.
- Offline `run.py`, output validator, and fixture contract that never call a
  live model.
- Persistent worker registry with deterministic worker task numbering.
- Background task submission, status polling, result retrieval, raw log
  retrieval, cancellation, and same-session follow-up prompting.
- Scope-locked task packets, read-only defaults, explicit write enablement,
  narrow write-root requirements, and commit/push/branch prohibitions.
- Headless Windows process creation using `CREATE_NO_WINDOW`.
- Atomic state writes and bounded task timeouts.
- Cross-process worker/task locks that prevent duplicate task allocation and
  lost state updates.
- Fail-closed recovery for runner launch failures, crashed background runners,
  cancellation/finalization races, timeouts, and excessive log growth.
- Separate shell opt-in, disabled Pi extensions by default, strict identifier
  and artifact-path validation, capped log/result returns, and executable
  resolution that correctly locates `pi.CMD` on Windows.
- Strict JSONL integrity checks for settlement, stop reason, non-empty result,
  malformed records, and observed `edit`/`write` paths, plus hashed task
  receipts for the prompt, spec, stdout, stderr, and result.

## What Was Demonstrated

- Local Pi CLI detected: version `0.80.3`.
- Focused tests: 20 passed.
- Offline skill fixture: passed.
- Repository contract runner: all fixtures passed.
- Governance critic: all checks passed.
- Root audit: passed after explicitly setting the repository root on
  `PYTHONPATH` for the canonical script invocation.
- A simulated Pi JSON event stream demonstrated exact session-ID propagation,
  final-assistant-text extraction, and successful task finalization without
  spending model tokens.
- A real headless lifecycle probe demonstrated deterministic worker creation,
  background runner startup, polling, missing-executable failure capture,
  integrity receipt emission, and worker release without invoking a model.

## Real vs Simulated

The Pi executable and its installed version were checked from the real local
environment. Worker state, task records, command construction, filesystem
placement, and process-launch behavior were exercised by real code. LLM
inference was intentionally simulated in tests so validation could not consume
API tokens or mutate the repository through an uncontrolled live agent turn.

## Metrics

- Skill files: 15
- Python files: 11
- Python lines: 1,428
- Focused tests: 20
- Persistent identity layers: worker ID plus Pi session UUID
- Default task timeout: 1,800 seconds; hard maximum: 86,400 seconds

## Conclusion

The local Pi agent can now be treated as a persistent background worker by
OpenCode, Codex, or another local orchestrator. The parent can submit a task,
poll it by task ID, collect the final response and logs, and issue a later prompt
against the same Pi session. No commit, push, merge, or live model call was made
during implementation.
