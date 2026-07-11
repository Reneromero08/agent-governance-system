---
uuid: "4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452"
title: "Pi Harness Governed Shell Implementation Report"
section: "report"
bucket: "implementation/agent_harness"
author: "GPT-5 Codex"
priority: "High"
created: "2026-07-09 14:14"
modified: "2026-07-09 14:14"
status: "Complete"
summary: "Replacement of Pi free-form bash execution with an allowlisted governed shell tool."
tags: [pi, harness, shell, security, integrity]
---
<!-- CONTENT_HASH: a05e6ff323d2564895774fda9e396327189f380156c3797972bb15bdba6070de -->
# Pi Harness Governed Shell Implementation Report

**Date:** 2026-07-09

**Status:** VERIFIED COMPLETE

**Agent:** GPT-5 Codex@4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452 | 2026-07-09

## Executive Summary

Replaced the Pi harness's opt-in free-form built-in `bash` capability with a
harness-owned governed shell extension. The new tool does not accept command
strings. It accepts only a worker-authorized native executable alias, a literal
argument array, a workspace-confined working directory, and a bounded timeout.

## What Was Built

- A Pi TypeScript extension that overrides the `bash` tool.
- Worker-time resolution of explicit program aliases to absolute executables.
- Rejection of Windows script launchers (`.cmd`, `.bat`, and PowerShell) and
  non-executable Unix programs.
- Literal argument arrays with count, per-argument, and total-size limits.
- Workspace-confined `cwd` resolution.
- Child environment filtering that excludes model/API credentials.
- Timeout and output caps with structured result details.
- Pi JSONL postflight checks for program allowlist and `cwd` violations.
- Prompt packets that expose the exact authorized program aliases and prohibit
  pipes, redirects, chaining, environment assignments, and shell strings.
- A governed-shell contract fixture and expanded regression tests.

## What Was Demonstrated

- The installed Pi 0.80.3 loader successfully loaded the extension in offline
  mode without an LLM request.
- The worker command includes the explicit extension even while automatic
  extension discovery remains disabled.
- Shell configuration reaches the child Pi process through deterministic JSON
  environment values.
- Non-allowlisted program calls fail the harness integrity gate.
- The offline governed-shell fixture validates the task-packet contract.
- Focused Pi harness tests pass.
- Repository fixtures, governance critic, and root audit pass.

## Real vs Simulated

The real local Pi executable loaded and type-checked the extension. Python
control-plane tests exercised real filesystem state, locks, command construction,
environment construction, receipts, and integrity parsing. LLM inference and
agent-authored shell calls were intentionally simulated so verification did not
spend API tokens or permit uncontrolled repository mutation.

## Metrics

- Python implementation lines: 1,535
- TypeScript governed-shell lines: 137
- Focused tests: 22
- Maximum allowlisted programs per worker: 32
- Maximum arguments per command: 64
- Maximum shell timeout: 300 seconds
- Maximum returned shell output: 50,000 characters and 2,000 lines

## Integrity Boundary

The tool mechanically proves program authorization, literal argument transport,
working-directory confinement, environment filtering, resource caps, and
structured auditability. A trusted executable can still interpret its arguments
as paths or mutate files beyond its `cwd`; therefore shell tasks continue to set
`shell_scope_verifiable=false` and require an independent workspace diff review.
This boundary is explicit and fail-visible rather than silently presented as a
filesystem sandbox.

## Conclusion

Pi no longer receives a general-purpose command-string shell from the harness.
Shell execution is explicit, allowlisted, structured, bounded, and audited, with
the remaining operating-system filesystem boundary documented truthfully.
