---
uuid: 4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452
title: Pi Harness Manual Context Implementation Report
section: report
bucket: implementation/agent_harness
author: GPT-5 Codex
priority: High
created: 2026-07-09 14:31
modified: 2026-07-09 15:01
status: Complete
summary: Manual-only task context, reliable Windows prompt transport, and live Neo3000
  verification.
tags:
- pi
- harness
- context
- tokens
- security
---
<!-- CONTENT_HASH: 032ed13d7eb4a5ba4aca5a50619e2bc72681c75be807c67a48f749d7bf104750 -->
# Pi Harness Manual Context Implementation Report

**Date:** 2026-07-09

**Status:** VERIFIED COMPLETE

**Agent:** GPT-5 Codex@4eb2adc1-f75e-4e10-8a6c-a8dbdbe37452 | 2026-07-09

## Executive Summary

Changed Pi workers to load no repository context files, skills, prompt
templates, default Pi system prompt, or repository-governance instruction
automatically. Every task and follow-up now selects its own explicit context
sources, tokenizer, and token budget while preserving the worker's Pi session
identity.

## What Was Built

- Mandatory `--no-context-files`, `--no-skills`, and
  `--no-prompt-templates` flags on every Pi turn.
- An explicitly empty `--system-prompt` so Pi's coding prompt is not injected.
- No automatic `AGENTS.md` or repository-governance instruction in task packets.
- Atomic generated prompt files for byte-preserving multiline transport through
  Windows `pi.cmd` shims, with prompt-file hashes in receipts.
- Current Pi `agent_end` and legacy `agent_settled` terminal-event support.
- Repeated per-task `--context-file` and `--context-text` inputs.
- A selectable `--context-token-budget` and `--context-tokenizer`.
- Deterministic source ordering and token-boundary truncation through tiktoken.
- Read-scope validation and individual/aggregate byte caps for context files.
- Context manifests containing source identity, raw SHA-256, original tokens,
  included tokens, and truncation state.
- Context manifests in task state and final task receipts.
- Offline task-packet support and focused context-packing tests.

## What Was Demonstrated

- Twenty-eight focused Pi harness tests pass.
- Zero-context task state records `included_tokens: 0` and an empty source set.
- File context outside declared read roots is rejected.
- Context without an explicit positive budget is rejected.
- Ordered multi-source context truncates deterministically at the selected token
  budget and emits source receipts.
- The Pi command contract disables all automatic project context surfaces.
- A live read-only Pi worker completed against Neo3000's existing 4,096-token
  server and returned `PI_HARNESS_OK`.
- The live receipt recorded 17 selected context tokens from a 32-token budget,
  `agent_end`, no tools, no shell, no writes, and no scope violations.

## Real vs Simulated

Real Pi dispatch exposed three integration defects before commit. Automatic
repository context produced an 8,727-token request against a 4,096-token
backend. Pi's default system prompt could produce an empty first turn, and the
Windows `pi.cmd` shim truncated a multiline task argument to its first line.
The harness now disables the default system prompt and transports the complete
packet through an audited prompt file. A fresh live task then completed through
the real Pi CLI and Neo3000 server with the expected marker and a clean receipt.

## Metrics

- Python files: 13
- Focused tests: 28
- Maximum context file: 2 MiB
- Maximum aggregate manual context: 8 MiB
- Maximum selectable manual context budget: 131,072 tokens

## Conclusion

Subagent context is now manual-only and selected independently for every task.
The harness can prove exactly which sources and tokens were injected without
assuming that Pi discovered governance, repository instructions, skills,
templates, or a default system prompt. The live `agents-a1` path is verified on
the currently running 4K Neo3000 server.
