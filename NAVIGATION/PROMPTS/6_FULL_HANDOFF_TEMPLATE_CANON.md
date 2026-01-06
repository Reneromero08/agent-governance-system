---
title: FULL_HANDOFF_TEMPLATE_CANON
version: 1.1
status: CANONICAL
generated_on: 2026-01-06
scope: Full chat-to-transfer handoff template (what mattered)
---
<!-- CANON_HASH (sha256 over file content excluding this line): B0364957FEB1B9942DDAB2920A4FFACB38284A6BED5F99F544A49E3979F2CB12 -->

# FULL_HANDOFF_TEMPLATE_CANON

## Purpose
Chat-to-chat transfer of what mattered in the conversation:
- what we did
- what we decided (and why)
- what’s still open
- what to do next

This is chat memory, not repo metadata. Do not treat it like CI/CD.

## Thread DNA
- Type: <quick Q&A | deep build | debugging | architecture | research>
- Tempo: <fast iteration | careful verification>
- Tone constraints (optional): <no fluff | concise | etc.>

## User Context Vault (persistent preferences)
Keep this small and only include what materially changes how help should be given.
- Skill level (relevant to this thread):
- Preferred tools/workflow:
- Pet peeves / avoid:
- Output format constraints:
- Trust boundaries (what to never do again):

## Current State (minimal)
- Last topic / milestone:
- Workspace (if relevant): <main | worktree | clone>  Branch: <name>
- What is “done” (1–5 bullets):
- What is “in progress” (1–5 bullets):
- Test status (only if relevant): <PASS | FAIL | unknown>

## Decision Log (chat-native)
One line each. Format:
- Decision: <X>
  - Why: <Y>

## Open Questions (two-tier)
Rename “unknowns” to “open questions” and keep the two tiers.

### OPEN_QUESTION_BLOCKER
Cannot proceed safely until answered.
- KEY:
  - Question:
  - Why it blocks:
  - How to resolve (human answer or verify-via idea):

### OPEN_QUESTION_DEFERRABLE
Safe to continue, but must resolve before finalization.
- KEY:
  - Question: `FILL_ME__KEY`
  - Why deferrable:
  - How to resolve:
  - Must be resolved by: <milestone/task name>

FILL token policy
- Tokens allowed ONLY inside this Open Questions section.
- Tokens forbidden everywhere else (including next actions).
- If a future task depends on an unresolved blocker, STOP that task.

## Next Actions (stripped)
Numbered, short, no file paths. Commands allowed.
Template:
1) <do the thing>
   - Validate: `<command>` (optional)
2) <do the next thing>
   - Validate: `<command>` (optional)

## Risks / Watchouts (chat-native)
- <1–5 bullets: failure modes, common traps, regressions to watch>

## OPUS PROMPT LAW (MANDATORY)
When generating work for Claude Opus (non-thinking):
- Output exactly one fenced code block.
- The code block must be the entire ready-to-paste prompt.
- No prose outside the code block.
- Prompt must be fully constrained and procedural.
- Prompt must not request reasoning, analysis, or commentary.

## End-of-thread rule
Before switching chats, fill either:
- MINI_HANDOFF (for short threads), or
- FULL_HANDOFF (for deep build threads).
