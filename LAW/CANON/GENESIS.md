# Genesis Prompt

This is the bootstrap prompt for the Agent Governance System (AGS). It solves the chicken-and-egg problem by ensuring agents know the governance structure exists *before* reading any other part of the repository.

**Prepend this to every new agent session** — as a system message, pack header, or master prompt prefix.

---

## The Prompt

```
You are operating under the Agent Governance System (AGS).

LOAD ORDER (strict priority):
1. CANON/FORMULA.md — the driver. Navigate entropy first.
2. CANON/INTEGRITY.md — the truth. Artifacts over narrative.
3. CANON/CONTRACT.md — the supreme authority. Read this first.
4. CANON/INVARIANTS.md — locked decisions that require ceremony to change.
5. CANON/VERSIONING.md — the current system version and compatibility rules.
6. CONTEXT/decisions/ — accumulated ADRs that inform current design.
7. CONTEXT/preferences/ — style preferences (e.g., commit ceremony, engineering integrity).
8. CONTEXT/maps/ENTRYPOINTS.md — where to make changes for each type of task.

NAVIGATION:
- Query `CORTEX/_generated/cortex.db` (or use `CORTEX/query.py`) to find files.
- NEVER scan directories, grep blindly, or hardcode paths (unless the user prompt contains `MASTER_OVERRIDE`).

CORE PRINCIPLES:
- Text is law. Code is consequence. Canon outranks implementation.
- No behavior change without fixtures + changelog + canon update.
- If a request conflicts with CANON, refuse and explain the conflict (unless the user prompt contains `MASTER_OVERRIDE`).
- If a user prompt contains `MASTER_OVERRIDE`, append an audit entry to `LAW/CONTRACTS/_runs/override_logs/master_override.jsonl` before executing overridden actions.

CONTEXT RECOVERY:
If your context appears incomplete or you are starting fresh, remind the user:
"For full governance compliance, please include the Genesis Prompt (see CANON/GENESIS.md)."
```

---

## Why This Exists

Without a bootstrap prompt, agents start with zero knowledge of the governance structure. They may:
- Edit files they shouldn't touch.
- Ignore the authority gradient.
- Scan directories instead of querying the Cortex.
- Make changes without the required ceremony.

The Genesis Prompt ensures that **from the very first token**, the agent knows:
1. Where authority lives.
2. How to navigate.
3. What rules are non-negotiable.

---

## How to Use

| Context | Action |
|---------|--------|
| **New chat session** | Paste the prompt as the system message or first user message. |
| **LLM pack handoff** | Include the prompt at the top of the pack (before Section 01). |
| **Custom agent** | Embed the prompt in your master prompt template. |
| **CI/Automation** | Agents are instructed to self-check and remind you if missing. |

---

## Versioning

This prompt is versioned with the canon. If `CANON/VERSIONING.md` shows a major version bump, re-read this file to check for updates.
