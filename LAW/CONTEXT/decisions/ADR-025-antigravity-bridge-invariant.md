---
id: "ADR-025"
title: "Antigravity Bridge as Invariant Infrastructure"
date: "2025-12-27"
status: "Accepted"
confidence: "High"
impact: "High"
tags: ["governance", "swarm", "infrastructure", "invariants"]
---

<!-- CONTENT_HASH: efdd8aacde3810338625e3063d5d58652103aa041b533a68902330535a739cb0 -->

# ADR-025: Antigravity Bridge as Invariant Infrastructure

## Context
The Swarm Architecture relies on the "Antigravity Bridge" (listening on localhost:4000) to spawn visible terminals inside VSCode. Previously, agents treated this as an optional or check-required component. Additionally, `SWARM_ARCHITECTURE.md` referenced `INV-014` (Prohibited External Windows), but this invariant was missing from the actual `CANON/INVARIANTS.md` file.

## Decision
1.  **Formalize INV-014:** We explicitly ban `subprocess.Popen` or shell commands that spawn external windows (e.g., `start wt`, `xterm`).
2.  **Bridge is Infrastructure:** The Antigravity Bridge is defined as **Invariant Infrastructure**. It is "Always On". Agents must assume it is available and use it for any task requiring a visible terminal. Failure to connect to the bridge is a system failure, not a feature unavailability.
3.  **Canon Update:** `CANON/INVARIANTS.md` will be updated to include INV-014.

## Consequences
**Positive:**
- Agents stop wasting tokens checking if the bridge is up.
- Enforces the "President monitors Governor" hierarchy by ensuring all terminals are visible in the host IDE.
- Closes the gap between Architecture docs and Canon.

**Negative:**
- The system depends on the external bridge process running.

## Compliance
- `CANON/INVARIANTS.md` updated.
- `CATALYTIC-DPT/LAB/ARCHITECTURE/SWARM_ARCHITECTURE.md` updated to remove "optional" language.