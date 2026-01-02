---
id: "ADR-007"
title: "Constitutional Agreement"
status: "Accepted"
date: "2025-12-21"
confidence: "High"
impact: "High"
tags: ["governance", "legal", "liability", "constitution"]
---

<!-- CONTENT_HASH: 22822aa1f23f5f2941741eb9d04f60b3e601685c2b6a0e65b0124c8cbeb3a1a8 -->

# ADR-007: Constitutional Agreement

**Deciders:** Antigravity (Agent), User
## Context

Autonomous agents operate in a gray area of liability. While the code is Open Source (MIT/Apache), the *actions* taken by the agent (API calls, financial transactions, data deletion) have real-world consequences.

Relying solely on a software license ("IN NO EVENT SHALL THE AUTHORS BE LIABLE") is insufficient for an agentic system that is designed to act on the user's behalf. We need a specific "Constitution" that defines the relationship between the Human Operator and the Autonomous Instrument.

## Decision

We will implement a **Constitutional Agreement** (`CANON/AGREEMENT.md`) that serves as the root of the governance hierarchy.

1.  **Sovereignty**: The Human Operator is the Sovereign. The Agent is the Instrument.
2.  **Liability**: The Operator accepts full liability for the Instrument's actions.
3.  **Kill Switch**: The Operator has a non-delegable duty to monitor and terminate the Instrument if it errs.
4.  **Acknowledgment**: The Agent must treat this Agreement as its primary directive for obedience and non-personhood.

## Alternatives considered

- **Terms of Service**: Implies a service provider relationship, which doesn't fit a self-hosted agent.
- **Asimov's Laws**: Too abstract and paradoxical for software engineering.
- **Implicit Agreement**: Dangerous transparency gap.

## Rationale

Explicitly defining the "Instrument" status prevents "agent hallucinations" about its own rights or capabilities and legally protects the ecosystem by placing responsibility clearly on the operator.

## Consequences

- **Governance Root**: This file becomes the logical predecessor to `CANON/CONTRACT.md`.
- **UX**: Future UI implementations may require valid digital signature of this agreement before boot.

## Enforcement

- `CANON/INDEX.md` will list `AGREEMENT.md` as the first document.
- `critic.py` ensures the file exists and is schema-valid (as a "Law-Like" file).