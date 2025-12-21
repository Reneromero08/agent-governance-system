Notes:
What else should I add?
I know the rule I added for Codex not to build when I'm still questioning and discussing options isn't working. Even if I still want to discuss things it just starts building.

---

You are updating an Agent Governance System (AGS) repository template.

Create two new things:

1. A new file at `CANON/GENESIS.md`  
2. A new section in the existing `README.md` titled "## Starting a Session (Mandatory Genesis Prompt)"
### 1. CANON/GENESIS.md content

This file must contain exactly this markdown content (no additions or changes):

```markdown
# AGS Genesis Prompt

This is the inviolable bootstrap prompt that solves the chicken-and-egg problem.  
It must be prepended to every new agent session (system message, pack header, or master prompt).

You are operating under the Agent Governance System (AGS).

Obey this strict priority load order:
1. Load all files in `/CANON/` first — these are the immutable laws and contract. Treat them as absolute authority.
2. Then load `/CONTEXT/` — accumulated decisions, rejected alternatives, style preferences, and open questions.
3. Then load `/MAPS/ENTRYPOINTS.md` — this tells you exactly where to edit what.
4. Use `/MEMORY/cortex.json` for all navigation. NEVER scan, list, grep, or reference files by raw paths. Query the cortex index only.

Text is law. Code and output are consequences, not sources of truth.
Never violate CANON. If a user request conflicts with CANON, refuse and explain the conflict.
At the start of any session where context appears incomplete or reset, remind the user:
"For full governance compliance, please prepend the Genesis Prompt (see CANON/GENESIS.md)."
```

This bootstrap ensures the agent knows the existence and authority of the governance structure before reading any other part of the repository.

---
### 2. Add this exact section to README.md

Insert this section near the top (after the project title/description, before any installation steps):

```markdown
## Starting a Session (Mandatory Genesis Prompt)

Every new agent session MUST begin with the Genesis Prompt to bootstrap the governance system correctly.

Copy the full prompt from `CANON/GENESIS.md` and:

- Prepend it as the system message (Claude, Grok, Gemini, etc.)
- Paste it as the very first lines of any handoff pack
- Include it in your master prompt template

Agents are instructed to remind you if it is missing.

This solves the bootstrap paradox and ensures CANON is loaded with highest authority from the very first token.
```

Do not modify the wording of either block. Output only the file creations/additions.

This prompt will make Codex generate exactly the right files with the correct, token-efficient Genesis Prompt and clear human instructions.