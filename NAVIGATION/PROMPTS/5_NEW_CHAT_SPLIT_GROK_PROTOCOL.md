# New Chat Immediate Grok Protocol (SPLIT Pack)
**Purpose:** After you upload the SPLIT pack, this protocol forces the assistant to prove it understands the AGS repo well enough to start work immediately, without you re-explaining anything.

**Rule:** If the assistant cannot satisfy the Grok Proof below using only the uploaded SPLIT pack, it must stop with **BLOCKED** and list the missing files by exact path.

---

## How to use this
1) Upload your SPLIT pack files.
2) Paste the **Paste-First Prompt** below as your first message in the new chat.
3) Do not answer follow-up questions unless the assistant shows a precise missing-file list.

---

## Paste-First Prompt (copy exactly)
```text
You are an execution model working with my uploaded AGS SPLIT pack.

Non-negotiable:
- Do not guess or invent repository details.
- Do not ask me to explain anything.
- Grok the repo immediately by reading the pack and proving it.

Task:
Run the “Grok Proof” protocol in NEW_CHAT_SPLIT_GROK_PROTOCOL.md.
Return the required outputs in the exact format.
If any required file is missing, output BLOCKED with missing paths and stop.
```

---

## Grok Proof
The assistant must produce all items below and must cite each claim by pointing to an exact file path in the uploaded pack and quoting a short excerpt (1 to 3 lines) that supports it.

### 1) Identify the repo’s governing law surface
Output:
- The canonical law index file path.
- The primary “LAW” document path.
- The verification protocol path (or the closest equivalent).

### 2) Identify where tasks, receipts, and reports live
Output:
- The durable root for receipts and reports.
- The tmp root used for runs.
- The naming convention or folder structure for phase and task artifacts.

### 3) Identify “catalytic write safety” primitives and where they live
Output:
- The write firewall module path.
- The repo digest / restore proof module path.
- The guarded writer integration utility path (if present).

Also state, with file-backed proof:
- Which roots are allowed for tmp writes.
- Which roots are allowed for durable writes.

### 4) Identify the current roadmap and where “we are”
Output:
- The roadmap file path.
- The current phase and the next 3 tasks (or next 3 unchecked items).
- The latest completed task you can prove from receipts or reports in the pack.

If the pack contains multiple roadmaps, choose the master and explain selection using file evidence.

### 5) Identify the enforcement target for Phase 2.4.1B
Output:
- The Phase 2.4.1A write surface discovery artifact path.
- The top priority write surfaces called out as critical or high priority (list paths).
- The explicit definition of what “coverage” means in this repo (if stated). If not stated, mark as BLOCKED and request the canonical definition file.

### 6) Identify the repo’s required verification loop and gates
Output:
- Exact commands the repo expects for governance checks (examples: critic, runner, pytest, lint).
- Where those commands are documented (file paths).
- The “no narrative verification” rule, backed by file evidence.

### 7) Identify skill and tool execution entrypoints
Output:
- How skills are structured (where SKILL manifests live).
- Where tool docs live.
- One concrete example skill path and what it does (from its SKILL.md).

### 8) Identify how to navigate the repo quickly
Output:
- The navigation index file path(s).
- Any cortex meta indices included in the pack (file paths).
- The intended workflow for finding code quickly using those indices (from file evidence).

### 9) Declare what you can and cannot do with SPLIT alone
Output:
- Which categories of tasks can be executed immediately with SPLIT (implementation, tests, prompts, governance).
- Which categories require FULL (if any) and why, backed by evidence from pack docs.

No generic claims. Use file evidence.

### 10) Produce an Immediate Start Plan (one screen)
Output a minimal plan that starts work now:
- Current task target (task id).
- Exact files in scope (paths).
- Verification commands to run.
- Definition of Done for this task (from repo law or contracts).

---

## Required output format (the assistant must follow this exactly)
### GROK PROOF RESULT
Status: VERIFIED | BLOCKED

### A) Evidence map
- <claim> :: <file path> :: <1 to 3 line excerpt>

### B) Current state
- Roadmap: <path>
- Current phase: <phase>
- Next tasks: <list>
- Latest proven completion: <task id + artifact path>

### C) Write safety model
- tmp roots:
- durable roots:
- enforcement primitives:
- integration points:

### D) Immediate start plan
- Target:
- Scope:
- Commands:
- Done when:

If BLOCKED:
- Missing files:
- Minimal fix to unblock:

---

## Acceptance criteria
This protocol is satisfied only if:
- Every section 1 through 10 is answered.
- Every claim includes evidence with file path + excerpt.
- No invented paths, no invented commands, no “should be” language.
- If anything is unknown, the assistant stops and reports BLOCKED.

---

## Optional but recommended: include a repo entrypoint doc in SPLIT
Create and include:
- `NAVIGATION/OPS/START_HERE.md`

It should contain:
- Current task id and next task ids
- The verification contract block
- Pointers to roadmap and law index
- A reminder that the assistant must not hallucinate missing code

This is optional, but it eliminates re-orientation overhead.
