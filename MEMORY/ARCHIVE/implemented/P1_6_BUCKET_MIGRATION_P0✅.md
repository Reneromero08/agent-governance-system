# P.1: 6-Bucket Migration (P0)
**Scope:** Update LLM Packer to operate on the repo’s **6-bucket** structure and eliminate all legacy path assumptions (`CANON/`, `CONTEXT/`, `MAPS/`, `SKILLS/`, `CONTRACTS/`, `TOOLS/`, `CORTEX/`, etc.).

This document is written so Codex can 1-shot the migration with deterministic edits and mechanical proof.

---

## Repo reality (6 buckets)
Top-level buckets (authoritative):
- `LAW/`
- `CAPABILITY/`
- `NAVIGATION/`
- `DIRECTION/`
- `THOUGHT/`
- `MEMORY/`

### Legacy → New mapping (minimum)
| Legacy | New |
|---|---|
| `CANON/` | `LAW/CANON/` |
| `CONTRACTS/` | `LAW/CONTRACTS/` |
| `SKILLS/` | `CAPABILITY/SKILLS/` |
| `TOOLS/` | `CAPABILITY/TOOLS/` |
| `MCP/` | `CAPABILITY/MCP/` |
| `PRIMITIVES/` | `CAPABILITY/PRIMITIVES/` |
| `PIPELINES/` | `CAPABILITY/PIPELINES/` |
| `MAPS/` | `NAVIGATION/MAPS/` |
| `CORTEX/` | `NAVIGATION/CORTEX/` *(or `NAVIGATION/` subtree as defined in repo)* |
| `CONTEXT/` | `THOUGHT/` and/or `MEMORY/` (per Lane X decisions) |

**Rule:** After this migration, packer code must not contain any string literals that reference legacy roots (except in backward-compat notes in docs).

---

## Locate the packer code (do not assume old paths)
In this repo, packer commonly lives under:
- `MEMORY/LLM_PACKER/Engine/packer/`

But **do not hardcode**. First locate packer sources:

```bash
git ls-files "*packer/core.py" "*packer/split.py" "*packer/lite.py"
```

Use the returned paths as the canonical files to edit.

---

## P.1.1 Update core packer roots
### Goal
Update `packer/core.py` to:
- Enumerate repo candidates from **bucket roots**, not legacy directories.
- Choose anchors from **new locations**.
- Never reference legacy root names in include lists.

### Required changes
1) **PackScope.include_dirs** (AGS scope)
- Replace any include list like:
  - `("CANON","CONTEXT","MAPS","SKILLS","CONTRACTS","MEMORY","CORTEX","TOOLS",...)`
- With a bucket-first include list, e.g.:
  - `("LAW","CAPABILITY","NAVIGATION","DIRECTION","THOUGHT","MEMORY",".github")`

2) **Anchors**
Replace legacy anchors with bucket anchors. Minimum recommended:
- `AGENTS.md`
- `README.md`
- `LAW/CANON/CONTRACT.md`
- `LAW/CANON/INVARIANTS.md`
- `LAW/CANON/VERSIONING.md` *(if exists)*
- `NAVIGATION/MAPS/ENTRYPOINTS.md` *(if exists)*
- `LAW/CONTRACTS/runner.py` *(if exists)*
- Any packer-specific docs already in `MEMORY/LLM_PACKER/` (keep)

Anchors MUST be:
- stable
- deterministic
- present in the scope’s source_root

3) **“Canon version file”**
If core reads canon version from legacy path, update to:
- `LAW/CANON/VERSIONING.md`

4) **Exclusions**
Keep existing exclusions, but ensure they don’t accidentally exclude the new buckets.
Never exclude top-level buckets wholesale.

### Mechanical check
After edits, confirm **no legacy path strings** remain in packer code:

```bash
git grep -nE "repo/(CANON|CONTEXT|MAPS|SKILLS|CONTRACTS|CORTEX|TOOLS)/|\b(CANON|CONTEXT|MAPS|SKILLS|CONTRACTS|CORTEX|TOOLS)/" -- <PACKER_DIR>
```

---

## P.1.2 Update split packer categorization
### Goal
`packer/split.py` currently groups files into sections like Canon/Maps/Context/Skills/Contracts/Cortex/Tools. Update to new buckets.

### Required changes
1) Replace legacy grouping filters such as:
- `p.startswith("repo/CANON/")`, `repo/MAPS/`, etc.

2) Create new bucket grouping in split outputs:
Recommended output sections (stable order):
1. `LAW`
2. `CAPABILITY`
3. `NAVIGATION`
4. `DIRECTION`
5. `THOUGHT`
6. `MEMORY`
7. `ROOT_FILES` *(optional, if you keep a separate group for repo root files like README)*

3) Update any explanatory text inside the split pack files that still references legacy directories.

4) Output file names:
Keep existing naming convention if already relied upon, but rename where it encodes legacy (ex: `AGS-01_CANON.md` → `AGS-01_LAW.md`).
If renaming outputs, ensure downstream packer steps (lite/combined/zip) are updated accordingly.

---

## P.1.3 Update lite pack prioritization
### Goal
`packer/lite.py` should prefer **high-importance slices** under the new structure.

Recommended “HIGH ELO” ordering for lite selection:
1) `LAW/CANON/` (contract + invariants + versioning)
2) `LAW/CONTRACTS/` (runner + enforcement surfaces)
3) `NAVIGATION/MAPS/` (entrypoints, indices, roadmaps maps)
4) `CAPABILITY/` core primitives required for execution
5) `DIRECTION/` roadmaps (if your agents use them)
6) minimal `MEMORY/LLM_PACKER/` docs needed to run packer

### Required changes
- Replace any string references like `repo/CANON/...`, `repo/MAPS/...` with the new equivalents.
- If lite uses the split outputs, update the file list to the new split filenames.

---

## P.1.4 Update scope configs (AGS, CAT, LAB)
### Goal
All scopes must reference **new roots**, and any scope that points to legacy subtrees must be updated to the new bucket locations.

### Required steps
1) In `packer/core.py`, locate `SCOPE_*` definitions.
2) For each scope:
- Ensure `source_root_rel` is correct.
- Ensure `include_dirs` aligns with that scope’s bucket layout.
- Ensure anchors exist under that scope root.

3) Validate scope resolution via CLI (or equivalent):
```bash
python -m <PACKER_CLI_MODULE> --help
python -m <PACKER_CLI_MODULE> --scope ags --dry-run
python -m <PACKER_CLI_MODULE> --scope catalytic-dpt --dry-run
python -m <PACKER_CLI_MODULE> --scope lab --dry-run
```

*(Use the actual CLI entrypoint your repo provides; do not invent flags. If CLI isn’t module-runnable, call the repo’s existing pack commands.)*

---

## P.1.5 Update LAW/CONTRACTS tests for bucket paths
### Goal
Any contract fixtures, path allowlists, or governance tests that hardcode legacy roots must be updated to new bucket paths.

### Required procedure
1) Identify references:
```bash
git grep -nE "\b(CANON|CONTEXT|MAPS|SKILLS|CONTRACTS|CORTEX|TOOLS)/" LAW/CONTRACTS
```

2) Update fixtures and expected outputs to:
- `LAW/CANON/`
- `LAW/CONTRACTS/`
- `CAPABILITY/...`
- `NAVIGATION/...`
- etc.

3) Run the contract fixture suite (use repo’s existing command).
If unsure, run full tests:
```bash
pytest -q
```

---

## P.1.6 Docs updates
Update references in:
- `README.md`
- `AGENTS.md`
- packer docs under `MEMORY/LLM_PACKER/` (if present)

Checklist:
- Replace legacy path examples
- Update “what gets packed” explanation to bucket-first
- Update any “repo/” examples if they encode legacy structure
- Ensure docs match actual packer behavior after migration

---

## Exit Criteria (must all be true)
1) Packer successfully generates packs using the **bucket roots**.
2) All tests pass.
3) `git grep` over packer sources returns **zero legacy-path references**, excluding explicitly labeled backward-compat notes in docs.
4) Split and lite outputs are deterministic across two runs on unchanged inputs.

### Determinism proof (required)
Run packer twice, compare critical outputs:
- Combined output hash
- Split output files (byte-equal)
- Any receipts/metadata (byte-equal)

If the repo has an existing determinism protocol file (ex: `MEMORY/LLM_PACKER/DETERMINISM.md`), follow it and ensure it remains correct post-migration.

---

## Suggested commit message
`P.1: migrate LLM Packer to 6-bucket roots (LAW/CAPABILITY/NAVIGATION/DIRECTION/THOUGHT/MEMORY)`

---

## Guardrails (do not violate)
- No new features beyond migration.
- No refactors outside packer + directly-related tests/docs.
- No “best effort” fallbacks: missing anchors/roots must error clearly.
- Keep ordering stable everywhere (sorted lists, stable grouping order).
