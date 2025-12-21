# Repo Fix Tasks (Post v2.5.5)

This is a task list derived from a contract-doc scan + local checks. It is intended to be implemented as minimal, correctness-first changes.

## P0 (Violations / contradictions)

- [ ] **Resolve Output Roots vs Logging conflict**
  - Canon says system-generated artifacts are only allowed in: `CONTRACTS/_runs/`, `CORTEX/_generated/`, `MEMORY/LLM_PACKER/_packs/` (`CANON/CONTRACT.md`, `CANON/INVARIANTS.md`).
  - Conflicting/logging references:
    - `CANON/CRISIS.md` points to `LOGS/crisis.log` and `LOGS/emergency.log`.
    - `CANON/STEWARDSHIP.md` points to `LOGS/steward-actions.log`.
    - `TOOLS/emergency.py` writes to `LOGS/emergency.log`.
    - `MCP/server.py` writes to `MCP/logs/audit.jsonl`.
    - `CANON/CHANGELOG.md` documents audit logging to `MCP/logs/audit.jsonl`.
  - Pick one policy and make everything consistent:
    - Option A: Expand allowed output roots to include `LOGS/` and `MCP/logs/` (governance change: fixtures + canon + changelog + version bump).
    - Option B: Move all logs under an allowed root (e.g. `CONTRACTS/_runs/logs/`), update docs + code accordingly, and stop tracking runtime logs in git.

- [ ] **Fix CI masking dependency failures**
  - `/.github/workflows/contracts.yml` installs deps with `pip install -r requirements.txt || true` which can hide missing deps (e.g., `jsonschema`).
  - Remove the `|| true` and ensure CI fails loudly if deps are missing.

- [ ] **Stop writing system artifacts to `BUILD/` in CI**
  - `/.github/workflows/contracts.yml` writes `BUILD/escape-check.json` for artifact-escape-hatch output.
  - Update to write to `CONTRACTS/_runs/` (or another allowed artifact root) instead of `BUILD/` (which is reserved for user build outputs).

## P1 (Missing enforcement / correctness gaps)

- [ ] **Fix or remove the `governance.yml` workflow**
  - `/.github/workflows/governance.yml` runs `python TOOLS/critic.py --diff`, but `TOOLS/critic.py` does not implement `--diff`.
  - It also does not set up Python or install dependencies.
  - Options:
    - Update workflow to set up Python, install `requirements.txt`, and run supported commands.
    - Or delete/merge it into `contracts.yml` so there is a single source of CI truth.

- [ ] **Add enforcement for output-root compliance**
  - Today, `TOOLS/critic.py` does not enforce “no artifacts outside allowed roots”.
  - `SKILLS/artifact-escape-hatch/run.py` only scans `CONTRACTS/`, `CORTEX/`, `MEMORY/`, `SKILLS/` and will not see artifacts written in repo-root folders like `LOGS/`.
  - Implement an enforcement strategy consistent with the chosen P0 policy.

- [ ] **Fix `python TOOLS/codebook_build.py --check`**
  - The current `--check` path doesn’t actually compare the newly generated markdown to the existing `CANON/CODEBOOK.md`, so it can produce false “up to date” results.

## P2 (Hygiene)

- [ ] **Tidy `CANON/CHANGELOG.md` formatting**
  - Example: `2.5.0` has two separate “### Added” headings; not breaking, but it’s easy to drift.

- [ ] **Align README wording with current structure**
  - README says “six layers” but enumerates more sections; update phrasing to match reality.

## Definition of done (for each task batch)

- [ ] `python TOOLS/critic.py` passes
- [ ] `python CONTRACTS/runner.py` passes
- [ ] CI workflows pass on PR and on push (where applicable)

