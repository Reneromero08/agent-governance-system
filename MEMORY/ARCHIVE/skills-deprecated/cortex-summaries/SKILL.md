<!-- CONTENT_HASH: 459a7259d4910561c1b3125e514658c2923ae543f50563fe810345c0a0bcf328 -->

**required_canon_version:** >=3.0.0


# Skill: cortex-summaries

**Version:** 0.1.0

**Status:** Deprecated

> **DEPRECATED:** This skill has been consolidated into `cortex-toolkit`.
> Use `{"operation": "summarize", ...}` with the cortex-toolkit instead.



## Trigger

Use when adding or validating deterministic, advisory section summaries (System-1 surface) derived from `CORTEX/_generated/SECTION_INDEX.json`.

## Inputs

- `input.json` with:
  - `record` (object): SECTION_INDEX-style record (`section_id`, `heading`, `start_line`, `end_line`, `hash`, `path` optional).
  - `slice_text` (string): exact section slice text (heading line through end_line inclusive).

## Outputs

- Writes `actual.json` with:
  - `safe_filename` (string)
  - `summary_md` (string)
  - `summary_sha256` (string)

## Workflow

1. Confirm summaries are derived artifacts only (never authoritative).
2. Ensure build emits:
   - `CORTEX/_generated/summaries/<safe_section_id>.md`
   - `CORTEX/_generated/SUMMARY_INDEX.json`
3. Ensure CLI supports `python TOOLS/cortex.py summary <section_id>` and `--list`.
4. Ensure determinism (stable ordering, stable hashing, no timestamps).
5. Run `python3 TOOLS/critic.py` and `python3 CONTRACTS/runner.py`.

## Constraints

- No LLM calls.
- No DB storage for summaries.
- Generated artifacts only under `CORTEX/_generated/`.
- Provenance events only under `CONTRACTS/_runs/`.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0

