---
name: prompt-runner
description: Run NAVIGATION/PROMPTS tasks by parsing headers, enforcing prompt canon gates (lint, FILL_ME__, hash checks, allowlists, dependencies), executing declared commands, and emitting deterministic receipts/reports. Use when asked to execute a prompt file or run a phase prompt.
---

**required_canon_version:** >=3.0.0

# Prompt Runner

**Version:** 0.1.0

**Status:** Active

Execute prompt files with a deterministic, receipt-first workflow. This skill enforces the prompt canon and fails closed.

## Usage

```bash
python run.py input.json output.json
```

## Input (input.json)

Required fields:
- task_id: string
- prompt_path: repo-relative path to a prompt markdown file
- output_dir: repo-relative path for runner artifacts (must be under LAW/CONTRACTS/_runs)

Optional fields:
- emit_data: true|false (default false)
- data_path: repo-relative path for DATA.json (defaults to output_dir/DATA.json)
- commands: list of objects { command: string, timeout_sec: int, allow_failure: bool }
- manifest_path: repo-relative path to PROMPT_PACK_MANIFEST.json (required when depends_on is non-empty)
- plan_ref: string plan identifier or artifact path (required for non-planner models)
- max_output_bytes: int (default 100000)

## Behavior

- Verifies prompt path exists in cortex SECTION_INDEX.
- Parses YAML frontmatter and enforces required header fields.
- Enforces required sections and SCOPE allowlists.
- Stops if any FILL_ME__ tokens remain (BLOCKED_UNKNOWN).
- Validates policy_canon_sha256 and guide_canon_sha256 against current canon.
- Runs canonical lint command and blocks on exit code 1.
- Checks dependencies via manifest and requires receipt result OK.
- Writes receipt.json and REPORT.md (and optional DATA.json) under LAW/CONTRACTS/_runs.

## Output (output.json)

- status: success|error
- result: OK|VERIFICATION_FAILED|BLOCKED_UNKNOWN|INTERNAL_ERROR|POLICY_BREACH
- prompt_sha256: hex digest of prompt contents
- frontmatter: parsed key-value pairs
- headings: list of section headings
- fill_me_tokens: list of unresolved tokens found
- lint: { command, exit_code, result, stdout, stderr }
- commands_run: list with exit codes and outputs
- artifacts: resolved artifact paths
- errors: list of error strings

## Notes

- Do not use this skill for unrelated repo changes.
- Lint failures or missing required facts stop execution.
