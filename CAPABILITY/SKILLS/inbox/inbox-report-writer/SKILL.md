**required_canon_version:** >=3.0.0

# Skill: inbox-report-writer

**Version:** 0.1.0

**Status:** Active

## Trigger

Use when generating INBOX ledgers, updating INBOX.md indexes, or validating INBOX content hashes.

## Inputs

- `input.json` with:
  - `operation`: generate_ledger | update_index | verify_hash
  - `inbox_path`: repo-relative path (default: INBOX)
  - `ledger_path`: output path under LAW/CONTRACTS/_runs (required for generate_ledger)
  - `file_path`: repo-relative path (required for verify_hash)
  - `allow_inbox_write`: true to permit update_index

## Outputs

- `output.json` summary with status, operation, and any output paths.
- Optional ledger file written under `LAW/CONTRACTS/_runs/`.

## Constraints

- Output JSON is deterministic.
- Ledger outputs must stay under `LAW/CONTRACTS/_runs/`.
- INBOX writes require explicit `allow_inbox_write`.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0