---
name: inbox-report-writer
description: "Use when generating INBOX ledgers, updating INBOX.md indexes, validating INBOX content hashes, or writing canonical reports to INBOX."
---
**required_canon_version:** >=3.0.0

# Skill: inbox-report-writer

**Version:** 0.2.0

**Status:** Active

## Trigger

Use when generating INBOX ledgers, updating INBOX.md indexes, validating INBOX content hashes, or **writing canonical reports** to INBOX.

## Inputs

- `input.json` with:
  - `operation`: generate_ledger | update_index | verify_hash | **write_report**
  - `inbox_path`: repo-relative path (default: INBOX)
  - `ledger_path`: output path under LAW/CONTRACTS/_runs (required for generate_ledger)
  - `file_path`: repo-relative path (required for verify_hash)
  - `allow_inbox_write`: true to permit update_index
  - **For write_report:**
    - `title`: (required) Report title
    - `body`: (required) Markdown body content
    - `uuid`: Agent session UUID (default: 00000000-0000-0000-0000-000000000000)
    - `section`: report | research | roadmap | guide (default: report)
    - `bucket`: Category path (default: reports)
    - `author`: Author name (default: System)
    - `priority`: High | Medium | Low (default: Medium)
    - `status`: Complete | Draft | In Progress (default: Complete)
    - `summary`: One-line summary (auto-generated if empty)
    - `tags`: List of tags (default: [])
    - `output_subdir`: INBOX subdirectory (default: reports)

## Outputs

- `output.json` summary with status, operation, and any output paths.
- Optional ledger file written under `LAW/CONTRACTS/_runs/`.
- **For write_report:** `report_path`, `filename`, `report_written` fields.

## Constraints

- Output JSON is deterministic.
- Ledger outputs must stay under `LAW/CONTRACTS/_runs/`.
- INBOX writes require explicit `allow_inbox_write` (for update_index).
- Reports written via `write_report` follow DOCUMENT_POLICY.md format.

## Fixtures

- `fixtures/basic/`

**required_canon_version:** >=3.0.0