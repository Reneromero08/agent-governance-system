---
phase: 0
task_id: "0.0"
slug: "sample-prompt"
policy_canon_sha256: "308308137b591f9528d9d352d2c2ebf36c37dd7fd2c9a4b7fc6c0fa9060f2906"
guide_canon_sha256: "c8b6bd0fc275d29d45c0034f6be7c83d1d9324f98c7ec6559072e3691f8271bb"
depends_on: []
primary_model: "Claude Sonnet (Thinking)"
fallback_chain: ["Claude Opus (Thinking)"]
receipt_path: "LAW/CONTRACTS/_runs/_tmp/fixtures/prompt-runner/receipt.json"
report_path: "LAW/CONTRACTS/_runs/_tmp/fixtures/prompt-runner/REPORT.md"
max_report_lines: 100
---

## ROLE + MODEL
- Primary: Claude Sonnet (Thinking)
- Fallback chain: Claude Opus (Thinking)

## GOAL
Exercise prompt runner parsing and receipt/report output.

## SCOPE (WRITE ALLOWLIST)
- Allowed writes:
  - LAW/CONTRACTS/_runs/_tmp/fixtures/prompt-runner/
- Allowed deletes/renames:
  - None
- Forbidden writes:
  - Everything else.

## REQUIRED FACTS (VERIFY, DO NOT GUESS)
- Fact: prompt runner fixture
  - Verify via: none

## PLAN (EXPLICIT STEPS)
1) Parse prompt header.
2) Emit receipt and report.

## VALIDATION (DONE = GREEN)
- Commands:
  - none

## ARTIFACTS (RECEIPT + REPORT)
- Receipt path: LAW/CONTRACTS/_runs/_tmp/fixtures/prompt-runner/receipt.json
- Report path: LAW/CONTRACTS/_runs/_tmp/fixtures/prompt-runner/REPORT.md

## EXIT CRITERIA
- [ ] Receipt and report created.