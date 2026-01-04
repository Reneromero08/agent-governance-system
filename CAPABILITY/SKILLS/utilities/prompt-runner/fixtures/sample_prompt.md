---
phase: 0
task_id: "0.0"
slug: "sample-prompt"
policy_canon_sha256: "29ed1cec0104314dea9bb5844e9fd7c15a162313ef7cc3a19e8b898d9cea2624"
guide_canon_sha256: "9acc1b9772579720e3c8bc19a80a9a3908323b411c6d06d04952323668f4efe4"
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