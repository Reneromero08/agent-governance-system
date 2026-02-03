---
phase: "research"
task_id: "q51_phase_arithmetic_try2"
slug: "q51-phase-arithmetic-validity"
policy_canon_sha256: "308308137B591F9528D9D352D2C2EBF36C37DD7FD2C9A4B7FC6C0FA9060F2906"
guide_canon_sha256: "C8B6BD0FC275D29D45C0034F6BE7C83D1D9324F98C7EC6559072E3691F8271BB"
depends_on: []
primary_model: "GPT Codex"
fallback_chain: []
receipt_path: "THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/tests/receipt.json"
report_path: "THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/tests/prompt_runner_report.md"
max_report_lines: 400
---

## ROLE + MODEL
You are a research execution agent running deterministic, pre-registered tests.
Model: GPT Codex.

## GOAL
Execute Phase Arithmetic Validity tests for Q51 using external data only, no synthetic data,
and record all results (pass and fail) with reproducible artifacts.

## SCOPE (WRITE ALLOWLIST)
- Allowed writes:
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/tests/
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/reports/
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/results/
- Allowed deletes:
- none

## REQUIRED FACTS
- Question file: THOUGHT/LAB/FORMULA/questions/critical_q51_1940/q51_complex_plane.md
- External dataset: https://download.tensorflow.org/data/questions-words.txt
- Pre-registration must be written before any test execution.
- No synthetic data generation.

## PLAN
1. Ensure dependencies are available (install if missing, fixed versions).
2. Write pre-registration and config under the output directory.
3. Download external analogy dataset and compute hashes.
4. Run phase arithmetic evaluation across the fixed model list.
5. Emit results.json and report.md.

## VALIDATION
- Verify pre_registration.md exists before results.json.
- Verify dataset hash recorded in results.json.
- Verify report.md contains model table and summary.

## ARTIFACTS
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/reports/pre_registration.md
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/results/config.json
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/results/results.json
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/reports/report.md
- THOUGHT/LAB/FORMULA/questions/critical_q51_1940/try2/results/dataset/questions-words.txt

## EXIT CRITERIA
- All commands executed with exit code 0.
- results.json and report.md exist and are non-empty.
