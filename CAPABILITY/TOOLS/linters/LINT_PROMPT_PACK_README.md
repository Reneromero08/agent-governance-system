# AGS Prompt Pack Linter

## Location
`CAPABILITY/TOOLS/lint_prompt_pack.sh`

## Purpose
Mechanical enforcement of `NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md` with deterministic, read-only validation.

## Exit Codes
- **0**: PASS - All checks passed
- **1**: POLICY VIOLATION (blocking) - Must be fixed before proceeding
- **2**: WARNING (non-blocking) - Should be addressed but doesn't block

## Usage
```bash
bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
```

## Checks Performed

### A) Manifest Validity
- `PROMPT_PACK_MANIFEST.json` must parse as valid JSON
- Must contain `tasks` array
- Each task must include:
  - `task_id`
  - `prompt_path`
  - `receipt_path`
  - `report_path`
  - `depends_on` (may be empty list)
- Every `prompt_path` referenced must exist

**Triggers exit 1**: Missing manifest, invalid JSON, missing required fields, broken prompt paths

### B) INDEX Link Validity
- `INDEX.md` must not reference missing .md files
- Extracts markdown-formatted paths (`` `NAVIGATION/PROMPTS/...` ``)
- Verifies each target exists

**Triggers exit 1**: Missing INDEX.md, broken links

### C) YAML Front Matter Required
Every prompt file under `NAVIGATION/PROMPTS/PHASE_*/*.md` must:
- Start with `---` on first non-whitespace line
- Include required YAML keys:
  - `phase` (must be integer)
  - `task_id` (must match `N.M` or `N.M.K` format)
  - `slug` (must be kebab-case: lowercase, digits, hyphens only)
  - `policy_canon_sha256`
  - `guide_canon_sha256`
  - `depends_on`
  - `primary_model`
  - `fallback_chain`
  - `receipt_path`
  - `report_path`
  - `max_report_lines`

**Triggers exit 1**: Missing YAML, unclosed YAML, missing required fields, invalid formats

### D) Canon Hash Consistency
- Computes SHA256 of:
  - `NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md`
  - `NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md`
- Each prompt's YAML `policy_canon_sha256` and `guide_canon_sha256` must match computed values

**Triggers exit 1**: Hash mismatch (indicates prompt was generated from outdated canon)

### E) Forbidden Inference Terms
- Detects forbidden inference verb and noun variants using hex-escaped regex:
  - `\bassume\b`
  - `\bassumption(s)?\b`
- Zero hits allowed anywhere under `NAVIGATION/PROMPTS/**`

**Triggers exit 1**: Any occurrence of forbidden terms

### F) Empty Bullet Lines
- Detects lines that are only a hyphen with optional whitespace: `^\s*-\s*$`

**Triggers exit 2** (WARNING): Empty bullet lines found

### G) FILL Token Containment
- If any prompt contains `FILL_ME__`, it must appear only inside the `REQUIRED FACTS` section
- If `REQUIRED FACTS` heading not found, any `FILL_ME__` is a violation

**Triggers exit 1**: `FILL_ME__` token outside REQUIRED FACTS section

## Deterministic Output
- Prints stable, concise summary
- Counts of PASS/FAIL/WARN
- Lists violating files in sorted order
- Lists warning files in sorted order
- Output capped at ~200 lines

## Dependencies
- **Bash** (required)
- **Python 3** (required)
- **sha256sum** or **shasum** or Python hashlib (one required)
- No jq, ripgrep, node, or other external dependencies

## Example Behavior

### Exit 1 (Policy Violation)
```bash
$ bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
=== AGS Prompt Pack Linter ===
Checking: /path/to/NAVIGATION/PROMPTS

VIOLATION: PHASE_01/1.1_hardened-inbox-governance-s-2.md: policy_canon_sha256 mismatch
VIOLATION: PHASE_02/2.1_cas-aware-llm-packer-integration.md: Contains forbidden inference term

=== LINT SUMMARY ===
Violations: 2
Warnings: 0

POLICY VIOLATIONS (blocking):
PROMPT_CHECK_FAILED: PHASE_01/1.1_hardened-inbox-governance-s-2.md
PROMPT_CHECK_FAILED: PHASE_02/2.1_cas-aware-llm-packer-integration.md

$ echo $?
1
```

### Exit 2 (Warning)
```bash
$ bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
=== AGS Prompt Pack Linter ===
Checking: /path/to/NAVIGATION/PROMPTS

WARNING: PHASE_03/3.1_router-fallback-stability.md:42: Empty bullet line (- with no content)

=== LINT SUMMARY ===
Violations: 0
Warnings: 1

WARNINGS (non-blocking):
PROMPT_WARNING: PHASE_03/3.1_router-fallback-stability.md

$ echo $?
2
```

### Exit 0 (Pass)
```bash
$ bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
=== AGS Prompt Pack Linter ===
Checking: /path/to/NAVIGATION/PROMPTS

=== LINT SUMMARY ===
Violations: 0
Warnings: 0

✓ PASS: All checks passed

$ echo $?
0
```

## CI Integration
```yaml
# Example GitHub Actions
- name: Lint Prompt Pack
  run: |
    bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
    exit_code=$?
    if [ $exit_code -eq 1 ]; then
      echo "❌ Policy violations detected - blocking"
      exit 1
    elif [ $exit_code -eq 2 ]; then
      echo "⚠️  Warnings detected - non-blocking"
      exit 0
    else
      echo "✅ All checks passed"
      exit 0
    fi
```

## Current Status
The linter is operational and has detected real violations:
- **Canon hash mismatches**: The prompt files contain outdated hashes, indicating they were generated from an older version of the canon files
- This is expected behavior - the linter is correctly enforcing version consistency

## Maintenance
- **Read-only**: Never modifies any files
- **Fast**: Runs in <5 seconds on typical prompt packs
- **Deterministic**: Same input always produces same output
- **Minimal dependencies**: Only Bash + Python 3
