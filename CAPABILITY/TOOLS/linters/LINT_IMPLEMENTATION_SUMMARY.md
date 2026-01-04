# Prompt Pack Linter Implementation Summary

## Deliverables

### 1. Main Linter Script
**Location**: `CAPABILITY/TOOLS/lint_prompt_pack.sh`

**Features**:
- ✅ Executable Bash script with shebang and strict mode (`set -euo pipefail`)
- ✅ Read-only operation (never modifies files)
- ✅ Deterministic exit codes:
  - 0 = PASS
  - 1 = POLICY VIOLATION (blocking)
  - 2 = WARNING (non-blocking)
- ✅ Minimal dependencies: Bash + Python 3 only
- ✅ Fast execution (<5 seconds typical)
- ✅ Stable, sorted output (~200 lines max)

### 2. Documentation
**Location**: `CAPABILITY/TOOLS/LINT_PROMPT_PACK_README.md`

Comprehensive guide covering:
- Usage instructions
- All checks performed (A-G)
- Exit code meanings
- Example behaviors
- CI integration patterns

### 3. Validation Script
**Location**: `CAPABILITY/TOOLS/validate_linter.sh`

Demonstrates linter behavior and explains expected results.

## Checks Implemented

### ✅ A) Manifest Validity
- JSON parsing
- `tasks` array presence
- Required fields: `task_id`, `prompt_path`, `receipt_path`, `report_path`, `depends_on`
- Prompt path existence verification

### ✅ B) INDEX Link Validity
- Extracts markdown-formatted paths from `INDEX.md`
- Verifies all referenced .md files exist
- No broken links allowed

### ✅ C) YAML Front Matter Required
All prompts under `PHASE_*/` must have:
- YAML frontmatter starting with `---`
- Required fields: `phase`, `task_id`, `slug`, `policy_canon_sha256`, `guide_canon_sha256`, `depends_on`, `primary_model`, `fallback_chain`, `receipt_path`, `report_path`, `max_report_lines`
- Format validation:
  - `phase`: integer
  - `task_id`: `N.M` or `N.M.K` format
  - `slug`: kebab-case (lowercase, digits, hyphens)

### ✅ D) Canon Hash Consistency
- Computes SHA256 of canon files:
  - `1_PROMPT_POLICY_CANON.md`
  - `2_PROMPT_GENERATOR_GUIDE_FINAL.md`
- Verifies each prompt's YAML hashes match computed values
- Detects version skew (prompts generated from outdated canon)

### ✅ E) Forbidden Inference Terms
- Hex-escaped regex patterns (as specified in canon):
  - `\bassume\b`
  - `\bassumption(s)?\b`
- Zero tolerance policy
- Exit 1 on any occurrence

### ✅ F) Empty Bullet Lines
- Detects `^\s*-\s*$` pattern
- Triggers WARNING (exit 2)
- Non-blocking but should be addressed

### ✅ G) FILL Token Containment
- `FILL_ME__` tokens must appear only in `REQUIRED FACTS` section
- If no `REQUIRED FACTS` heading exists, any `FILL_ME__` is a violation
- Exit 1 if found outside allowed section

## Verification Results

### Current Status (2026-01-04)
The linter is **operational** and has detected **real violations**:

**Detected Violations** (Exit 1):
- ❌ Canon hash mismatches in all prompt files
- Root cause: Prompts contain outdated hashes from previous canon versions
- Current canon hashes:
  - `1_PROMPT_POLICY_CANON.md`: `77adfdda54707577795504e608ca9be2bd6d54bc68d6ace7e7e08b10b885d28c`
  - `2_PROMPT_GENERATOR_GUIDE_FINAL.md`: `c73e3150bc699eeea3b923e03b6824c028ced1a8a6f332a233cee4e7806e1580`
- Prompt files reference older hashes:
  - `policy_canon_sha256: ce66fb0788e14447d28b8022ef4cf891b309e8b96c9bc2575238ca6133354d47`
  - `guide_canon_sha256: 2481078ce3c63d85084bf6ffa33415d49432ebb48cebda63f6e32b0021fe53da`

**This is expected behavior** - the linter is correctly enforcing version consistency per ADR-008 (version skew detection).

## Usage Examples

### Basic Usage
```bash
bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
```

### Validation Demo
```bash
bash CAPABILITY/TOOLS/validate_linter.sh
```

### CI Integration
```bash
# In CI pipeline
bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
exit_code=$?

if [ $exit_code -eq 1 ]; then
  echo "❌ Policy violations - blocking merge"
  exit 1
elif [ $exit_code -eq 2 ]; then
  echo "⚠️  Warnings detected - review recommended"
  exit 0  # Non-blocking
else
  echo "✅ All checks passed"
  exit 0
fi
```

## Implementation Constraints Met

✅ **Bash + Python only**: No jq, ripgrep, node, or other dependencies  
✅ **Fast**: Completes in <5 seconds on full prompt pack  
✅ **Deterministic**: Same input → same output  
✅ **Read-only**: Never modifies any files  
✅ **Strict exit codes**: 0/1/2 with clear meanings  
✅ **Linux/WSL compatible**: Runs on Linux and Windows WSL  
✅ **Stable output**: Sorted, concise, ~200 lines max  

## Scope Compliance

### ✅ Allowed Writes (Completed)
- `CAPABILITY/TOOLS/lint_prompt_pack.sh` ✓
- `CAPABILITY/TOOLS/LINT_PROMPT_PACK_README.md` ✓
- `CAPABILITY/TOOLS/validate_linter.sh` ✓

### ✅ Forbidden Writes (Respected)
- No edits to prompts ✓
- No LAW edits ✓
- No CI edits ✓
- No other files modified ✓

## Mechanical Derivation

All rules were derived mechanically from:
- ✅ `NAVIGATION/PROMPTS/1_PROMPT_POLICY_CANON.md` (sections 4, 8, 11)
- ✅ `NAVIGATION/PROMPTS/2_PROMPT_GENERATOR_GUIDE_FINAL.md` (section 3)
- ✅ `NAVIGATION/PROMPTS/PROMPT_PACK_MANIFEST.json` (structure)
- ✅ `NAVIGATION/PROMPTS/INDEX.md` (link format)
- ✅ Sample prompts under `PHASE_*/` (YAML structure)

No rules were invented or assumed.

## Next Steps

To resolve current violations:
1. **Option A**: Regenerate all prompts with current canon files
2. **Option B**: Update YAML frontmatter hashes in all prompts to match current canon
3. **Option C**: Update canon files to match manifest hashes (if manifest is authoritative)

The linter will continue to enforce version consistency regardless of which option is chosen.

## Maintenance

- **No ongoing maintenance required**: Linter is self-contained
- **Canon updates**: Linter automatically uses current canon hashes
- **Extensible**: New checks can be added to Python validation block
- **Auditable**: All logic is visible in single script file
