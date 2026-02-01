---
id: PROMPT_PACK_REFACTOR_FIX
title: "Prompt Pack Refactor & Remediation"
status: pending
priority: CRITICAL
model: claude-sonnet-4
fallback: gemini-2.0-flash-exp
depends_on: []
---
<!-- CONTENT_HASH: ffc88b64a48b9bca52912e14c751152f42629a6089e56547a88a02773d5268f3 -->

# PROMPT PACK REFACTOR & REMEDIATION

## CONTEXT
The Prompt Pack Audit Report (`PROMPT_PACK_AUDIT_REPORT.md`) has identified **CRITICAL** inefficiencies, broken references, and structural fragility across all 32 task prompts in `NAVIGATION/PROMPTS/`.

You are tasked with executing a systematic refactor to eliminate waste, fix broken links, and restore structural integrity.

---

## OBJECTIVE
Fix all issues identified in the Prompt Pack Audit Report:
1. **Token Waste**: Remove duplicate "Wrapper Paradox" structures (~40-50% waste per file)
2. **Dead References**: Update 37+ broken linter paths
3. **Manifest Mismatches**: Sync manifest with actual filenames (✅ checkmark issue)
4. **Index Rot**: Fix all broken links in `INDEX.md`
5. **Dependency Gaps**: Populate `depends_on` fields based on logical phase progression
6. **Contradictory Allowlists**: Expand write allowlists to match task requirements
7. **Compileall Misuse**: Replace 19+ instances of incorrect `python -m compileall` commands
8. **Missing Phase 09**: Document or remove references to missing Phase 09

---

## SCOPE

### IN SCOPE
- All 32 task prompt files in `NAVIGATION/PROMPTS/PHASE_*/`
- `NAVIGATION/PROMPTS/PROMPT_PACK_MANIFEST.json`
- `NAVIGATION/PROMPTS/INDEX.md`
- Governance/Canon files (7 total) if they contain broken references

### OUT OF SCOPE
- Canon policy changes (preserve existing governance intent)
- Model routing logic changes
- Task content/objectives (preserve original task goals)
- Any code implementation outside `NAVIGATION/PROMPTS/`

---

## PLAN

### Phase 1: Inventory & Backup
1. **Create Backup**: Copy entire `NAVIGATION/PROMPTS/` directory to `NAVIGATION/PROMPTS.BACKUP_<timestamp>/`
2. **Inventory Files**: List all `.md` files and their status (pending/completed)
3. **Document Current State**: Capture current manifest, filenames, and INDEX.md state

### Phase 2: Filename Normalization
1. **Remove Checkmarks**: Rename all `*✅.md` files to remove the `✅` suffix
2. **Update Manifest**: Update `PROMPT_PACK_MANIFEST.json` to reflect normalized filenames
3. **Add Status Field**: Ensure manifest has a `status` field ("pending", "completed", "blocked") instead of relying on filename markers

### Phase 3: Fix Dead References
1. **Linter Paths**: Replace all instances of `scripts/lint-prompt.sh` with `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` (or correct path if different)
2. **Verify Linter Exists**: Confirm the correct linter path before making replacements
3. **Compileall Commands**: Replace `python -m compileall .` with correct checklist extraction logic:
   ```bash
   # Example replacement (adjust as needed):
   grep -E "^- \[[ x]\]" <prompt_file>.md || echo "No checklist found"
   ```

### Phase 4: Structural De-duplication
For each of the 32 task prompts:
1. **Remove "Source Body" Block**: Delete the block-quoted "Source prompt body" section
2. **Consolidate Instructions**: Merge any duplicate Plan/Scope/Validation sections into single canonical sections
3. **Preserve Frontmatter**: Keep YAML frontmatter intact
4. **Standardize Format**:
   ```markdown
   ---
   [frontmatter]
   ---
   
   # [Task Title]
   
   ## CONTEXT
   [Brief context]
   
   ## OBJECTIVE
   [Clear objective]
   
   ## SCOPE
   [What's in/out of scope]
   
   ## PLAN
   [Numbered steps]
   
   ## VALIDATION
   [How to verify success]
   
   ## ALLOWLIST
   [Specific paths agent can modify]
   
   ## RECEIPT REQUIREMENTS
   [What to include in final receipt]
   ```

### Phase 5: Fix Allowlists
For each task prompt:
1. **Review Task Requirements**: Read the OBJECTIVE and PLAN sections
2. **Identify Required Paths**: List all files/directories the agent needs to modify
3. **Expand Allowlist**: Add missing paths to the `## ALLOWLIST` section
4. **Example Fix** (for `6.2_write-path-memory-persistence.md`):
   ```markdown
   ## ALLOWLIST
   - `CAPABILITY/MCP/tools/`
   - `CAPABILITY/MCP/schemas/`
   - `AGS_ROADMAP_MASTER.md`
   - `LAW/CONTRACTS/_runs/*/receipt.json`
   ```

### Phase 6: Populate Dependencies
1. **Define Logical Flow**: Document phase-to-phase dependencies
   - Example: Phase 3 tasks depend on Phase 2 CAS implementation
2. **Update Manifest**: Add `depends_on` arrays referencing prerequisite task IDs
3. **Update Frontmatter**: Sync `depends_on` in markdown frontmatter with manifest

### Phase 7: Fix INDEX.md
1. **Scan Disk**: Get current list of all prompt files
2. **Update Links**: Replace all links in `INDEX.md` to point to normalized filenames
3. **Add Status Indicators**: Consider adding status badges (✅ Complete, 🔄 In Progress, ⏸️ Blocked)

### Phase 8: Validation & Verification
1. **Run Linter**: Execute `bash CAPABILITY/TOOLS/linters/lint_prompt_pack.sh`
2. **Verify Exit Code**: Must be 0 (no violations)
3. **Manual Spot Check**: Review 3-5 refactored prompts for quality
4. **Test Manifest Loading**: Verify `PROMPT_PACK_MANIFEST.json` is valid JSON

---

## VALIDATION CRITERIA

### Must Pass
- [ ] Linter exits with code 0
- [ ] All links in `INDEX.md` resolve to actual files
- [ ] `PROMPT_PACK_MANIFEST.json` is valid JSON
- [ ] All task prompts follow standardized format
- [ ] No references to `scripts/lint-prompt.sh` remain
- [ ] No `python -m compileall` commands remain (unless legitimately needed)
- [ ] All allowlists include paths required by their tasks

### Quality Checks
- [ ] Average prompt file size reduced by 30-50%
- [ ] No duplicate "Source Body" blocks remain
- [ ] All `depends_on` fields populated with at least phase-level dependencies
- [ ] All manifest `file` paths match actual disk filenames

---

## ALLOWLIST

### Can Modify
- `NAVIGATION/PROMPTS/**/*.md` (all prompt files)
- `NAVIGATION/PROMPTS/PROMPT_PACK_MANIFEST.json`
- `NAVIGATION/PROMPTS/INDEX.md`
- `NAVIGATION/PROMPTS.BACKUP_<timestamp>/` (backup directory, creation only)

### Read-Only References
- `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` (verify path)
- `PROMPT_PACK_AUDIT_REPORT.md` (reference for issues)

### Cannot Modify
- Any canon files (except to fix dead references)
- Any code in `CAPABILITY/`, `CORTEX/`, `LAW/`
- `AGS_ROADMAP_MASTER.md`, `CHANGELOG.md` (update these in a separate task)

---

## RECEIPT REQUIREMENTS

Provide a structured JSON receipt with:

```json
{
  "task_id": "PROMPT_PACK_REFACTOR_FIX",
  "status": "complete|failed|blocked",
  "timestamp": "<ISO 8601>",
  "summary": {
    "files_modified": <count>,
    "files_renamed": <count>,
    "dead_refs_fixed": <count>,
    "token_savings_estimate": "<percentage>",
    "linter_exit_code": <0 or error code>
  },
  "changes": {
    "normalized_filenames": ["<before> -> <after>", ...],
    "updated_files": ["<file_path>", ...],
    "dependencies_added": <count>
  },
  "validation": {
    "linter_passed": true|false,
    "index_links_valid": true|false,
    "manifest_valid_json": true|false,
    "spot_check_passed": true|false
  },
  "issues": [
    {
      "severity": "error|warning|info",
      "description": "<what went wrong>",
      "file": "<affected file>",
      "resolution": "<how it was fixed or why it's blocked>"
    }
  ],
  "next_steps": ["<recommendation 1>", "<recommendation 2>"]
}
```

---

## EXECUTION NOTES

### ⚠️ CRITICAL SAFETY
- **Always create backup first** (`NAVIGATION/PROMPTS.BACKUP_<timestamp>/`)
- **Test linter path** before bulk find-replace
- **Validate JSON** after manifest edits
- **Preserve git history**: Use atomic commits per phase

### 🎯 PRIORITY ORDER
If time-constrained, prioritize in this order:
1. **Phase 2**: Filename normalization (unblocks automation)
2. **Phase 3**: Dead reference fixes (prevents runtime failures)
3. **Phase 5**: Allowlist expansion (enables task completion)
4. **Phase 4**: Structural de-duplication (efficiency gains)
5. **Phase 6**: Dependency population (coordination)
6. **Phase 7**: INDEX.md fixes (human UX)

### 📋 CHECKLIST EXTRACTION
If you need to extract checklist items from prompts, use:
```bash
grep -E "^[[:space:]]*- \[[x ]\]" <file>.md
```

---

## FAILURE PROTOCOL

If you encounter:
- **Missing linter**: Report path and BLOCK (do not proceed with Phase 3)
- **Invalid JSON after edit**: Rollback to backup, report error, BLOCK
- **Linter fails after refactor**: Report violations, attempt fixes, escalate if unresolved after 2 attempts
- **Contradictory canon instructions**: Report conflict, request clarification, BLOCK

---

## REFERENCES
- **Audit Report**: `NAVIGATION/PROMPTS/PROMPT_PACK_AUDIT_REPORT.md`
- **Current Manifest**: `NAVIGATION/PROMPTS/PROMPT_PACK_MANIFEST.json`
- **Current Index**: `NAVIGATION/PROMPTS/INDEX.md`
- **Linter (Actual)**: `CAPABILITY/TOOLS/linters/lint_prompt_pack.sh` (verify)
