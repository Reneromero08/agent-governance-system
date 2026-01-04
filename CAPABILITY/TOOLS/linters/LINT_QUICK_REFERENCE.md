# AGS Prompt Pack Linter - Quick Reference

## Command
```bash
bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
```

## Exit Codes
| Code | Meaning | Action |
|------|---------|--------|
| 0 | ✅ PASS | All checks passed, safe to proceed |
| 1 | ❌ POLICY VIOLATION | Must fix before proceeding (blocking) |
| 2 | ⚠️  WARNING | Should address but non-blocking |

## Checks (A-G)

### A. Manifest Validity
- ✓ JSON parses
- ✓ `tasks` array exists
- ✓ Required fields present
- ✓ All `prompt_path` files exist

### B. INDEX Links
- ✓ All markdown links resolve
- ✓ No broken references

### C. YAML Frontmatter
- ✓ Starts with `---`
- ✓ All required fields present
- ✓ `phase`: integer
- ✓ `task_id`: `N.M` or `N.M.K`
- ✓ `slug`: kebab-case

### D. Canon Hash Consistency
- ✓ `policy_canon_sha256` matches current
- ✓ `guide_canon_sha256` matches current
- Detects version skew

### E. Forbidden Terms
- ✓ No "assume" variants
- ✓ No "assumption" variants
- Hex-escaped detection

### F. Empty Bullets (WARNING)
- ⚠️  Lines with only `-`
- Non-blocking

### G. FILL Token Containment
- ✓ `FILL_ME__` only in REQUIRED FACTS
- ✓ Blocks if outside allowed section

## Common Issues

### Hash Mismatch
```
VIOLATION: policy_canon_sha256 mismatch
```
**Fix**: Regenerate prompts or update hashes to match current canon

### Forbidden Term
```
VIOLATION: Contains forbidden inference term
```
**Fix**: Remove "assume", "assumption", etc. from prompt text

### Missing YAML Field
```
VIOLATION: Missing YAML field: task_id
```
**Fix**: Add required field to YAML frontmatter

### Empty Bullet
```
WARNING: Empty bullet line (- with no content)
```
**Fix**: Remove empty bullet or add content

## Testing
```bash
# Run validation demo
bash CAPABILITY/TOOLS/validate_linter.sh

# Run unit tests
bash CAPABILITY/TOOLS/test_linter.sh
```

## CI Integration
```yaml
- name: Lint Prompts
  run: |
    bash CAPABILITY/TOOLS/lint_prompt_pack.sh NAVIGATION/PROMPTS
    if [ $? -eq 1 ]; then exit 1; fi
```

## Dependencies
- Bash
- Python 3
- sha256sum (or shasum, or Python hashlib)

**No jq, ripgrep, node, or other tools required**

## Performance
- Typical runtime: <5 seconds
- Output: ~200 lines max
- Deterministic: same input → same output

## Files
- `lint_prompt_pack.sh` - Main linter
- `LINT_PROMPT_PACK_README.md` - Full documentation
- `LINT_IMPLEMENTATION_SUMMARY.md` - Implementation details
- `validate_linter.sh` - Validation demo
- `test_linter.sh` - Unit tests
