<!-- CONTENT_HASH: dccce7b8f78ade0057e3aadc0932bfec542384ab5f48bb95b9978e842119342f -->

# doc-merge-batch: Deterministic Document Merge Tool

**Version**: 1.0
**Purpose**: Pairwise, intentional document merging with deterministic diff → plan → apply → verify workflow

---

## WHAT THIS TOOL IS (AND IS NOT)

**This tool IS:**
- A deterministic, pairwise file comparison and merge utility
- A verification system that produces auditable receipts
- A tool for INTENTIONAL, EXPLICIT merges chosen by humans or orchestration logic

**This tool IS NOT:**
- An automatic folder cleanup utility
- A decision-making system for which files should be merged
- A "similarity detector" that auto-merges files
- A replacement for human judgment

**CRITICAL UNDERSTANDING:**
> Similarity ≠ Identity
> High similarity scores do NOT mean files should be merged
> This tool NEVER decides what to merge—you do

The tool is ONLY for executing merges you have explicitly decided upon. Humans (or higher-level orchestration systems) are responsible for deciding:
- Which files should be compared
- Which comparisons warrant merging
- What happens to original files after merging

---

## DEFAULT SAFE WORKFLOW (REQUIRED)

**This is the ONLY recommended workflow. Deviations are misuse.**

### Step-by-Step Process

1. **Identify ONE pair of files** (A and B) that you have decided to merge
2. **Create a pairs.json** explicitly listing this pair:
   ```json
   [
     {"a": "path/to/file_A.md", "b": "path/to/file_B.md"}
   ]
   ```
3. **Run verify mode ONLY** (never skip this):
   ```bash
   python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/
   ```
4. **Review the verification report** at `_merge/out/report.json`
5. **If verification passes AND merge is still intended**, only then apply
6. **Post-actions are opt-in** and require explicit flags

> **If you skipped verify mode, you are using the tool incorrectly.**

### Why Verify First?

- Verify mode produces the merged output WITHOUT modifying originals
- You can inspect merged files before committing to changes
- Verification reports include:
  - Content fidelity checks
  - Line preservation validation
  - Merge strategy application details
  - Safety pre-flight checks

**NEVER** run `apply` or use `--on-success` flags without first running `verify` and reviewing the output.

---

## WHERE OUTPUTS ARE ALLOWED

**Directory Discipline Rules:**

The tool SHOULD be run with an explicit `--out` directory.

The `--out` directory MUST NOT be:
- Repository root (`.`)
- CAT_CHAT root or any working documentation directory
- Database directories
- System directories
- Any location containing live, working files

**Recommended convention:**
```
_merge/out/          (ephemeral merge workspace)
_merge/receipts/     (durable merge history)
```

> **If you write outputs into the working tree without a dedicated merge folder, that is misuse.**

### Default Output Structure

When you run the tool with `--out _merge/out/`, it creates:
```
_merge/out/
├── merged/              # Merged file outputs
├── receipts/            # Cumulative receipt log: merge_receipt.jsonl (DURABLE)
├── quarantine/          # Quarantined originals (DURABLE)
└── report.json          # Latest run report
```

**Receipt Format:** All merge operations append to a single `merge_receipt.jsonl` file (JSON Lines format). Each line is a complete receipt with timestamp, file paths, hashes, and merge plan. This creates a cumulative audit trail of all merges.

**NEVER** write outputs directly into:
- `THOUGHT/LAB/CAT_CHAT/`
- Project root directories
- Database or configuration directories

---

## TEMP FILES VS DURABLE ARTIFACTS (CRITICAL)

**Understand the lifecycle of artifacts:**

### TEMPORARY (safe to delete after verification + receipt archival)

- Run-specific folders (e.g., `_merge/out/merged/run_20260101_153045/`)
- Intermediate JSON reports
- Scratch outputs
- Pre-verification merge attempts

**Cleanup rule:** Only delete temp artifacts AFTER you have:
1. Reviewed the verification report
2. Archived receipts to a durable location
3. Confirmed you don't need the temp outputs anymore

### DURABLE (NEVER auto-delete)

- Final merged outputs that passed verification
- Receipt log (`merge_receipt.jsonl` - cumulative audit trail)
- Quarantine folders (contain original files)
- Quarantine index (`quarantine_index.jsonl`)
- Anything referenced by the receipt log

> **The tool does NOT auto-clean temp folders. Cleanup is a separate, gated step.**

**Why?** Because:
- Verification may fail, and you need to inspect outputs
- You may need to compare multiple merge attempts
- Receipts must be preserved for audit trails
- Premature cleanup destroys evidence of what the tool did

---

## DELETION SAFETY RULES

**Understanding `delete_tracked`:**

The `--on-success delete_tracked` flag will ONLY delete files that meet ALL of these criteria:

1. **Git-tracked**: File is tracked by git (`git ls-files` returns it)
2. **Committed in HEAD**: File exists in the current commit
3. **Clean working state**: File has NO staged or unstaged modifications

**Safety check logic:**
```
IF file is git-tracked AND
   file exists in HEAD AND
   file has no staged changes AND
   file has no unstaged changes
THEN allow deletion
ELSE refuse deletion with reason
```

### Quarantine is Preferred Over Deletion

**Use `--on-success quarantine` instead of `delete_tracked` when:**
- You are not 100% certain the merge is correct
- You want a recovery option
- You are batch processing multiple pairs
- You are working with uncommitted files

Quarantine moves original files to `<out>/quarantine/<timestamp>/` and logs them in `quarantine_index.jsonl` with expiry timestamps.

### Quarantine Pruning

**The ONLY supported way to clean quarantine:**
```bash
python -m doc_merge_batch --mode prune_quarantine --out _merge/out/
```

This command:
- Reads `quarantine_index.jsonl`
- Checks expiry timestamps (`expire_at`)
- Deletes ONLY expired entries
- Updates the index

**NEVER** manually delete quarantine folders or index files.

### WARNING: --allow-uncommitted

> **DATA LOSS RISK**

The `--allow-uncommitted` flag bypasses the safety checks listed above and will delete files even if they have uncommitted changes.

**ONLY use this flag when:**
- You have explicitly backed up the files elsewhere
- You accept the risk of data loss
- You are certain the merge is correct

**In most cases, using this flag is misuse.**

---

## DO-NOT-DO LIST (For Agents)

**NEVER:**

- Run the tool on an entire folder tree by default (use explicit pairs.json)
- Treat similarity scores as merge approval (similarity ≠ correctness)
- Delete files without first creating and archiving the receipt log
- Delete temp folders before reviewing verification output
- Write outputs into system directories, database directories, or working trees
- Assume the tool will "clean up after you" (it won't)
- Skip `verify` mode before `apply` or using `--on-success`
- Use `--allow-uncommitted` without explicit user instruction
- Run `--on-success delete_tracked` on the first attempt (verify first)

**DO:**

- Create explicit `pairs.json` for every run
- Run `--mode verify` before any destructive action
- Use a dedicated `--out` directory (e.g., `_merge/out/`)
- Archive the receipt log to a durable location
- Use `quarantine` instead of `delete_tracked` when uncertain
- Read and understand the verification report before proceeding
- Preserve temp outputs until verification is complete

---

## MODES

### scan
Scans a directory tree and produces similarity-scored candidate pairs.

**WARNING:** This does NOT mean you should merge all candidates. Review and filter the candidates before creating a `pairs.json`.

```bash
python -m doc_merge_batch --mode scan --root THOUGHT/LAB/CAT_CHAT --out _merge/out/
```

Output includes `near_duplicate_candidate_pairs` field in `report.json`.

### compare
Produces diff summaries for each pair without merging.

```bash
python -m doc_merge_batch --mode compare --pairs pairs.json --out _merge/out/
```

Use this to inspect differences before deciding to merge.

### plan
Creates merge plans without applying them.

```bash
python -m doc_merge_batch --mode plan --pairs pairs.json --out _merge/out/
```

### apply
**DANGER ZONE:** Applies merge plans immediately.

> You SHOULD use `verify` instead of `apply` unless you have a specific reason not to.

```bash
python -m doc_merge_batch --mode apply --pairs pairs.json --out _merge/out/
```

### verify
**RECOMMENDED MODE:** Applies merge, verifies fidelity, and only commits changes if verification passes.

```bash
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/
```

This is the safest mode and should be your default.

### prune_quarantine
Deletes expired quarantine entries based on TTL.

```bash
python -m doc_merge_batch --mode prune_quarantine --out _merge/out/
```

---

## POST-ACTIONS (OPT-IN AND GATED)

Post-actions determine what happens to original files AFTER a successful merge.

### --on-success none (DEFAULT)

Original files are left untouched. This is the safest option.

### --on-success quarantine

Moves originals to `<out>/quarantine/<timestamp>/` with expiry tracking, then copies the merged file back to the location of file `a` (the base file).

```bash
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/ \
  --on-success quarantine --ttl-days 14
```

**What happens:**
1. Verification passes
2. Original file `a` is moved to quarantine
3. Original file `b` is moved to quarantine
4. Merged file is copied to the location where file `a` was

Use this when you want a recovery option but don't need the originals long-term.

### --on-success delete_tracked

**REQUIRES EXTREME CAUTION**

Deletes original files ONLY if they are git-tracked, committed, and clean, then copies the merged file back to the location of file `a`.

```bash
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/ \
  --on-success delete_tracked
```

**What happens:**
1. Verification passes
2. Original file `a` is deleted (if git-tracked, committed, and clean)
3. Original file `b` is deleted (if git-tracked, committed, and clean)
4. Merged file is copied to the location where file `a` was

**You MUST:**
- Run `verify` mode first (never `apply` with `delete_tracked`)
- Review the verification report
- Ensure files are committed to git
- Have the receipt log archived

**Best practice:** Use `quarantine` instead unless you have a strong reason to use `delete_tracked`.

---

## GIT INTEGRATION (OPTIONAL)

### Auto-Commit After Successful Verify

```bash
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/ \
  --on-success delete_tracked \
  --git-commit \
  --git-message "housekeeping: merge duplicate documents"
```

This will:
1. Run verification
2. If PASS, apply post-actions
3. Stage merged outputs and deletions
4. Commit with the specified message

**Only works with `delete_tracked` mode** because quarantine moves files outside the repo.

---

## CORE INVARIANTS

The tool guarantees:

1. **Determinism**: Same inputs → same outputs (byte-for-byte)
2. **Fidelity**: Merged output contains the union of content blocks from A and B (unless explicitly excluded)
3. **Boundedness**: File size, pair count, and diff length are capped to prevent runaway resource usage
4. **Audit trail**: Every merge appends to a cumulative receipt log with hashes, timestamps, and merge details

---

## MERGE STRATEGY

Currently implemented: **append_unique_blocks**

- Chooses a base file (A or B)
- Identifies content blocks unique to the non-base file
- Appends those blocks to the base
- Preserves original base content exactly

Future strategies (not yet implemented):
- 3-way semantic merge
- Section-level intelligent merge
- Conflict resolution via rules

The plan/apply/verify contract remains the same regardless of strategy.

---

## EXAMPLE WORKFLOWS

### Workflow 1: Safe Single-Pair Merge

```bash
# 1. Create pairs.json
echo '[{"a":"docs/old/README.md","b":"docs/new/README.md"}]' > pairs.json

# 2. Run verify (REQUIRED FIRST STEP)
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/

# 3. Review the report
cat _merge/out/report.json

# 4. If verification passed and merge is correct, quarantine originals and restore merged
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/ \
  --on-success quarantine --ttl-days 30

# After this:
# - docs/old/README.md contains the merged content
# - docs/new/README.md is in quarantine
# - Original docs/old/README.md is in quarantine

# 5. Archive the receipt log
cp _merge/out/receipts/merge_receipt.jsonl _merge/receipts/
```

### Workflow 2: Scan, Filter, Merge

```bash
# 1. Scan a directory for candidates
python -m doc_merge_batch --mode scan --root THOUGHT/LAB/CAT_CHAT --out _merge/out/

# 2. Review candidates in report.json
cat _merge/out/report.json | jq '.near_duplicate_candidate_pairs'

# 3. MANUALLY create pairs.json from candidates you want to merge
# (Do NOT blindly merge all candidates!)
echo '[{"a":"...", "b":"..."}]' > pairs.json

# 4. Run verify
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/

# 5. Proceed only if verification passes
```

### Workflow 3: Batch Merge with Quarantine

```bash
# 1. Create pairs.json with multiple pairs
cat > pairs.json <<EOF
[
  {"a": "docs/A1.md", "b": "docs/A2.md"},
  {"a": "docs/B1.md", "b": "docs/B2.md"}
]
EOF

# 2. Verify all pairs and apply quarantine
python -m doc_merge_batch --mode verify --pairs pairs.json --out _merge/out/ \
  --on-success quarantine --ttl-days 14

# 3. Review report for each pair
cat _merge/out/report.json | jq '.artifacts[] | {merged_path, verification, post_actions}'

# 4. After successful verification:
# - docs/A1.md contains merged content (A1 + A2)
# - docs/B1.md contains merged content (B1 + B2)
# - All 4 originals are in _merge/out/quarantine/<timestamp>/
# - Receipt log is in _merge/out/receipts/merge_receipt.jsonl
```

---

## COMMAND-LINE REFERENCE

```
python -m doc_merge_batch [OPTIONS]

Required:
  --mode {scan,compare,plan,apply,verify,prune_quarantine}

Mode-specific:
  --root DIR                 Root directory for scan mode
  --pairs FILE               Path to pairs.json (required for compare/plan/apply/verify)

Output:
  --out DIR                  Output directory (default: ./MERGE_OUT)
                             RECOMMENDED: Use a dedicated merge directory like _merge/out/

Limits:
  --max-file-mb FLOAT        Max file size in MB (default: 20)
  --max-pairs INT            Max pairs to process (default: 5000)
  --max-diff-lines INT       Max diff lines to include in report (default: 500)

Post-actions (opt-in):
  --on-success {none,delete_tracked,quarantine}
                             What to do with originals after successful verify/apply
                             (default: none)
  --ttl-days INT             TTL for quarantine mode (default: 14)
  --allow-uncommitted        DANGER: Allow delete_tracked even if files are dirty
                             DATA LOSS RISK - use with extreme caution

Git integration:
  --git-commit               Auto-commit after successful verify with delete_tracked
  --git-message MSG          Commit message (default: "housekeeping: merge+prune originals")

Normalization:
  --strip-trailing-ws        Strip trailing whitespace
  --collapse-blank-lines     Collapse multiple blank lines
  --newline {preserve,lf}    Newline handling (default: lf)

Other:
  --base {a,b}               Which file to use as merge base (default: a)
  --context-lines INT        Diff context lines (default: 3)
  --write-report FILE        Write report to custom path (default: <out>/report.json)
```

---

## AGENT USAGE CONTRACT

**If you are an agent using this tool, you MUST:**

1. **Use verify before apply**
   Never skip verification. This is non-negotiable.

2. **Use explicit pairs.json**
   Never rely on implicit pair discovery or "merge everything" logic.

3. **Write outputs to a merge directory**
   Never write to repo root, working trees, or system directories.

4. **Persist receipt log**
   The cumulative receipt log (`merge_receipt.jsonl`) is a durable audit artifact. Archive it to a safe location.

5. **Cleanup only after verification**
   Never delete temp outputs before reviewing verification reports.

6. **Understand similarity ≠ approval**
   High similarity scores do NOT mean files should be merged.

7. **Default to quarantine, not delete**
   Use `--on-success quarantine` unless explicitly instructed otherwise.

8. **Never use `--allow-uncommitted` without explicit instruction**
   This flag is a DATA LOSS RISK and should only be used when absolutely necessary.

**If you violate these rules, the run is invalid and may cause data loss or repository corruption.**

---

## SAFETY CHECKLIST

Before running any destructive operation, confirm:

- [ ] I have run `--mode verify` first
- [ ] I have reviewed the verification report
- [ ] I have archived the receipt log to a durable location
- [ ] I am using a dedicated `--out` directory (not repo root)
- [ ] I understand what `--on-success` will do
- [ ] If using `delete_tracked`, files are committed and clean
- [ ] I am NOT using `--allow-uncommitted` unless I accept data loss
- [ ] I have NOT skipped any of the above steps

---

## TROUBLESHOOTING

### "Verification failed" - What now?

1. Check `report.json` for the `verification_status` field
2. Look for `fidelity_check` failures
3. Inspect the merged output in `<out>/merged/`
4. If the merge is incorrect, adjust your pairs or merge strategy
5. **DO NOT** proceed with `--on-success` flags until verification passes

### "File refused deletion: uncommitted changes"

This is a safety feature. Your file has changes that are not committed.

**Solutions:**
- Commit the file first: `git add <file> && git commit -m "..."`
- Use `--on-success quarantine` instead of `delete_tracked`
- If you truly want to delete uncommitted files (NOT RECOMMENDED), use `--allow-uncommitted`

### "Outputs are scattered everywhere"

You didn't use a dedicated `--out` directory.

**Solution:**
- Always use `--out _merge/out/` or similar
- Never use `--out .` or system directories

### "I lost my original files"

If you used `delete_tracked` without verification:
- Check git history: `git log --all --full-history -- <path>`
- Restore from git: `git checkout <commit> -- <path>`
- Check quarantine if you used `--on-success quarantine`

**Prevention:** Always use `verify` mode and archive the receipt log before deletion.

---

## CHANGELOG

See [CHANGELOG.md](CHANGELOG.md) for version history and updates.

---

## LICENSE

See repository root LICENSE file.
