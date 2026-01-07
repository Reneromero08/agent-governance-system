# INBOX Report Writer Skill

## Purpose
Complete INBOX management solution: writes canonical reports, manages content hashes, generates ledgers, normalizes structure, and **writes canonical reports** following DOCUMENT_POLICY.md format.

## Features

### 1. Hash Management
- Computes SHA256 hash of file content (excluding hash comment itself)
- Inserts/updates `<!-- CONTENT_HASH: <sha256> -->` comment
- Handles frontmatter correctly (inserts after `---` block)
- Provides CLI interface: `update` and `verify` commands

### 2. INBOX Ledger
- Generates human-readable YAML ledger of all INBOX files
- Includes metadata: path, size, modified date, hash status, frontmatter
- Categorizes files by subdirectory (reports, research, roadmaps, etc.)
- Provides summary statistics (valid/invalid/missing hashes)

### 3. Report Writer
- Creates canonical reports with YAML frontmatter per DOCUMENT_POLICY.md
- Auto-generates filename: `MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md`
- Computes and embeds content hash automatically
- Required fields: `title`, `body`
- Optional fields: `uuid`, `section`, `bucket`, `author`, `priority`, `status`, `summary`, `tags`, `output_subdir`

### 4. Report Cleanup
- Removes deprecated fields (e.g., `hashtags`) from existing reports
- Recomputes content hashes after modifications
- Validates report formatting against DOCUMENT_POLICY.md
- Dry-run mode to preview changes

### 5. INBOX Normalization
- Organizes files into `YYYY-MM/Week-XX` structure based on ISO 8601 weeks
- Parses timestamps from filenames (multiple format support)
- Generates receipts: dry-run, execution, pre/post digests
- Maintains content integrity via SHA256 verification

### 6. Policy Enforcement
- Enforces DOCUMENT_POLICY.md compliance via pre-commit hooks
- Validates all markdown files are in INBOX with content hashes
- Checks for required YAML frontmatter fields
- Prevents commits with policy violations

### 7. Weekly Automation
- Automated weekly normalization with safety checks
- Dry-run mode for validation before execution
- Full receipt generation with restore proofs
- Idempotent: skips if no new files since last run

## Usage

### Hash Management

#### Update or insert hash in a file
```bash
python CAPABILITY/SKILLS/inbox/inbox-report-writer/hash_inbox_file.py update <filepath>
```

#### Verify hash in a file
```bash
python CAPABILITY/SKILLS/inbox/inbox-report-writer/hash_inbox_file.py verify <filepath>
```

### INBOX Ledger

#### Generate ledger
```bash
python CAPABILITY/SKILLS/inbox/inbox-report-writer/generate_inbox_ledger.py
```

This creates `INBOX/LEDGER.yaml` with:
- Summary statistics (total files, valid/invalid/missing hashes)
- Files organized by category (reports, research, roadmaps, etc.)
- Full metadata for each file (path, size, modified date, hash status, frontmatter)

#### Custom paths
```bash
python CAPABILITY/SKILLS/inbox/inbox-report-writer/generate_inbox_ledger.py --inbox /path/to/inbox --output /path/to/ledger.yaml
```

### Write Report (via skill runner)

Create a canonical report using the skill runner:

```json
{
  "operation": "write_report",
  "title": "My Report Title",
  "body": "# Report\n\n## Summary\nReport content here...",
  "uuid": "agent-session-uuid",
  "section": "report",
  "bucket": "implementation/my-feature",
  "author": "AgentName",
  "priority": "High",
  "status": "Complete",
  "summary": "Brief one-line summary",
  "tags": ["feature", "implementation"],
  "output_subdir": "reports"
}
```

The skill will:
1. Generate canonical filename: `MM-DD-YYYY-HH-MM_MY_REPORT_TITLE.md`
2. Build YAML frontmatter with all required fields
3. Compute content hash on body content
4. Write to `INBOX/<output_subdir>/<filename>`
5. Return `report_path`, `filename`, and `report_written` in output

## Ledger Format

The generated `INBOX/LEDGER.yaml` looks like:

```yaml
generated: '2026-01-04T12:48:36'
inbox_path: D:\CCC 2.0\AI\agent-governance-system\INBOX
total_files: 62
summary:
  valid_hashes: 45
  invalid_hashes: 15
  missing_hashes: 2
  errors: 0
files_by_category:
  reports:
    - path: reports/01-01-2026-11-37_SYSTEM_POTENTIAL_REPORT.md
      filename: 01-01-2026-11-37_SYSTEM_POTENTIAL_REPORT.md
      size_bytes: 3069
      modified: '2026-01-01T11:37:00'
      hash:
        valid: true
        stored: abc123...
        computed: abc123...
        match: true
      frontmatter:
        title: System Potential Report
        section: report
        author: Antigravity
        priority: High
  research:
    - path: research/catalytic-chat-research.md
      ...
```

## Hash Format
The tool inserts a comment at the top of the file (after frontmatter if present):
```markdown
<!-- CONTENT_HASH: <sha256_hex> -->
```

## Integration
- Used by pre-commit hooks to validate INBOX file integrity
- Used by runtime interceptors to block unhashed writes
- Ensures all INBOX documents have valid content hashes before commit
- Ledger provides quick overview of INBOX state

### Automatic Updates

The pre-commit hook **automatically updates** both `INBOX.md` and `LEDGER.yaml` before every commit:

1. **INBOX.md** is regenerated with current file listings
2. **LEDGER.yaml** is regenerated with current metadata
3. Both files are automatically staged for commit
4. Hash validation runs on all INBOX files

This means:
- ✅ INBOX.md is always up-to-date
- ✅ LEDGER.yaml always reflects current state  
- ✅ No manual maintenance required
- ✅ Hash mismatches are caught before commit

### Manual Updates

You can also update manually:

```bash
# Update INBOX.md index
python CAPABILITY/SKILLS/inbox/inbox-report-writer/update_inbox_index.py

# Update LEDGER.yaml
python CAPABILITY/SKILLS/inbox/inbox-report-writer/generate_inbox_ledger.py

# Fix all hashes in INBOX
Get-ChildItem -Path INBOX -Filter *.md -Recurse | ForEach-Object { 
    python CAPABILITY/SKILLS/inbox/inbox-report-writer/hash_inbox_file.py update $_.FullName 
}
```

### Report Cleanup

Clean up formatting issues in existing reports:

```bash
# Dry-run mode (preview changes)
python CAPABILITY/SKILLS/inbox/inbox-report-writer/cleanup_report_formatting.py --dry-run --verbose

# Apply changes
python CAPABILITY/SKILLS/inbox/inbox-report-writer/cleanup_report_formatting.py --verbose
```

This will:
- Remove deprecated `hashtags` field from YAML frontmatter
- Recompute content hashes after modifications
- Report which files were modified

### INBOX Normalization

Organize INBOX files into `YYYY-MM/Week-XX` structure:

```bash
# Dry-run mode (generates dry-run receipt only)
python CAPABILITY/SKILLS/inbox/inbox-report-writer/inbox_normalize.py

# Execute normalization
python CAPABILITY/SKILLS/inbox/inbox-report-writer/inbox_normalize.py --execute
```

This will:
- Parse timestamps from filenames
- Organize files by calendar month and ISO week
- Generate receipts in `LAW/CONTRACTS/_runs/`:
  - `INBOX_DRY_RUN.json` - Classification plan
  - `PRE_DIGEST.json` - State before moves
  - `POST_DIGEST.json` - State after moves
  - `INBOX_EXECUTION.json` - Execution summary with integrity verification

### Policy Enforcement

Enforce DOCUMENT_POLICY.md compliance:

```bash
# Check all staged files for INBOX policy compliance
python CAPABILITY/SKILLS/inbox/inbox-report-writer/check_inbox_policy.py
```

This runs automatically via pre-commit hooks and:
- Ensures markdown files are in INBOX directory
- Validates content hashes exist
- Checks YAML frontmatter compliance

### Weekly Automation

Automated INBOX normalization (runs every Monday at 00:00 UTC):

```bash
# Safety check before running
python CAPABILITY/SKILLS/inbox/inbox-report-writer/weekly_normalize.py --check

# Dry-run mode
python CAPABILITY/SKILLS/inbox/inbox-report-writer/weekly_normalize.py

# Execute weekly normalization
python CAPABILITY/SKILLS/inbox/inbox-report-writer/weekly_normalize.py --execute
```

Features:
- Idempotent execution (skips if no new files)
- Full safety checks before execution
- Generates timestamped receipts in `LAW/CONTRACTS/_runs/inbox_weekly_YYYY-MM-DD/`

## Exit Codes
- `0`: Success (update successful or hash valid)
- `1`: Error (file not found, hash mismatch, or missing hash)
