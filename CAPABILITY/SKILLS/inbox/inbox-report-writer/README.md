# INBOX Report Writer Skill

## Purpose
Automatically computes and manages SHA256 content hashes for INBOX markdown files to ensure integrity and prevent unauthorized modifications.

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

## Exit Codes
- `0`: Success (update successful or hash valid)
- `1`: Error (file not found, hash mismatch, or missing hash)
