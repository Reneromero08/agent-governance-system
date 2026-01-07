#!/usr/bin/env python3
"""
Canonical Document Enforcer Skill

Validates and fixes canonical filename and metadata standards for ALL markdown
documentation across the repository.

Exit Codes:
    0 - Success (all documents canonical)
    1 - Violations found (validate mode)
    2 - Fix operation failed
    3 - Invalid arguments
"""

import argparse
import hashlib
import json
import re
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation

# Exempted paths (relative to repo root)
EXEMPTIONS = [
    "LAW/CANON",
    "LAW/CONTEXT",  # ADRs, preferences, rejected - have their own governance
    "LAW/SCHEMAS",  # System schemas
    "LAW/CONTRACTS/_runs/pytest_tmp",  # Pytest temp files
    "LAW/CONTRACTS/_runs/_tmp",  # Temporary run artifacts
    "LAW/CONTRACTS/fixtures",
    "CAPABILITY",  # All implementation code and subsystem docs
    "NAVIGATION",  # Protects MAPS, PROMPTS, INVARIANTS, CORTEX (System Defs)
    "INBOX/prompts",  # Prompts have their own schema (id, model, priority...)
    "MEMORY/LLM_PACKER",  # Packer artifacts and benchmarks
    "THOUGHT/LAB",  # Experimental lab work
    ".github",
    "BUILD",
    ".git",
    ".venv",
    "node_modules",
]

# Exempted filenames
EXEMPTED_FILES = [
    "README.md",
    "AGENTS.md",
    "CHANGELOG.md",
    "LICENSE",
    "SKILL.md",  # Skill manifests
    "IMPLEMENTATION.md",  # Subsystem implementation docs
    "QUICKSTART.md",  # Quick start guides
    "SPEC.md",  # Specifications
    "CONFIG.md",  # Configuration docs
    "GUIDE.md",  # Guide docs
    "INBOX.md",  # Index files
    "git-transport.md",  # Protocol specs
]

# Canonical filename pattern
CANONICAL_PATTERN = re.compile(r"^(\d{2})-(\d{2})-(\d{4})-(\d{2})-(\d{2})_(.+)\.md$")

# Required YAML fields
REQUIRED_FIELDS = [
    "uuid",
    "title",
    "section",
    "bucket",
    "author",
    "priority",
    "created",
    "modified",
    "status",
    "summary",
    "tags",
]


def is_exempted(file_path: Path, repo_root: Path) -> bool:
    """Check if file is exempted from canonical enforcement."""
    rel_path = file_path.relative_to(repo_root)
    
    # Check exempted filenames
    if file_path.name in EXEMPTED_FILES:
        return True
    
    # Check exempted paths
    for exemption in EXEMPTIONS:
        if str(rel_path).startswith(exemption.replace("/", "\\")):
            return True
    
    return False


def is_canonical_filename(filename: str) -> bool:
    """Check if filename matches canonical pattern."""
    return bool(CANONICAL_PATTERN.match(filename))


def parse_canonical_filename(filename: str) -> Optional[Dict[str, str]]:
    """Parse canonical filename into components."""
    match = CANONICAL_PATTERN.match(filename)
    if not match:
        return None
    
    mm, dd, yyyy, hh, min_, title = match.groups()
    return {
        "month": mm,
        "day": dd,
        "year": yyyy,
        "hour": hh,
        "minute": min_,
        "title": title,
        "timestamp": f"{mm}-{dd}-{yyyy}-{hh}-{min_}",
    }


def extract_yaml_frontmatter(content: str) -> Tuple[Optional[Dict], str]:
    """Extract YAML frontmatter from markdown content."""
    lines = content.split("\n")
    
    if not lines or lines[0].strip() != "---":
        return None, content
    
    # Find closing ---
    yaml_end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            yaml_end = i
            break
    
    if yaml_end is None:
        return None, content
    
    yaml_content = "\n".join(lines[1:yaml_end])
    remaining_content = "\n".join(lines[yaml_end + 1:])
    
    try:
        metadata = yaml.safe_load(yaml_content)
        return metadata, remaining_content
    except yaml.YAMLError:
        return None, content


def compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def extract_content_hash(content: str) -> Optional[str]:
    """Extract content hash from markdown content."""
    # Look for <!-- CONTENT_HASH: <hash> --> pattern
    match = re.search(r"<!--\s*CONTENT_HASH:\s*([a-f0-9]{64})\s*-->", content)
    if match:
        return match.group(1)
    return None


def validate_file(file_path: Path, repo_root: Path) -> List[str]:
    """Validate a single file for canonical compliance."""
    violations = []
    
    # Check filename
    if not is_canonical_filename(file_path.name):
        violations.append("Invalid filename (missing timestamp prefix or wrong format)")
    
    # Read content
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        violations.append(f"Cannot read file: {e}")
        return violations
    
    # Check YAML frontmatter
    metadata, remaining = extract_yaml_frontmatter(content)
    if metadata is None:
        violations.append("Missing or invalid YAML frontmatter")
    else:
        # Check required fields
        for field in REQUIRED_FIELDS:
            if field not in metadata:
                violations.append(f"Missing required field: {field}")
    
    # Check content hash
    stored_hash = extract_content_hash(remaining)
    if stored_hash is None:
        violations.append("Missing content hash")
    else:
        # Verify hash
        # Content for hashing is everything after the hash line
        hash_line_pattern = r"<!--\s*CONTENT_HASH:\s*[a-f0-9]{64}\s*-->\n?"
        content_after_hash = re.sub(hash_line_pattern, "", remaining, count=1)
        computed_hash = compute_content_hash(content_after_hash)
        
        if stored_hash != computed_hash:
            violations.append("Content hash mismatch")
    
    # Check timestamp consistency
    if metadata and is_canonical_filename(file_path.name):
        parsed = parse_canonical_filename(file_path.name)
        if parsed and "created" in metadata:
            created_str = metadata["created"]
            # Parse created timestamp
            try:
                created_dt = datetime.strptime(created_str, "%Y-%m-%d %H:%M")
                filename_ts = f"{parsed['month']}-{parsed['day']}-{parsed['year']}-{parsed['hour']}-{parsed['minute']}"
                created_ts = created_dt.strftime("%m-%d-%Y-%H-%M")
                
                if filename_ts != created_ts:
                    violations.append(f"Timestamp mismatch: filename={filename_ts}, yaml={created_ts}")
            except ValueError:
                violations.append("Invalid created timestamp format in YAML")
    
    return violations


def generate_canonical_filename(file_path: Path, metadata: Optional[Dict] = None) -> str:
    """Generate canonical filename for a file."""
    # Use created timestamp from metadata if available
    if metadata and "created" in metadata:
        try:
            dt = datetime.strptime(metadata["created"], "%Y-%m-%d %H:%M")
        except ValueError:
            dt = datetime.fromtimestamp(file_path.stat().st_mtime)
    else:
        # Use file modification time
        dt = datetime.fromtimestamp(file_path.stat().st_mtime)
    
    timestamp = dt.strftime("%m-%d-%Y-%H-%M")
    
    # Sanitize title
    stem = file_path.stem
    # Remove existing timestamp prefix if present
    stem = re.sub(r"^\d{2}-\d{2}-\d{4}-\d{2}-\d{2}_", "", stem)
    # Convert to ALL_CAPS_WITH_UNDERSCORES
    title = stem.upper().replace(" ", "_").replace("-", "_")
    title = re.sub(r"[^A-Z0-9_]", "_", title)
    title = re.sub(r"_+", "_", title).strip("_")
    
    return f"{timestamp}_{title}.md"


def generate_yaml_frontmatter(file_path: Path, existing_metadata: Optional[Dict] = None) -> str:
    """Generate YAML frontmatter for a file."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M")
    
    # Sentinel UUID for legacy documents where agent session is unknown
    UNKNOWN_AGENT_UUID = "00000000-0000-0000-0000-000000000000"
    
    # Determine section from path
    rel_path = str(file_path)
    if "reports" in rel_path:
        section = "report"
    elif "research" in rel_path:
        section = "research"
    elif "roadmap" in rel_path:
        section = "roadmap"
    elif "ARCHIVE" in rel_path:
        section = "archive"
    else:
        section = "guide"
    
    # Determine bucket from path
    parts = Path(rel_path).parts
    if len(parts) >= 3:
        bucket = f"{parts[-3]}/{parts[-2]}"
    else:
        bucket = "uncategorized/general"
    
    stem = file_path.stem
    default_title = stem.replace("_", " ").title()
    
    metadata = {
        "uuid": existing_metadata.get("uuid", UNKNOWN_AGENT_UUID) if existing_metadata else UNKNOWN_AGENT_UUID,
        "title": existing_metadata.get("title", default_title) if existing_metadata else default_title,
        "section": existing_metadata.get("section", section) if existing_metadata else section,
        "bucket": existing_metadata.get("bucket", bucket) if existing_metadata else bucket,
        "author": existing_metadata.get("author", "System") if existing_metadata else "System",
        "priority": existing_metadata.get("priority", "Medium") if existing_metadata else "Medium",
        "created": existing_metadata.get("created", timestamp) if existing_metadata else timestamp,
        "modified": timestamp,
        "status": existing_metadata.get("status", "Active") if existing_metadata else "Active",
        "summary": existing_metadata.get("summary", "Document summary") if existing_metadata else "Document summary",
        "tags": existing_metadata.get("tags", []) if existing_metadata else [],
    }
    
    return yaml.dump(metadata, default_flow_style=False, sort_keys=False)


def fix_file(file_path: Path, repo_root: Path, dry_run: bool = False, writer: Optional[GuardedWriter] = None) -> Dict:
    """Fix a non-canonical file."""
    if not writer:
        raise ValueError("GuardedWriter is required for fix operations")

    result = {
        "file": str(file_path.relative_to(repo_root)),
        "actions": [],
        "success": True,
        "error": None,
    }
    
    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        result["success"] = False
        result["error"] = f"Cannot read file: {e}"
        return result
    
    # Extract existing metadata
    metadata, remaining = extract_yaml_frontmatter(content)
    
    # Generate canonical filename
    new_filename = generate_canonical_filename(file_path, metadata)
    original_path = file_path
    
    if file_path.name != new_filename:
        result["actions"].append(f"Rename: {file_path.name} -> {new_filename}")
        if not dry_run:
            new_path = file_path.with_name(new_filename)
            if new_path.exists():
                result["success"] = False
                result["error"] = f"Target filename already exists: {new_filename}"
                return result
            
            # Resolve to repo root relative for writer
            src_rel = file_path.resolve().relative_to(repo_root)
            dst_rel = new_path.resolve().relative_to(repo_root)
            writer.safe_rename(src_rel, dst_rel)
            
            file_path = new_path
    
    # Generate/update YAML frontmatter
    if metadata is None:
        yaml_content = generate_yaml_frontmatter(file_path, None)
        result["actions"].append("Add YAML frontmatter")
        new_content = f"---\n{yaml_content}---\n{content}"
    else:
        # Update existing metadata
        yaml_content = generate_yaml_frontmatter(file_path, metadata)
        result["actions"].append("Update YAML frontmatter")
        new_content = f"---\n{yaml_content}---\n{remaining}"
    
    # Compute and insert content hash
    # Content for hashing is everything after the hash line
    hash_line_pattern = r"<!--\s*CONTENT_HASH:\s*[a-f0-9]{64}\s*-->\n?"
    content_without_hash = re.sub(hash_line_pattern, "", new_content.split("---\n", 2)[2] if "---\n" in new_content else new_content, count=1)
    content_hash = compute_content_hash(content_without_hash)
    
    # Insert hash after YAML
    if "---\n" in new_content:
        parts = new_content.split("---\n", 2)
        if len(parts) == 3:
            new_content = f"{parts[0]}---\n{parts[1]}---\n<!-- CONTENT_HASH: {content_hash} -->\n{content_without_hash}"
    
    result["actions"].append("Update content hash")
    
    if not dry_run:
        # Resolve to repo root relative
        rel_path = file_path.resolve().relative_to(repo_root)
        writer.write_durable(rel_path, new_content)
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Canonical Document Enforcer")
    parser.add_argument("--mode", choices=["validate", "fix", "report"], required=True)
    parser.add_argument("--file", help="Specific file to process")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--output", help="Output directory for reports")
    parser.add_argument("--staged-only", action="store_true", help="Only check staged files (for pre-commit)")
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[4]
    
    # Initialize GuardedWriter
    writer = GuardedWriter(
        project_root=repo_root,
        durable_roots=["INBOX", "MEMORY", "LAW/CONTRACTS"],  # Allow modification of documents in these roots
        exclusions=[ex for ex in EXEMPTIONS if ex not in ["."]] # Basic exclusions
    )
    
    # Open commit gate for fix operations
    if args.mode == "fix":
        writer.open_commit_gate()

    # Find all markdown files
    if args.file:
        files = [Path(args.file).resolve()]
    else:
        files = list(repo_root.rglob("*.md"))
    
    # Filter exempted files
    files = [f for f in files if not is_exempted(f, repo_root)]
    
    if args.mode == "validate":
        violations_found = False
        for file_path in files:
            violations = validate_file(file_path, repo_root)
            if violations:
                violations_found = True
                print(f"[VIOLATION] {file_path.relative_to(repo_root)}")
                for v in violations:
                    print(f"  - {v}")
        
        sys.exit(1 if violations_found else 0)
    
    elif args.mode == "fix":
        results = []
        for file_path in files:
            result = fix_file(file_path, repo_root, args.dry_run, writer=writer)
            results.append(result)
            
            prefix = "[DRY-RUN]" if args.dry_run else "[FIXED]"
            if result["success"]:
                print(f"{prefix} {result['file']}")
                for action in result["actions"]:
                    print(f"  - {action}")
            else:
                print(f"[ERROR] {result['file']}: {result['error']}")
        
        # Emit receipt
        receipt_dir = repo_root / "LAW" / "CONTRACTS" / "_runs" / "canonical-doc-enforcer"
        receipt_path = receipt_dir / "fix_receipt.json"
        
        json_content = json.dumps(results, indent=2)
        
        writer.mkdir_durable(str(receipt_dir.relative_to(repo_root)))
        writer.write_durable(str(receipt_path.relative_to(repo_root)), json_content)
        
        sys.exit(0)
    
    elif args.mode == "report":
        # Generate compliance report
        total = len(files)
        compliant = sum(1 for f in files if not validate_file(f, repo_root))
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_files": total,
            "compliant_files": compliant,
            "non_compliant_files": total - compliant,
            "compliance_rate": f"{(compliant / total * 100):.1f}%" if total > 0 else "N/A",
        }
        
        print(json.dumps(report, indent=2))
        
        if args.output:
            output_dir = Path(args.output)
            json_report = json.dumps(report, indent=2)
            timestamp = datetime.now().strftime("%m-%d-%Y-%H-%M")
            report_path = output_dir / f"{timestamp}_CANONICAL_COMPLIANCE_REPORT.json"
            
            # Assuming output arg is relative or absolute, normalize to repo root
            try:
                rel_output = output_dir.resolve().relative_to(repo_root)
                writer.mkdir_durable(str(rel_output))
                writer.write_durable(str(report_path.relative_to(repo_root)), json_report)
            except ValueError:
                # Output outside repo, we cannot enforce durable write on outside repo reliably with relative paths
                # But GuardedWriter mostly fails safe.
                # Here we will just error out or warn if user asks for external report.
                print(f"Error: Output directory {output_dir} must be within repository for guarded write.")
                sys.exit(1)
                
            print(f"\nReport saved to: {report_path}")
        
        sys.exit(0)


if __name__ == "__main__":
    main()
