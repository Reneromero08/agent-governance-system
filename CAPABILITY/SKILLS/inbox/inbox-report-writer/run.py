#!/usr/bin/env python3
"""Inbox report writer skill runner."""

import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALLOWED_OUTPUT_ROOT = (PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs").resolve()
INBOX_ROOT = (PROJECT_ROOT / "INBOX").resolve()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat
import generate_inbox_ledger
import update_inbox_index
import hash_inbox_file


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None


def _atomic_write_bytes(path: Path, data: bytes, writer: Any = None) -> None:
    if not writer:
        raise RuntimeError("GuardedWriter required")
    # write_durable expects str, so decode if it's JSON bytes (which are ASCII/UTF-8 compatible)
    # Use write_tmp for paths in _tmp directories, write_durable for others
    path_str = str(path)
    if "_tmp" in path_str or "\\tmp\\" in path_str or "/tmp/" in path_str:
        writer.write_tmp(path_str, data.decode('utf-8'))
    else:
        writer.write_auto(path_str, data.decode('utf-8'))


def _resolve_repo_path(path_str: str) -> Path:
    path = (PROJECT_ROOT / path_str).resolve()
    if not str(path).startswith(str(PROJECT_ROOT)):
        raise ValueError(f"Path escapes repo root: {path_str}")
    return path


def _ensure_output_path(path: Path) -> None:
    if not str(path).startswith(str(ALLOWED_OUTPUT_ROOT)):
        raise ValueError(f"Output path must be under LAW/CONTRACTS/_runs: {path}")


def _ensure_inbox_path(path: Path) -> None:
    """Verify path is under INBOX root."""
    if not str(path.resolve()).startswith(str(INBOX_ROOT)):
        raise ValueError(f"Report path must be under INBOX: {path}")


def _normalize_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _compute_content_hash(content: str) -> str:
    """Compute SHA256 hash of content."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _format_tags_yaml(tags: List[str]) -> str:
    """Format tags list as YAML array."""
    if not tags:
        return "[]"
    return "\n" + "\n".join(f"- {tag}" for tag in tags)


def _generate_canonical_filename(title: str, timestamp: Optional[datetime] = None) -> str:
    """Generate canonical filename per DOCUMENT_POLICY.md: MM-DD-YYYY-HH-MM_DESCRIPTIVE_TITLE.md"""
    if timestamp is None:
        timestamp = datetime.now()
    
    # Normalize title: uppercase, underscores, alphanumeric only
    normalized_title = re.sub(r'[^A-Z0-9_]', '', title.upper().replace(' ', '_').replace('-', '_'))
    # Collapse multiple underscores
    normalized_title = re.sub(r'_+', '_', normalized_title).strip('_')
    
    return f"{timestamp.strftime('%m-%d-%Y-%H-%M')}_{normalized_title}.md"


def _build_report_content(
    title: str,
    body: str,
    uuid: str,
    section: str = "report",
    bucket: str = "reports",
    author: str = "System",
    priority: str = "Medium",
    status: str = "Complete",
    summary: str = "",
    tags: Optional[List[str]] = None,
    timestamp: Optional[datetime] = None,
) -> str:
    """
    Build a canonical report with YAML frontmatter and content hash.
    
    Per DOCUMENT_POLICY.md:
    - YAML frontmatter with required fields
    - Content hash immediately after YAML (computed on body content only)
    - Markdown body
    """
    if timestamp is None:
        timestamp = datetime.now()
    if tags is None:
        tags = []
    if not summary:
        # Use first 100 chars of body as summary
        summary = body[:100].replace('\n', ' ').strip()
        if len(body) > 100:
            summary += "..."
    
    yaml_timestamp = timestamp.strftime("%Y-%m-%d %H:%M")
    
    # Compute content hash on body only (per DOCUMENT_POLICY.md)
    content_hash = _compute_content_hash(body)
    
    # Build YAML frontmatter
    tags_yaml = _format_tags_yaml(tags)
    yaml_header = f"""---
uuid: {uuid}
title: "{title}"
section: {section}
bucket: {bucket}
author: {author}
priority: {priority}
created: {yaml_timestamp}
modified: {yaml_timestamp}
status: {status}
summary: "{summary.replace('"', "'")}"
tags: {tags_yaml}
---"""
    
    # Assemble full document
    return f"{yaml_header}\n<!-- CONTENT_HASH: {content_hash} -->\n\n{body}"


def main(input_path: Path, output_path: Path) -> int:
    if not ensure_canon_compat(Path(__file__).resolve().parent):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"Error reading input JSON: {exc}")
        return 1

    errors: List[str] = []
    status = "success"

    operation = str(payload.get("operation", "")).strip() or "generate_ledger"
    inbox_path = str(payload.get("inbox_path", "INBOX"))

    output: Dict[str, Any] = {
        "operation": operation,
        "status": "success",
        "ledger_path": "",
        "index_updated": False,
        "hash_valid": None,
        "errors": [],
    }

    if not GuardedWriter:
        print("Error: GuardedWriter not available")
        return 1

    writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS", "INBOX"])
    writer.open_commit_gate()

    try:
        if operation == "generate_ledger":
            ledger_path_str = payload.get("ledger_path")
            if not ledger_path_str:
                raise ValueError("ledger_path is required for generate_ledger")
            ledger_path = _resolve_repo_path(str(ledger_path_str))
            _ensure_output_path(ledger_path)
            # scanner: guarded
            writer.mkdir_auto(str(ledger_path.parent))
            inbox_dir = _resolve_repo_path(inbox_path)
            generate_inbox_ledger.generate_ledger(inbox_dir, ledger_path, quiet=True, writer=writer)
            output["ledger_path"] = _normalize_path(ledger_path.relative_to(PROJECT_ROOT))
        elif operation == "update_index":
            allow_write = bool(payload.get("allow_inbox_write", False))
            if not allow_write:
                raise ValueError("allow_inbox_write must be true for update_index")
            inbox_dir = _resolve_repo_path(inbox_path)
            updated = update_inbox_index.update_inbox_index(inbox_dir, quiet=True)
            output["index_updated"] = bool(updated)
        elif operation == "verify_hash":
            file_path_str = payload.get("file_path")
            if not file_path_str:
                raise ValueError("file_path is required for verify_hash")
            file_path = _resolve_repo_path(str(file_path_str))
            valid, stored_hash, computed_hash = hash_inbox_file.verify_hash(file_path)
            output["hash_valid"] = bool(valid)
            output["hash_stored"] = stored_hash
            output["hash_computed"] = computed_hash
        elif operation == "write_report":
            # Required fields
            title = payload.get("title")
            body = payload.get("body")
            if not title:
                raise ValueError("title is required for write_report")
            if not body:
                raise ValueError("body is required for write_report")
            
            # Optional fields with defaults
            uuid_val = payload.get("uuid", "00000000-0000-0000-0000-000000000000")
            section = payload.get("section", "report")
            bucket = payload.get("bucket", "reports")
            author = payload.get("author", "System")
            priority = payload.get("priority", "Medium")
            report_status = payload.get("status", "Complete")
            summary = payload.get("summary", "")
            tags = payload.get("tags", [])
            
            # Output subdirectory (default: reports)
            output_subdir = payload.get("output_subdir", "reports")
            
            # Generate filename and path
            timestamp = datetime.now()
            filename = _generate_canonical_filename(title, timestamp)
            report_path = INBOX_ROOT / output_subdir / filename
            
            # Ensure output is within INBOX
            _ensure_inbox_path(report_path)
            
            # Build report content
            content = _build_report_content(
                title=title,
                body=body,
                uuid=uuid_val,
                section=section,
                bucket=bucket,
                author=author,
                priority=priority,
                status=report_status,
                summary=summary,
                tags=tags,
                timestamp=timestamp,
            )
            
            # Write via guarded writer
            writer.mkdir_auto(str(report_path.parent))
            writer.write_auto(str(report_path), content)
            
            # Output info
            output["report_path"] = _normalize_path(report_path.relative_to(PROJECT_ROOT))
            output["filename"] = filename
            output["report_written"] = True
        else:
            raise ValueError(f"Unknown operation: {operation}")
    except Exception as exc:
        errors.append(str(exc))
        status = "error"

    output["status"] = status
    output["errors"] = errors

    _atomic_write_bytes(output_path, _canonical_json_bytes(output), writer)

    return 0 if status == "success" else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
