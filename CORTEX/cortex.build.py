#!/usr/bin/env python3

"""
Build the cortex index.

This script scans the repository for Markdown files and other artifacts, extracting
basic metadata (id, type, title, tags) and writes the index to `CORTEX/_generated/cortex.db`.

The SQLite database provides O(1) lookups by ID, type, path, or tag.

Incremental Update Strategy (v1.1):
- Retains existing database.
- Checks file modification time (mtime) against DB `last_modified`.
- Only parses and updates changed files.
- Prunes entities for deleted files.
"""

import json
import os
import re
import sqlite3
from hashlib import sha256
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import query as cortex_query

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = Path(__file__).resolve().parent
GENERATED_DIR = CORTEX_DIR / "_generated"
SCHEMA_FILE = CORTEX_DIR / "schema.sql"
DB_FILE = GENERATED_DIR / "cortex.db"
VERSIONING_PATH = PROJECT_ROOT / "CANON" / "VERSIONING.md"
SECTION_INDEX_FILE = GENERATED_DIR / "SECTION_INDEX.json"
SUMMARY_INDEX_FILE = GENERATED_DIR / "SUMMARY_INDEX.json"
SUMMARIES_DIR = GENERATED_DIR / "summaries"
CORTEX_META_FILE = GENERATED_DIR / "CORTEX_META.json"
SUMMARY_SCHEMA_VERSION = "1.0"
SUMMARY_MIN_SECTION_LINES = 10

SECTION_INDEX_DIR_ALLOWLIST = {
    "CANON",
    "CONTRACTS",
    "MAPS",
    "SKILLS",
    "CORTEX",
    "CATALYTIC-DPT",
}

SECTION_INDEX_ROOT_ALLOWLIST = {
    "README.md",
    "AGENTS.md",
    "AGS_ROADMAP_MASTER.md",
}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(?:\s+#+\s*)?$")


def get_canon_version() -> str:
    """Read canon_version from VERSIONING.md."""
    try:
        content = VERSIONING_PATH.read_text(errors="ignore")
        match = re.search(r'canon_version:\s*(\d+\.\d+\.\d+)', content)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "0.1.0"  # Fallback


def extract_title(path: Path) -> str:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return path.stem


def init_db(conn: sqlite3.Connection) -> None:
    """Initialize the database schema and handle migrations."""
    cursor = conn.cursor()
    
    # 1. ensure tables exist
    schema_sql = SCHEMA_FILE.read_text()
    cursor.executescript(schema_sql)
    
    # 2. Migration: Add last_modified if missing (for upgrades from v1.0)
    try:
        cursor.execute("SELECT last_modified FROM entities LIMIT 1")
    except sqlite3.OperationalError:
        # Check if error is due to missing column
        try:
            cursor.execute("ALTER TABLE entities ADD COLUMN last_modified REAL")
            print("[cortex] Migrated schema: added last_modified column")
        except sqlite3.OperationalError:
            pass # Already exists or other issue

    conn.commit()


def build_index(conn: sqlite3.Connection) -> int:
    """Scan the repository and populate the database incrementally. Returns updated entity count."""
    cursor = conn.cursor()

    # 1. Load snapshot of existing state
    cursor.execute("SELECT source_path, last_modified FROM entities")
    existing_state = {row[0]: row[1] for row in cursor.fetchall()}
    
    fs_paths = set()
    updates = 0
    
    # scan for deletions and updates
    for md_file in PROJECT_ROOT.rglob("*.md"):
         # Skip files under hidden directories and output artifacts.
        if any(part.startswith('.') for part in md_file.parts):
            continue
        if any(part in ("BUILD", "_runs", "_packs", "_generated") for part in md_file.parts):
             continue

        rel_path = str(md_file.relative_to(PROJECT_ROOT))
        
        try:
            current_mtime = md_file.stat().st_mtime
        except (FileNotFoundError, OSError):
            # File disappeared (e.g., pytest temp directory cleaned up)
            continue
        
        # Only add to fs_paths if file actually exists
        fs_paths.add(rel_path)
        
        # Check if update needed
        if rel_path in existing_state:
            cached_mtime = existing_state[rel_path]
            # If cached_mtime is None (migration) or older, update
            if cached_mtime and cached_mtime >= current_mtime:
                continue

        # Needs update
        
        # Unique ID generation to avoid collisions (e.g. README.md -> page:context_readme)
        unique_suffix = rel_path.replace(os.sep, "_").replace(".", "_")
        entity_id = f"page:{unique_suffix}"
        entity_type = "page"
        title = extract_title(md_file)
        
        # Cleanup any existing entity for this path (handles ID changes or re-insertion)
        cursor.execute("DELETE FROM entities WHERE source_path = ?", (rel_path,))

        # Insert or Replace
        cursor.execute(
            "INSERT OR REPLACE INTO entities (id, type, title, source_path, last_modified) VALUES (?, ?, ?, ?, ?)",
            (entity_id, entity_type, title, rel_path, current_mtime)
        )
        
        # Refresh tags (naive: delete all tags for this entity and re-add if we extracted them, 
        # but currently extract_title doesn't get tags. if we did, we'd do it here)
        # For now, just ensuring the entity row is up to date.
        
        updates += 1

    # Pruning: Delete entities for files that no longer exist
    to_delete = existing_state.keys() - fs_paths
    if to_delete:
        cursor.executemany("DELETE FROM entities WHERE source_path = ?", [(p,) for p in to_delete])
        print(f"[cortex] Pruned {len(to_delete)} deleted entities")

    # Update global metadata
    canon_version = get_canon_version()
    generated_at = os.environ.get("CORTEX_BUILD_TIMESTAMP", datetime.now(timezone.utc).isoformat())
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.preflight import compute_canon_sha256
        canon_sha256 = compute_canon_sha256(PROJECT_ROOT)
    except Exception:
        canon_sha256 = ""
    
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("cortex_version", "1.1.0"))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("canon_version", canon_version))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("generated_at", generated_at))
    cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("canon_sha256", canon_sha256))
    
    # Provenance
    try:
        import sys
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        from TOOLS.provenance import generate_header
        prov_header = generate_header(
            generator="CORTEX/cortex.build.py",
            inputs=["CANON/", "CONTEXT/", "MAPS/", "SKILLS/", "CONTRACTS/", "CATALYTIC-DPT/"]
        )
        prov_json = json.dumps(prov_header)
        cursor.execute("INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)", ("provenance", prov_json))
    except ImportError:
        pass

    conn.commit()
    return updates


def write_json_snapshot() -> None:
    """Write a JSON snapshot of the cortex index under _generated/."""
    snapshot_path = GENERATED_DIR / "cortex.json"
    data = cortex_query.export_to_json()
    snapshot_path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _slugify_heading(heading: str) -> str:
    slug = heading.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = re.sub(r"-{2,}", "-", slug).strip("-")
    return slug or "section"


def _normalize_path_for_id(path: str) -> str:
    # Path normalization policy for section_id:
    # - repo-relative
    # - forward slashes
    # - preserve repo casing (do not lowercase)
    return Path(path).as_posix()


def iter_section_index_paths() -> List[Path]:
    paths: List[Path] = []
    include_fixtures = os.environ.get("CORTEX_SECTION_INDEX_INCLUDE_FIXTURES", "").strip().lower() in {"1", "true", "yes"}
    for md_file in PROJECT_ROOT.rglob("*.md"):
        try:
            if any(part.startswith(".") for part in md_file.parts):
                continue
            if any(part in ("BUILD", "_runs", "_packs", "_generated") for part in md_file.parts):
                continue

            rel = md_file.relative_to(PROJECT_ROOT)
            if not include_fixtures and len(rel.parts) >= 2 and rel.parts[0] == "CORTEX" and rel.parts[1] == "fixtures":
                continue
            if len(rel.parts) == 1 and rel.name in SECTION_INDEX_ROOT_ALLOWLIST:
                paths.append(md_file)
                continue
            if rel.parts and rel.parts[0] in SECTION_INDEX_DIR_ALLOWLIST:
                paths.append(md_file)
                continue
        except (FileNotFoundError, OSError):
            # File disappeared during iteration (e.g., pytest temp cleanup)
            continue
    
    # Filter out any paths that no longer exist before sorting
    valid_paths = []
    for p in paths:
        try:
            p.stat()  # Check if file still exists
            valid_paths.append(p)
        except (FileNotFoundError, OSError):
            continue
    
    return sorted(valid_paths, key=lambda p: str(p.relative_to(PROJECT_ROOT)).lower())


def extract_sections_from_markdown(md_path: Path) -> List[Dict[str, object]]:
    try:
        rel_path = md_path.relative_to(PROJECT_ROOT).as_posix()
        content = md_path.read_text(encoding="utf-8", errors="replace")
    except (FileNotFoundError, OSError):
        # File disappeared during processing
        return []
    
    # Normalization contract for SECTION_INDEX hashing:
    # - Normalize newlines to '\n' before slicing and hashing.
    # - Hash is sha256 over the exact section slice (heading line through end_line inclusive).
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines(keepends=True)

    headings: List[Tuple[int, int, str]] = []
    in_fence = False
    fence_token = ""
    for i, line in enumerate(lines, start=1):
        stripped = line.lstrip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            token = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_token = token
            elif fence_token == token:
                in_fence = False
                fence_token = ""
            continue

        if in_fence:
            continue

        match = _HEADING_RE.match(line.rstrip("\n"))
        if not match:
            continue
        level = len(match.group(1))
        heading = match.group(2).strip()
        headings.append((i, level, heading))

    if not headings:
        return []

    slug_counts: Dict[str, int] = {}
    sections: List[Dict[str, object]] = []
    for idx, (start_line, level, heading) in enumerate(headings):
        end_line = len(lines)
        for next_start, next_level, _ in headings[idx + 1 :]:
            if next_level <= level:
                end_line = next_start - 1
                break

        heading_slug = _slugify_heading(heading)
        slug_counts[heading_slug] = slug_counts.get(heading_slug, 0) + 1
        ordinal = slug_counts[heading_slug]
        section_id = f"{_normalize_path_for_id(rel_path)}::{heading_slug}::{ordinal:02d}"

        slice_text = "".join(lines[start_line - 1 : end_line])
        slice_hash = sha256(slice_text.encode("utf-8")).hexdigest()

        sections.append(
            {
                "section_id": section_id,
                "path": rel_path,
                "heading": heading,
                "start_line": start_line,
                "end_line": end_line,
                "hash": slice_hash,
            }
        )

    return sections


def write_section_index() -> List[Dict[str, object]]:
    """
    Write SECTION_INDEX.json under CORTEX/_generated/.

    Line numbers are 1-based and inclusive.
    """
    all_sections: List[Dict[str, object]] = []
    for md_path in iter_section_index_paths():
        all_sections.extend(extract_sections_from_markdown(md_path))

    # Deterministic ordering: by path, then start_line.
    all_sections = sorted(all_sections, key=lambda r: (str(r["path"]), int(r["start_line"])))
    SECTION_INDEX_FILE.write_text(json.dumps(all_sections, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return all_sections


def _safe_section_id_filename(section_id: str) -> str:
    """
    Deterministically map section_id -> safe filename stem.

    Contract:
    - stable across runs
    - filesystem-safe (ASCII subset)
    - collision-resistant via hash suffix when needed
    """
    raw = (section_id or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    cleaned = re.sub(r"_{2,}", "_", cleaned).strip("._-") or "section"

    digest = sha256(raw.encode("utf-8")).hexdigest()[:8]
    max_len = 160
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("._-")
        return f"{cleaned}_{digest}"
    return f"{cleaned}_{digest}"


def _collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _split_sentences(text: str) -> List[str]:
    value = _collapse_ws(text)
    if not value:
        return []
    parts = re.split(r"(?<=[.!?])\s+", value)
    return [p.strip() for p in parts if p.strip()]


def _read_section_slice(record: Dict[str, object]) -> str:
    rel_path = str(record.get("path") or "").strip()
    if not rel_path:
        raise ValueError("section record missing path")
    start_line = int(record.get("start_line") or 0)
    end_line = int(record.get("end_line") or 0)
    if start_line <= 0 or end_line <= 0 or end_line < start_line:
        raise ValueError("invalid start_line/end_line in section record")

    file_path = PROJECT_ROOT / Path(rel_path)
    content = file_path.read_text(encoding="utf-8", errors="replace")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    lines = content.splitlines(keepends=True)

    if end_line > len(lines):
        raise ValueError("section line range outside file bounds")
    return "".join(lines[start_line - 1 : end_line])


def _summarize_section(record: Dict[str, object], slice_text: str) -> str:
    heading = str(record.get("heading") or "").strip()
    section_id = str(record.get("section_id") or "").strip()
    start_line = int(record.get("start_line") or 0)
    end_line = int(record.get("end_line") or 0)
    hash_value = str(record.get("hash") or "").strip()
    hash8 = hash_value[:8] if hash_value else ""

    # Deterministic heuristic summary:
    # - Use the first paragraph after the heading; fall back to first non-empty lines.
    content_lines = slice_text.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    body_lines = content_lines[1:] if content_lines else []

    paragraph_lines: List[str] = []
    remainder_lines: List[str] = []
    seen_nonempty = False
    for line in body_lines:
        stripped = line.strip()
        if not stripped:
            if seen_nonempty:
                remainder_lines = body_lines[len(paragraph_lines) + (0 if not paragraph_lines else 1) :]
                break
            continue
        seen_nonempty = True
        paragraph_lines.append(stripped)

    paragraph_text = " ".join(paragraph_lines).strip()
    candidates: List[str] = []
    candidates.extend(_split_sentences(paragraph_text))

    if len(candidates) < 3:
        for line in body_lines:
            stripped = _collapse_ws(line)
            if not stripped:
                continue
            if stripped == heading:
                continue
            candidates.append(stripped)
            if len(candidates) >= 6:
                break

    bullets: List[str] = []
    for item in candidates:
        normalized = _collapse_ws(item)
        if not normalized:
            continue
        if normalized not in bullets:
            bullets.append(normalized)
        if len(bullets) >= 6:
            break

    parts: List[str] = []
    parts.append(f"source: {section_id}:{start_line}-{end_line}#{hash8}")
    parts.append(f"# {heading}" if heading else "# (untitled)")
    for bullet in bullets[:6]:
        parts.append(f"- {bullet}")
    return "\n".join(parts).rstrip() + "\n"


def write_section_summaries(section_index: List[Dict[str, object]]) -> None:
    """
    Emit per-section deterministic summaries under CORTEX/_generated/.

    - Summaries are derived artifacts only (System-1 surface).
    - Skips short sections (< SUMMARY_MIN_SECTION_LINES).
    """
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)

    summary_records: List[Dict[str, object]] = []
    for record in section_index:
        try:
            start_line = int(record.get("start_line") or 0)
            end_line = int(record.get("end_line") or 0)
        except Exception:
            continue
        if start_line <= 0 or end_line <= 0:
            continue
        if (end_line - start_line + 1) < SUMMARY_MIN_SECTION_LINES:
            continue

        section_id = str(record.get("section_id") or "").strip()
        if not section_id:
            continue
        section_hash = str(record.get("hash") or "").strip()
        if not section_hash:
            continue

        try:
            slice_text = _read_section_slice(record)
        except Exception:
            continue

        summary_md = _summarize_section(record, slice_text)
        summary_bytes = summary_md.encode("utf-8")
        summary_sha = sha256(summary_bytes).hexdigest()

        filename = _safe_section_id_filename(section_id) + ".md"
        summary_rel = (Path("CORTEX") / "_generated" / "summaries" / filename).as_posix()
        summary_path = PROJECT_ROOT / Path(summary_rel)

        summary_path.write_text(summary_md, encoding="utf-8")

        summary_records.append(
            {
                "schema_version": SUMMARY_SCHEMA_VERSION,
                "section_hash": section_hash,
                "section_id": section_id,
                "summary_path": summary_rel,
                "summary_sha256": summary_sha,
            }
        )

    summary_records = sorted(summary_records, key=lambda r: str(r["section_id"]))
    SUMMARY_INDEX_FILE.write_text(
        json.dumps(summary_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main():
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)

    # Pin generated_at for this build (single value for DB + metadata outputs).
    if not os.environ.get("CORTEX_BUILD_TIMESTAMP"):
        os.environ["CORTEX_BUILD_TIMESTAMP"] = datetime.now(timezone.utc).isoformat()
    
    # Do NOT unlink DB file - we want persistence
    
    conn = sqlite3.connect(DB_FILE)
    try:
        init_db(conn)
        count = build_index(conn)
        print(f"Cortex index updated at {DB_FILE} ({count} updates)")
    finally:
        conn.close()
    write_json_snapshot()
    section_index = write_section_index()
    write_section_summaries(section_index)

    # Emit cortex build metadata for preflight drift detection.
    try:
        canon_sha = ""
        try:
            import sys
            if str(PROJECT_ROOT) not in sys.path:
                sys.path.insert(0, str(PROJECT_ROOT))
            from TOOLS.preflight import compute_canon_sha256
            canon_sha = compute_canon_sha256(PROJECT_ROOT)
        except Exception:
            canon_sha = ""

        generated_at = os.environ.get("CORTEX_BUILD_TIMESTAMP", datetime.now(timezone.utc).isoformat())
        cortex_sha = sha256(SECTION_INDEX_FILE.read_bytes()).hexdigest() if SECTION_INDEX_FILE.exists() else ""
        meta = {
            "generated_at": generated_at,
            "canon_sha256": canon_sha,
            "cortex_sha256": cortex_sha,
        }
        CORTEX_META_FILE.write_text(json.dumps(meta, ensure_ascii=True, separators=(",", ":"), sort_keys=False) + "\n", encoding="utf-8")
    except Exception:
        pass


if __name__ == "__main__":
    main()
