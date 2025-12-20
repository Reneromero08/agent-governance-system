#!/usr/bin/env python3

"""
Build the cortex index.

This script scans the repository for Markdown files and other artifacts, extracting
basic metadata (id, type, title, tags) and writes the index to `CORTEX/_generated/cortex.json`.

In a real implementation, more sophisticated parsing and tagging would occur.
"""

import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = Path(__file__).resolve().parent
GENERATED_DIR = CORTEX_DIR / "_generated"
OUTPUT_FILE = GENERATED_DIR / "cortex.json"

def extract_title(path: Path) -> str:
    for line in path.read_text(errors="ignore").splitlines():
        if line.startswith("#"):
            return line.lstrip("# ").strip()
    return path.stem

def build_index() -> dict:
    entities = []
    for md_file in PROJECT_ROOT.rglob("*.md"):
        # Skip files under hidden directories and output artifacts.
        if any(part.startswith('.') for part in md_file.parts):
            continue
        if any(part in ("BUILD", "_runs", "_packs", "_generated") for part in md_file.parts):
            continue
        entity = {
            "id": f"page:{md_file.stem}",
            "type": "page",
            "title": extract_title(md_file),
            "tags": [],
            "paths": {
                "source": str(md_file.relative_to(PROJECT_ROOT))
            }
        }
        entities.append(entity)
    return {
        "cortex_version": "0.1.0",
        "canon_version": "0.1.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "entities": entities
    }

def main():
    index = build_index()
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(index, indent=2))
    print(f"Cortex index written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
