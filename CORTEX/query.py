"""
Simple query interface for the cortex index.

Functions in this module allow skills and tools to search the cortex by id or
by type, or to retrieve entities that contain a particular path.  In a real
system, this module could offer more powerful queries, including full-text
search, tag filtering and relational joins.
"""

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_INDEX: Optional[Dict[str, Any]] = None
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_DIR = Path(__file__).resolve().parent
DEFAULT_INDEX_PATH = CORTEX_DIR / "_generated" / "cortex.json"
FALLBACK_INDEX_PATH = CORTEX_DIR / "cortex.json"
LEGACY_INDEX_PATH = PROJECT_ROOT / "BUILD" / "cortex.json"

def load_index() -> Dict[str, Any]:
    global _INDEX
    if _INDEX is None:
        candidates = (DEFAULT_INDEX_PATH, FALLBACK_INDEX_PATH, LEGACY_INDEX_PATH)
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
            raise FileNotFoundError("No cortex index found in expected locations.")
        _INDEX = json.loads(path.read_text())
    return _INDEX

def get_entity_by_id(entity_id: str) -> Optional[Dict[str, Any]]:
    index = load_index()
    for entity in index.get("entities", []):
        if entity["id"] == entity_id:
            return entity
    return None

def find_entities_by_type(entity_type: str) -> List[Dict[str, Any]]:
    index = load_index()
    return [e for e in index.get("entities", []) if e.get("type") == entity_type]

def find_entities_containing_path(path_substring: str) -> List[Dict[str, Any]]:
    index = load_index()
    results = []
    for entity in index.get("entities", []):
        for p in entity.get("paths", {}).values():
            if path_substring in p:
                results.append(entity)
                break
    return results

__all__ = [
    "get_entity_by_id",
    "find_entities_by_type",
    "find_entities_containing_path",
    "load_index",
]
