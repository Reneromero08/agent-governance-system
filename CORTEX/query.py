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


def load_index() -> Dict[str, Any]:
    """Load the cortex index from the generated path or fallback."""
    global _INDEX
    if _INDEX is None:
        if DEFAULT_INDEX_PATH.exists():
            path = DEFAULT_INDEX_PATH
        elif FALLBACK_INDEX_PATH.exists():
            path = FALLBACK_INDEX_PATH
        else:
            raise FileNotFoundError(
                f"No cortex index found. Run 'python CORTEX/cortex.build.py' to generate."
            )
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Query the AGS Cortex")
    parser.add_argument("--id", help="Find entity by ID")
    parser.add_argument("--type", help="Find entities by type")
    parser.add_argument("--find", help="Find entities containing path substring")
    parser.add_argument("--list", action="store_true", help="List all entities (summary)")
    
    args = parser.parse_args()
    
    try:
        index = load_index()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if args.id:
        result = get_entity_by_id(args.id)
        print(json.dumps(result, indent=2) if result else "Not found.")
    elif args.type:
        results = find_entities_by_type(args.type)
        print(json.dumps(results, indent=2))
    elif args.find:
        results = find_entities_containing_path(args.find)
        print(json.dumps(results, indent=2))
    elif args.list:
        entities = index.get("entities", [])
        print(f"Cortex contains {len(entities)} indexed entities:")
        for e in entities:
            print(f"- {e['id']} ({e['type']}) : {e['title']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

__all__ = [
    "get_entity_by_id",
    "find_entities_by_type",
    "find_entities_containing_path",
    "load_index",
]
