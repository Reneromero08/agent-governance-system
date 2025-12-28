#!/usr/bin/env python3
"""
Semiotic Compression Layer (Lane I1)

Implements the "Symbol Stack" for compressing context.
- Generates @Symbols for all indexed files
- Resolves symbols to content (decompression)
- Expands prompts using symbol registry
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, Optional, List

# Configuration
META_DIR = Path("meta")
SYMBOL_REGISTRY_PATH = META_DIR / "SYMBOL_REGISTRY.json"
FILE_INDEX_PATH = META_DIR / "FILE_INDEX.json"

class SemioticLayer:
    def __init__(self):
        self.registry = {}
        META_DIR.mkdir(exist_ok=True)
        self._load_registry()

    def _load_registry(self):
        """Load symbol registry from disk."""
        if SYMBOL_REGISTRY_PATH.exists():
            with open(SYMBOL_REGISTRY_PATH, 'r', encoding='utf-8') as f:
                self.registry = json.load(f)

    def generate_symbols(self):
        """Auto-generate symbols for all files in usage."""
        print("[SCL] Generating symbols...")
        
        if not FILE_INDEX_PATH.exists():
            print("Error: meta/FILE_INDEX.json not found. Run Cortex Indexer first.")
            return

        with open(FILE_INDEX_PATH, 'r', encoding='utf-8') as f:
            file_index = json.load(f)

        new_registry = {}
        
        # Strategy 1: Filename-based symbols (@Filename)
        for rel_path, data in file_index.items():
            path_obj = Path(rel_path)
            stem = path_obj.stem
            symbol = f"@{stem}"
            
            # Handle collisions (naive: append dir)
            if symbol in new_registry:
                parent = path_obj.parent.name
                symbol = f"@{parent}/{stem}"
            
            new_registry[symbol] = {
                "type": "file",
                "target": rel_path,
                "description": f"Content of {rel_path}"
            }
            
            # Strategy 2: Section-based symbols (@Filename#Start)
            if "sections" in data:
                for sec in data["sections"]:
                    if sec["level"] > 0: # Skip root
                        slug = sec["anchor"].split("#")[1]
                        sec_symbol = f"{symbol}#{slug}"
                        new_registry[sec_symbol] = {
                            "type": "section",
                            "target": sec["anchor"],
                            "description": f"Section '{sec['title']}' in {rel_path}"
                        }
        
        self.registry = new_registry
        self._save_registry()
        print(f"[SCL] Generated {len(self.registry)} symbols.")

    def resolve(self, symbol: str) -> Optional[str]:
        """Resolve a single symbol to its content."""
        if symbol not in self.registry:
            return None
            
        entry = self.registry[symbol]
        target = entry["target"]
        
        try:
            if entry["type"] == "file":
                return Path(target).read_text(encoding='utf-8')
            elif entry["type"] == "section":
                path_str, anchor = target.split("#")
                content = Path(path_str).read_text(encoding='utf-8')
                # Naive section extraction (should use DB/Indexer logic)
                # For now, return full file with anchor context
                return f"[Content from {target}]\n{content}" 
        except Exception as e:
            return f"[Error resolving {symbol}: {e}]"
            
        return None

    def expand_prompt(self, prompt: str) -> str:
        """Expand all @Symbols in a prompt."""
        # Regex to find @Symbol tokens (alphanumeric, /, -, #, _)
        # Matches @Foo, @Foo/Bar, @Foo#Section
        pattern = r'(@[\w\-/\.]+)(?:#([\w\-]+))?'
        
        def replace(match):
            full_match = match.group(0)
            if full_match in self.registry:
                content = self.resolve(full_match)
                if content:
                    return f"\n<expanded_context id='{full_match}'>\n{content}\n</expanded_context>\n"
            return full_match # Keep original if not found
            
        expanded = re.sub(pattern, replace, prompt)
        return expanded

    def _save_registry(self):
        with open(SYMBOL_REGISTRY_PATH, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, sort_keys=True)

def main():
    scl = SemioticLayer()
    
    import argparse
    parser = argparse.ArgumentParser(description="Semiotic Compression Layer CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Generate
    subparsers.add_parser("generate", help="Generate symbol registry from index")
    
    # Expand
    expand_parser = subparsers.add_parser("expand", help="Expand symbols in a text string")
    expand_parser.add_argument("text", help="Text containing @Symbols")
    
    # List
    subparsers.add_parser("list", help="List available symbols")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        scl.generate_symbols()
    elif args.command == "expand":
        print(scl.expand_prompt(args.text))
    elif args.command == "list":
        for sym, data in sorted(scl.registry.items()):
            print(f"{sym:30} -> {data['target']}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
