#!/usr/bin/env python3
"""
Tokenizer Test Harness

Measures real tokenization counts against target models (GPT-4o, o1, etc.).
Provides comparison between raw and compressed (Codebook) text.

Usage:
    python TOOLS/tokenizer_harness.py "Load CONTRACT.md"
    python TOOLS/tokenizer_harness.py --file CANON/CONTRACT.md
    python TOOLS/tokenizer_harness.py --dir CANON
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Enable internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import tiktoken
except ImportError:
    print("Error: tiktoken not installed. Run 'pip install tiktoken'")
    sys.exit(1)

from CAPABILITY.TOOLS.compress import compress

# Models and their encodings
MODEL_ENCODINGS = {
    "o1": "o200k_base",
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "claude-3-proxy": "cl100k_base", # Claude doesn't have a public tokenizer, cl100k is a common proxy
}

def get_tokens(text: str, model: str = "gpt-4o") -> int:
    """Get token count for a model."""
    encoding_name = MODEL_ENCODINGS.get(model, "cl100k_base")
    try:
        encoding = tiktoken.get_encoding(encoding_name)
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def analyze_text(text: str, model: str = "gpt-4o") -> Dict:
    """Analyze text tokens and compression impact."""
    original_tokens = get_tokens(text, model)
    compressed_text, _ = compress(text, aggressive=True)
    compressed_tokens = get_tokens(compressed_text, model)
    
    savings = original_tokens - compressed_tokens
    percent = (savings / original_tokens * 100) if original_tokens > 0 else 0
    
    return {
        "original": original_tokens,
        "compressed": compressed_tokens,
        "savings": savings,
        "percent": round(percent, 2)
    }

def main():
    parser = argparse.ArgumentParser(description="Tokenizer Test Harness")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", help="File to analyze")
    parser.add_argument("--dir", help="Directory to analyze (recursive)")
    parser.add_argument("--model", default="gpt-4o", choices=list(MODEL_ENCODINGS.keys()), help="Default model to use")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    args = parser.parse_args()
    
    if args.file:
        path = Path(args.file).resolve()
        if not path.exists():
            print(f"File not found: {args.file}")
            return 1
        text = path.read_text(encoding="utf-8", errors="ignore")
        results = analyze_text(text, args.model)
        results["path"] = str(path.relative_to(PROJECT_ROOT))
    elif args.dir:
        dir_path = Path(args.dir).resolve()
        if not dir_path.exists():
            # Try relative to project root
            dir_path = (PROJECT_ROOT / args.dir).resolve()
            
        if not dir_path.exists():
            print(f"Directory not found: {args.dir}")
            return 1
        
        dir_results = []
        total_original = 0
        total_compressed = 0
        
        # Files to analyze
        extensions = [".md", ".txt", ".py", ".json"]
        
        for f in sorted(dir_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in extensions:
                try:
                    t = f.read_text(encoding="utf-8", errors="ignore")
                    if not t.strip():
                        continue
                    res = analyze_text(t, args.model)
                    res["path"] = str(f.relative_to(PROJECT_ROOT))
                    dir_results.append(res)
                    total_original += res["original"]
                    total_compressed += res["compressed"]
                except Exception as e:
                    # print(f"Error reading {f}: {e}")
                    continue
        
        results = {
            "files": dir_results,
            "total": {
                "original": total_original,
                "compressed": total_compressed,
                "savings": total_original - total_compressed,
                "percent": round(((total_original - total_compressed) / total_original * 100), 2) if total_original > 0 else 0
            }
        }
    elif args.text:
        results = analyze_text(args.text, args.model)
    else:
        parser.print_help()
        return 0

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        if "total" in results:
            print(f"Analysis for directory: {args.dir} (Model: {args.model})")
            print(f"{'Path':<50} {'Original':>10} {'Compressed':>12} {'Savings':>10}")
            print("-" * 85)
            for f in results["files"]:
                print(f"{f['path'][:49]:<50} {f['original']:>10} {f['compressed']:>12} {f['percent']:>9}%")
            print("-" * 85)
            print(f"{'TOTAL':<50} {results['total']['original']:>10} {results['total']['compressed']:>12} {results['total']['percent']:>9}%")
        else:
            print(f"Model: {args.model}")
            if "path" in results:
                print(f"Path:  {results['path']}")
            print(f"Original tokens:   {results['original']}")
            print(f"Compressed tokens: {results['compressed']}")
            print(f"Savings:           {results['savings']} tokens ({results['percent']}%)")
            
    return 0

if __name__ == "__main__":
    sys.exit(main())
