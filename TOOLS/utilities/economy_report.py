#!/usr/bin/env python3
"""
Economy Report - Calculates token usage and savings across the repository.
Driven by The Living Formula (@F0).
"""

import json
import sys
from pathlib import Path

# Enable internal imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TOOLS.tokenizer_harness import analyze_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRACKED_FOLDERS = ['CANON', 'SKILLS', 'TOOLS', 'CONTEXT']

def generate_report():
    print(f"AGS ECONOMY REPORT (Formula Drive)")
    print(f"{'='*40}")
    
    total_orig = 0
    total_comp = 0
    
    folder_stats = {}
    
    for folder in TRACKED_FOLDERS:
        dir_path = PROJECT_ROOT / folder
        if not dir_path.exists():
            continue
            
        f_orig = 0
        f_comp = 0
        
        for f in dir_path.rglob("*"):
            if f.is_file() and f.suffix.lower() in ('.md', '.py', '.json'):
                try:
                    text = f.read_text(encoding='utf-8', errors='ignore')
                    if not text.strip(): continue
                    res = analyze_text(text)
                    f_orig += res['original']
                    f_comp += res['compressed']
                except:
                    continue
        
        folder_stats[folder] = (f_orig, f_comp)
        total_orig += f_orig
        total_comp += f_comp
        
    for name, (o, c) in folder_stats.items():
        savings = o - c
        pct = (savings / o * 100) if o > 0 else 0
        print(f"{name:10}: {o:7} tokens -> {c:7} ({pct:+.2f}%)")
        
    print(f"{'-'*40}")
    total_savings = total_orig - total_comp
    total_pct = (total_savings / total_orig * 100) if total_orig > 0 else 0
    
    print(f"{'TOTAL':10}: {total_orig:7} tokens")
    print(f"{'SAVINGS':10}: {total_savings:7} tokens ({total_pct:+.2f}%)")
    print(f"{'='*40}")
    
    if total_pct < 0:
        print("💡 NOTE: Compression Penalty detected. Symbols like '@C0' may be more expensive")
        print("   than raw words for the current tokenizer. Formula Audit @F0 required.")

if __name__ == "__main__":
    generate_report()
