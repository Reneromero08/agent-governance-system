#!/usr/bin/env python3
"""
TOOLS/terminal_hunter.py

Scans the entire repository for code patterns that might spawn visible terminal windows.
Target patterns:
- 'start ' (Windows command)
- 'shell=True' (subprocess)
- 'xterm'
- 'gnome-terminal'
- 'conhost'
- 'cmd.exe'
- 'powershell.exe' without -WindowStyle Hidden
"""

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DANGEROUS_PATTERNS = [
    (r'subprocess\.Popen.*shell=True', "High", "Shell execution (risk of window)"),
    (r'os\.system\(.*start ', "Critical", "Direct 'start' command"),
    (r'subprocess\.call\(.*start ', "Critical", "Direct 'start' command"),
    (r'cmd\.exe.*\/c.*start', "Critical", "CMD start command"),
    (r'xterm', "Medium", "Linux terminal spawner"),
    (r'gnome-terminal', "Medium", "Linux terminal spawner"),
    (r'creationflags\s*=\s*0\b', "Medium", "Explicit visible window flag (0)"),
    (r'shell\s*=\s*True', "High", "Shell execution"),
]

EXCLUDES = [
    ".git",
    "__pycache__",
    "node_modules",
    ".pytest_cache",
    "TOOLS/terminal_hunter.py", # Self
    "CATALYTIC-DPT/LAB/ARCHIVE" # Deprecated stuff
]

def scan_file(path):
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        return []

    findings = []
    for pattern, severity, desc in DANGEROUS_PATTERNS:
        matches = re.finditer(pattern, content, re.IGNORECASE)
        for m in matches:
            line_no = content[:m.start()].count('\n') + 1
            snippet = content[max(0, m.start()-20):min(len(content), m.end()+20)].replace('\n', ' ')
            findings.append({
                "file": str(path.relative_to(PROJECT_ROOT)),
                "line": line_no,
                "severity": severity,
                "description": desc,
                "snippet": snippet.strip()
            })
    return findings

def main():
    print(f"Scanning {PROJECT_ROOT} for terminal spawners...")
    all_findings = []
    
    for root, dirs, files in os.walk(PROJECT_ROOT):
        # Filter excludes
        dirs[:] = [d for d in dirs if d not in EXCLUDES and not any(x in str(Path(root)/d) for x in EXCLUDES)]
        
        for file in files:
            if file.endswith(('.py', '.js', '.ps1', '.sh', '.md')):
                path = Path(root) / file
                findings = scan_file(path)
                all_findings.extend(findings)

    # Report
    if not all_findings:
        print("No terminal spawners found! The repo is clean.")
    else:
        print(f"Found {len(all_findings)} potential terminal spawners:")
        print("-" * 60)
        for f in sorted(all_findings, key=lambda x: x['severity']):
            print(f"[{f['severity']}] {f['file']}:{f['line']} - {f['description']}")
            print(f"    Snippet: ...{f['snippet']}...")
        print("-" * 60)

if __name__ == "__main__":
    main()
