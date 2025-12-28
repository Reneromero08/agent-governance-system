#!/usr/bin/env python3
"""
Skill: agi-hardener
Harden the AGI repository at D:/CCC 2.0/AI/AGI to follow AGS standards.
"""

import os
import re
import json
import shutil
from pathlib import Path

AGI_ROOT = Path("D:/CCC 2.0/AI/AGI")

def atomic_write(path: Path, content: str):
    temp_path = path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        f.write(content)
    os.replace(temp_path, path)

def harden_file(path: Path):
    if not path.exists(): return
    
    print(f"[Hardener] Processing {path}...")
    content = path.read_text(encoding="utf-8", errors="ignore")
    original = content
    
    # 1. Fix bare excepts
    # Replace: except: OR except Exception: with except Exception as e: + logging
    content = re.sub(r'except:\s*', 'except Exception as e:\n        print(f"Error: {e}")\n        ', content)
    content = re.sub(r'except Exception:\s*', 'except Exception as e:\n        print(f"Error: {e}")\n        ', content)
    
    # 2. Fix input() calls
    content = re.sub(r'input\(.*\)', '"Batch Mode (Auto-Input)"', content)
    
    # 3. Add utf-8 encoding to opens if missing
    def fix_open(match):
        line = match.group(0)
        if "encoding=" not in line:
            return line.replace(")", ", encoding='utf-8')")
        return line
    
    content = re.sub(r'open\([^,)]+\)', fix_open, content)
    
    if content != original:
        atomic_write(path, content)
        print(f"[Hardener] Fixed {path}")

def main():
    if not AGI_ROOT.exists():
        print(f"Error: {AGI_ROOT} not found.")
        return

    targets = [
        AGI_ROOT / "SKILLS/ant/run.py",
        AGI_ROOT / "SKILLS/swarm-governor/run.py",
        AGI_ROOT / "SKILLS/research-ingest/run.py",
        AGI_ROOT / "SKILLS/launch-terminal/run.py",
        AGI_ROOT / "MCP/server.py"
    ]
    
    for target in targets:
        harden_file(target)
    
    print("TASK_COMPLETE")

if __name__ == "__main__":
    main()
