
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

def fix_doubled_paths(directory: Path):
    print(f"Scanning {directory} for double paths...")
    count = 0
    for file_path in directory.rglob("*.py"):
        if not file_path.is_file():
            continue
            
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Fix LAW/LAW
            new_content = content.replace("LAW/LAW/", "LAW/")
            # Fix duplicate parents
            new_content = new_content.replace('parents[2] / "CATALYTIC-DPT"', 'parents[2]')
            
            # Common agent stutter: REPO_ROOT / "LAW" / "LAW"
            new_content = new_content.replace('REPO_ROOT / "LAW" / "LAW"', 'REPO_ROOT / "LAW"')
            
            # Fix parents calculation if it was messed up
            # old: REPO_ROOT = Path(__file__).resolve().parents[1] -> likely wrong for deep files
            # new files are usually deep in CAPABILITY/TESTBENCH so parents[2] is correct for REPO_ROOT
            
            if content != new_content:
                file_path.write_text(new_content, encoding='utf-8')
                print(f"Fixed {file_path.relative_to(REPO_ROOT)}")
                count += 1
        except Exception as e:
            print(f"Skipping {file_path}: {e}")
            
    print(f"Fixed {count} files.")

if __name__ == "__main__":
    fix_doubled_paths(REPO_ROOT / "CAPABILITY" / "TESTBENCH")
