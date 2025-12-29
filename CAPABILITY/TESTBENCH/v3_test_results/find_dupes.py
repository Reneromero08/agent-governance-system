import hashlib
from pathlib import Path
from collections import defaultdict

def find_duplicates():
    files_by_hash = defaultdict(list)
    exclude = {".git", "__pycache__", "node_modules", "_runs", "_generated", "_packs", "BUILD"}
    
    for p in Path(".").rglob("*"):
        if not p.is_file():
            continue
        if any(x in p.parts for x in exclude):
            continue
            
        try:
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            files_by_hash[h].append(str(p))
        except Exception:
            continue
            
    for h, paths in files_by_hash.items():
        if len(paths) > 1:
            print(f"{h}:")
            for path in paths:
                print(f"  - {path}")

if __name__ == "__main__":
    find_duplicates()
