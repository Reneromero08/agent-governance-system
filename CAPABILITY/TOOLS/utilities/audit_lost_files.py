
import subprocess
from pathlib import Path

def get_git_deleted():
    # Get list of deleted files (only those that were tracked)
    out = subprocess.check_output(["git", "ls-files", "-d"], text=True)
    return set(out.splitlines())

def get_all_new_files():
    # Get all files in the new buckets
    new_files = set()
    buckets = ["LAW", "CAPABILITY", "NAVIGATION", "DIRECTION", "THOUGHT", "MEMORY"]
    for bucket in buckets:
        for p in Path(bucket).rglob("*"):
            if p.is_file():
                # We want the basename or some way to match
                # Actually, most files were moved with their relative structure preserved
                # e.g. CANON/SOP.md -> LAW/CANON/SOP.md
                new_files.add(p.as_posix())
    return new_files

def audit():
    deleted = get_git_deleted()
    new_paths = get_all_new_files()
    
    missing = []
    
    for d in deleted:
        # Check if this file (by basename or relative subpath) exists in any of the buckets
        # Most moves follow: bucket/original_path
        # But some moved directly, e.g. CANON -> LAW/CANON
        found = False
        
        # Check standard bucket moves
        buckets = ["LAW", "CAPABILITY", "NAVIGATION", "DIRECTION", "THOUGHT", "MEMORY"]
        for b in buckets:
            # Try bucket/d
            if (Path(b) / d).exists():
                found = True
                break
            # Try specific exceptions if known
            # (In my case, I moved them like Move-Item CANON LAW/CANON)
            # So if 'CANON/SOP.md' was deleted, it should be in 'LAW/CANON/SOP.md'
            # Path(b)/d works for that.
            
        if not found:
            # Special check for root files that moved
            # e.g. AGS_ROADMAP_MASTER.md -> DIRECTION/AGS_ROADMAP_MASTER.md
            for b in buckets:
                if (Path(b) / Path(d).name).exists():
                    found = True # Likely moved to bucket root
                    break
        
        if not found:
            missing.append(d)
            
    if missing:
        print("POTENTIALLY LOST FILES (Deleted from git, not found in buckets):")
        for m in missing:
            print(f"  {m}")
    else:
        print("No files lost during the 6-bucket move.")

if __name__ == "__main__":
    audit()
