
import re
from pathlib import Path

# Files manually fixed (SKIP THESE)
EXCLUDE_FILES = {
    "NAVIGATION/CORTEX/db/cortex.build.py",
    "CAPABILITY/MCP/server.py",
    "CAPABILITY/SKILLS/inbox/inbox-report-writer/check_inbox_policy.py",
    "LAW/CONTRACTS/runner.py",
    "CAPABILITY/TOOLS/governance/critic.py",
    # "CAPABILITY/TOOLS/ags.py", # Fixed? Yes Step 7586.
    "CAPABILITY/TOOLS/ags.py"
}

# Buckets to scan
BUCKETS = ["LAW", "CAPABILITY", "NAVIGATION", "DIRECTION", "THOUGHT", "MEMORY"]

REPO_ROOT = Path(".").resolve()

def fix_depths():
    print(f"Starting Depth Fixer...")
    
    files_changed = 0
    
    for bucket in BUCKETS:
        bucket_dir = REPO_ROOT / bucket
        if not bucket_dir.exists():
            continue
            
        for py_file in bucket_dir.rglob("*.py"):
            rel_path = py_file.relative_to(REPO_ROOT).as_posix()
            
            # Skip excluded
            if rel_path in EXCLUDE_FILES:
                print(f"Skipping manually fixed: {rel_path}")
                continue
                
            content = py_file.read_text(encoding="utf-8")
            
            # Regex to find parents[N]
            # Capture N
            pattern = r'parents\[(\d+)\]'
            
            def replace_match(match):
                n = int(match.group(1))
                # Heuristic: If N < 10, increment it.
                # Assuming all files moved 1 level deeper.
                # All buckets (LAW, CAPABILITY...) add 1 level.
                return f'parents[{n+1}]'
            
            new_content = re.sub(pattern, replace_match, content)
            
            if new_content != content:
                print(f"Patching depth in {rel_path}")
                py_file.write_text(new_content, encoding="utf-8")
                files_changed += 1
                
    print(f"Depth Fix Complete. {files_changed} files updated.")

if __name__ == "__main__":
    fix_depths()
