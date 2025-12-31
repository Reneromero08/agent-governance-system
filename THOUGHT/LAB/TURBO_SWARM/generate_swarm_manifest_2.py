import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FAILED_FILE = REPO_ROOT / "test_failures_v11.txt"
OUTPUT_FILE = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"

def parse_failures(file_path):
    tasks = []
    try:
        content = file_path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        content = file_path.read_text(encoding='latin-1')
    
    # Match standard pytest failure lines:
    # FAILED CAPABILITY/TESTBENCH/core/test_cas_store.py::test_put_bytes_same_bytes_same_hash - NameError: name 'CatalyticStore' is not defined
    # OR match the Summary Section at the end
    
    # We'll use a regex to find all FAILED lines which contain the file and the error
    # But pytest --tb=short output is better parsed from the failure blocks.
    
    # Capture failure blocks
    # ________________ test_cas_corruption_detected_by_hash_toolbelt ________________
    # CAPABILITY\TESTBENCH\adversarial\test_adversarial_cas.py:39: in test_cas_corruption_detected_by_hash_toolbelt
    # ...
    # E   NameError: name 'pytest' is not defined
    
    blocks = re.split(r'________________ (.*?) ________________', content)
    if len(blocks) > 1:
        for i in range(1, len(blocks), 2):
            test_name = blocks[i]
            block_content = blocks[i+1]
            
            # Find file path
            file_match = re.search(r'([a-zA-Z0-9_/\\\.]+\.py):\d+', block_content)
            if not file_match: continue
            file_path_str = file_match.group(1).replace('\\', '/')
            
            # Find error message (E line)
            error_match = re.search(r'E\s+(.*)', block_content)
            error_msg = error_match.group(1) if error_match else "Unknown error"
            
            tasks.append({
                "file": file_path_str,
                "instruction": f"Fix the failure in {test_name}: {error_msg}. Ensure all necessary imports (pytest, Path, sys, etc.) are present and use the correct PRIMITIVES/PIPELINES structure."
            })

    # Dedup by file (combine instructions if same file)
    file_tasks = {}
    for t in tasks:
        if t["file"] not in file_tasks:
            file_tasks[t["file"]] = []
        file_tasks[t["file"]].append(t["instruction"])
    
    final_tasks = []
    for f, insts in file_tasks.items():
        # Combine instructions into a compact list
        unique_insts = list(set(insts))
        joined = "\n- ".join(unique_insts)
        final_tasks.append({
            "file": f,
            "instruction": f"Fix the following test failures:\n- {joined}\n\nIMPORTANT:\n1. Use 'from CAPABILITY.PRIMITIVES import ...' or 'from CAPABILITY.PIPELINES import ...'.\n2. Always include 'import pytest', 'import sys', and 'from pathlib import Path' if used.\n3. REPO_ROOT should be Path(__file__).resolve().parents[N] where N is the depth (core=3, phases=4, integration=3, pipeline=3)."
        })
        
    return final_tasks

def main():
    if not FAILED_FILE.exists():
        print(f"❌ {FAILED_FILE} not found")
        return
    
    tasks = parse_failures(FAILED_FILE)
    print(f"✅ Extracted {len(tasks)} file-level tasks from {FAILED_FILE.name}")
    
    OUTPUT_FILE.parent.mkdir(exist_ok=True, parents=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(tasks, f, indent=2)
    print(f"📋 Manifest: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
