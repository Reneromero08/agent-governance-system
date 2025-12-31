import json
from pathlib import Path
import os

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTBENCH_ROOT = REPO_ROOT / "CAPABILITY" / "TESTBENCH"

tasks = []

# Walk the new structure
for root, dirs, files in os.walk(TESTBENCH_ROOT):
    for file in files:
        if not file.endswith(".py") or file.startswith("_") or file.startswith("fix_"):
            continue
            
        full_path = Path(root) / file
        rel_path = full_path.relative_to(REPO_ROOT)
        
        # Calculate depth for parents[N]
        # REPO_ROOT is base.
        # CAPABILITY/TESTBENCH/core/file.py -> Depth 3 from REPO_ROOT (CAPABILITY, TESTBENCH, core)
        # So parents[N] where N is number of parts in relative path - 1 (file itself)
        # Actually:
        # file in REPO/A/B/C/file.py
        # parents[0] = C
        # parents[1] = B
        # parents[2] = A
        # parents[3] = REPO
        
        parts = rel_path.parts # ('CAPABILITY', 'TESTBENCH', 'core', 'test_cas_store.py')
        depth = len(parts) - 1
        
        instruction = (
            f"This file has been moved to `{rel_path.parent}`. "
            f"Update the REPO_ROOT calculation to use `parents[{depth}]`. "
            f"Ensure `sys.path` correctly includes the repo root relative to this new location. "
            "Fix any local relative imports if they broke."
        )
        
        tasks.append({
            "file": str(rel_path).replace("\\", "/"),
            "instruction": instruction
        })

output_path = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tasks, f, indent=2)

print(f"Generated {len(tasks)} reorg tasks.")
