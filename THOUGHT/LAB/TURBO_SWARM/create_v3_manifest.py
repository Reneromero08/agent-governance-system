import json
from pathlib import Path

# Read the failures manifest
manifest_path = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\failures.txt")
with open(manifest_path) as f:
    lines = [line.strip() for line in f if line.strip() and "FAILED" in line and "::" in line]

# Extract unique test files and their failing tests
failures_dict = {}
for line in lines:
    if line.startswith("FAILED "):
        line = line.replace("FAILED ", "")
    parts = line.split("::")
    if len(parts) >= 2:
        file_path = parts[0].replace("CAPABILITY/TESTBENCH/", "")
        test_name = parts[1].split()[0]
        if file_path not in failures_dict:
            failures_dict[file_path] = []
        if test_name not in failures_dict[file_path]:
            failures_dict[file_path].append(test_name)

# Write to V3 manifest
output_path = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TURBO_SWARM\SWARM_MANIFEST_V3.json")
with open(output_path, 'w') as f:
    json.dump(failures_dict, f, indent=2)

print(f"Created {len(failures_dict)} file entries with {sum(len(v) for v in failures_dict.values())} total failing tests")
