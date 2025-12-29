import json
from pathlib import Path

# Paths
MANIFEST_V4_PATH = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TURBO_SWARM\SWARM_MANIFEST_V4.json")
OUTPUT_MANIFEST_PATH = Path(r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\SWARM_MANIFEST.json")

# Load V4
with open(MANIFEST_V4_PATH, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

# Write to the location the orchestrator expects
with open(OUTPUT_MANIFEST_PATH, 'w', encoding='utf-8') as f:
    json.dump(tasks, f, indent=2)

print(f"Successfully deployed {len(tasks)} prompts to {OUTPUT_MANIFEST_PATH}")
