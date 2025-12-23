#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List


DOC_SYSTEM = [
    "CANON/ (rules + invariants)",
    "CONTEXT/ (ADRs, preferences, rejections, research)",
    "MAPS/ (entrypoints, system map, data flow)",
    "README.md (project overview + session bootstrap)",
    "MCP/ (protocol integration docs)",
    "SKILLS/ (procedures + fixtures)",
    "CONTRACTS/ (fixtures + schemas)",
    "TOOLS/ (critics, linters, migration helpers)",
]

TARGETS_BY_TOPIC = {
    "mcp": [
        "README.md",
        "MAPS/ENTRYPOINTS.md",
        "MCP/README.md",
        "MCP/MCP_SPEC.md",
        "CONTEXT/decisions/ADR-004-mcp-integration.md",
    ],
}

NOTES_BY_TOPIC = {
    "mcp": [
        "Update only the minimal set of docs needed to inform all agents.",
        "Include entrypoint, log location, and verification skills for MCP.",
        "Prefer README + MAPS/ENTRYPOINTS for agent-visible onboarding.",
    ],
}


def load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def merge_targets(base: List[str], extra: List[str]) -> List[str]:
    combined = []
    for item in base + extra:
        if item and item not in combined:
            combined.append(item)
    return combined


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        return 1

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    payload = load_json(input_path)

    topic = str(payload.get("topic", "")).strip().lower()
    extra_targets = payload.get("extra_targets", [])

    base_targets = TARGETS_BY_TOPIC.get(topic, [])
    merged_targets = merge_targets(base_targets, extra_targets)
    recommended_targets = sorted(merged_targets)

    notes = NOTES_BY_TOPIC.get(topic, [])
    result = {
        "topic": topic,
        "doc_system": DOC_SYSTEM,
        "recommended_targets": recommended_targets,
        "notes": notes,
    }

    write_json(output_path, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
