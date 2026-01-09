#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None

DOC_SYSTEM = [
    "CANON/ (rules + invariants)",
    "CONTEXT/ (ADRs, preferences, rejections, research)",
    "MAPS/ (entrypoints, system map, data flow)",
    "README.md (project overview + session bootstrap)",
    "MCP/ (protocol integration docs)",
    "SKILLS/ (procedures + fixtures)",
    "LAW/CONTRACTS/ (fixtures + schemas)",
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


def write_json(path: Path, payload: Dict, writer: GuardedWriter) -> None:
    # Use GuardedWriter for safe write
    if writer:
        try:
            rel_path = str(path.resolve().relative_to(writer.project_root))
        except ValueError:
            # Fallback if path is outside project root (should not happen in standard flow)
             # But for strict firewall, we fail or assume tmp if specific roots allow.
             # We'll rely on writer to handle absolute paths if it supports it, 
             # or just fail if we can't resolve relative.
             # Actually GuardedWriter currently expects relative or absolute.
            rel_path = str(path)
        
        # Decide if tmp or durable? 
        # Skill outputs are usually tmp unless they are final artifacts.
        # But this skill just outputs a plan (JSON).
        # We will assume tmp for now or check if it matches durable roots.
        # The safest is to try tmp, then durable if that fails? No, that's ambiguous.
        # Given this is "doc-update", it produces a plan.
        # Let's assume tmp write for skill output.
        writer.write_auto(rel_path, json.dumps(payload, indent=2))
    else:
        raise RuntimeError("GuardedWriter required for write_json")


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

    # Initialize GuardedWriter
    if GuardedWriter is None:
        print("Failed to initialize GuardedWriter: import failed")
        return 1
    try:
        writer = GuardedWriter(PROJECT_ROOT, durable_roots=["LAW/CONTRACTS/_runs", "CAPABILITY/SKILLS"])
        # We are writing to output_path which is likely in _runs/_tmp or similar.
    except Exception:
        print("Failed to initialize GuardedWriter")
        return 1

    write_json(output_path, result, writer)
    return 0


if __name__ == "__main__":
    sys.exit(main())
