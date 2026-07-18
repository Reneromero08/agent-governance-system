#!/usr/bin/env python3
"""Render the machine-readable capability graph as Markdown."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def root() -> Path:
    return Path(__file__).resolve().parents[1]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def render(graph: dict[str, Any]) -> str:
    lines = [
        "# CAT_CAS Capability Graph",
        "",
        "This view is generated from `control/capability_graph.json`. It answers what",
        "each load-bearing experiment contributes to the final machine. It is not an",
        "evidence ledger and does not replace experiment reports.",
        "",
        f"Mission: `{graph['mission_id']}`",
        "",
        "## Compute-leverage ladder",
        "",
    ]
    for rung in graph.get("compute_rungs", []):
        lines.extend(
            [
                f"### {rung['id']}",
                "",
                rung["definition"],
                "",
            ]
        )
    lines.extend(["## Capability lineage", ""])
    for node in graph.get("nodes", []):
        lines.extend(
            [
                f"### {node['id']} - {node['title']}",
                "",
                f"- Path: `{node['path']}`",
                f"- Mission rungs: {', '.join(f'`{item}`' for item in node.get('mission_rungs', []))}",
                f"- Task class: `{node['task_class']}`",
                f"- Status: `{node['status']}`",
                f"- Capabilities: {', '.join(node.get('capabilities', [])) or 'none'}",
                f"- Open blockers: {', '.join(node.get('open_blockers', [])) or 'none'}",
                "",
            ]
        )
        if node.get("code_entrypoints"):
            lines.append("Code entrypoints:")
            lines.append("")
            for entrypoint in node["code_entrypoints"]:
                lines.append(f"- `{entrypoint}`")
            lines.append("")
    lines.extend(["## Transfer edges", ""])
    for source, target, relation in graph.get("edges", []):
        lines.append(f"- `{source}` -> `{target}`: {relation}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if CAPABILITY_GRAPH.md is stale.")
    parser.add_argument("--write", action="store_true", help="Write CAPABILITY_GRAPH.md.")
    args = parser.parse_args()
    if args.check == args.write:
        parser.error("choose exactly one of --check or --write")

    lab = root()
    rendered = render(load(lab / "control" / "capability_graph.json"))
    target = lab / "CAPABILITY_GRAPH.md"
    if args.write:
        target.write_text(rendered, encoding="utf-8")
        print(target)
        return 0

    if not target.exists() or target.read_text(encoding="utf-8") != rendered:
        print("CAT_CAS_CAPABILITY_GRAPH_STALE")
        return 1
    print("CAT_CAS_CAPABILITY_GRAPH_CURRENT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
