#!/usr/bin/env python3
"""Compile a task-scoped CAT_CAS phase-lock packet.

The packet is intentionally small. It binds the constitutional mission, current
frontier state, branch context, relevant capability lineage, and exact code paths.
Agents complete the generated receipt before implementation.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fnmatch
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


MISSION_ID = "CAT_CAS_UNBOUNDED_COMPUTE_V1"
TASK_CLASSES = (
    "flagship_compute",
    "enabling_infrastructure",
    "external_product",
    "calibration",
    "evidence_audit",
)
MODES = ("exploration", "engineering", "verification", "compression")


def lab_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def git_value(root: Path, *args: str, default: str = "UNKNOWN") -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        return result.stdout.strip() or default
    except (OSError, subprocess.CalledProcessError):
        return default


def normalize_words(text: str) -> set[str]:
    punctuation = ".,:;()[]{}<>/\\-_`'\""
    return {
        word.strip(punctuation).lower()
        for word in text.split()
        if len(word.strip(punctuation)) > 2
    }


def node_score(task: str, node: dict[str, Any]) -> int:
    haystack = canonical_json(node).decode("utf-8").lower()
    score = sum(1 for word in normalize_words(task) if word in haystack)

    aliases = {
        "audio": "AUDIO_SIDEQUEST",
        "pi": "AUDIO_SIDEQUEST",
        "torus": "AUDIO_SIDEQUEST",
        "wave": "AUDIO_SIDEQUEST",
        "bounty": "TRACK8",
        "external": "TRACK8",
        "holo": "HOLO_GEN5",
        "wall": "EXP50",
        "physical": "EXP50",
        "decoder": "EXP49",
        "fixed-point": "EXP49",
        "fixed": "EXP49",
        "restoration": "EXP01",
        "multiplex": "EXP08",
        "cache": "EXP12",
        "throughput": "EXP14",
    }
    task_lower = task.lower()
    for alias, node_id in aliases.items():
        if alias in task_lower and node.get("id") == node_id:
            score += 5
    return score


def select_nodes(task: str, graph: dict[str, Any], limit: int) -> list[dict[str, Any]]:
    nodes = graph.get("nodes", [])
    ranked = sorted(nodes, key=lambda item: (node_score(task, item), item.get("id", "")), reverse=True)
    selected = [node for node in ranked if node_score(task, node) > 0][:limit]
    return selected or ranked[: min(limit, len(ranked))]


def branch_record(branch: str, registry: dict[str, Any]) -> dict[str, Any] | None:
    records = registry.get("branches", [])
    exact = [record for record in records if record.get("pattern") == branch]
    if exact:
        return exact[0]
    for record in records:
        if fnmatch.fnmatch(branch, record.get("pattern", "")):
            return record
    return None


def load_branch_context(root: Path, branch: str, registry: dict[str, Any]) -> dict[str, Any]:
    record = branch_record(branch, registry)
    if record is None:
        return {
            "branch": branch,
            "status": "UNREGISTERED_BRANCH",
            "warning": "Inspection is allowed, but mission or claim decisions require registered context.",
        }

    source = record.get("context_source")
    context: dict[str, Any]
    if source:
        path = root / source
        context = read_json(path) if path.exists() else {
            "status": "MISSING_CONTEXT_SOURCE",
            "expected": source,
        }
    else:
        context = {"status": "REGISTERED_WITHOUT_CONTEXT_SOURCE"}

    context = dict(context)
    context.setdefault("branch", branch)
    context["registry_record"] = record
    return context


def context_claim_ceiling(branch_context: dict[str, Any], state: dict[str, Any]) -> str:
    if branch_context.get("claim_ceiling"):
        return str(branch_context["claim_ceiling"])
    tokens: list[str] = []
    for frontier in state.get("active_frontiers", []):
        tokens.extend(frontier.get("claim_tokens", []))
    return " | ".join(tokens) if tokens else "SEE_CURRENT_STATE"


def build_digest(
    mission: dict[str, Any],
    state: dict[str, Any],
    branch_context: dict[str, Any],
    selected: list[dict[str, Any]],
    task: str,
    mode: str,
    task_class: str,
) -> str:
    return sha256_bytes(
        canonical_json(
            {
                "mission": mission,
                "state": state,
                "branch_context": branch_context,
                "selected": selected,
                "task": task,
                "mode": mode,
                "task_class": task_class,
            }
        )
    )


def write_packet(
    path: Path,
    *,
    mission: dict[str, Any],
    state: dict[str, Any],
    branch_context: dict[str, Any],
    selected: list[dict[str, Any]],
    task: str,
    mode: str,
    task_class: str,
    branch: str,
    commit: str,
    digest: str,
) -> None:
    lines = [
        "# CAT_CAS Phase-Lock Packet",
        "",
        f"- Mission: `{mission['mission_id']}`",
        f"- Context digest: `{digest}`",
        f"- Task: {task}",
        f"- Mode: `{mode}`",
        f"- Task class: `{task_class}`",
        f"- Branch: `{branch}`",
        f"- Commit: `{commit}`",
        "",
        "## Mission",
        "",
        mission["one_sentence"],
        "",
        "## Native chain",
        "",
        "```text",
        " -> ".join(mission["native_chain"]),
        "```",
        "",
        "## Compute target",
        "",
        f"`{mission['compute_metric']['formula']}`",
        "",
        f"Strong target: `{mission['compute_metric']['strong_target']}`",
        "",
        "## Current state",
        "",
        "```json",
        json.dumps(state, indent=2, sort_keys=True),
        "```",
        "",
        "## Branch context",
        "",
        "```json",
        json.dumps(branch_context, indent=2, sort_keys=True),
        "```",
        "",
        "## Relevant capability lineage",
        "",
    ]
    for node in selected:
        lines.extend(
            [
                f"### {node.get('id')}: {node.get('title')}",
                "",
                "```json",
                json.dumps(node, indent=2, sort_keys=True),
                "```",
                "",
            ]
        )
    lines.extend(
        [
            "## Required reconstruction",
            "",
            "Complete `PHASE_LOCK_RECEIPT.json` before implementation. A flagship task",
            "must identify the classical explosion, `.holo` process-object, native",
            "operator, invariant or fixed point, final classical boundary, restoration",
            "law, no-smuggle killer control, and current claim ceiling.",
            "",
            "The receipt must be validated with:",
            "",
            "```text",
            "python tools/validate_control_plane.py --receipt _agent/PHASE_LOCK_RECEIPT.json",
            "```",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--task-class", choices=TASK_CLASSES, required=True)
    parser.add_argument("--branch", help="Override the current Git branch for branch-scoped priming.")
    parser.add_argument("--output", default="_agent")
    parser.add_argument("--max-nodes", type=int, default=6)
    args = parser.parse_args()

    root = lab_root()
    mission = read_json(root / "control" / "mission.json")
    if mission.get("mission_id") != MISSION_ID:
        raise SystemExit("Mission ID mismatch")
    state = read_json(root / "control" / "current_state.json")
    graph = read_json(root / "control" / "capability_graph.json")
    registry = read_json(root / "control" / "branch_registry.json")

    branch = args.branch or git_value(root, "branch", "--show-current")
    commit = git_value(root, "rev-parse", "HEAD")
    context = load_branch_context(root, branch, registry)
    selected = select_nodes(args.task, graph, args.max_nodes)
    digest = build_digest(mission, state, context, selected, args.task, args.mode, args.task_class)

    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / "PHASE_LOCK_PACKET.md"
    receipt_path = output / "PHASE_LOCK_RECEIPT.json"
    task_contract_path = output / "TASK_CONTRACT.json"

    write_packet(
        packet_path,
        mission=mission,
        state=state,
        branch_context=context,
        selected=selected,
        task=args.task,
        mode=args.mode,
        task_class=args.task_class,
        branch=branch,
        commit=commit,
        digest=digest,
    )

    receipt = read_json(root / "templates" / "PHASE_LOCK_RECEIPT.json")
    code_paths = sorted(
        {
            path
            for node in selected
            for path in node.get("code_entrypoints", [])
        }
    )
    receipt.update(
        {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "branch": branch,
            "commit": commit,
            "mode": args.mode,
            "task_class": args.task_class,
            "task": args.task,
            "context_digest": digest,
            "current_claim_ceiling": context_claim_ceiling(context, state),
            "selected_capability_nodes": [node.get("id") for node in selected],
            "selected_code_paths": code_paths,
        }
    )
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    contract = read_json(root / "templates" / "TASK_CONTRACT.json")
    contract.update(
        {
            "task_id": digest[:16],
            "mode": args.mode,
            "task_class": args.task_class,
            "branch": branch,
            "requested_operation": args.task,
            "current_claim_ceiling": context_claim_ceiling(context, state),
            "selected_capability_nodes": [node.get("id") for node in selected],
            "selected_code_paths": code_paths,
        }
    )
    task_contract_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(packet_path)
    print(receipt_path)
    print(task_contract_path)
    print(f"context_digest={digest}")


if __name__ == "__main__":
    main()
