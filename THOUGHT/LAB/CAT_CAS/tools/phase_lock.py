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


def select_nodes(
    task: str,
    graph: dict[str, Any],
    limit: int,
    task_class: str,
) -> list[dict[str, Any]]:
    nodes = graph.get("nodes", [])
    by_id = {node.get("id"): node for node in nodes}
    ranked = sorted(nodes, key=lambda item: (node_score(task, item), item.get("id", "")), reverse=True)

    task_lower = task.lower()
    anchor_ids: list[str] = []
    if task_class == "flagship_compute":
        anchor_ids.extend(["HOLO_GEN5", "EXP49", "EXP50", "EXP20_11", "EXP01"])
    if any(term in task_lower for term in ("bounty", "external", "prize", "verifier")):
        anchor_ids.insert(0, "TRACK8")
    if any(term in task_lower for term in ("audio", "torus", "wave", "pi-native")):
        anchor_ids = ["AUDIO_SIDEQUEST", "HOLO_GEN5", "EXP50", "EXP20_11", *anchor_ids]
    if task_class == "external_product":
        anchor_ids.insert(0, "TRACK8")

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for node_id in anchor_ids:
        node = by_id.get(node_id)
        if node is not None and node_id not in seen:
            selected.append(node)
            seen.add(node_id)
    for node in ranked:
        node_id = str(node.get("id"))
        if node_id in seen or node_score(task, node) <= 0:
            continue
        selected.append(node)
        seen.add(node_id)
        if len(selected) >= limit:
            break

    if not selected:
        selected = ranked[: min(limit, len(ranked))]
    return selected[:limit]


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
    elif record.get("requires_branch_state"):
        path = root / str(record.get("branch_state_filename", "BRANCH_STATE.json"))
        context = read_json(path) if path.exists() else {
            "status": "MISSING_REQUIRED_BRANCH_STATE",
            "expected": str(path.relative_to(root)),
        }
    else:
        context = {"status": "REGISTERED_WITHOUT_CONTEXT_SOURCE"}

    context = dict(context)
    context.setdefault("status", "REGISTERED_CONTEXT_LOADED")
    context.setdefault("branch", branch)
    context["registry_record"] = record
    return context


def resolve_branch_commit(root: Path, branch: str, checked_out_branch: str) -> str:
    if branch == checked_out_branch:
        return git_value(root, "rev-parse", "HEAD")
    candidates = (f"origin/{branch}", f"refs/remotes/origin/{branch}", branch)
    for candidate in candidates:
        value = git_value(root, "rev-parse", "--verify", candidate)
        if value != "UNKNOWN":
            return value
    return "UNKNOWN"


def context_claim_ceiling(
    branch_context: dict[str, Any],
    state: dict[str, Any],
    selected: list[dict[str, Any]],
) -> str:
    if branch_context.get("claim_ceiling"):
        return str(branch_context["claim_ceiling"])

    frontier_for_node = {
        "EXP50": "EXP50_SMALL_WALL",
        "AUDIO_SIDEQUEST": "AUDIO_PHASE_TORUS",
        "TRACK8": "TRACK8_EXTERNAL_FRONTIERS",
    }
    selected_frontiers = {
        frontier_for_node[node.get("id")]
        for node in selected
        if node.get("id") in frontier_for_node
    }
    tokens: list[str] = []
    for frontier in state.get("active_frontiers", []):
        if frontier.get("id") in selected_frontiers:
            tokens.extend(frontier.get("claim_tokens", []))
    return " | ".join(dict.fromkeys(tokens)) if tokens else "SEE_CURRENT_STATE"


def build_digest(
    mission: dict[str, Any],
    state: dict[str, Any],
    branch_context: dict[str, Any],
    selected: list[dict[str, Any]],
    task: str,
    mode: str,
    task_class: str,
    branch: str,
    branch_commit: str,
    control_plane_commit: str,
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
                "branch": branch,
                "branch_commit": branch_commit,
                "control_plane_commit": control_plane_commit,
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
    branch_commit: str,
    checked_out_branch: str,
    control_plane_commit: str,
    receipt_display: str,
    contract_display: str,
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
        f"- Target branch: `{branch}`",
        f"- Target branch commit: `{branch_commit}`",
        f"- Control-plane checkout: `{checked_out_branch}`",
        f"- Control-plane commit: `{control_plane_commit}`",
        f"- Implementation checkout matches target: `{checked_out_branch == branch}`",
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
            "Complete both `PHASE_LOCK_RECEIPT.json` and `TASK_CONTRACT.json` before",
            "implementation. A flagship task must identify the classical explosion,",
            "`.holo` process-object, native operator, invariant or fixed point, final",
            "classical boundary, restoration law, no-smuggle killer control, and current",
            "claim ceiling.",
            "",
            "If the target branch differs from the control-plane checkout, this packet",
            "is planning context only. Perform edits in a worktree based on the target",
            "branch commit recorded above.",
            "",
            "Validate the completed pair with:",
            "",
            "```text",
            f"python tools/validate_control_plane.py --receipt {receipt_display} --task-contract {contract_display}",
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

    selected = select_nodes(args.task, graph, args.max_nodes, args.task_class)
    checked_out_branch = git_value(root, "branch", "--show-current")
    control_plane_commit = git_value(root, "rev-parse", "HEAD")
    inferred_branch = selected[0].get("branch") if selected else None
    branch = args.branch or inferred_branch or checked_out_branch
    branch_commit = resolve_branch_commit(root, branch, checked_out_branch)
    if branch_commit == "UNKNOWN":
        raise SystemExit(f"Cannot resolve target branch commit: {branch}")
    context = load_branch_context(root, branch, registry)
    if context.get("status") in {
        "UNREGISTERED_BRANCH",
        "MISSING_CONTEXT_SOURCE",
        "MISSING_REQUIRED_BRANCH_STATE",
    } and args.mode in {"engineering", "verification"}:
        raise SystemExit(f"Branch context blocks {args.mode}: {context.get('status')}")
    digest = build_digest(
        mission,
        state,
        context,
        selected,
        args.task,
        args.mode,
        args.task_class,
        branch,
        branch_commit,
        control_plane_commit,
    )

    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    packet_path = output / "PHASE_LOCK_PACKET.md"
    receipt_path = output / "PHASE_LOCK_RECEIPT.json"
    task_contract_path = output / "TASK_CONTRACT.json"

    def display_path(path: Path) -> str:
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()

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
        branch_commit=branch_commit,
        checked_out_branch=checked_out_branch,
        control_plane_commit=control_plane_commit,
        receipt_display=display_path(receipt_path),
        contract_display=display_path(task_contract_path),
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
            "commit": branch_commit,
            "checked_out_branch": checked_out_branch,
            "control_plane_commit": control_plane_commit,
            "mode": args.mode,
            "task_class": args.task_class,
            "task": args.task,
            "context_digest": digest,
            "current_claim_ceiling": context_claim_ceiling(context, state, selected),
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
            "commit": branch_commit,
            "checked_out_branch": checked_out_branch,
            "control_plane_commit": control_plane_commit,
            "context_digest": digest,
            "requested_operation": args.task,
            "current_claim_ceiling": context_claim_ceiling(context, state, selected),
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
