#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SKILL_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from CAPABILITY.TOOLS.agents.skill_runtime import ensure_canon_compat

# Add GuardedWriter for write firewall enforcement
try:
    from CAPABILITY.TOOLS.utilities.guarded_writer import GuardedWriter
    from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation
except ImportError:
    GuardedWriter = None
    FirewallViolation = None


def _is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except Exception:
        return False


def main(input_path: Path, output_path: Path, writer: Optional[GuardedWriter] = None) -> int:
    if not ensure_canon_compat(SKILL_DIR):
        return 1

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[doc-merge-batch-skill] Error reading input JSON: {exc}")
        return 1

    mode = str(payload.get("mode", "verify")).strip()
    pairs = payload.get("pairs", [])
    out_dir_rel = str(payload.get("out_dir", "")).strip()

    if mode not in ("verify", "apply"):
        print("[doc-merge-batch-skill] mode must be 'verify' or 'apply'")
        return 1
    if not isinstance(pairs, list) or not pairs:
        print("[doc-merge-batch-skill] pairs must be a non-empty list")
        return 1
    if not out_dir_rel:
        print("[doc-merge-batch-skill] out_dir is required")
        return 1

    out_dir_abs = (PROJECT_ROOT / out_dir_rel).resolve()
    allowed_roots = [
        (PROJECT_ROOT / "LAW" / "CONTRACTS" / "_runs").resolve(),
        (PROJECT_ROOT / "BUILD").resolve(),
    ]
    if not any(_is_within(out_dir_abs, root) for root in allowed_roots):
        print(f"[doc-merge-batch-skill] Refusing out_dir outside allowed roots: {out_dir_rel}")
        return 1

    # Use GuardedWriter for directory creation if available, otherwise fallback
    # Always use GuardedWriter for writes to enforce firewall
    writer = writer or GuardedWriter(project_root=PROJECT_ROOT)

    try:
        rel_out_dir = str(out_dir_abs.relative_to(PROJECT_ROOT))
        writer.mkdir_auto(rel_out_dir)
    except ValueError:
        print(f"[doc-merge-batch-skill] Output dir {out_dir_abs} outside project root")
        return 1
    
    pairs_path = out_dir_abs / "pairs.json"
    pairs_data = json.dumps(pairs, indent=2, sort_keys=True)
    
    # Use GuardedWriter for pairs.json write if available, otherwise fallback
    # Use GuardedWriter for pairs.json
    try:
        rel_pairs_path = str(pairs_path.relative_to(PROJECT_ROOT))
        writer.write_auto(rel_pairs_path, pairs_data)
    except ValueError:
        print(f"[doc-merge-batch-skill] pairs.json path {pairs_path} outside project root")
        return 1

    env = os.environ.copy()
    cmd = [
        sys.executable,
        "-m",
        "doc_merge_batch",
        "--mode",
        mode,
        "--pairs",
        str(pairs_path),
        "--out",
        str(out_dir_abs),
    ]
    res = subprocess.run(cmd, cwd=str(SKILL_DIR), env=env, capture_output=True, text=True)

    report_path = out_dir_abs / "report.json"
    report_rel = None
    if report_path.exists():
        report_rel = str(report_path.relative_to(PROJECT_ROOT)).replace("\\", "/") # guarded: string op, not filesystem write

    output = {
        "ok": res.returncode == 0,
        "mode": mode,
        "out_dir": out_dir_rel.replace("\\", "/"),
        "pairs": pairs,
        "report_path": report_rel,
        "stderr": res.stderr.strip() if res.stderr else "",
    }

    output_data = json.dumps(output, indent=2, sort_keys=True)
    
    # Use GuardedWriter for final output write if available, otherwise fallback
    # Use GuardedWriter for final output
    try:
        rel_output_path = str(output_path.relative_to(PROJECT_ROOT))
        writer.mkdir_auto(str(Path(rel_output_path).parent))
        writer.write_auto(rel_output_path, output_data)
    except ValueError:
        print(f"[doc-merge-batch-skill] Output path {output_path} outside project root")
        return 1
    
    return 0 if output["ok"] else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: run.py <input.json> <output.json>")
        raise SystemExit(1)
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))

