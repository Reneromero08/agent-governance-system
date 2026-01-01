"""doc_merge_batch.core - Deterministic document merge engine

WHAT THIS MODULE DOES:
    Executes pairwise, intentional document merges with deterministic diff → plan → apply → verify workflow.
    Produces auditable receipts and enforces safety checks.

WHAT THIS MODULE DOES NOT DO:
    - Automatically decide which files should be merged
    - Auto-clean directories or manage temp files
    - Treat similarity as proof of mergability

SAFETY REQUIREMENTS:
    1. ALWAYS use verify mode before any destructive action
    2. ALWAYS use explicit pairs.json (never implicit discovery)
    3. ALWAYS write outputs to a dedicated merge directory (NOT repo root)
    4. NEVER delete files without receipts
    5. NEVER skip verification before applying post-actions

MENTAL MODEL:
    - This is a tool for INTENTIONAL merges only
    - Similarity ≠ Identity
    - Humans (or orchestration) decide what to merge
    - This module executes those decisions safely

CORE INVARIANTS:
    - Determinism: same inputs → same outputs
    - Fidelity: merged output contains union of content blocks
    - Boundedness: file size, pair count, diff length caps enforced
    - Audit trail: every merge produces a receipt

POST-ACTION SAFETY:
    - delete_tracked ONLY deletes git-tracked, committed, clean files
    - quarantine is preferred over deletion
    - --allow-uncommitted bypasses safety (DATA LOSS RISK)

See README.md for full usage documentation and safety rules.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import Normalization, ensure_outdir, read_bytes, sha256_bytes, normalize_text, sha256_text, relpath_safe, find_git_root, iso_utc_now, git_stage_and_commit, parse_iso_z, git_file_committed_in_head, find_git_root, iso_utc_now

def _is_git_tracked(path: Path) -> bool:
    gr = find_git_root(path)
    if gr is None:
        return False
    try:
        rel = path.resolve().relative_to(gr.resolve()).as_posix()
    except Exception:
        return False
    try:
        r = subprocess.run(["git","-C",str(gr),"ls-files","--error-unmatch",rel],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return r.returncode == 0
    except Exception:
        return False

def _post_action(on_success: str, ttl_days: int, require_committed: bool, a_path: Path, b_path: Path, out_dir: Path, merged_path: Path | None = None) -> dict:
    res = {"on_success": on_success, "ttl_days": ttl_days, "actions": []}
    if on_success == "none":
        return res

    quarantine_root = out_dir / "quarantine"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    stamp = __import__("datetime").datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    q_dir = quarantine_root / stamp
    q_dir.mkdir(parents=True, exist_ok=True)

    def quarantine_one(p: Path) -> str:
        dest = q_dir / p.name
        if dest.exists():
            i = 1
            while True:
                cand = q_dir / f"{p.stem}.{i}{p.suffix}"
                if not cand.exists():
                    dest = cand
                    break
                i += 1
        shutil.move(p.as_posix(), dest.as_posix())
        return dest.as_posix()

    idx_path = quarantine_root / "quarantine_index.jsonl"
    expire_at = (__import__("datetime").datetime.utcnow() + __import__("datetime").timedelta(days=int(ttl_days))).isoformat() + "Z"

    for p in (a_path, b_path):
        if not p.exists():
            continue
        tracked = _is_git_tracked(p)
        if on_success == "delete_tracked":
            if tracked:
                if require_committed:
                    st = git_file_committed_in_head(p)
                    if not st.get("ok"):
                        res["actions"].append({"path": p.as_posix(), "action": "refused", "git_tracked": True, "reason": st.get("reason")})
                    else:
                        os.remove(p)
                        res["actions"].append({"path": p.as_posix(), "action": "deleted", "git_tracked": True})
                else:
                    os.remove(p)
                    res["actions"].append({"path": p.as_posix(), "action": "deleted", "git_tracked": True, "note": "require_committed=false"})
            else:
                res["actions"].append({"path": p.as_posix(), "action": "skipped", "git_tracked": False})
        elif on_success == "quarantine":
            moved_to = quarantine_one(p)
            entry = {"ts": iso_utc_now(), "src": p.as_posix(), "dst": moved_to, "expire_at": expire_at, "git_tracked": tracked}
            with idx_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
            res["actions"].append({"path": p.as_posix(), "action": "quarantined", "moved_to": moved_to, "git_tracked": tracked, "expire_at": expire_at})

    # Copy merged file back to original location (use a_path as target) AFTER originals are moved/deleted
    if merged_path and merged_path.exists():
        shutil.copy2(merged_path.as_posix(), a_path.as_posix())
        res["actions"].append({"path": a_path.as_posix(), "action": "restored_from_merged", "source": merged_path.as_posix()})

    return res

from .diffing import similarity_ratio, unique_lines, diff_summary
from .planner import plan_append_unique_blocks, apply_append_unique_blocks, MergePlan

def _load_pairs(pairs_path: Path) -> List[Dict[str,str]]:
    data = json.loads(pairs_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("pairs json must be a list")
    out: List[Dict[str,str]] = []
    for item in data:
        if not isinstance(item, dict) or "a" not in item or "b" not in item:
            raise ValueError("each pair must be an object with keys 'a' and 'b'")
        out.append({"a": str(item["a"]), "b": str(item["b"])})
    return out

def _scan_root(root: Path, max_file_mb: float) -> List[Path]:
    # Deterministic walk order
    paths = [p for p in root.rglob("*") if p.is_file()]
    paths.sort(key=lambda p: p.as_posix())
    # size filter applied in read_bytes; here we just return
    return paths


def prune_quarantine(out_dir: Path) -> dict:
    qroot = out_dir / "quarantine"
    idx = qroot / "quarantine_index.jsonl"
    if not idx.exists():
        return {"pruned": 0, "kept": 0, "missing": 0, "index": idx.as_posix(), "note": "no index"}
    now = __import__("time").time()
    pruned = 0
    kept = 0
    missing = 0
    kept_lines = []
    for line in idx.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        exp = parse_iso_z(str(rec.get("expire_at","")))
        dst = Path(str(rec.get("dst","")))
        if exp and exp <= now:
            if dst.exists():
                try:
                    dst.unlink()
                    pruned += 1
                except Exception:
                    # if can't delete, keep record
                    kept += 1
                    kept_lines.append(line)
            else:
                missing += 1
            # do not keep expired records
        else:
            kept += 1
            kept_lines.append(line)
    idx.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""), encoding="utf-8")
    return {"pruned": pruned, "kept": kept, "missing": missing, "index": idx.as_posix()}

def scan(root: Path, norm: Normalization, max_file_mb: float, max_pairs: int) -> Dict[str, Any]:
    paths = _scan_root(root, max_file_mb)
    by_bytes: Dict[str, List[str]] = {}
    by_text: Dict[str, List[str]] = {}
    errors: List[str] = []

    for p in paths:
        try:
            b = read_bytes(p, max_file_mb)
            hb = sha256_bytes(b)
            by_bytes.setdefault(hb, []).append(p.as_posix())
            try:
                t = b.decode("utf-8")
            except UnicodeDecodeError:
                continue
            nt = normalize_text(t, norm)
            ht = sha256_text(nt)
            by_text.setdefault(ht, []).append(p.as_posix())
        except Exception as e:
            errors.append(f"{p}: {e}")

    exact_duplicates = [v for v in by_bytes.values() if len(v) > 1]
    near_duplicates = [v for v in by_text.values() if len(v) > 1]

    # cap candidate pair explosion by slicing deterministically
    def to_pairs(groups: List[List[str]]) -> List[Dict[str,str]]:
        pairs: List[Dict[str,str]] = []
        for g in groups:
            g_sorted = sorted(g)
            for i in range(len(g_sorted)):
                for j in range(i+1, len(g_sorted)):
                    pairs.append({"a": g_sorted[i], "b": g_sorted[j]})
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    return {
        "exact_duplicates": exact_duplicates,
        "near_duplicates": near_duplicates,
        "near_duplicate_candidate_pairs": to_pairs(near_duplicates),
        "errors": errors,
    }

def compare_pair(a_path: Path, b_path: Path, norm: Normalization, max_file_mb: float, context_lines: int, max_diff_lines: int, base: Path | None = None) -> Dict[str, Any]:
    a_bytes = read_bytes(a_path, max_file_mb)
    b_bytes = read_bytes(b_path, max_file_mb)

    a_hash = sha256_bytes(a_bytes)
    b_hash = sha256_bytes(b_bytes)
    identical_bytes = (a_hash == b_hash)

    a_text = None
    b_text = None
    try:
        a_text = a_bytes.decode("utf-8")
        b_text = b_bytes.decode("utf-8")
    except UnicodeDecodeError:
        # binary: only report byte equality
        return {
            "file_a": relpath_safe(a_path, base),
            "file_b": relpath_safe(b_path, base),
            "identical": identical_bytes,
            "similarity": 1.0 if identical_bytes else 0.0,
            "unique_to_a": [],
            "unique_to_b": [],
            "diff_summary": "(binary file; diff skipped)",
            "hashes": {"a_bytes": a_hash, "b_bytes": b_hash},
        }

    na = normalize_text(a_text, norm)
    nb = normalize_text(b_text, norm)
    a_lines = na.splitlines(True)
    b_lines = nb.splitlines(True)

    sim = similarity_ratio(a_lines, b_lines)
    u_a, u_b = unique_lines(a_lines, b_lines)
    ds = diff_summary(a_lines, b_lines, context_lines=context_lines, max_diff_lines=max_diff_lines)

    return {
        "file_a": relpath_safe(a_path, base),
        "file_b": relpath_safe(b_path, base),
        "identical": identical_bytes,
        "similarity": float(sim),
        "unique_to_a": u_a,
        "unique_to_b": u_b,
        "diff_summary": ds,
        "hashes": {
            "a_bytes": a_hash,
            "b_bytes": b_hash,
            "a_text_norm": sha256_text(na),
            "b_text_norm": sha256_text(nb),
        }
    }

def plan_pair(a_path: Path, b_path: Path, norm: Normalization, max_file_mb: float, base_side: str) -> Dict[str, Any]:
    a_bytes = read_bytes(a_path, max_file_mb)
    b_bytes = read_bytes(b_path, max_file_mb)
    a_text = a_bytes.decode("utf-8", errors="replace")
    b_text = b_bytes.decode("utf-8", errors="replace")
    na = normalize_text(a_text, norm)
    nb = normalize_text(b_text, norm)

    plan = plan_append_unique_blocks(na, nb, base=base_side)
    return {"a": a_path.as_posix(), "b": b_path.as_posix(), "plan": asdict(plan)}

def apply_pair(a_path: Path, b_path: Path, plan: MergePlan, norm: Normalization, max_file_mb: float, out_dir: Path, post_actions: dict | None = None) -> Dict[str, Any]:
    a_bytes = read_bytes(a_path, max_file_mb)
    b_bytes = read_bytes(b_path, max_file_mb)
    a_text = a_bytes.decode("utf-8", errors="replace")
    b_text = b_bytes.decode("utf-8", errors="replace")
    na = normalize_text(a_text, norm)
    nb = normalize_text(b_text, norm)

    merged = apply_append_unique_blocks(na, nb, plan)

    merged_dir = out_dir / "merged"
    receipts_dir = out_dir / "receipts"
    ensure_outdir(merged_dir)
    ensure_outdir(receipts_dir)

    # deterministic output filename based on source hashes
    a_h = sha256_bytes(a_bytes)[:12]
    b_h = sha256_bytes(b_bytes)[:12]
    merged_name = f"merged_{a_h}_{b_h}.md"
    merged_path = merged_dir / merged_name
    merged_path.write_text(merged, encoding="utf-8")

    receipt = {
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "a_path": a_path.as_posix(),
        "b_path": b_path.as_posix(),
        "a_bytes_hash": sha256_bytes(a_bytes),
        "b_bytes_hash": sha256_bytes(b_bytes),
        "merged_bytes_hash": sha256_bytes(merged.encode("utf-8")),
        "merged_name": merged_name,
        "plan": asdict(plan),
    }

    # Append to cumulative receipt file instead of creating individual files
    receipt_path = receipts_dir / "merge_receipt.jsonl"
    with receipt_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(receipt) + "\n")

    post = None
    if post_actions:
        post = _post_action(post_actions.get("on_success","none"), int(post_actions.get("ttl_days",14)), bool(post_actions.get("require_committed", True)), a_path, b_path, out_dir, merged_path)

    return {"merged_path": merged_path.as_posix(), "receipt_path": receipt_path.as_posix(), "receipt": receipt, "post_actions": post}

def verify_pair(a_path: Path, b_path: Path, merged_text: str, norm: Normalization, max_file_mb: float) -> Dict[str, Any]:
    # Verify block-union completeness under the planner's block fingerprinting.
    from .blocks import split_blocks, block_fingerprint

    a_bytes = read_bytes(a_path, max_file_mb)
    b_bytes = read_bytes(b_path, max_file_mb)
    a_text = normalize_text(a_bytes.decode("utf-8", errors="replace"), norm)
    b_text = normalize_text(b_bytes.decode("utf-8", errors="replace"), norm)
    m_text = normalize_text(merged_text, norm)

    a_set = {block_fingerprint(bl) for bl in split_blocks(a_text) if block_fingerprint(bl) != "BLANK"}
    b_set = {block_fingerprint(bl) for bl in split_blocks(b_text) if block_fingerprint(bl) != "BLANK"}
    m_set = {block_fingerprint(bl) for bl in split_blocks(m_text) if block_fingerprint(bl) != "BLANK"}

    missing_from_merged = sorted(list((a_set | b_set) - m_set))
    return {
        "pass": len(missing_from_merged) == 0,
        "missing_block_fingerprints": missing_from_merged[:50],  # cap
        "missing_count": len(missing_from_merged),
    }

def run_job(payload: Dict[str, Any]) -> Dict[str, Any]:
    mode = payload["mode"]
    out_dir = Path(payload.get("out_dir", "./MERGE_OUT"))
    ensure_outdir(out_dir)

    norm_cfg = payload.get("normalization", {}) or {}
    norm = Normalization(
        newline=norm_cfg.get("newline", "lf"),
        strip_trailing_ws=bool(norm_cfg.get("strip_trailing_ws", False)),
        collapse_blank_lines=bool(norm_cfg.get("collapse_blank_lines", False)),
    )

    max_file_mb = float(payload.get("max_file_mb", 20))
    max_pairs = int(payload.get("max_pairs", 5000))

    diff_cfg = payload.get("diff", {}) or {}
    max_diff_lines = int(diff_cfg.get("max_diff_lines", 500))
    context_lines = int(diff_cfg.get("context_lines", 3))

    merge_cfg = payload.get("merge", {}) or {}
    post_cfg = payload.get("post_actions", {}) or {}
    post_actions = {"on_success": post_cfg.get("on_success","none"), "ttl_days": int(post_cfg.get("ttl_days",14)), "require_committed": bool(post_cfg.get("require_committed", True))}
    git_cfg = payload.get("git_commit", {}) or {}
    git_commit = {"enabled": bool(git_cfg.get("enabled", False)), "message": str(git_cfg.get("message", "housekeeping: merge+prune originals"))}
    base_side = merge_cfg.get("base", "a")

    errors: List[str] = []
    comparisons: List[Dict[str, Any]] = []
    plans: List[Dict[str, Any]] = []
    artifacts: List[Dict[str, Any]] = []
    receipts: List[Dict[str, Any]] = []

    if mode == "prune_quarantine":
        rep = {
            "mode": mode,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "post_actions": post_actions,
        "git_commit": git_commit,
            "scan": None,
            "comparisons": [],
            "plans": [],
            "artifacts": [],
            "receipts": [],
            "errors": [],
            "prune_report": prune_quarantine(out_dir),
        }
        return rep

    if mode == "scan":
        root = payload.get("root")
        if not root:
            raise ValueError("scan mode requires 'root'")
        scan_out = scan(Path(root), norm=norm, max_file_mb=max_file_mb, max_pairs=max_pairs)
        errors.extend(scan_out.get("errors", []))
        report = {
            "mode": mode,
            "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "scan": {k: scan_out[k] for k in ("exact_duplicates","near_duplicates","near_duplicate_candidate_pairs")},
            "comparisons": [],
            "plans": [],
            "artifacts": [],
            "receipts": [],
            "errors": errors,
        }
        return report

    # Resolve pairs
    pairs: List[Dict[str,str]] = []
    if "pairs" in payload and isinstance(payload["pairs"], list):
        pairs = [{"a": p["a"], "b": p["b"]} for p in payload["pairs"]]
    elif "pairs_path" in payload:
        pairs = _load_pairs(Path(payload["pairs_path"]))
    elif "files" in payload and isinstance(payload["files"], list) and len(payload["files"]) == 2:
        pairs = [{"a": payload["files"][0], "b": payload["files"][1]}]
    else:
        raise ValueError("provide 'pairs' (list), or 'pairs_path', or 'files' of length 2")

    if len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]

    for p in pairs:
        try:
            a = Path(p["a"])
            b = Path(p["b"])
            if mode in ("compare","plan","apply","verify"):
                comp = compare_pair(a, b, norm=norm, max_file_mb=max_file_mb, context_lines=context_lines, max_diff_lines=max_diff_lines, base=None)
                comparisons.append(comp)

            if mode in ("plan","apply","verify"):
                plan_obj = plan_pair(a, b, norm=norm, max_file_mb=max_file_mb, base_side=base_side)
                plans.append(plan_obj)
                plan = MergePlan(**plan_obj["plan"])

            if mode in ("apply","verify"):
                applied = apply_pair(a, b, plan=plan, norm=norm, max_file_mb=max_file_mb, out_dir=out_dir, post_actions=(post_actions if mode=="apply" else None))
                art = {"merged_path": applied["merged_path"], "receipt_path": applied["receipt_path"]}
                if applied.get("post_actions") is not None:
                    art["post_actions"] = applied["post_actions"]
                    if git_commit.get("enabled") and post_actions.get("on_success") == "delete_tracked":
                        paths_to_commit = [Path(art["merged_path"])]
                        for act in (applied.get("post_actions") or {}).get("actions", []):
                            if act.get("action") == "deleted" and act.get("git_tracked"):
                                paths_to_commit.append(Path(act.get("path")))
                        art["git_commit_result"] = git_stage_and_commit(paths_to_commit, git_commit.get("message","housekeeping: merge+prune originals"))
                artifacts.append(art)
                receipts.append(applied["receipt"])

            if mode == "verify":
                merged_path = Path(artifacts[-1]["merged_path"])
                merged_text = merged_path.read_text(encoding="utf-8")
                v = verify_pair(a, b, merged_text=merged_text, norm=norm, max_file_mb=max_file_mb)
                artifacts[-1]["verification"] = v
                if v.get("pass") and post_actions.get("on_success","none") != "none":
                    post = _post_action(post_actions.get("on_success","none"), int(post_actions.get("ttl_days",14)), bool(post_actions.get("require_committed", True)), a, b, out_dir, merged_path)
                    artifacts[-1]["post_actions"] = post
                    if git_commit.get("enabled") and post_actions.get("on_success") == "delete_tracked":
                        paths_to_commit = []
                        # merged output might be inside repo; stage if so
                        mp = Path(artifacts[-1]["merged_path"])
                        paths_to_commit.append(mp)
                        # include originals (deleted) if they were git-tracked
                        for act in post.get("actions", []):
                            if act.get("action") == "deleted" and act.get("git_tracked"):
                                paths_to_commit.append(Path(act.get("path")))
                        artifacts[-1]["git_commit_result"] = git_stage_and_commit(paths_to_commit, git_commit.get("message","housekeeping: merge+prune originals"))

        except Exception as e:
            errors.append(f"{p}: {e}")

    report = {
        "mode": mode,
        "generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "post_actions": post_actions,
        "git_commit": git_commit,
        "scan": None,
        "comparisons": comparisons,
        "plans": plans,
        "artifacts": artifacts,
        "receipts": receipts,
        "errors": errors,
    }
    return report
