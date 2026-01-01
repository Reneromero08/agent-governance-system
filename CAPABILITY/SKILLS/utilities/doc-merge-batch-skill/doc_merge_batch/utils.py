from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

@dataclass(frozen=True)
class Normalization:
    newline: str = "lf"                 # "preserve" or "lf"
    strip_trailing_ws: bool = False
    collapse_blank_lines: bool = False

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_bytes(path: Path, max_file_mb: float) -> bytes:
    size = path.stat().st_size
    if size > int(max_file_mb * 1024 * 1024):
        raise ValueError(f"File too large ({size} bytes) for cap {max_file_mb} MB: {path}")
    return path.read_bytes()

def normalize_text(text: str, norm: Normalization) -> str:
    if norm.newline == "lf":
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    if norm.strip_trailing_ws:
        text = "\n".join([line.rstrip() for line in text.splitlines()])
        if text and not text.endswith("\n"):
            text += "\n"
    if norm.collapse_blank_lines:
        out_lines: List[str] = []
        blank = 0
        for line in text.splitlines(True):
            if line.strip() == "":
                blank += 1
                if blank > 1:
                    continue
            else:
                blank = 0
            out_lines.append(line)
        text = "".join(out_lines)
    return text

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def ensure_outdir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

def relpath_safe(path: Path, base: Path | None) -> str:
    try:
        if base:
            return path.resolve().relative_to(base.resolve()).as_posix()
    except Exception:
        pass
    return path.as_posix()


def find_git_root(start: Path) -> Path | None:
    cur = start.resolve()
    if cur.is_file():
        cur = cur.parent
    for _ in range(30):
        if (cur / ".git").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return None

def iso_utc_now() -> str:
    import datetime as _dt
    return _dt.datetime.utcnow().isoformat() + "Z"


def git_stage_and_commit(paths: list[Path], message: str) -> dict:
    # Stages and commits given repo-relative paths. Fail-closed.
    if not paths:
        return {"committed": False, "reason": "no paths to commit"}
    gr = find_git_root(paths[0])
    if gr is None:
        return {"committed": False, "reason": "no git root found"}
    gr = gr.resolve()

    rels: list[str] = []
    for p in paths:
        rp = p.resolve()
        try:
            rel = rp.relative_to(gr).as_posix()
        except Exception:
            return {"committed": False, "reason": f"path outside git repo: {p.as_posix()}"}
        rels.append(rel)

    try:
        r_add = subprocess.run(["git","-C",str(gr),"add","--"] + rels, capture_output=True, text=True)
        if r_add.returncode != 0:
            return {"committed": False, "reason": "git add failed", "stderr": r_add.stderr.strip()}
        r_commit = subprocess.run(["git","-C",str(gr),"commit","-m",message], capture_output=True, text=True)
        if r_commit.returncode != 0:
            # common case: nothing to commit
            return {"committed": False, "reason": "git commit failed", "stderr": r_commit.stderr.strip()}
        # get commit hash
        r_hash = subprocess.run(["git","-C",str(gr),"rev-parse","HEAD"], capture_output=True, text=True)
        ch = r_hash.stdout.strip() if r_hash.returncode == 0 else None
        return {"committed": True, "commit": ch, "stdout": r_commit.stdout.strip()}
    except Exception as e:
        return {"committed": False, "reason": f"exception: {e}"}

def parse_iso_z(ts: str) -> float:
    # returns unix seconds; expects trailing Z or naive iso; best-effort
    try:
        if ts.endswith("Z"):
            ts = ts[:-1]
        dt = __import__("datetime").datetime.fromisoformat(ts)
        return dt.replace(tzinfo=__import__("datetime").timezone.utc).timestamp()
    except Exception:
        return 0.0


def git_file_committed_in_head(path: Path) -> dict:
    gr = find_git_root(path)
    if gr is None:
        return {"ok": False, "reason": "no git root"}
    gr = gr.resolve()
    try:
        rel = path.resolve().relative_to(gr).as_posix()
    except Exception:
        return {"ok": False, "reason": "path outside repo"}
    # exists in HEAD?
    r = subprocess.run(["git","-C",str(gr),"cat-file","-e",f"HEAD:{rel}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0:
        return {"ok": False, "reason": "not in HEAD"}
    # ensure no unstaged changes
    r1 = subprocess.run(["git","-C",str(gr),"diff","--quiet","--",rel], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r1.returncode != 0:
        return {"ok": False, "reason": "working tree modified"}
    # ensure no staged changes
    r2 = subprocess.run(["git","-C",str(gr),"diff","--cached","--quiet","--",rel], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r2.returncode != 0:
        return {"ok": False, "reason": "index modified"}
    return {"ok": True, "reason": "committed+clean"}
