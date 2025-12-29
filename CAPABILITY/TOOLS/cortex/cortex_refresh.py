#!/usr/bin/env python3

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CORTEX_META_PATH = PROJECT_ROOT / "CORTEX" / "_generated" / "CORTEX_META.json"

# Ensure repo root is importable when running as a file.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_meta() -> Optional[Dict[str, Any]]:
    if not CORTEX_META_PATH.exists():
        return None
    return json.loads(CORTEX_META_PATH.read_text(encoding="utf-8"))

def _git_head_timestamp() -> Optional[str]:
    try:
        res = subprocess.run(
            ["git", "log", "-1", "--format=%cI"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            return None
        value = res.stdout.strip()
        return value or None
    except Exception:
        return None


def _needs_refresh() -> bool:
    try:
        from CAPABILITY.TOOLS.governance.preflight import compute_canon_sha256
    except Exception as exc:
        raise RuntimeError(f"PRECHECK_IMPORT_ERROR: {exc}")

    meta = _read_meta()
    if meta is None:
        return True

    canon_sha = compute_canon_sha256(PROJECT_ROOT)
    recorded = str(meta.get("canon_sha256") or "")
    if not recorded:
        return True
    return recorded != canon_sha


def main() -> int:
    try:
        if not _needs_refresh():
            return 0
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = "0"
        head_ts = _git_head_timestamp()
        if head_ts is not None:
            env["CORTEX_BUILD_TIMESTAMP"] = head_ts
        res = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "CORTEX" / "cortex.build.py")],
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        if res.returncode != 0:
            return int(res.returncode)
        # Ensure metadata exists after refresh.
        _ = _read_meta()
        return 0
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
