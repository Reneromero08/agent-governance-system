#!/usr/bin/env python3
"""Create a completed ZIP after downloads and repository-context import."""
from __future__ import annotations
import hashlib, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / (ROOT.name + "_complete.zip")
with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS.txt"}:
            z.write(path, arcname=str(Path(ROOT.name) / path.relative_to(ROOT)))
h = hashlib.sha256(OUT.read_bytes()).hexdigest()
OUT.with_suffix(OUT.suffix + ".sha256").write_text(f"{h}  {OUT.name}\n", encoding="utf-8")
print(OUT)
print(h)
