#!/usr/bin/env python3
"""Align the valid authorization fixture with the frozen 8 kHz V2 runtime."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE.parent.parent / "holo_runtime_v2" / "test_combined_pdn_runner.py"
text = path.read_text(encoding="utf-8")
old = '        "read_hz": 4000,\n'
new = '        "read_hz": 8000,\n'
if new in text:
    print("ROUND2_ALREADY_PRESENT")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("ROUND2_APPLIED")
else:
    raise RuntimeError(f"expected read_hz fixture anchor missing: {path}")
