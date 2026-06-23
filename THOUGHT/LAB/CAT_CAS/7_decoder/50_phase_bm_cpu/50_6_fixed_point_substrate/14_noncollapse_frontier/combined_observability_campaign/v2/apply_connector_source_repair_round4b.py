#!/usr/bin/env python3
"""Point the frozen threshold identity test at the shared C contract header."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
path = HERE / "test_calibration_contract.py"
text = path.read_text(encoding="utf-8")
old = '''        source = (Path(__file__).resolve().parents[2] / "holo_runtime_v2" /
                  "combined_pdn_hardware.c").read_text(encoding="utf-8")
'''
new = '''        source = (Path(__file__).resolve().parents[2] / "holo_runtime_v2" /
                  "capture_quality_contract.h").read_text(encoding="utf-8")
'''
if new in text:
    print("ROUND4B_ALREADY_PRESENT")
elif old in text:
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("ROUND4B_APPLIED")
else:
    raise RuntimeError(f"expected threshold identity anchor missing: {path}")
