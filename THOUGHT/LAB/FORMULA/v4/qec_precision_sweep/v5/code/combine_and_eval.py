"""Combine high-shot low-p data with original v2 data and re-run v5 evaluation."""
from __future__ import annotations
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

V2_DIR = ROOT.parent / "v2" / "results"
HIGH_DIR = RESULTS

def combine(v2_name: str, high_name: str, out_name: str, nm: str):
    v2 = json.loads((V2_DIR / v2_name / "qec_precision_sweep_v2.json").read_text(encoding="utf-8"))
    high = json.loads((HIGH_DIR / high_name / "highshot_sweep.json").read_text(encoding="utf-8"))

    low_ps = {0.0005, 0.001, 0.002, 0.004}
    v2_rows = [r for r in v2["conditions"] if float(r["physical_error_rate"]) not in low_ps]
    high_rows = high["conditions"]

    combined = {"config": v2["config"], "conditions": v2_rows + high_rows, "run_id": f"combined_{nm}"}

    out = RESULTS / out_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "combined_sweep.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return str(out)

depol_dir = combine("depol", "v5_highshot_depol", "v5_combined_depol", "depol")
meas_dir = combine("meas", "v5_highshot_meas", "v5_combined_meas", "meas")
print(depol_dir)
print(meas_dir)
