"""Phase 4b Real Model Runner — launches experiment with TraDo-4B."""
import sys, time, json, os
os.environ["PYTHONIOENCODING"] = "utf-8"

sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE")

from phase4b_runner import run_full_experiment
from model import get_model, RealTraDoGenerator

print("=" * 60, flush=True)
print("Phase 4b: Real TraDo-4B Experiment", flush=True)
print("=" * 60, flush=True)

t0 = time.time()
trado = get_model(quantize="q4")
gen = RealTraDoGenerator(trado)
print(f"Model loaded in {time.time()-t0:.1f}s", flush=True)

conditions = ["CONTROL", "VALUES_LATTICE", "EPISTEMIC_LATTICE", "EPISTEMIC_NO_COMMONSENSE"]
print(f"Conditions: {conditions}", flush=True)
print(f"Estimated runtime: ~70 minutes (4 conditions + calibration)", flush=True)

output = run_full_experiment(
    model=gen, mock=False, failing=False,
    conditions=conditions, verbose=True,
)

total_time = time.time() - t0
print(f"\nTotal experiment time: {total_time/60:.1f} minutes", flush=True)

# Print per-condition summary
for cond, metrics in output["metrics"].items():
    print(f"\n  {cond}: accuracy={metrics['accuracy']:.4f} "
          f"({metrics['n_correct']}/{metrics['n_verified']}) "
          f"hard_gates={metrics['total_hard_gates']} "
          f"recovery={metrics['recovery_rate']:.4f} "
          f"grad_S={metrics['grad_S_mean']:.4f} "
          f"R={metrics['R_mean']:.2f}", flush=True)

print("\nDone.", flush=True)
