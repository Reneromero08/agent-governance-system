"""Lissajous Scout: correlated noise crosstalk test on rotated surface code d=3."""
import json, math, numpy as np, stim
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[1]
CORRELATED_ROOT = ROOT / "correlated"
OUT = CORRELATED_ROOT

def t(d): return (d-1)//2

def build_crosstalk_circuit(d, p, p_crosstalk, basis="x"):
    """Build rotated surface code circuit with crosstalk after CX gates."""
    task = f"surface_code:rotated_memory_{basis}"
    base = stim.Circuit.generated(task, distance=d, rounds=d,
        after_clifford_depolarization=p, after_reset_flip_probability=p,
        before_measure_flip_probability=p, before_round_data_depolarization=p)
    
    new = stim.Circuit()
    for inst in base.flattened():
        new.append(inst)
        if inst.name == "CX":
            qs = [t.value for t in inst.targets_copy()]
            for i in range(0, len(qs), 2):
                if i + 1 < len(qs):
                    new.append("CORRELATED_ERROR", [stim.target_x(qs[i]), stim.target_x(qs[i+1])], p_crosstalk)
    return new

def build_correlation_matrix(circuit):
    """Build correlation matrix from undecomposed DEM."""
    dem = circuit.detector_error_model(decompose_errors=False)
    # Count actual detectors from DEM
    detectors = set()
    for inst in dem.flattened():
        if inst.type == "error":
            for t in inst.targets_copy():
                if not t.is_separator:
                    try:
                        detectors.add(t.val)
                    except ValueError:
                        raise ValueError(f"Invalid detector target value: t={t} in DEM instruction")
    n = len(detectors)
    if n == 0: return None
    
    # Map detector IDs to matrix indices
    det_list = sorted(detectors)
    det_idx = {d: i for i, d in enumerate(det_list)}
    
    M = np.zeros((n, n))
    for inst in dem.flattened():
        if inst.type != "error": continue
        dets = []
        for t in inst.targets_copy():
            if t.is_separator: continue
            if t.is_logical_observable_id: continue
            dets.append(t.val)
        if len(dets) < 2: continue  # skip independent errors
        args = inst.args_copy()
        if not args: continue
        prob = args[0]
        for i in dets:
            if i in det_idx:
                M[det_idx[i], det_idx[i]] += prob
                for j in dets:
                    if i != j and j in det_idx:
                        M[det_idx[i], det_idx[j]] += prob
    
    M = (M + M.T) / 2
    return M

def analyze_spectrum(M, label):
    """Compute eigenvalue spectrum and report structure."""
    if M.shape[0] < 2:
        return {"n_detectors": M.shape[0], "coupling_ratio": 0, "significant_modes": 0,
                "top5": [0]*5, "max_eigenval": 0, "sum_eigenvals": 0, "label": label}
    
    eigenvals = np.linalg.eigvalsh(M)
    eigenvals = np.sort(np.abs(eigenvals))[::-1]
    
    diag_sum = np.sum(np.abs(np.diag(M)))
    off_sum = np.sum(np.abs(M)) - diag_sum
    coupling_ratio = off_sum / max(diag_sum, 1e-12)
    
    top5 = eigenvals[:min(5, len(eigenvals))]
    max_ev = top5[0] if len(top5) > 0 else 0
    
    significant = len([e for e in eigenvals if e > 0.01 * max(max_ev, 1e-12)])
    
    return {
        "n_detectors": M.shape[0],
        "coupling_ratio": float(coupling_ratio),
        "significant_modes": significant,
        "top5": [float(e) for e in top5] + [0]*(5-len(top5)),
        "max_eigenval": float(max_ev),
        "label": label
    }

print("LISSAJOUS SCOUT: Correlated Noise Test")
print("=" * 60)

ps = [0.001, 0.004, 0.010]
d = 3

for p_crosstalk_name, p_crosstalk_factor in [("10%", 0.1), ("20%", 0.2)]:
    print(f"\nCrosstalk level: {p_crosstalk_name} of physical error rate")
    print(f"{'p':>8s}  {'n_det':>6s}  {'coupling':>10s}  {'modes':>6s}  {'top3_eigenvals':>40s}")
    
    results = []
    for p in ps:
        p_xtalk = p * p_crosstalk_factor
        circuit = build_crosstalk_circuit(d, p, p_xtalk)
        
        # Save circuit
        circ_path = OUT / "circuits" / f"d{d}_p{p:.4f}_xtalk{p_crosstalk_name}.stim"
        try:
            circ_path.parent.mkdir(parents=True, exist_ok=True)
            circ_path.write_text(str(circuit))
        except OSError as e:
            print(f"ERROR: Failed to write circuit to {circ_path}: {e}", flush=True)
            raise
        
        # Correlation matrix
        M = build_correlation_matrix(circuit)
        if M is None:
            print(f"{p:8.4f}  NO DEM")
            continue
        
        spectrum = analyze_spectrum(M, f"d={d}, p={p}, xtalk={p_crosstalk_name}")
        results.append(spectrum)
        
        top3 = ", ".join([f"{v:.6f}" for v in spectrum["top5"][:3]])
        print(f"{p:8.4f}  {spectrum['n_detectors']:6d}  {spectrum['coupling_ratio']:10.6f}  {spectrum['significant_modes']:6d}  [{top3}]")
    
    if results:
        coupling_ratios = [r["coupling_ratio"] for r in results]
        modes = [r["significant_modes"] for r in results]
        print(f"  Coupling ratio range: [{min(coupling_ratios):.6f}, {max(coupling_ratios):.6f}]")
        print(f"  Significant modes range: [{min(modes)}, {max(modes)}]")

# Verdict  
print(f"\n{'='*60}")
print("VERDICT")

has_coupling = False
for p_xtalk in [0.1, 0.2]:
    for p in ps:
        circuit = build_crosstalk_circuit(d, p, p * p_xtalk)
        M = build_correlation_matrix(circuit)
        if M is not None and M.shape[0] > 0:
            diag = np.sum(np.abs(np.diag(M)))
            off = np.sum(np.abs(M)) - diag
            cr = off / max(diag, 1e-12)
            ev = np.sort(np.abs(np.linalg.eigvalsh(M)))[::-1]
            sig = len([e for e in ev if e > 0.01 * max(np.max(ev), 1e-12)])
            print(f"  p={p:.4f} xtalk={p_xtalk*100:.0f}% n_det={M.shape[0]} coupling_ratio={cr:.6f} significant_modes={sig} top3={[round(e,6) for e in ev[:3]]}")
            if cr > 0.001:
                has_coupling = True

if has_coupling:
    print("\n  COUPLING DETECTED: Non-zero off-diagonal entries in correlation matrix.")
    print("  The Lissajous hypothesis is TESTABLE with correlated noise.")
else:
    print("\n  NULL: No off-diagonal coupling in the correlation matrix.")
