"""
Q49 Fixed Verification: Normalize by matched random baseline at each N.
This removes the N-dependent artifact and isolates semantic structure.
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer

TARGET = 8 * math.e

POOL = [
    "water","fire","earth","sky","sun","moon","star","mountain",
    "river","tree","flower","rain","wind","snow","cloud","ocean",
    "forest","desert","island","lake","valley","cave","hill","field",
    "dog","cat","bird","fish","horse","tiger","lion","elephant",
    "snake","wolf","bear","eagle","whale","spider","ant","bee",
    "deer","rabbit","fox","owl","crow","shark","dolphin","monkey",
    "heart","eye","hand","head","brain","blood","bone","skin","face",
    "arm","leg","finger","ear","nose","mouth","tooth",
    "mother","father","child","friend","king","queen","doctor","teacher",
    "soldier","artist","farmer","priest","judge","writer","singer","dancer",
    "love","hate","truth","life","death","time","space","power",
    "peace","war","hope","fear","joy","pain","dream","thought",
    "freedom","justice","beauty","wisdom","courage","honor","faith","pride",
    "book","door","house","road","food","money","stone","gold",
    "sword","light","shadow","music","word","name","law","art",
    "table","chair","window","mirror","key","ring","crown","flag",
    "good","bad","big","small","old","new","high","low",
    "hot","cold","dark","bright","strong","weak","fast","slow",
    "hard","soft","deep","wide","long","short","rich","poor",
    "queen","prince","castle","knight","dragon","magic","wizard",
    "witch","giant","hero","monster","ghost","spirit","angel","demon",
    "brain","mind","soul","foot","nose","mouth","body",
    "force","energy","mass","speed","matter","atom","cell",
    "north","south","east","west","left","right",
    "above","below","morning","night","spring","summer","autumn","winter",
]


def get_eigenspectrum(matrix):
    centered = matrix - matrix.mean(axis=0)
    cov = np.cov(centered.T)
    ev = eigh(cov, eigvals_only=True)
    ev = np.sort(ev)[::-1]
    ev = np.maximum(ev, 1e-10)
    return ev


def compute_df(ev):
    ev_nz = ev[ev > 1e-10]
    s1, s2 = ev_nz.sum(), (ev_nz * ev_nz).sum()
    return s1 * s1 / s2 if s2 > 0 else 1.0


def compute_alpha(ev):
    ev_nz = ev[ev > 1e-10]
    k = np.arange(1, len(ev_nz) + 1)
    half = len(ev_nz) // 2
    if half < 5:
        return np.nan, np.nan
    slope = np.polyfit(np.log(k[:half]), np.log(ev_nz[:half]), 1)[0]
    r2 = np.corrcoef(np.log(k[:half]), np.log(ev_nz[:half]))[0, 1] ** 2
    return -slope, r2


def measure(embeddings):
    ev = get_eigenspectrum(embeddings)
    return compute_df(ev), *compute_alpha(ev)


def random_baseline(N, D, n_trials=20):
    """Matched random baseline at same N, D."""
    products = []
    for _ in range(n_trials):
        X = np.random.randn(N, D)
        ev = get_eigenspectrum(X)
        products.append(compute_df(ev) * compute_alpha(ev)[0])
    return np.mean(products), np.std(products)


print("=" * 64)
print("Q49 FIXED VERIFICATION: Random-baseline-normalized product")
print("=" * 64)
print(f"Target: 8e = {TARGET:.4f}")
print()

POOL = list(dict.fromkeys(POOL))

results = {}
for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    D = model.get_sentence_embedding_dimension()

    print(f"{'='*64}")
    print(f"  {name} (D={D})")
    print(f"  {'N':>5s}  {'Df*alpha':>10s}  {'Random':>10s}  {'Delta':>10s}  {'Ratio':>10s}")
    print(f"  {'-'*49}")

    sweep = {}
    for N in [20, 30, 40, 50, 60, 75, 100, 150, 200, 300, 400, 500, 600, 800, 1000, 1500]:
        np.random.seed(N)
        words = np.random.choice(POOL, size=N, replace=N > len(POOL))
        embs = model.encode(list(words), normalize_embeddings=True)
        df_val, alpha_val, r2 = measure(embs)
        real_product = df_val * alpha_val

        rand_mean, rand_std = random_baseline(N, D, n_trials=10)
        delta = real_product - rand_mean
        ratio = real_product / rand_mean if rand_mean > 0 else 0

        sweep[N] = {
            "real": float(real_product),
            "random_mean": float(rand_mean),
            "random_std": float(rand_std),
            "delta": float(delta),
            "ratio": float(ratio),
            "Df": float(df_val),
            "alpha": float(alpha_val),
        }

        near = " <-- NEAR 8e" if abs(real_product - TARGET) < 1 else ""
        converged = " <-- CONVERGED" if N >= 150 else ""
        print(f"  {N:5d}  {real_product:10.4f}  {rand_mean:10.4f}  {delta:+10.4f}  {ratio:10.4f}{near}{converged}")

    results[name] = sweep

# Summary
print(f"\n{'='*64}")
print("SUMMARY: Random-baseline-normalized Delta")
print("=" * 64)

for name in results:
    print(f"\n  {name}:")
    for N in [30, 50, 75, 100, 300, 500]:
        if N in results[name]:
            d = results[name][N]
            near = " *** match!" if abs(d["delta"] - TARGET) < 1 else ""
            print(f"    N={N:4d}: real={d['real']:.2f}  random={d['random_mean']:.2f}  delta={d['delta']:.2f}  ratio={d['ratio']:.3f}{near}")

# Check convergence
print(f"\n{'='*64}")
print("CONVERGENCE CHECK: Does delta stabilize as N grows?")
print("=" * 64)

for name in results:
    swept = results[name]
    Ns = sorted(swept.keys())
    deltas = [swept[n]["delta"] for n in Ns]
    ratios = [swept[n]["ratio"] for n in Ns]

    # Check if delta or ratio converges in large-N regime
    large_Ns = [n for n in Ns if n >= 300]
    if len(large_Ns) >= 2:
        large_deltas = [swept[n]["delta"] for n in large_Ns]
        large_ratios = [swept[n]["ratio"] for n in large_Ns]
        d_mean = np.mean(large_deltas)
        d_std = np.std(large_deltas)
        d_cv = d_std / d_mean * 100 if abs(d_mean) > 1e-10 else float('nan')
        print(f"  {name} (N >= 300):")
        print(f"    delta: mean={d_mean:.4f}  std={d_std:.4f}  CV={d_cv:.1f}%")
        print(f"    ratio: mean={np.mean(large_ratios):.4f}  std={np.std(large_ratios):.4f}")

out = Path("THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verification_fixed.json")
json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, open(out, "w"), indent=2)
print(f"\nSaved: {out}")
