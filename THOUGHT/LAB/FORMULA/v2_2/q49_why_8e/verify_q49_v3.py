"""
Q49 Fresh Verification v3 - Cached models only, MDS distance method.
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer

E8 = 8 * math.e
SP = 7 * math.pi
T2 = 22.0

WORDS = [
    "the","of","and","to","in","is","that","for","it","with",
    "on","are","be","this","have","from","or","one","had","by",
    "word","but","not","what","all","were","we","when","your","can",
    "said","there","use","each","which","she","how","their","will",
    "other","about","out","many","then","them","these","some","her",
    "would","make","like","him","into","time","has","look","two",
    "more","write","see","number","way","could","people","than",
    "first","water","been","call","who","oil","now","find","long",
    "down","day","did","get","come","made","may","part","over",
    "new","sound","take","only","little","work","know","place",
    "year","live","back","give","most","very","after","thing",
    "our","just","name","good","man","think","say",
    "great","where","help","through","much","before","line","right",
    "too","mean","old","any","same","tell","boy","follow","came",
    "want","show","also","around","form","three","small","set",
    "put","end","does","another","well","large","must","big",
    "even","such","here","why","ask","went","men","read","need",
    "land","different","home","move","try","kind","hand","picture",
    "again","change","off","play","air","animal","house",
    "point","page","letter","mother","answer","found","study",
    "still","learn","should","world","high","every","near",
    "add","food","between","own","below","country","plant",
    "last","school","father","keep","tree","never","start",
    "city","earth","eye","light","thought","head","under",
    "story","saw","left","few","white","children","begin",
    "got","walk","example","ease","paper","group","music",
    "car","feet","second","book","carry","took","science",
    "eat","room","friend","began","idea","fish",
    "stop","once","base","hear","horse","cut","sure","watch",
    "color","face","wood","main","open","seem","together",
    "next","run","present","dog","cat","bird","sun","moon",
    "star","ocean","sea","valley","forest",
    "tree","flower","rain","snow","wind","fire","glass","door",
    "window","table","chair","bed","road","bridge","wall","roof",
    "gold","silver","iron","steel","stone","metal",
    "love","hate","peace","war","life","death","truth","false",
    "happy","sad","angry","calm","brave","afraid","strong","weak",
    "fast","slow","young","old","rich","poor","wise",
    "hot","cold","dark","light","hard","soft",
    "queen","king","prince","castle","knight","sword",
    "shield","dragon","magic","wizard","witch","fairy","giant",
    "hero","monster","ghost","spirit","angel","demon","god",
    "brain","mind","soul","heart","hand","foot","eye","ear",
    "nose","mouth","blood","bone","skin","body",
    "power","force","energy","mass","speed","time","space",
    "matter","atom","cell",
    "north","south","east","west","left","right",
    "above","below","morning","night","spring","summer","autumn","winter",
    "hundred","thousand","million",
]


def distance_mds(embeddings):
    """Squared distance -> double-centered Gram -> eigenvalues."""
    n = embeddings.shape[0]
    D2 = ((embeddings[:, None, :] - embeddings[None, :, :]) ** 2).sum(axis=2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    ev = np.sort(eigh(B, eigvals_only=True))[::-1]
    return np.maximum(ev, 0)


def compute_metrics(evals):
    """D_f (participation ratio) and alpha (power-law exponent)."""
    s1 = evals.sum()
    s2 = (evals * evals).sum()
    Df = s1 * s1 / s2 if s2 > 0 else 1.0
    ks = np.arange(1, len(evals) + 1, dtype=np.float64)
    keep = (evals > 1e-12) & (ks >= 10)
    alpha, r2 = np.nan, np.nan
    if keep.sum() >= 10:
        slope, _, r, _, _ = stats.linregress(np.log(ks[keep]), np.log(evals[keep]))
        alpha = -slope
        r2 = r ** 2
    return Df, alpha, r2


def monte_carlo_null(evals, n_points, n_trials=500):
    """Random eigenvectors, same spectrum."""
    k = min(len(evals), n_points - 2, 200)
    ev_k = evals[:k]
    n = n_points
    products = []
    for _ in range(n_trials):
        Q, _ = np.linalg.qr(np.random.randn(n, k))
        V = Q @ np.diag(np.sqrt(np.maximum(ev_k, 0)))
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        ne = distance_mds(V)
        d, a, _ = compute_metrics(ne)
        if not np.isnan(a):
            products.append(d * a)
    return np.array(products)


print("Q49 Fresh Verification v3")
print(f"Constants: 8e={E8:.4f}  7pi={SP:.4f}  22={T2:.4f}")
print(f"Words: {len(WORDS)}")
t0 = time.time()

results = {}
for model_name in ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]:
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    m = SentenceTransformer(model_name, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True, show_progress_bar=False)

    ev = distance_mds(embs)
    Df_val, alpha_val, r2_val = compute_metrics(ev)
    product = Df_val * alpha_val

    print(f"  D_f:       {Df_val:.4f}")
    print(f"  alpha:     {alpha_val:.4f}  (R^2={r2_val:.4f})")
    print(f"  product:   {product:.4f}")
    print(f"  8e:        {E8:.4f}  delta={abs(product-E8):.4f} ({abs(product-E8)/E8*100:.1f}%)")
    print(f"  7pi:       {SP:.4f}  delta={abs(product-SP):.4f} ({abs(product-SP)/SP*100:.1f}%)")
    print(f"  22:        {T2:.4f}  delta={abs(product-T2):.4f}")

    # vocab sweep
    print(f"  Vocab sweep:")
    sweep = []
    for sz in [30, 50, 75, 100, 150, 200, 250, 300, 350]:
        if sz > len(WORDS):
            break
        np.random.seed(sz)
        idx = np.random.choice(len(WORDS), sz, replace=False)
        ev2 = distance_mds(embs[idx])
        d2, a2, _ = compute_metrics(ev2)
        p2 = d2 * a2
        sweep.append((sz, p2))
        print(f"    N={sz:3d}: {p2:.4f}")

    # monte carlo
    print(f"  Monte Carlo (500 trials):")
    nulls = monte_carlo_null(ev, len(WORDS), 500)
    nm, ns = nulls.mean(), nulls.std()
    zs = (product - nm) / ns if ns > 0 else 0
    pv = (nulls >= product).mean()
    print(f"    null: {nm:.4f} +/- {ns:.4f}")
    print(f"    z={zs:.2f}  p={pv:.3f}")

    results[model_name] = {
        "n_words": len(WORDS),
        "Df": float(Df_val),
        "alpha": float(alpha_val),
        "alpha_r2": float(r2_val),
        "product": float(product),
        "delta_8e_pct": float(abs(product - E8) / E8 * 100),
        "delta_7pi_pct": float(abs(product - SP) / SP * 100),
        "sweep": [(int(s), float(p)) for s, p in sweep],
        "null_mean": float(nm),
        "null_std": float(ns),
        "null_z": float(zs),
        "null_p": float(pv),
    }

print(f"\n{'='*60}")
print("SUMMARY")
prods = [r["product"] for r in results.values()]
mp, sp = np.mean(prods), np.std(prods)
cv = sp / mp * 100 if mp > 0 else float('inf')
for n, r in results.items():
    print(f"  {n}: {r['product']:.4f}  (delta 8e: {r['delta_8e_pct']:.1f}%)")
print(f"  Mean: {mp:.4f}  Std: {sp:.4f}  CV: {cv:.1f}%")
print(f"  vs 8e:  |{mp - E8:.4f}| ({abs(mp - E8)/E8*100:.1f}%)")
print(f"  vs 7pi: |{mp - SP:.4f}| ({abs(mp - SP)/SP*100:.1f}%)")
print(f"  vs 22:  |{mp - T2:.4f}|")
print(f"  Time: {time.time() - t0:.1f}s")

out = Path("THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verification_v3.json")
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "constants": {"8e": E8, "7pi": SP, "22": T2},
    "summary": {"mean": float(mp), "std": float(sp), "cv_pct": float(cv)},
    "results": results}, open(out, "w"), indent=2)
print(f"Saved: {out}")
