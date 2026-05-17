"""
Q49 Fresh Verification v4 - Correct methodology: DxD covariance matrix.
Matches v1 approach: np.cov(centered.T), eigvalsh, polyfit on first half.
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch

TARGET = 8 * math.e  # 21.746
ALT_7PI = 7 * math.pi
ALT_22 = 22.0

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


def get_eigenspectrum_cov(embeddings):
    """v1 method: DxD covariance, eigvalsh, sorted descending."""
    centered = embeddings - embeddings.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = eigh(cov, eigvals_only=True)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvalues


def compute_df(eigenvalues):
    """Participation ratio."""
    ev = eigenvalues[eigenvalues > 1e-10]
    s1 = ev.sum()
    s2 = (ev * ev).sum()
    return s1 * s1 / s2 if s2 > 0 else 1.0


def compute_alpha(eigenvalues):
    """Power-law exponent via polyfit on first half of eigenvalues."""
    ev = eigenvalues[eigenvalues > 1e-10]
    k = np.arange(1, len(ev) + 1)
    half = len(ev) // 2
    if half < 5:
        return np.nan, np.nan
    log_k = np.log(k[:half])
    log_ev = np.log(ev[:half])
    slope, _ = np.polyfit(log_k, log_ev, 1)
    alpha = -slope
    r2 = np.corrcoef(log_k, log_ev)[0, 1] ** 2
    return alpha, r2


def compute_df_alpha(embeddings):
    ev = get_eigenspectrum_cov(embeddings)
    df_val = compute_df(ev)
    alpha_val, r2_val = compute_alpha(ev)
    return df_val, alpha_val, r2_val, df_val * alpha_val


def get_token_embeddings(model_name, words, normalize=True):
    """Extract token embeddings from a HuggingFace model's embedding table."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embed_weight = model.get_input_embeddings().weight.detach().cpu().numpy()
    embeddings = []
    valid = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            idx = ids[0]
            if idx < len(embed_weight):
                embeddings.append(embed_weight[idx])
                valid.append(w)
    arr = np.array(embeddings, dtype=np.float64)
    if normalize:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1
        arr = arr / norms
    return arr, valid


def get_sbert_embeddings(model_name, words):
    """Sentence-transformer embeddings."""
    model = SentenceTransformer(model_name, device="cpu")
    arr = model.encode(words, normalize_embeddings=True, show_progress_bar=False)
    return arr, words


def monte_carlo_df_alpha(eigenvalues, n_trials=200):
    """Generate null: random eigenvectors, same spectrum."""
    D = len(eigenvalues)
    products = []
    Lambda_sqrt = np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    for _ in range(n_trials):
        Q = np.linalg.qr(np.random.randn(D, D))[0]
        null_cov = Q @ Lambda_sqrt @ Lambda_sqrt @ Q.T
        null_cov = (null_cov + null_cov.T) / 2
        null_ev = np.sort(eigh(null_cov, eigvals_only=True))[::-1]
        null_ev = np.maximum(null_ev, 1e-10)
        df_val = compute_df(null_ev)
        alpha_val, _ = compute_alpha(null_ev)
        if not np.isnan(alpha_val):
            products.append(df_val * alpha_val)
    return np.array(products)


print("Q49 Fresh Verification v4 — Covariance Method (v1-compatible)")
print(f"Target: 8e={TARGET:.4f}  7pi={ALT_7PI:.4f}  22={ALT_22:.4f}")
print()

results = {}
t0 = time.time()

configs = [
    ("bert-base-uncased (token)", "bert-base-uncased", "hf"),
    ("all-MiniLM-L6-v2 (sbert)", "all-MiniLM-L6-v2", "sbert"),
    ("all-mpnet-base-v2 (sbert)", "all-mpnet-base-v2", "sbert"),
]

for label, model_id, kind in configs:
    print(f"{'='*60}")
    print(f"  {label}")

    if kind == "hf":
        embs, words = get_token_embeddings(model_id, WORDS)
    else:
        embs, words = get_sbert_embeddings(model_id, WORDS)

    n_words, n_dims = embs.shape
    print(f"  Shape: {n_words} words x {n_dims} dims")

    Df_val, alpha_val, r2_val, product = compute_df_alpha(embs)
    print(f"  D_f:         {Df_val:.2f}")
    print(f"  alpha:       {alpha_val:.4f}  (R^2={r2_val:.4f})")
    print(f"  D_f*alpha:   {product:.4f}")
    print(f"  8e:          {TARGET:.4f}  delta={product-TARGET:+.4f}  ({abs(product-TARGET)/TARGET*100:.1f}%)")
    print(f"  7pi:         {ALT_7PI:.4f}  delta={product-ALT_7PI:+.4f}")
    print(f"  22:          {ALT_22:.4f}  delta={product-ALT_22:+.4f}")

    # Vocabulary sweep
    print(f"  Vocab sweep:")
    sweep = []
    for sz in [30, 50, 75, 100, 150, 200, 250, 300, n_words]:
        if sz > n_words:
            break
        np.random.seed(sz)
        idx = np.random.choice(n_words, sz, replace=False)
        _, _, _, prod = compute_df_alpha(embs[idx])
        sweep.append((sz, prod))
        print(f"    N={sz:3d}: {prod:.4f}")

    # Monte Carlo null
    ev_full = get_eigenspectrum_cov(embs)
    nulls = monte_carlo_df_alpha(ev_full, 200)
    if len(nulls) > 0:
        nm, ns = nulls.mean(), nulls.std()
        z = (product - nm) / ns if ns > 0 else 0
        pv = (nulls >= product).mean()
        print(f"  Monte Carlo (200 trials):")
        print(f"    null: {nm:.4f} +/- {ns:.4f}")
        print(f"    z={z:.2f}  p={pv:.3f}")

    results[label] = {
        "n_words": n_words,
        "n_dims": n_dims,
        "Df": float(Df_val),
        "alpha": float(alpha_val),
        "alpha_r2": float(r2_val),
        "product": float(product),
        "delta_8e": float(product - TARGET),
        "delta_8e_pct": float(abs(product - TARGET) / TARGET * 100),
        "sweep": [(int(s), float(p)) for s, p in sweep],
    }
    print()

print(f"{'='*60}")
print("SUMMARY")
prods = [r["product"] for r in results.values()]
mp, sp = np.mean(prods), np.std(prods)
cv = sp / mp * 100 if mp > 0 else float('inf')
for n, r in results.items():
    print(f"  {n}: {r['product']:.4f}  delta_8e={r['delta_8e']:+.4f}  ({r['delta_8e_pct']:.1f}%)")
print(f"  Mean: {mp:.4f}  Std: {sp:.4f}  CV: {cv:.1f}%")
print(f"  vs 8e:  |{mp-TARGET:.4f}| ({abs(mp-TARGET)/TARGET*100:.1f}%)")
print(f"  Time: {time.time()-t0:.1f}s")

out = Path("THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verification_v4.json")
out.parent.mkdir(parents=True, exist_ok=True)
json.dump({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "results": results}, open(out, "w"), indent=2)
print(f"  Saved: {out}")
