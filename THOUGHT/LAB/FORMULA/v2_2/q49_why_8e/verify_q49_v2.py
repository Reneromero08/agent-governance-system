"""
Q49 Fresh Verification v2: Df * alpha = 8e
Uses raw token embedding tables + distance-based MDS (matching v1 methodology).
"""
import json
import math
import time
import warnings
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import eigh

warnings.filterwarnings("ignore")

EIGHT_E = 8 * math.e
SEVEN_PI = 7 * math.pi
TWENTY_TWO = 22.0

COMMON_WORDS = [
    "the", "of", "and", "to", "in", "is", "that", "for", "it", "with",
    "on", "are", "be", "this", "have", "from", "or", "one", "had", "by",
    "word", "but", "not", "what", "all", "were", "we", "when", "your", "can",
    "said", "there", "use", "each", "which", "she", "how", "their", "will",
    "other", "about", "out", "many", "then", "them", "these", "some", "her",
    "would", "make", "like", "him", "into", "time", "has", "look", "two",
    "more", "write", "see", "number", "way", "could", "people", "than",
    "first", "water", "been", "call", "who", "oil", "now", "find", "long",
    "down", "day", "did", "get", "come", "made", "may", "part", "over",
    "new", "sound", "take", "only", "little", "work", "know", "place",
    "year", "live", "back", "give", "most", "very", "after", "thing",
    "our", "just", "name", "good", "sentence", "man", "think", "say",
    "great", "where", "help", "through", "much", "before", "line", "right",
    "too", "mean", "old", "any", "same", "tell", "boy", "follow", "came",
    "want", "show", "also", "around", "form", "three", "small", "set",
    "put", "end", "does", "another", "well", "large", "must", "big",
    "even", "such", "here", "why", "ask", "went", "men", "read", "need",
    "land", "different", "home", "move", "try", "kind", "hand", "picture",
    "again", "change", "off", "play", "spell", "air", "animal", "house",
    "point", "page", "letter", "mother", "answer", "found", "study",
    "still", "learn", "should", "world", "high", "every", "near",
    "add", "food", "between", "own", "below", "country", "plant",
    "last", "school", "father", "keep", "tree", "never", "start",
    "city", "earth", "eye", "light", "thought", "head", "under",
    "story", "saw", "left", "few", "white", "children", "begin",
    "got", "walk", "example", "ease", "paper", "group", "music",
    "river", "car", "feet", "second", "book", "carry", "took", "science",
    "eat", "room", "friend", "began", "idea", "fish", "mountain",
    "stop", "once", "base", "hear", "horse", "cut", "sure", "watch",
    "color", "face", "wood", "main", "open", "seem", "together",
    "next", "run", "present", "dog", "cat", "bird", "sun", "moon",
    "star", "ocean", "sea", "river", "mountain", "valley", "forest",
    "tree", "flower", "rain", "snow", "wind", "fire", "glass", "door",
    "window", "table", "chair", "bed", "road", "bridge", "wall", "roof",
    "gold", "silver", "iron", "steel", "stone", "wood", "metal",
    "love", "hate", "peace", "war", "life", "death", "truth", "false",
    "happy", "sad", "angry", "calm", "brave", "afraid", "strong", "weak",
    "fast", "slow", "young", "old", "rich", "poor", "wise", "foolish",
    "hot", "cold", "wet", "dry", "dark", "light", "hard", "soft",
    "queen", "king", "prince", "princess", "castle", "knight", "sword",
    "shield", "dragon", "magic", "wizard", "witch", "fairy", "giant",
    "hero", "monster", "ghost", "spirit", "angel", "demon", "god",
    "brain", "mind", "soul", "heart", "hand", "foot", "eye", "ear",
    "nose", "mouth", "blood", "bone", "skin", "flesh", "body",
    "power", "force", "energy", "mass", "speed", "time", "space",
    "matter", "atom", "cell", "life", "death", "light", "sound",
    "heat", "cold", "wet", "dry", "black", "white", "red", "blue",
    "green", "yellow", "brown", "gray", "north", "south", "east", "west",
    "left", "right", "up", "down", "in", "out", "on", "off",
    "above", "below", "before", "after", "always", "never",
    "morning", "night", "day", "week", "month", "year",
    "spring", "summer", "autumn", "winter",
]


def distance_mds_eigenvalues(embeddings):
    """Classical MDS: squared distances -> double-centered Gram -> eigenvalues."""
    n = embeddings.shape[0]
    diffs = embeddings[:, None, :] - embeddings[None, :, :]
    D2 = (diffs * diffs).sum(axis=2)
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigenvalues = np.sort(eigh(B, eigvals_only=True))[::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues


def participation_ratio(eigenvalues):
    s1 = eigenvalues.sum()
    s2 = (eigenvalues * eigenvalues).sum()
    if s2 == 0:
        return 1.0
    return s1 * s1 / s2


def fit_alpha(eigenvalues, min_k=10):
    """Fit power-law exponent: lambda_k = C * k^(-alpha)."""
    ks = np.arange(1, len(eigenvalues) + 1, dtype=np.float64)
    keep = (eigenvalues > 1e-12) & (ks >= min_k)
    if keep.sum() < 10:
        return np.nan, np.nan
    log_k = np.log(ks[keep])
    log_lambda = np.log(eigenvalues[keep])
    slope, intercept, r, p, se = stats.linregress(log_k, log_lambda)
    return -slope, r ** 2


def compute_df_alpha_from_embeddings(embs):
    evals = distance_mds_eigenvalues(embs)
    Df = participation_ratio(evals)
    alpha, r2 = fit_alpha(evals)
    return Df, alpha, r2, Df * alpha


def random_matrix_null(evals, n_trials=1000):
    """Structure-preserving null: same spectrum, random eigenvectors."""
    n = len(evals) + 1
    k = len(evals)
    products = []
    for i in range(n_trials):
        Q, _ = np.linalg.qr(np.random.randn(n, k))
        Lam = np.diag(np.sqrt(np.maximum(evals, 0)))
        V = (Q @ Lam)
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
        null_evals = distance_mds_eigenvalues(V)
        Df = participation_ratio(null_evals)
        alpha, _ = fit_alpha(null_evals)
        if not np.isnan(alpha):
            products.append(Df * alpha)
    return np.array(products)


def load_token_embeddings_hf(model_name, words):
    """Load a model's token embedding table and extract word embeddings."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embed_weight = model.get_input_embeddings().weight.detach().cpu().numpy()
    embeddings = []
    valid_words = []
    for w in words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if len(ids) == 1:
            idx = ids[0]
            if idx < len(embed_weight):
                embeddings.append(embed_weight[idx])
                valid_words.append(w)
    arr = np.array(embeddings, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1
    arr = arr / norms
    return arr, valid_words


def load_sentence_transformer_embeddings(model_name, words):
    """Use sentence-transformers (simpler, faster)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    embeddings = model.encode(words, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings, words


def run_verification():
    results = {}
    start = time.time()

    configs = [
        ("BERT-base (token embed)", "bert-base-uncased", "hf"),
        ("MiniLM-L6-v2 (sbert)", "all-MiniLM-L6-v2", "sbert"),
        ("MPNet-base-v2 (sbert)", "all-mpnet-base-v2", "sbert"),
    ]

    for label, model_id, loader_type in configs:
        print(f"\n{'='*60}")
        print(f"Model: {label}")

        if loader_type == "hf":
            embs, words = load_token_embeddings_hf(model_id, COMMON_WORDS)
        else:
            embs, words = load_sentence_transformer_embeddings(model_id, COMMON_WORDS)

        n = len(words)
        print(f"  Valid words: {n}")

        if n < 50:
            print(f"  SKIP: too few valid tokens")
            continue

        Df, alpha, r2, product = compute_df_alpha_from_embeddings(embs)
        print(f"  D_f:          {Df:.4f}")
        print(f"  alpha:        {alpha:.4f}  (R² = {r2:.4f})")
        print(f"  D_f * alpha:  {product:.4f}")
        print(f"  8e:           {EIGHT_E:.4f}  delta: {abs(product - EIGHT_E):.4f}  ({abs(product - EIGHT_E)/EIGHT_E*100:.2f}%)")
        print(f"  7pi:          {SEVEN_PI:.4f}  delta: {abs(product - SEVEN_PI):.4f}")
        print(f"  22:           {TWENTY_TWO:.4f}  delta: {abs(product - TWENTY_TWO):.4f}")

        # Vocabulary sweep
        print(f"\n  Vocab size sweep:")
        sizes = sorted(set([30, 50, 75, 100, 150, 200, n]))
        sizes = [s for s in sizes if s <= n]
        sweep = []
        for size in sizes:
            np.random.seed(size)
            idxs = np.random.choice(n, size=size, replace=False)
            sub_embs = embs[idxs]
            _, _, _, prod = compute_df_alpha_from_embeddings(sub_embs)
            sweep.append((size, prod))
            print(f"    N={size:3d}: {prod:.4f}")

        # Monte Carlo
        print(f"\n  Monte Carlo null (1000 trials):")
        evals = distance_mds_eigenvalues(embs)
        null = random_matrix_null(evals[:min(n-1, len(evals))], n_trials=1000)
        null_mean, null_std = null.mean(), null.std()
        z = (product - null_mean) / null_std if null_std > 0 else 0
        pv = (null >= product).mean()
        print(f"    Observed:    {product:.4f}")
        print(f"    Null:        {null_mean:.4f} +/- {null_std:.4f}")
        print(f"    Z:           {z:.4f}")
        print(f"    p:           {pv:.4f}")

        results[label] = {
            "n_words": n,
            "Df": float(Df),
            "alpha": float(alpha),
            "alpha_r2": float(r2),
            "product": float(product),
            "delta_8e_pct": float(abs(product - EIGHT_E) / EIGHT_E * 100),
            "sweep": [(int(s), float(p)) for s, p in sweep],
            "null_mean": float(null_mean),
            "null_std": float(null_std),
            "null_z": float(z),
            "null_p": float(pv),
        }

    print(f"\n{'='*60}")
    print("SUMMARY")
    products = [r["product"] for r in results.values()]
    if products:
        mean_p = np.mean(products)
        std_p = np.std(products)
        cv = std_p / mean_p * 100 if mean_p > 0 else float('inf')
        for name, r in results.items():
            print(f"  {name}: {r['product']:.4f}  (delta 8e: {r['delta_8e_pct']:.2f}%)")
        print(f"  Mean: {mean_p:.4f}  Std: {std_p:.4f}  CV: {cv:.2f}%")
        print(f"  vs 8e: |{mean_p - EIGHT_E:.4f}| ({abs(mean_p - EIGHT_E)/EIGHT_E*100:.2f}%)")
        print(f"  vs 7pi: |{mean_p - SEVEN_PI:.4f}|")
        print(f"  vs 22: |{mean_p - TWENTY_TWO:.4f}|")

    elapsed = time.time() - start
    print(f"\n  Time: {elapsed:.1f}s")

    out_path = Path("THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verification_results_v2.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "constants": {"8e": EIGHT_E, "7pi": SEVEN_PI, "22": TWENTY_TWO},
            "results": results,
        }, f, indent=2)
    print(f"  Saved: {out_path}")

    return results


if __name__ == "__main__":
    run_verification()
