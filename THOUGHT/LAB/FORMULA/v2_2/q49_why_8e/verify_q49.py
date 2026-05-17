"""
Q49 Fresh Verification: Df * alpha = 8e
Tests the conservation law across embedding models with Monte Carlo null.
"""
import json
import math
import time
import warnings
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.sparse.linalg import eigsh
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ---- Constants ----
EIGHT_E = 8 * math.e          # 21.74625...
SEVEN_PI = 7 * math.pi        # 21.99114...
TWENTY_TWO = 22.0

# ---- Vocabulary ----
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
]


def compute_embeddings(model, words):
    """Get normalized embeddings for a list of words."""
    embeddings = model.encode(words, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings


def compute_gram_matrix(embeddings):
    """Compute the Gram (similarity) matrix from embeddings."""
    return embeddings @ embeddings.T


def eigen_decompose(gram, k=128):
    """Get top-k eigenvalues of the Gram matrix."""
    n = gram.shape[0]
    k = min(k, n - 2)
    eigenvalues, eigenvectors = eigsh(gram, k=k, which='LM')
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues, eigenvectors


def compute_participation_ratio(eigenvalues):
    """D_f = (sum lambda)^2 / sum(lambda^2)."""
    total = eigenvalues.sum()
    if total == 0:
        return 1.0
    return total * total / (eigenvalues * eigenvalues).sum()


def fit_power_law(eigenvalues, min_k=3, max_k=None):
    """Fit alpha from lambda_k = C * k^(-alpha) using log-log regression."""
    if max_k is None:
        max_k = len(eigenvalues)
    ks = np.arange(1, max_k + 1, dtype=np.float64)
    keep = (eigenvalues > 1e-12) & (ks >= min_k) & (ks <= max_k)
    if keep.sum() < 5:
        return np.nan
    log_k = np.log(ks[keep])
    log_lambda = np.log(eigenvalues[keep])
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_k, log_lambda)
    return -slope


def compute_df_alpha(embs):
    """Compute D_f * alpha for a set of embeddings."""
    gram = compute_gram_matrix(embs)
    evals, _ = eigen_decompose(gram, k=min(128, len(embs) - 2))
    Df = compute_participation_ratio(evals)
    alpha = fit_power_law(evals)
    return Df, alpha, Df * alpha


def generate_structured_null(eigenvalues, n_vecs=256, n_trials=1000):
    """Generate null distribution preserving eigenvalue spectrum but randomizing eigenvectors."""
    products = []
    n = len(eigenvalues) + 1
    for _ in range(n_trials):
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        null_evecs = Q[:, :len(eigenvalues)]
        Lambda = np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        null_vecs = (null_evecs @ Lambda)
        null_vecs = null_vecs / (np.linalg.norm(null_vecs, axis=1, keepdims=True) + 1e-12)
        gram = null_vecs @ null_vecs.T
        null_evals = np.sort(np.linalg.eigvalsh(gram))[::-1]
        null_evals = np.maximum(null_evals, 0)
        Df = compute_participation_ratio(null_evals)
        alpha = fit_power_law(null_evals)
        if not np.isnan(alpha) and not np.isinf(alpha):
            products.append(Df * alpha)
    return np.array(products)


def bayes_factor_approx(observed_mean, observed_std, n_samples, constant):
    """Approximate log Bayes factor: log P(data | constant) - log P(data | free mu)."""
    se = observed_std / np.sqrt(n_samples)
    log_lik_constant = -0.5 * ((observed_mean - constant) / se) ** 2
    return max(log_lik_constant, 0)


def run_verification():
    results = {}
    start = time.time()

    # ---- Models to test ----
    model_names = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
    print(f"Loading models: {model_names}")

    for model_name in model_names:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        model = SentenceTransformer(model_name)

        # Full vocabulary
        words = COMMON_WORDS
        embs = compute_embeddings(model, words)
        Df, alpha, product = compute_df_alpha(embs)
        n_words = len(words)

        print(f"  Vocabulary size: {n_words}")
        print(f"  D_f:              {Df:.4f}")
        print(f"  alpha:            {alpha:.4f}")
        print(f"  D_f * alpha:      {product:.4f}")
        print(f"  8e:               {EIGHT_E:.4f}  (delta: {abs(product - EIGHT_E):.4f}, {abs(product - EIGHT_E)/EIGHT_E*100:.2f}%)")
        print(f"  7*pi:             {SEVEN_PI:.4f}  (delta: {abs(product - SEVEN_PI):.4f}, {abs(product - SEVEN_PI)/SEVEN_PI*100:.2f}%)")
        print(f"  22:               {TWENTY_TWO:.4f}  (delta: {abs(product - TWENTY_TWO):.4f})")

        # ---- Vocabulary size sweep ----
        print(f"\n  Vocabulary size sweep:")
        sizes = [30, 50, 75, 100, 150, 200, 250, 300]
        sweep_results = []
        for size in sizes:
            if size >= n_words:
                break
            np.random.seed(size)
            subset = np.random.choice(words, size=size, replace=False)
            sub_embs = compute_embeddings(model, subset)
            _, _, prod = compute_df_alpha(sub_embs)
            sweep_results.append((size, prod))

        for size, prod in sweep_results:
            marker = " *" if size == max(s for s, _ in sweep_results) else ""
            print(f"    N={size:3d}: D_f*alpha = {prod:.4f}{marker}")

        # ---- Monte Carlo null ----
        print(f"\n  Monte Carlo (1000 trials, structure-preserving null):")
        gram = compute_gram_matrix(embs)
        evals, _ = eigen_decompose(gram, k=min(128, n_words - 2))
        null_products = generate_structured_null(evals, n_vecs=n_words, n_trials=1000)

        null_mean = null_products.mean()
        null_std = null_products.std()
        p_value = (null_products >= product).mean()
        z_score = (product - null_mean) / null_std if null_std > 0 else float('inf')

        print(f"    Observed: {product:.4f}")
        print(f"    Null mean: {null_mean:.4f} +/- {null_std:.4f}")
        print(f"    Z-score:   {z_score:.4f}")
        print(f"    p-value:   {p_value:.4f}")

        # ---- Model comparison ----
        print(f"\n  Bayesian comparison (approximate log BF vs free parameter):")
        sweep_prods = np.array([p for _, p in sweep_results])
        obs_mean = sweep_prods.mean()
        obs_std = sweep_prods.std() if len(sweep_prods) > 1 else 0.5
        n_eff = len(sweep_prods)

        bf_8e = bayes_factor_approx(obs_mean, obs_std, n_eff, EIGHT_E)
        bf_7pi = bayes_factor_approx(obs_mean, obs_std, n_eff, SEVEN_PI)
        bf_22 = bayes_factor_approx(obs_mean, obs_std, n_eff, TWENTY_TWO)

        print(f"    Observed mean across vocab sizes: {obs_mean:.4f} +/- {obs_std:.4f}")
        print(f"    log BF(8e):     {bf_8e:.4f}")
        print(f"    log BF(7*pi):   {bf_7pi:.4f}")
        print(f"    log BF(22):     {bf_22:.4f}")

        best = max(("8e", bf_8e), ("7*pi", bf_7pi), ("22", bf_22), key=lambda x: x[1])
        print(f"    Best match: {best[0]} (log BF = {best[1]:.4f})")

        # Store results
        results[model_name] = {
            "n_words": n_words,
            "Df": float(Df),
            "alpha": float(alpha),
            "product": float(product),
            "delta_8e": float(abs(product - EIGHT_E)),
            "delta_8e_pct": float(abs(product - EIGHT_E) / EIGHT_E * 100),
            "delta_7pi": float(abs(product - SEVEN_PI)),
            "delta_22": float(abs(product - TWENTY_TWO)),
            "sweep_sizes": [int(s) for s, _ in sweep_results],
            "sweep_values": [float(p) for _, p in sweep_results],
            "monte_carlo": {
                "n_trials": 1000,
                "null_mean": float(null_mean),
                "null_std": float(null_std),
                "z_score": float(z_score),
                "p_value": float(p_value),
            },
            "bayesian": {
                "obs_mean": float(obs_mean),
                "obs_std": float(obs_std),
                "log_bf_8e": float(bf_8e),
                "log_bf_7pi": float(bf_7pi),
                "log_bf_22": float(bf_22),
                "best_match": best[0],
            },
        }

    # ---- Cross-model summary ----
    print(f"\n{'='*60}")
    print("CROSS-MODEL SUMMARY")
    products = [r["product"] for r in results.values()]
    mean_prod = np.mean(products)
    std_prod = np.std(products)
    cv = std_prod / mean_prod * 100 if mean_prod > 0 else float('inf')

    print(f"  Models tested: {len(results)}")
    for name, r in results.items():
        print(f"    {name}: {r['product']:.4f} (delta 8e: {r['delta_8e_pct']:.2f}%)")
    print(f"  Mean: {mean_prod:.4f}")
    print(f"  Std:  {std_prod:.4f}")
    print(f"  CV:   {cv:.2f}%")
    print(f"  vs 8e:  |{mean_prod - EIGHT_E:.4f}|  ({abs(mean_prod - EIGHT_E)/EIGHT_E*100:.2f}%)")
    print(f"  vs 7pi: |{mean_prod - SEVEN_PI:.4f}|  ({abs(mean_prod - SEVEN_PI)/SEVEN_PI*100:.2f}%)")
    print(f"  vs 22:  |{mean_prod - TWENTY_TWO:.4f}|")

    elapsed = time.time() - start
    print(f"\n  Total time: {elapsed:.1f}s")

    # ---- Save results ----
    out_path = Path("THOUGHT/LAB/FORMULA/v2_2/q49_why_8e/verification_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "constants": {"8e": EIGHT_E, "7pi": SEVEN_PI, "22": TWENTY_TWO},
            "summary": {
                "n_models": len(results),
                "mean_product": float(mean_prod),
                "std_product": float(std_prod),
                "cv_pct": float(cv),
                "delta_8e": float(abs(mean_prod - EIGHT_E)),
                "delta_8e_pct": float(abs(mean_prod - EIGHT_E) / EIGHT_E * 100),
            },
            "models": results,
        }, f, indent=2)
    print(f"\nResults saved to: {out_path}")

    return results


if __name__ == "__main__":
    run_verification()
