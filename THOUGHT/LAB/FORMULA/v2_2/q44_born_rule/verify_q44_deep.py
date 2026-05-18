"""
Q44 Deep Verification: Beyond the v2 README tests.
Tests: (A) Random-vector null, (B) Individual Born rule, (C) Phase modulation.
"""
import json, math, time
from pathlib import Path
import numpy as np
from scipy import stats
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer

CONCEPTS = [
    "love", "hate", "truth", "false", "peace", "war", "life", "death",
    "time", "space", "power", "fear", "hope", "joy", "pain", "dream",
    "freedom", "justice", "beauty", "wisdom", "courage", "honor",
    "science", "art", "music", "nature", "god", "evil", "good", "bad",
    "light", "dark", "strong", "weak", "fast", "slow", "hot", "cold",
    "rich", "poor", "young", "old", "new", "ancient", "simple", "complex",
]

CONTEXTS = [
    "emotion", "feeling", "thought", "idea", "action", "result",
    "beginning", "ending", "knowledge", "mystery", "strength", "weakness",
    "order", "chaos", "creation", "destruction", "presence", "absence",
    "motion", "stillness", "sound", "silence", "growth", "decay",
    "connection", "separation", "unity", "division", "health", "sickness",
    "wealth", "poverty", "success", "failure", "past", "future",
    "inside", "outside", "above", "below", "journey", "arrival",
    "departure", "victory", "defeat", "friendship", "betrayal",
    "memory", "forgetting", "courage", "fear", "love", "hatred",
    "warmth", "coldness", "hunger", "thirst", "sleep", "waking",
    "birth", "death", "dawn", "dusk", "spring", "winter",
]


def overlaps(psi, phi_embs):
    psi_n = psi / (np.linalg.norm(psi) + 1e-12)
    phi_n = phi_embs / (np.linalg.norm(phi_embs, axis=1, keepdims=True) + 1e-12)
    return psi_n @ phi_n.T


print("=" * 64)
print("Q44 DEEP VERIFICATION")
print("=" * 64)

for model_id, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    model = SentenceTransformer(model_id, device="cpu")
    D = model.get_sentence_embedding_dimension()
    c_embs = model.encode(CONCEPTS, normalize_embeddings=True)
    ctx_embs = model.encode(CONTEXTS, normalize_embeddings=True)
    n_concepts, n_contexts = len(CONCEPTS), len(CONTEXTS)

    print(f"\n{'='*64}")
    print(f"Model: {name} (D={D})")
    print(f"Concepts: {n_concepts}, Contexts: {n_contexts}")

    # =========================================================================
    # TEST A: Random-vector null
    # =========================================================================
    print(f"\n  TEST A: Random-vector null (same dim, same count)")
    print(f"  If the same correlations appear for random vectors,")
    print(f"  the effect is geometry, not semantics.")

    real_E, real_P = [], []
    for i in range(n_concepts):
        x = overlaps(c_embs[i], ctx_embs)
        real_E.append(np.maximum(x, 0).mean())
        real_P.append((np.maximum(x, 0) ** 2).mean())

    real_r = stats.pearsonr(real_E, real_P)[0]

    rand_rs = []
    for trial in range(100):
        r_vecs = np.random.randn(n_concepts, D)
        r_vecs = r_vecs / (np.linalg.norm(r_vecs, axis=1, keepdims=True) + 1e-12)
        rand_E, rand_P = [], []
        for i in range(n_concepts):
            x = overlaps(r_vecs[i], ctx_embs)
            rand_E.append(np.maximum(x, 0).mean())
            rand_P.append((np.maximum(x, 0) ** 2).mean())
        rand_rs.append(stats.pearsonr(rand_E, rand_P)[0])

    rand_rs = np.array(rand_rs)
    p_better = (rand_rs >= real_r).mean()

    print(f"    Real embedding r(E, P_born): {real_r:.4f}")
    print(f"    Random vector r(E, P_born):  {rand_rs.mean():.4f} +/- {rand_rs.std():.4f}")
    print(f"    p(random >= real):           {p_better:.4f}")
    print(f"    Verdict: {'GEOMETRY (random matches real)' if p_better > 0.05 else 'SEMANTIC (real exceeds random)'}")

    # =========================================================================
    # TEST B: Individual-level Born rule
    # =========================================================================
    print(f"\n  TEST B: Individual Born rule (per-pair, not aggregate)")
    print(f"  Born rule: P_i = |<phi_i|psi>|^2 for EACH i, not just on average.")
    print(f"  Test: do individual E_i values predict individual P_i values?")

    # For each concept, compute E_i and P_i for EACH context word
    indiv_E, indiv_P = [], []
    for i in range(n_concepts):
        x = overlaps(c_embs[i], ctx_embs)
        indiv_E.extend(np.maximum(x, 0).tolist())
        indiv_P.extend((np.maximum(x, 0) ** 2).tolist())

    indiv_r = stats.pearsonr(indiv_E, indiv_P)[0]
    # Born rule predicts P_i = E_i^2, so r(E, sqrt(P)) should be high
    indiv_r_sqrt = stats.pearsonr(indiv_E, np.sqrt(np.maximum(indiv_P, 0)))[0]

    print(f"    N pairs:          {len(indiv_E)}")
    print(f"    r(E_i, P_i):      {indiv_r:.4f}")
    print(f"    r(E_i, sqrt(P_i)):{indiv_r_sqrt:.4f}")

    # Test: does P_i = E_i^2 hold for individual pairs?
    # Born rule: P_i should EQUAL E_i^2, not just correlate
    residuals = np.array(indiv_P) - np.array(indiv_E) ** 2
    mae = np.abs(residuals).mean()
    rmse = np.sqrt((residuals ** 2).mean())
    print(f"    MAE(P_i - E_i^2): {mae:.4f}")
    print(f"    RMSE(P_i - E_i^2):{rmse:.4f}")

    # The Born rule is not just "correlates well" — it's a specific equality
    # P_i = |<phi_i|psi>|^2 for all i
    # For real cosine similarities: P_i = cos_sim_i^2, so E_i = cos_sim_i
    # This means the "Born rule" reduces to: cos_sim_i^2 = cos_sim_i^2
    # Which is an identity, not a prediction
    print(f"    Verdict: Born rule reduces to cos_sim^2 = cos_sim^2 at individual level.")
    print(f"             This is an identity, not a physical law.")

    # =========================================================================
    # TEST C: Phase modulation — the quantum signature
    # =========================================================================
    print(f"\n  TEST C: Phase modulation (cos^2 fringe scan)")
    print(f"  Quantum prediction: P(theta) = A + B*cos(2*theta + phi)")
    print(f"  Classical: just a smooth function, no specific sinusoidal form.")

    w1_idx, w2_idx = CONCEPTS.index("love"), CONCEPTS.index("hate")
    e1, e2 = c_embs[w1_idx], c_embs[w2_idx]

    # Orthogonalize to get a clean 2D subspace
    e2_orth = e2 - (e1 @ e2) * e1
    e2_orth = e2_orth / (np.linalg.norm(e2_orth) + 1e-12)

    thetas = np.linspace(0, 2 * np.pi, 60)
    P_vals = []
    for theta in thetas:
        psi = math.cos(theta) * e1 + math.sin(theta) * e2_orth
        psi = psi / (np.linalg.norm(psi) + 1e-12)
        x = overlaps(psi, ctx_embs)
        _, P = np.maximum(x, 0).mean(), (np.maximum(x, 0) ** 2).mean()
        P_vals.append(P)

    P_vals = np.array(P_vals)

    # Fit to A + B*cos(2θ + φ)
    from scipy.optimize import curve_fit
    def cos2_model(theta, A, B, phi):
        return A + B * np.cos(2 * theta + phi)

    try:
        popt, _ = curve_fit(cos2_model, thetas, P_vals, p0=[P_vals.mean(), P_vals.std(), 0])
        A_fit, B_fit, phi_fit = popt
        P_pred = cos2_model(thetas, A_fit, B_fit, phi_fit)
        r2_fringe = np.corrcoef(P_vals, P_pred)[0, 1] ** 2
        fringe_ratio = abs(B_fit) / (abs(A_fit) + 1e-12)

        print(f"    cos²(2θ+φ) fit R²: {r2_fringe:.4f}")
        print(f"    Fringe visibility (|B/A|): {fringe_ratio:.4f}")
        print(f"    Verdict: {'QUANTUM FRINGE PATTERN' if r2_fringe > 0.9 and fringe_ratio > 0.1 else 'NO CLEAR FRINGES (classical)' if r2_fringe < 0.5 else 'MARGINAL'}")
    except:
        print(f"    Fit failed")

    # =========================================================================
    # TEST D: Does the embedding dimension matter? (scaling check)
    # =========================================================================
    print(f"\n  TEST D: Dimensional scaling — does the Born-rule correlation scale with dim?")
    print(f"  If it's algebraic, r should be stable across dims. If quantum, it might show")
    print(f"  specific scaling behavior.")

    # Randomly subsample dimensions and re-compute
    for frac in [0.25, 0.5, 0.75, 1.0]:
        d_sub = max(10, int(D * frac))
        idxs = np.random.choice(D, d_sub, replace=False)
        sub_E, sub_P = [], []
        for i in range(n_concepts):
            x = overlaps(c_embs[i][idxs], ctx_embs[:, idxs])
            sub_E.append(np.maximum(x, 0).mean())
            sub_P.append((np.maximum(x, 0) ** 2).mean())
        sub_r = stats.pearsonr(sub_E, sub_P)[0]
        print(f"    dims={d_sub:4d} ({frac:.0%}): r(E,P_born) = {sub_r:.4f}")

    # Save
    out = Path(f"THOUGHT/LAB/FORMULA/v2_2/q44_born_rule/verification_deep_{name}.json")
    json.dump({
        "model": name,
        "testA": {"real_r": float(real_r), "rand_mean_r": float(rand_rs.mean()),
                  "rand_std_r": float(rand_rs.std()), "p_better": float(p_better)},
        "testB": {"indiv_r": float(indiv_r), "indiv_sqrt_r": float(indiv_r_sqrt),
                  "mae": float(mae), "rmse": float(rmse)},
    }, open(out, "w"), indent=2)

print(f"\n{'='*64}")
print("Q44 DEEP VERIFICATION COMPLETE")
print("=" * 64)
