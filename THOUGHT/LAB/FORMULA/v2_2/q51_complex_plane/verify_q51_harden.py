"""Q51 hardening: PCA sweep, causal test, predictive test, seed stability."""
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer

WORDS_POOL = ["water","fire","earth","sky","sun","moon","star","mountain","river","tree","flower","rain","wind","snow","cloud","ocean","dog","cat","bird","fish","horse","tiger","lion","elephant","heart","eye","hand","head","brain","blood","mother","father","child","friend","king","queen","love","hate","truth","life","death","time","space","power","peace","war","hope","fear","joy","pain","dream","thought","book","door","house","road","food","money","stone","gold","car","bicycle","train","airplane","school","hospital","church","bank","gun","sword","knife","army","navy","police","doctor","lawyer","teacher","engineer","student","piano","guitar","violin","drum","apple","banana","orange","grape","shirt","pants","shoes","hat","summer","winter","spring","autumn","morning","night","dawn","dusk","moon","sun","star","planet"]

ANALOGIES_CORRECT = [
    ["king","man","woman","queen"],
    ["paris","france","berlin","germany"],
    ["walk","walking","run","running"],
    ["good","better","bad","worse"],
    ["cat","kitten","dog","puppy"],
    ["doctor","hospital","teacher","school"],
    ["hot","cold","fast","slow"],
    ["buy","sell","give","take"],
    ["japan","tokyo","france","paris"],
    ["apple","fruit","carrot","vegetable"],
    ["dog","bark","cat","meow"],
    ["big","bigger","small","smaller"],
]

ANALOGIES_WRONG = [
    ["king","man","car","bicycle"],
    ["paris","france","apple","banana"],
    ["walk","walking","shirt","pants"],
    ["good","better","moon","sun"],
    ["cat","kitten","gun","sword"],
    ["doctor","hospital","piano","guitar"],
    ["hot","cold","summer","winter"],
    ["buy","sell","doctor","lawyer"],
    ["japan","tokyo","water","fire"],
    ["apple","fruit","train","airplane"],
    ["dog","bark","church","bank"],
    ["big","bigger","dawn","dusk"],
]

def berry_phase(states):
    total = np.complex128(0.0)
    for i in range(len(states)-1):
        overlap = np.conj(states[i]).dot(states[i+1])
        if abs(overlap) < 1e-15:
            return 0.0, 0.0
        total += np.log(overlap)
    phi = (-np.imag(total)) % (2*np.pi)
    return phi, abs(np.exp(1j * total))

def test_loop(words, K, evecs, center, kind):
    embs = m.encode(words, normalize_embeddings=True)
    cw = embs - center
    proj = cw @ evecs[:, :K]
    norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
    proj = proj / norms
    if kind == "hilbert":
        z = hilbert(proj, axis=0).astype(np.complex128)
    elif kind == "random":
        z = proj.astype(np.complex128) * np.exp(1j * 2 * np.pi * np.random.rand(*proj.shape))
    else:  # real
        z = proj.astype(np.complex128)
    zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)
    return berry_phase([z[i] for i in range(len(z))])

for mid, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    all_embs = m.encode(WORDS_POOL, normalize_embeddings=True)
    center = all_embs.mean(axis=0)
    centered = all_embs - center
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
    D = m.get_sentence_embedding_dimension()

    print(f"\n{'='*64}")
    print(f"{name}")
    print(f"{'='*64}")

    # ---- ANGLE 1: PCA sweep for Berry phase separation ----
    print(f"\n  ANGLE 1: PCA sweep — optimal K for analogy vs random separation")
    for K in [8, 16, 32, 64, 96, 128, 192, 256, 384, 768]:
        if K > D: continue
        np.random.seed(K)
        correct_phis = []
        for words in ANALOGIES_CORRECT:
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, "hilbert")
            correct_phis.append(phi)
        wrong_phis = []
        for words in ANALOGIES_WRONG:
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, "hilbert")
            wrong_phis.append(phi)
        rand_phis = []
        for _ in range(50):
            words = list(np.random.choice(WORDS_POOL, 5, replace=False))
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, "hilbert")
            rand_phis.append(phi)

        c_arr = np.array(correct_phis); w_arr = np.array(wrong_phis); r_arr = np.array(rand_phis)
        ks_cr = stats.ks_2samp(c_arr, r_arr)
        ks_wr = stats.ks_2samp(w_arr, r_arr)
        ks_cw = stats.ks_2samp(c_arr, w_arr)
        best = "*** BEST" if ks_cr.pvalue < 0.01 and ks_cw.pvalue < 0.05 else ""
        print(f"  K={K:3d}  correct_vs_random p={ks_cr.pvalue:.2e}  wrong_vs_random p={ks_wr.pvalue:.2e}  correct_vs_wrong p={ks_cw.pvalue:.4e}{best}")

    # ---- ANGLE 2: Generalization — random phases also work? ----
    print(f"\n  ANGLE 2: Generalization across complexification methods (K=96)")
    K = 96
    for method, label in [("hilbert", "Hilbert"), ("random", "Random phases"), ("real", "Real (no complex)")]:
        np.random.seed(42)
        c_phis = []
        for words in ANALOGIES_CORRECT:
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, method)
            c_phis.append(phi)
        r_phis = []
        for _ in range(50):
            words = list(np.random.choice(WORDS_POOL, 5, replace=False))
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, method)
            r_phis.append(phi)
        c_arr = np.array(c_phis); r_arr = np.array(r_phis)
        ks = stats.ks_2samp(c_arr, r_arr)
        print(f"  {label:<20s}: correct_vs_random KS p={ks.pvalue:.2e}")

    # ---- ANGLE 3: Seed stability ----
    print(f"\n  ANGLE 3: Seed stability (K=96, Hilbert, 10 seeds)")
    K = 96
    p_vals = []
    for seed in range(10):
        np.random.seed(seed)
        c_phis = []
        for words in ANALOGIES_CORRECT:
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, "hilbert")
            c_phis.append(phi)
        r_phis = []
        for _ in range(50):
            words = list(np.random.choice(WORDS_POOL, 5, replace=False))
            loop = words + [words[0]]
            phi, _ = test_loop(loop, K, evecs, center, "hilbert")
            r_phis.append(phi)
        ks = stats.ks_2samp(np.array(c_phis), np.array(r_phis))
        p_vals.append(ks.pvalue)
    p_vals = np.array(p_vals)
    print(f"  KS p range: [{p_vals.min():.4f}, {p_vals.max():.4f}]  mean={p_vals.mean():.4f}")
    print(f"  p<0.01 in {sum(p_vals<0.01)}/10 seeds")

    # ---- ANGLE 4: Predictive test — AUROC for correct vs wrong ----
    print(f"\n  ANGLE 4: Predictive — can phi separate correct from wrong analogies?")
    K = 96
    np.random.seed(42)
    labels = [1]*len(ANALOGIES_CORRECT) + [0]*len(ANALOGIES_WRONG)
    scores = []
    for words in ANALOGIES_CORRECT + ANALOGIES_WRONG:
        loop = words + [words[0]]
        phi, _ = test_loop(loop, K, evecs, center, "hilbert")
        scores.append(phi)
    from sklearn.metrics import roc_auc_score
    auroc = roc_auc_score(labels, scores)
    auroc_inv = roc_auc_score(labels, [-s for s in scores])
    print(f"  AUROC = {max(auroc, auroc_inv):.4f}  (0.5 = random, 1.0 = perfect separation)")

print(f"\n{'='*64}")
print("Q51 HARDENING COMPLETE")
