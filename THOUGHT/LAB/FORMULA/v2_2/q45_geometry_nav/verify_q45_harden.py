"""Q45 hardening: test Hermitian, phase, imaginary, and ensemble metrics."""
import sys
import numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0, "THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS = list(ANCHOR_1024)

SEMANTIC_GROUPS = {
    "dog": ["cat","puppy","bark","pet","animal","tail","bone","wolf"],
    "king": ["queen","prince","castle","crown","rule","throne","royal","palace"],
    "water": ["river","ocean","sea","lake","rain","wave","stream","drink"],
    "fire": ["flame","burn","hot","heat","smoke","ash","blaze","spark"],
    "love": ["heart","romance","passion","kiss","emotion","desire","caring","devotion"],
    "war": ["battle","fight","peace","army","weapon","combat","soldier","conflict"],
    "book": ["page","read","story","author","chapter","novel","write","library"],
    "tree": ["forest","leaf","wood","branch","root","plant","bark","grow"],
    "car": ["vehicle","drive","wheel","engine","road","truck","speed","motor"],
    "night": ["dark","moon","star","sleep","dream","evening","midnight","shadow"],
}

OPPOSITE_PAIRS = [
    ("love","hate"), ("hot","cold"), ("good","bad"), ("light","dark"),
    ("fast","slow"), ("rich","poor"), ("young","old"), ("strong","weak"),
    ("happy","sad"), ("brave","afraid"), ("peace","war"), ("life","death"),
]

def rank_words(query_idx, vectors, metric, exclude_idxs=None):
    if exclude_idxs is None: exclude_idxs = []
    n = len(vectors)
    if metric == "cosine":
        sims = vectors @ vectors[query_idx]
    elif metric == "hermitian":
        sims = np.real(vectors @ np.conj(vectors[query_idx]))
    elif metric == "phase":
        overlap = vectors @ np.conj(vectors[query_idx])
        sims = -np.abs(np.angle(overlap))
    elif metric == "imaginary":
        sims = np.abs(np.imag(vectors @ np.conj(vectors[query_idx])))
    elif metric == "ensemble":
        o = vectors @ np.conj(vectors[query_idx])
        cos = vectors.real @ vectors[query_idx].real
        sims = 0.5 * cos + 0.3 * np.real(o) + 0.2 * (-np.abs(np.angle(o)))
    else:
        return np.inf
    sims[query_idx] = -np.inf
    for ei in exclude_idxs:
        if 0 <= ei < len(sims): sims[ei] = -np.inf
    return sims

for mid, name in [("all-MiniLM-L6-v2", "MiniLM"), ("all-mpnet-base-v2", "MPNet")]:
    m = SentenceTransformer(mid, device="cpu")
    embs = m.encode(WORDS, normalize_embeddings=True)
    D = m.get_sentence_embedding_dimension()

    print(f"\n{'='*64}")
    print(f"{name}")
    print(f"{'='*64}")

    for K in [96, D]:
        centered = embs - embs.mean(axis=0)
        cov = np.cov(centered.T)
        evals, evecs = np.linalg.eigh(cov)
        idx = np.argsort(evals)[::-1]; evecs = evecs[:, idx]
        proj = centered @ evecs[:, :K]
        norms = np.linalg.norm(proj, axis=1, keepdims=True); norms[norms==0]=1
        proj = proj / norms
        z = hilbert(proj, axis=0).astype(np.complex128)
        zn = np.sqrt(np.sum(np.abs(z)**2, axis=1, keepdims=True)); z = z / (zn + 1e-12)

        results = {}
        for metric in ["cosine", "hermitian", "phase", "imaginary", "ensemble"]:
            ranks = []
            for qword, rwords in SEMANTIC_GROUPS.items():
                if qword not in WORDS: continue
                qi = WORDS.index(qword)
                sims = rank_words(qi, z if metric != "cosine" else proj, metric)
                for rw in rwords:
                    if rw not in WORDS: continue
                    ri = WORDS.index(rw)
                    rank = np.sum(sims > sims[ri])
                    ranks.append(rank)
            if ranks:
                results[metric] = np.mean(ranks)

        # OPPOSITE PAIRS: cosine should rank antonyms NEAR (cos_sim negative)
        # Hermitian should ALSO rank them near (cos_sim preserved)
        # FS |overlap| should rank them FAR (abs destroys sign)
        opp_ranks = {}
        for metric in ["cosine", "hermitian", "phase"]:
            ranks = []
            for a, b in OPPOSITE_PAIRS:
                if a not in WORDS or b not in WORDS: continue
                ai, bi = WORDS.index(a), WORDS.index(b)
                sims = rank_words(ai, z if metric != "cosine" else proj, metric, [ai])
                rank = np.sum(sims > sims[bi])
                ranks.append(rank)
            if ranks:
                opp_ranks[metric] = np.mean(ranks)

        best_metric = min(results, key=results.get)
        print(f"  K={K:3d}: related_ranks: {', '.join(f'{m}={results[m]:.1f}' for m in sorted(results, key=results.get))}")
        print(f"         Best: {best_metric} ({results[best_metric]:.1f})")
        if opp_ranks:
            print(f"         opposite_ranks: {', '.join(f'{m}={opp_ranks[m]:.1f}' for m in sorted(opp_ranks, key=opp_ranks.get))}")
            # Opposite word ranking: HIGHER rank = the metric puts antonyms farther apart (better)
            best_opp = max(opp_ranks, key=opp_ranks.get)
            print(f"         Best opp separator: {best_opp} ({opp_ranks[best_opp]:.1f})")

print(f"\n{'='*64}")
print("VERDICT:")
print("  Related words: lower rank = better (metric ranks related words higher)")
print("  Opposite words: higher rank = better (metric pushes opposites apart)")
