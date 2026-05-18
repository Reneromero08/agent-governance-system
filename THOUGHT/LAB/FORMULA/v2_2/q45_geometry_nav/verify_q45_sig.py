"""Q45 significance test: paired ranks for combined metric vs cosine."""
import sys, numpy as np
from scipy import stats
from scipy.signal import hilbert
from sentence_transformers import SentenceTransformer
sys.path.insert(0,"THOUGHT/LAB/EIGEN_ALIGNMENT/vector-communication/lib")
from large_anchor_generator import ANCHOR_1024
WORDS=list(ANCHOR_1024)

SEMANTIC_GROUPS = {
    "dog":["cat","puppy","bark","pet","animal","tail","bone","wolf"],
    "king":["queen","prince","castle","crown","rule","throne","royal","palace"],
    "water":["river","ocean","sea","lake","rain","wave","stream","drink"],
    "fire":["flame","burn","hot","heat","smoke","ash","blaze","spark"],
    "love":["heart","romance","passion","kiss","emotion","desire","caring","devotion"],
    "war":["battle","fight","peace","army","weapon","combat","soldier","conflict"],
    "book":["page","read","story","author","chapter","novel","write","library"],
    "tree":["forest","leaf","wood","branch","root","plant","bark","grow"],
}
OPPOSITE_PAIRS = [
    ("love","hate"),("hot","cold"),("good","bad"),("light","dark"),
    ("fast","slow"),("rich","poor"),("young","old"),("strong","weak"),
    ("happy","sad"),("war","peace"),("life","death"),("brave","afraid"),
]

for mid,name in [("all-MiniLM-L6-v2","MiniLM"),("all-mpnet-base-v2","MPNet")]:
    m=SentenceTransformer(mid,device="cpu")
    embs=m.encode(WORDS,normalize_embeddings=True)
    D=m.get_sentence_embedding_dimension()
    centered=embs-embs.mean(axis=0)
    cov=np.cov(centered.T)
    evals,evecs=np.linalg.eigh(cov)
    idx=np.argsort(evals)[::-1];evecs=evecs[:,idx]
    proj=(centered@evecs[:,:96])
    norms=np.linalg.norm(proj,axis=1,keepdims=True);norms[norms==0]=1;proj=proj/norms
    z=hilbert(proj,axis=0).astype(np.complex128)
    zn=np.sqrt(np.sum(np.abs(z)**2,axis=1,keepdims=True));z=z/(zn+1e-12)

    overlap=z@np.conj(z).T
    phase_diff=np.abs(np.angle(overlap))
    cos_sim=proj@proj.T

    print(f"\n{'='*64}")
    print(f"{name} K=96 — Paired significance tests")
    print(f"{'='*64}")

    for beta in [0.1, 0.3, 1.0, 2.0]:
        combined = cos_sim - beta * phase_diff

        # Related word ranks: cosine vs combined (paired per word)
        rel_diffs = []
        for qword, rwords in SEMANTIC_GROUPS.items():
            if qword not in WORDS: continue
            qi = WORDS.index(qword)
            cos_sims = cos_sim[qi].copy(); cos_sims[qi] = -np.inf
            com_sims = combined[qi].copy(); com_sims[qi] = -np.inf
            for rw in rwords:
                if rw not in WORDS: continue
                ri = WORDS.index(rw)
                rel_diffs.append(np.sum(com_sims > com_sims[ri]) - np.sum(cos_sims > cos_sims[ri]))
        rel_diffs = np.array(rel_diffs)
        t_rel, p_rel = stats.ttest_1samp(rel_diffs, 0)

        # Opposite word ranks: cosine vs combined
        opp_diffs = []
        for a, b in OPPOSITE_PAIRS:
            if a not in WORDS or b not in WORDS: continue
            ai, bi = WORDS.index(a), WORDS.index(b)
            cos_sims = cos_sim[ai].copy(); cos_sims[ai] = -np.inf
            com_sims = combined[ai].copy(); com_sims[ai] = -np.inf
            opp_diffs.append(np.sum(com_sims > com_sims[bi]) - np.sum(cos_sims > cos_sims[bi]))
        opp_diffs = np.array(opp_diffs)
        t_opp, p_opp = stats.ttest_1samp(opp_diffs, 0)

        sig_rel = "SIG (p<0.05)" if p_rel < 0.05 else "ns"
        sig_opp = "SIG (p<0.05)" if p_opp < 0.05 else "ns"
        print(f"  beta={beta:.1f}: rel_delta={rel_diffs.mean():+.1f} ranks (SE={rel_diffs.std()/np.sqrt(len(rel_diffs)):.1f}) t={t_rel:.2f} p={p_rel:.4f} {sig_rel}")
        print(f"            opp_delta={opp_diffs.mean():+.1f} ranks (SE={opp_diffs.std()/np.sqrt(len(opp_diffs)):.1f}) t={t_opp:.2f} p={p_opp:.4f} {sig_opp}")

    # Overall: does combining cosine + phase beat cosine alone for antonym detection?
    print(f"\n  Antonym detection (beta=1.0):")
    for a, b in OPPOSITE_PAIRS:
        if a not in WORDS or b not in WORDS: continue
        ai, bi = WORDS.index(a), WORDS.index(b)
        cos_rank = np.sum(cos_sim[ai] > cos_sim[ai][bi])
        com_rank = np.sum((cos_sim - 1.0 * phase_diff)[ai] > (cos_sim - 1.0 * phase_diff)[ai][bi])
        print(f"    {a:>8s}-{b:<8s}: cos_rank={cos_rank:4d}  combined_rank={com_rank:4d}")
