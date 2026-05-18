"""Q45: Combined metric sweep — cosine + phase for optimal navigation."""
import sys, numpy as np
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

    print(f"\n{name} K=96:")
    print(f"{'beta':>8s} {'cosine_rank':>14s} {'combined_rank':>14s} {'opp_sep':>10s}")
    print(f"{'='*50}")

    for beta in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0]:
        combined = cos_sim - beta * phase_diff

        rel_ranks = []
        for qword, rwords in SEMANTIC_GROUPS.items():
            if qword not in WORDS: continue
            qi = WORDS.index(qword)
            sims = combined[qi].copy(); sims[qi] = -np.inf
            for rw in rwords:
                if rw not in WORDS: continue
                ri = WORDS.index(rw)
                rel_ranks.append(np.sum(sims > sims[ri]))

        opp_ranks = []
        for a, b in OPPOSITE_PAIRS:
            if a not in WORDS or b not in WORDS: continue
            ai, bi = WORDS.index(a), WORDS.index(b)
            sims = combined[ai].copy(); sims[ai] = -np.inf
            opp_ranks.append(np.sum(sims > sims[bi]))

        mr = np.mean(rel_ranks); mo = np.mean(opp_ranks)
        marker = ""
        if beta == 0.0:
            cosine_rel = mr; cosine_opp = mo
        else:
            delta_rel = mr - cosine_rel
            delta_opp = mo - cosine_opp
            marker = f"  rel:{delta_rel:+.0f} opp:{delta_opp:+.0f}"
        print(f"{beta:8.1f} {mr:14.1f} {mr:14.1f} {mo:10.1f}{marker}")
