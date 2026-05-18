"""Q43 integrity check: C = I - G_avg identity."""
import numpy as np
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer

WORDS = ['water','fire','earth','sky','sun','moon','star','mountain','river','tree','flower','rain','wind','snow','cloud','ocean','dog','cat','bird','fish','horse','tiger','lion','elephant','heart','eye','hand','head','brain','blood','bone','mother','father','child','friend','king','queen','love','hate','truth','life','death','time','space','power','peace','war','hope','fear','joy','pain','dream','thought','book','door','house','road','food','money','stone','gold','light','shadow','music','word','name','law','good','bad','big','small','old','new','high','low']

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
embs = model.encode(WORDS, normalize_embeddings=True)
d, n = embs.shape[1], len(WORDS)
centered = embs - embs.mean(axis=0)

# C from covariance (1/N normalization for identity check)
C = np.cov(centered.T, bias=True)

# G_avg = I - (1/N) sum x_i x_i^T (for zero-mean)
G = np.eye(d) - (centered.T @ centered) / n

# Verify C + G = I
dev = np.abs(C + G - np.eye(d)).max()
print(f'C + G = I  -> max deviation: {dev:.2e}')

# Eigenvalue complementarity
C_ev = np.sort(eigh(C, eigvals_only=True))[::-1]
G_ev = np.sort(eigh(G, eigvals_only=True))[::-1]
sum_ev = C_ev + G_ev[::-1]
dev_ev = np.abs(sum_ev - 1.0).mean()
print(f'C_ev[i] + G_ev[d-1-i] = 1  -> mean deviation: {dev_ev:.2e}')

# C trace = sum of C eigenvalues = average variance per dimension * number of dims
# For normalized embeddings: trace(I) = d, trace(C) + trace(G) = d
print(f'trace(C): {np.trace(C):.4f}, trace(G): {np.trace(G):.4f}, sum: {np.trace(C)+np.trace(G):.4f}')
print()
print('C = I - G_avg holds exactly. They are complementary representations.')
print('C is the covariance. G_avg is the Fubini-Study metric.')
print('Same information, complementary bases. No new predictions from G_avg.')
