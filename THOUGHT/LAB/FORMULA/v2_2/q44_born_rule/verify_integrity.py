"""Q44 integrity check: algebraic null test."""
import numpy as np
from scipy import stats
from sentence_transformers import SentenceTransformer

concepts = ['love','hate','truth','false','peace','war','life','death','time','space','power','fear','hope','joy','pain','dream','freedom','justice','beauty','wisdom','courage','honor','science','art','music','nature','god','evil','good','bad','light','dark','strong','weak','fast','slow','hot','cold','rich','poor','young','old','new','ancient','simple','complex']
contexts = ['emotion','feeling','thought','idea','action','result','beginning','ending','knowledge','mystery','strength','weakness','order','chaos','creation','destruction','presence','absence','motion','stillness','sound','silence','growth','decay','connection','separation','unity','division','health','sickness']

model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
c_embs = model.encode(concepts, normalize_embeddings=True)
ctx_embs = model.encode(contexts, normalize_embeddings=True)

flat = (c_embs @ ctx_embs.T).flatten()
print(f'Pairs: {len(flat)}, Positive: {(flat>0).mean()*100:.1f}%, Mean: {flat.mean():.4f}')

# Real
real_sims = np.array([c_embs[i] @ ctx_embs.T for i in range(len(concepts))])
real_E, real_P = real_sims.mean(axis=1), (real_sims**2).mean(axis=1)
real_r = stats.pearsonr(real_E, real_P)[0]
print(f'Real  r(E,P_born): {real_r:.4f}')

# Null 1: random unit vectors (no semantic structure)
D = c_embs.shape[1]
null1_rs = []
for _ in range(200):
    rv = np.random.randn(len(concepts), D)
    rv /= np.linalg.norm(rv, axis=1, keepdims=True)
    sims = rv @ ctx_embs.T
    null1_rs.append(stats.pearsonr(sims.mean(axis=1), (sims**2).mean(axis=1))[0])
null1_rs = np.array(null1_rs)
print(f'Null1 r(E,P_born): {null1_rs.mean():.4f} +/- {null1_rs.std():.4f}')
print(f'  (random unit vectors, cos_sim ~ 0)')

# Null 2: algebraic — rand sample from SAME distribution, shuffled per concept
all_vals = real_sims.flatten()
null2_rs = []
for _ in range(200):
    null_E, null_P = [], []
    for i in range(len(concepts)):
        vals = np.random.choice(all_vals, size=len(contexts), replace=True)
        null_E.append(vals.mean())
        null_P.append((vals**2).mean())
    null2_rs.append(stats.pearsonr(null_E, null_P)[0])
null2_rs = np.array(null2_rs)
print(f'Null2 r(E,P_born): {null2_rs.mean():.4f} +/- {null2_rs.std():.4f}')
print(f'  (random samples from real distribution, scrambled per concept)')

# Null 3: preserve per-concept mean, scramble within
null3_rs = []
for _ in range(200):
    null_E, null_P = [], []
    for i in range(len(concepts)):
        vals = np.random.choice(all_vals, size=len(contexts), replace=True)
        null_E.append(vals.mean())
        null_P.append((vals**2).mean())
    # Same as null2, this is equivalent
# Actually null3 should preserve concept identity
null3_rs = []
for _ in range(200):
    shuffled = real_sims.copy()
    for i in range(len(concepts)):
        np.random.shuffle(shuffled[i])
    null3_rs.append(stats.pearsonr(shuffled.mean(axis=1), (shuffled**2).mean(axis=1))[0])
null3_rs = np.array(null3_rs)
print(f'Null3 r(E,P_born): {null3_rs.mean():.4f} +/- {null3_rs.std():.4f}')
print(f'  (preserve per-concept distribution, shuffle context assignment)')

# Summary
print(f'\n--- VERDICT ---')
z_vs_null1 = (real_r - null1_rs.mean()) / null1_rs.std()
z_vs_null2 = (real_r - null2_rs.mean()) / null2_rs.std()
z_vs_null3 = (real_r - null3_rs.mean()) / null3_rs.std()

v1 = "SEMANTIC (real >> random)" if z_vs_null1 > 3 else "ALGEBRAIC (real ~ random)"
v2 = "SEMANTIC" if abs(z_vs_null2) > 3 else "ALGEBRAIC"
v3 = "SEMANTIC" if abs(z_vs_null3) > 3 else "ALGEBRAIC"
print(f'vs random vectors:    z={z_vs_null1:.1f}  -> {v1}')
print(f'vs scrambled dist:    z={z_vs_null2:.1f}  -> {v2}')
print(f'vs shuffled within:   z={z_vs_null3:.1f}  -> {v3}')
