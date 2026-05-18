"""Q50: cross-architecture conservation test."""
import numpy as np
from scipy.linalg import eigh
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel

WORDS = ['water','fire','earth','sky','sun','moon','star','mountain','river','tree','flower','rain','wind','snow','cloud','ocean','dog','cat','bird','fish','horse','tiger','lion','elephant','heart','eye','hand','head','brain','blood','bone','mother','father','child','friend','king','queen','love','hate','truth','life','death','time','space','power','peace','war','hope','fear','joy','pain','dream','thought','book','door','house','road','food','money','stone','gold','light','shadow','music','word','name','law','good','bad','big','small','old','new','high','low']
E8 = 8 * np.e

def compute_product(embs):
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.maximum(ev, 1e-10)
    evnz = ev[ev > 1e-10]
    Df = evnz.sum()**2 / (evnz**2).sum()
    k = np.arange(1, len(evnz)+1)
    h = len(evnz)//2
    slope, _ = np.polyfit(np.log(k[:h]), np.log(evnz[:h]), 1)
    alpha = -slope
    return Df * alpha, Df, alpha

results = []
N = len(WORDS)

# BERT token embeddings
tok = AutoTokenizer.from_pretrained('bert-base-uncased')
mod = AutoModel.from_pretrained('bert-base-uncased')
wgt = mod.get_input_embeddings().weight.detach().numpy()
embs_bert = []
for w in WORDS:
    ids = tok.encode(w, add_special_tokens=False)
    if len(ids) == 1 and ids[0] < len(wgt):
        embs_bert.append(wgt[ids[0]])
embs_bert = np.array(embs_bert)
embs_bert /= np.linalg.norm(embs_bert, axis=1, keepdims=True)
p, d, a = compute_product(embs_bert)
results.append(('bert-base-uncased (token)', embs_bert.shape[1], d, a, p))

# Sentence-transformers
for mid in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-MiniLM-L6-v2']:
    model = SentenceTransformer(mid, device='cpu')
    embs = model.encode(WORDS, normalize_embeddings=True)
    D = model.get_sentence_embedding_dimension()
    p, d, a = compute_product(embs)
    results.append((mid, D, d, a, p))

print(f'Vocabulary size N = {N}')
print(f'Target: 8e = {E8:.4f}')
print()
print(f'{"Model":<35s} {"D":>5s} {"Df":>8s} {"alpha":>8s} {"product":>10s} {"delta 8e":>10s}')
print('-' * 80)
for name, dim, df, alpha, prod in results:
    print(f'{name:<35s} {dim:>5d} {df:>8.2f} {alpha:>8.4f} {prod:>10.4f} {prod-E8:>+10.4f}')

prods = [r[4] for r in results]
mp, sp = np.mean(prods), np.std(prods)
print(f'\nMean: {mp:.4f}  Std: {sp:.4f}  CV: {sp/mp*100:.1f}%')
print(f'Min: {min(prods):.4f}  Max: {max(prods):.4f}  Range: {max(prods)-min(prods):.4f}')
print(f'Models agree? {"YES (CV<10%)" if sp/mp*100 < 10 else "NO (CV>=10%)"}')
