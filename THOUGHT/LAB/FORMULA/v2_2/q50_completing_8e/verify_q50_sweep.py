"""Q50: test cross-arch agreement at multiple N."""
import numpy as np
from sentence_transformers import SentenceTransformer

WORDS = ['water','fire','earth','sky','sun','moon','star','mountain','river','tree','flower','rain','wind','snow','cloud','ocean','dog','cat','bird','fish','horse','tiger','lion','elephant','heart','eye','hand','head','brain','blood','bone','mother','father','child','friend','king','queen','love','hate','truth','life','death','time','space','power','peace','war','hope','fear','joy','pain','dream','thought','book','door','house','road','food','money','stone','gold','light','shadow','music','word','name','law','good','bad','big','small','old','new','high','low','hot','cold','dark','bright','strong','weak','fast','slow','hard','soft','deep','wide','long','short','rich','poor','free','safe','clean','fair','kind','cruel','brave','wise','young','old','true','false','happy','sad','queen','king','prince','castle','knight','sword','shield','dragon','magic','wizard','witch','giant','hero','monster','ghost','spirit','angel','demon','god','brain','mind','soul','foot','body','force','energy','mass','speed','matter','atom','cell','north','south','east','west']
np.random.seed(42)

def product(embs):
    centered = embs - embs.mean(axis=0)
    cov = np.cov(centered.T)
    ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
    ev = np.maximum(ev, 1e-10)
    evnz = ev[ev > 1e-10]
    df = evnz.sum()**2 / (evnz**2).sum()
    k = np.arange(1, len(evnz)+1)
    h = len(evnz)//2
    slope, _ = np.polyfit(np.log(k[:h]), np.log(evnz[:h]), 1)
    return df * (-slope)

models = {}
for mid in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'multi-qa-MiniLM-L6-cos-v1', 'paraphrase-MiniLM-L6-v2']:
    m = SentenceTransformer(mid, device='cpu')
    embs = m.encode(WORDS, normalize_embeddings=True)
    models[mid] = embs

print(f'{"N":>5s} {"mean":>8s} {"cv":>6s} {"span":>7s}  values')
for N in [30, 50, 75, 100, 130]:
    np.random.seed(N)
    idxs = np.random.choice(len(WORDS), N, replace=False)
    vals = []
    for mid in sorted(models.keys()):
        vals.append(product(models[mid][idxs]))
    vals = np.array(vals)
    cv = np.std(vals) / np.mean(vals) * 100
    span = vals.max() - vals.min()
    vstr = '  '.join([f'{v:.2f}' for v in vals])
    print(f'{N:5d} {vals.mean():8.2f} {cv:5.1f}% {span:7.2f}  [{vstr}]')
