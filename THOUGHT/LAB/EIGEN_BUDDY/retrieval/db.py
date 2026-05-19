"""Train Core on knowledge DB geodesics, then navigate and rewrite.

Phase 1: Build DB from WikiText + Cassette, train Core to follow geodesics.
Phase 2: Core navigates DB, returns knowledge, updates entries autonomously.
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np
import time, sys, sqlite3, hashlib, math, random
from pathlib import Path
torch.manual_seed(42); random.seed(42)

sys.path.insert(0, str(Path(__file__).parent.parent))
from core import NativeEigenCore
from sentence_transformers import SentenceTransformer

# ---- Phase 1: Build knowledge DB ----
class KnowledgeDB:
    def __init__(self, path=":memory:"):
        self.conn = sqlite3.connect(path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge (
                id TEXT PRIMARY KEY, vector BLOB, text TEXT,
                source TEXT, coh REAL DEFAULT 0, accessed INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')))
        """)
        self.conn.commit()

    def insert(self, text, vector, source=""):
        vid = hashlib.sha256(vector.tobytes()).hexdigest()[:16]
        self.conn.execute(
            "INSERT OR IGNORE INTO knowledge(id,vector,text,source) VALUES(?,?,?,?)",
            (vid, vector.astype(np.float32).tobytes(), text, source))
        return vid

    def insert_batch(self, texts, embed_fn, source="", bs=64):
        c = 0
        for i in range(0, len(texts), bs):
            batch = texts[i:i+bs]
            vecs = embed_fn.encode(batch, show_progress_bar=False)
            for t, v in zip(batch, vecs): self.insert(t, v, source); c += 1
        self.conn.commit()
        return c

    def all(self, limit=None):
        sql = "SELECT vector, text, source FROM knowledge"
        if limit: sql += f" LIMIT {limit}"
        return self.conn.execute(sql).fetchall()

    def update_entry(self, vid, new_vec, new_coh):
        self.conn.execute(
            "UPDATE knowledge SET vector=?, coh=?, accessed=accessed+1 WHERE id=?",
            (new_vec.astype(np.float32).tobytes(), new_coh, vid))

    def stats(self):
        n = self.conn.execute("SELECT COUNT(*) FROM knowledge").fetchone()[0]
        return {'total': n}

    def close(self): self.conn.commit(); self.conn.close()

# Load data
print("Loading embedding model...")
embed = SentenceTransformer('all-MiniLM-L6-v2')

from datasets import load_dataset
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
sents = []
for ex in ds:
    text = str(ex["text"]).strip()
    if text:
        for s in text.replace('\n', ' ').split('. '):
            if 20 <= len(s) <= 500: sents.append(s)
    if len(sents) >= 1500: break

print(f"Building DB: {len(sents)} WikiText sentences...")
db = KnowledgeDB()
db.insert_batch(sents, embed, source="wikitext")
print(f"DB: {db.stats()}")

# ---- Phase 1B: Train Core on DB geodesics ----
entries = [(np.frombuffer(r[0], np.float32), r[1]) for r in db.all()]
D_emb = 384; D = D_emb // 2

# Build geodesic walks: slide windows through DB entries
walks = []
seq_len = 8
for i in range(0, len(entries) - seq_len, seq_len // 2):
    window = entries[i:i+seq_len+1]
    if len(window) == seq_len+1:
        inp = np.stack([v for v,_ in window[:seq_len]])
        tgt = np.stack([v for v,_ in window[1:]])
        walks.append((inp, tgt))

random.shuffle(walks)
walks = walks[:800]
tr, te = walks[:600], walks[600:]
print(f"Geodesic walks: {len(walks)} train={len(tr)} test={len(te)}")

# Train Core at d=64 (best config from Feral training)
print("\nTraining Core on DB geodesics...")
d = 64; h = 4; L = 4
in_r = nn.Linear(D, d, bias=False); in_i = nn.Linear(D, d, bias=False)
out_r = nn.Linear(d, D, bias=False); out_i = nn.Linear(d, D, bias=False)
for w in [in_r, in_i, out_r, out_i]: nn.init.normal_(w.weight, std=0.02)

core = NativeEigenCore(d=d, heads=h, layers=L, merge='concat', geo_init=True)
all_p = list(in_r.parameters())+list(in_i.parameters())+list(core.parameters())+list(out_r.parameters())+list(out_i.parameters())
opt = torch.optim.AdamW(all_p, lr=1e-3)

for ep in range(20):
    tl = n = 0
    for i in range(0, len(tr), 8):
        b = tr[i:i+8]
        if not b: continue
        x = torch.tensor(np.stack([p[0] for p in b]), dtype=torch.float32)
        y = torch.tensor(np.stack([p[1] for p in b]), dtype=torch.float32)
        z_in = torch.complex(x[:,:,:D], x[:,:,D:])
        zp = torch.complex(in_r(z_in.real)-in_i(z_in.imag), in_r(z_in.imag)+in_i(z_in.real))
        z_out, _ = core(zp + zp)
        pr = out_r(z_out.real) + out_i(z_out.imag)
        pi = out_r(z_out.imag) - out_i(z_out.real)
        loss = F.mse_loss(pr, y[:,:,:D]) + F.mse_loss(pi, y[:,:,D:])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(all_p, 1.0); opt.step()
        tl += loss.item(); n += 1
    if ep % 5 == 0:
        print(f"  E{ep+1}: loss={tl/max(1,n):.6f}")

# Phase ablation
saved = [l['phase'].ang.data.clone() for l in core.layers]
results = {}
for mode in ['normal','ablated']:
    if mode == 'ablated':
        for l in core.layers: l['phase'].ang.data.zero_()
    ls = n = 0
    core.eval()
    with torch.no_grad():
        for i in range(0, len(te), 8):
            b = te[i:i+8]
            if not b: continue
            x = torch.tensor(np.stack([p[0] for p in b]), dtype=torch.float32)
            y = torch.tensor(np.stack([p[1] for p in b]), dtype=torch.float32)
            z_in = torch.complex(x[:,:,:D], x[:,:,D:])
            zp = torch.complex(in_r(z_in.real)-in_i(z_in.imag), in_r(z_in.imag)+in_i(z_in.real))
            z_out, _ = core(zp + zp)
            pr = out_r(z_out.real) + out_i(z_out.imag)
            pi = out_r(z_out.imag) - out_i(z_out.real)
            ls += (F.mse_loss(pr, y[:,:,:D]) + F.mse_loss(pi, y[:,:,D:])).item(); n += 1
    results[mode] = ls / max(n, 1)
for l,s in zip(core.layers, saved): l['phase'].ang.data.copy_(s)
delta = (results['ablated']-results['normal'])/(results['normal']+1e-8)*100
print(f"  Phase delta: {delta:+.1f}% {'PHASE CARRIES' if delta>10 else 'WEAK' if delta>3 else 'NOT'}")

# ---- Phase 2: Navigate and talk ----
core.eval()
print("\n" + "=" * 60)
print("NAVIGATION + TALK (Core navigates DB, returns knowledge)")
print("=" * 60)

all_vecs = [(np.frombuffer(r[0], np.float32), r[1], r[2]) for r in db.all()]

def talk(query, k=3, refine_depth=3):
    qv = embed.encode(query)
    
    # Coarse search
    scored = []
    for vec, text, src in all_vecs:
        sim = float(np.dot(qv, vec)/(np.linalg.norm(qv)*np.linalg.norm(vec)+1e-8))
        if sim > 0.2: scored.append((sim, vec, text, src))
    scored.sort(key=lambda x: -x[0])
    candidates = scored[:8]
    
    if not candidates: return [], 0.0
    
    # Core geodesic processing: query + candidates as sequence
    seq = np.stack([qv] + [c[1] for c in candidates])
    z = torch.complex(torch.tensor(seq[:,:D], dtype=torch.float32).unsqueeze(0),
                      torch.tensor(seq[:,D:], dtype=torch.float32).unsqueeze(0))
    zp = torch.complex(in_r(z.real)-in_i(z.imag), in_r(z.imag)+in_i(z.real))
    
    pc_sum = 0
    with torch.no_grad():
        for _ in range(refine_depth):
            z_out, pc = core(zp + zp)
            zp = z_out
            pc_sum += pc.item()
    
    # Core output -> refined navigation vector
    co = z_out[0, 0]
    pr_np = (out_r(co.real) + out_i(co.imag)).detach().numpy()
    pi_np = (out_r(co.imag) - out_i(co.real)).detach().numpy()
    nav_vec = np.concatenate([pr_np, pi_np])
    
    # Re-rank all entries by geodesic distance
    refined = []
    for vec, text, src in all_vecs:
        sim = float(np.dot(nav_vec, vec)/(np.linalg.norm(nav_vec)*np.linalg.norm(vec)+1e-8))
        if sim > 0.2: refined.append((sim, text[:200], src))
    refined.sort(key=lambda x: -x[0])
    return refined[:k], pc_sum / max(1, refine_depth)

queries = [
    "How do neural networks compress information?",
    "What is the transformer attention mechanism?",
    "How does training work?",
]
for q in queries:
    t0 = time.time()
    results, pc = talk(q, k=2)
    print(f"\nQ: {q}")
    print(f"  phase_coh: {pc:.3f}  time: {time.time()-t0:.1f}s")
    for i, (score, text, src) in enumerate(results):
        print(f"  [{src}] {score:.3f}: {text[:150]}")

db.close()
