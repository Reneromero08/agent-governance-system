"""Feral Resident geodesics — paper walks as semantic paths for the Core.

Feral DB: 8904 vectors (384-dim), 4381 semantic edges, paper sequences.
Each paper paragraph chain is a geodesic through semantic space.
The Core learns to follow these geodesics — predict next vector in sequence.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time, struct
import numpy as np, sqlite3
torch.manual_seed(42); random.seed(42)
import sys
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen')
from native_eigen_core import NativeEigenCore

# ---- Load Feral DB ----
DB = r'THOUGHT/LAB/FERAL_RESIDENT/data/db/feral_eternal.db'
conn = sqlite3.connect(DB)

# Load vectors as sequences
seqs = conn.execute("""
    SELECT sequence_id, vector_id, vec_blob, sequence_idx
    FROM vectors WHERE sequence_id IS NOT NULL
    ORDER BY sequence_id, sequence_idx
""").fetchall()

# Group by sequence
from collections import defaultdict
sequences = defaultdict(list)
for seq_id, vec_id, blob, idx in seqs:
    vec = np.frombuffer(blob, dtype=np.float32)
    sequences[seq_id].append((vec_id, vec, idx))

# Filter to sequences with >= 8 vectors
walks = []
for seq_id, items in sequences.items():
    items.sort(key=lambda x: x[2])
    vecs = [np.array(v) for _, v, _ in items]
    if len(vecs) >= 8:
        # Slide windows of 8
        for i in range(0, len(vecs) - 8, 4):
            window = vecs[i:i+9]
            if len(window) == 9:
                walks.append((window[:8], window[1:]))

# Also add e_edge walks (jump between related vectors)
edges = conn.execute("""
    SELECT a.vec_blob, a.sequence_id, b.vec_blob, e_score
    FROM e_edges e
    JOIN vectors a ON e.vector_id_a = a.vector_id
    JOIN vectors b ON e.vector_id_b = b.vector_id
    WHERE e_score > 0.8
    LIMIT 1000
""").fetchall()

for a_blob, a_seq, b_blob, score in edges:
    a_vec = np.frombuffer(a_blob, dtype=np.float32)
    b_vec = np.frombuffer(b_blob, dtype=np.float32)
    # Simple edge walk: a -> b
    walks.append((np.stack([a_vec, a_vec, a_vec, a_vec, a_vec, a_vec, a_vec, a_vec]),
                  np.stack([b_vec, b_vec, b_vec, b_vec, b_vec, b_vec, b_vec, b_vec])))
conn.close()

print(f"Geodesic walks: {len(walks)} (from {len(sequences)} sequences + edges)")

D_emb = 384
D = D_emb // 2  # 192 complex dims
walks = walks[:2000]  # cap for speed

# ---- Build training data ----
data = []
for inp, tgt in walks:
    inp_v = np.array(inp, dtype=np.float32)
    tgt_v = np.array(tgt, dtype=np.float32)
    z = torch.complex(torch.tensor(inp_v[:, :D]), torch.tensor(inp_v[:, D:]))
    tr = torch.tensor(tgt_v[:, :D])
    ti = torch.tensor(tgt_v[:, D:])
    data.append((z, tr, ti))

n_train = len(data) * 3 // 4
tr, te = data[:n_train], data[n_train:]

# ---- Iterative model ----
class IterativeCore(nn.Module):
    def __init__(self, d=16, heads=4, layers=2, max_cycles=4):
        super().__init__()
        self.max_cycles = max_cycles
        self.core = NativeEigenCore(d=d, heads=heads, layers=layers, merge='concat', geo_init=True)
        self.stop_threshold = nn.Parameter(torch.tensor(0.5))
    def forward(self, z):
        z_init = z; z_current = z
        for _ in range(self.max_cycles):
            z_next, pc = self.core(z_current + z_init)
            z_current = z_next
            if pc > self.stop_threshold: break
        return z_current

print("=" * 55)
print("FERAL GEODESICS: paper walks through semantic space")
print("=" * 55)

for d, h, L, cycles in [(16,4,4,4), (32,4,4,4), (64,8,4,4)]:
    t0 = time.time()
    in_r = nn.Linear(D, d, bias=False); in_i = nn.Linear(D, d, bias=False)
    nn.init.normal_(in_r.weight, std=0.02); nn.init.normal_(in_i.weight, std=0.02)
    iterative = IterativeCore(d=d, heads=h, layers=L, max_cycles=cycles)
    head_r = nn.Linear(d, D, bias=False); head_i = nn.Linear(d, D, bias=False)
    nn.init.normal_(head_r.weight, std=0.02); nn.init.normal_(head_i.weight, std=0.02)
    all_p = list(in_r.parameters()) + list(in_i.parameters()) + list(iterative.parameters()) + list(head_r.parameters()) + list(head_i.parameters())
    opt = torch.optim.AdamW(all_p, lr=1e-3)
    P = sum(p.numel() for p in all_p)

    for ep in range(15):
        for i in range(0, len(tr), 8):
            b = tr[i:i+8]
            if not b: continue
            x = torch.stack([p[0] for p in b])
            yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
            zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
            zo = iterative(zp)
            pr = head_r(zo.real)+head_i(zo.imag); pi = head_r(zo.imag)-head_i(zo.real)
            loss = F.mse_loss(pr, yr) + F.mse_loss(pi, yi)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0); opt.step()

    saved = [l['phase'].ang.data.clone() for l in iterative.core.layers]
    results = {}
    for mode in ['normal','ablated']:
        if mode == 'ablated':
            for l in iterative.core.layers: l['phase'].ang.data.zero_()
        ls = n = 0
        iterative.eval(); head_r.eval(); head_i.eval()
        with torch.no_grad():
            for i in range(0, len(te), 8):
                b = te[i:i+8]
                if not b: continue
                x = torch.stack([p[0] for p in b])
                yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
                zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
                zo = iterative(zp)
                pr = head_r(zo.real)+head_i(zo.imag); pi = head_r(zo.imag)-head_i(zo.real)
                ls += (F.mse_loss(pr, yr) + F.mse_loss(pi, yi)).item(); n += 1
        results[mode] = ls / max(n, 1)
    for l,s in zip(iterative.core.layers, saved): l['phase'].ang.data.copy_(s)
    delta = (results['ablated']-results['normal'])/(results['normal']+1e-8)*100
    compr = D / d
    print(f"d={d:>2} h={h} L={L} cyc={cycles} compr={compr:.0f}x P={P:>7,} norm={results['normal']:.4f} abl={results['ablated']:.4f} delta={delta:+.1f}% time={time.time()-t0:.0f}s")
    print(f"  {'PHASE CARRIES' if delta>10 else 'WEAK' if delta>3 else 'NOT LOAD-BEARING'}")
