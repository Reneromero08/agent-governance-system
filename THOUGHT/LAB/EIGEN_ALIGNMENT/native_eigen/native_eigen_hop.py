"""Phase hopping: concept network -> context bridge -> paper sequence.

Feral DB has two networks:
1. Concept vectors (5417, no sequence_id) connected by 4381 e_edges
2. Paper vectors (3487, with sequence_id) chained by prev/next pointers
Bridge: each paper vector has a context_vec_blob matching concept vectors.

Test: can phase survive the hop across these structural boundaries?
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
import numpy as np, sqlite3
torch.manual_seed(42); random.seed(42)
import sys
sys.path.insert(0, r'THOUGHT/LAB/EIGEN_ALIGNMENT/native_eigen')
from native_eigen_core import NativeEigenCore

DB = r'THOUGHT/LAB/FERAL_RESIDENT/data/db/feral_eternal.db'
conn = sqlite3.connect(DB)
D_emb = 384; D = D_emb // 2

# ---- Build concept walks via e_edges ----
# Each edge connects two concept vectors. Chain them into walks.
edges_raw = conn.execute("""
    SELECT a.vec_blob, b.vec_blob, e_score
    FROM e_edges e
    JOIN vectors a ON e.vector_id_a = a.vector_id
    JOIN vectors b ON e.vector_id_b = b.vector_id
    WHERE e_score > 0.7
    LIMIT 2000
""").fetchall()

# Build adjacency for concept walks
import random as rnd
concept_walks = []
edges_list = [(np.frombuffer(a, np.float32), np.frombuffer(b, np.float32), s) for a,b,s in edges_raw]
# Create short concept walks by chaining edges with high similarity
for _ in range(500):
    walk = []
    e = rnd.choice(edges_list)
    walk.append(e[0]); walk.append(e[1])
    # Try to extend
    for _ in range(6):
        candidates = [x for x in edges_list if np.dot(walk[-1], x[0]) > 0.85]
        if candidates:
            nxt = rnd.choice(candidates)
            walk.append(nxt[1])
        else:
            break
    if len(walk) == 9:
        walk_arr = np.stack(walk)
        concept_walks.append((walk_arr[:8], walk_arr[1:9]))
    elif len(walk) == 8:
        walk_arr = np.stack(walk)
        concept_walks.append((walk_arr, walk_arr))  # autoencoder

print(f"Concept walks: {len(concept_walks)}")

# ---- Build paper walks (within-sequence) ----
paper_walks = []
seqs = conn.execute("""
    SELECT sequence_id, vec_blob, sequence_idx
    FROM vectors WHERE sequence_id IS NOT NULL
    ORDER BY sequence_id, sequence_idx
""").fetchall()

from collections import defaultdict
sequences = defaultdict(list)
for seq_id, blob, idx in seqs:
    sequences[seq_id].append(np.frombuffer(blob, np.float32))

for seq_id, vecs in sequences.items():
    if len(vecs) >= 9:
        for i in range(0, len(vecs)-8, 4):
            w = vecs[i:i+9]
            if len(w) == 9:
                paper_walks.append((np.stack(w[:8]), np.stack(w[1:])))

print(f"Paper walks: {len(paper_walks)}")

# ---- Build HOP walks: concept -> paper transition (lower threshold) ----
paper_starts = [(seq_id, vecs[0]) for seq_id, vecs in sequences.items() if len(vecs) >= 8]
hop_walks = []
for seq_id, start_vec in paper_starts[:80]:
    concept_near = []
    for a_vec, b_vec, score in edges_list[:1000]:
        if np.dot(start_vec, a_vec) > 0.4: concept_near.append((a_vec, np.dot(start_vec, a_vec)))
        if np.dot(start_vec, b_vec) > 0.4: concept_near.append((b_vec, np.dot(start_vec, b_vec)))
    if not concept_near: continue
    concept_near.sort(key=lambda x: -x[1])
    bridge_vec = concept_near[0][0]
    walk = [bridge_vec, bridge_vec, bridge_vec, start_vec]
    vecs = sequences[seq_id]
    for j in range(4):
        if j+1 < len(vecs): walk.append(vecs[j+1])
    if len(walk) == 8:
        hop_walks.append((np.stack(walk[:8]), np.stack(walk[:8])))  # autoencoder on hops

print(f"Hop walks (concept->paper): {len(hop_walks)}")
conn.close()

# ---- Train model on mixed walks ----
data = []
for inp, tgt in concept_walks + paper_walks + hop_walks:
    inp_v = np.array(inp, np.float32); tgt_v = np.array(tgt, np.float32)
    z = torch.complex(torch.tensor(inp_v[:, :D]), torch.tensor(inp_v[:, D:]))
    tr = torch.tensor(tgt_v[:, :D]); ti = torch.tensor(tgt_v[:, D:])
    data.append((z, tr, ti))

rnd.shuffle(data)
data = data[:1500]
n_tr = len(data)*3//4
tr, te = data[:n_tr], data[n_tr:]
print(f"Total walks: {len(data)}  train={len(tr)} test={len(te)}")
print(f"Concept: {len(concept_walks)}  Paper: {len(paper_walks)}  Hop: {len(hop_walks)}")

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

print("\n" + "=" * 55)
print("PHASE HOPPING: concept network -> paper sequences")
print("=" * 55)

for d, h, L, cycles in [(32, 4, 4, 4)]:
    t0 = time.time()
    in_r = nn.Linear(D, d, bias=False); in_i = nn.Linear(D, d, bias=False)
    nn.init.normal_(in_r.weight, std=0.02); nn.init.normal_(in_i.weight, std=0.02)
    it = IterativeCore(d=d, heads=h, layers=L, max_cycles=cycles)
    hr = nn.Linear(d, D, bias=False); hi = nn.Linear(d, D, bias=False)
    nn.init.normal_(hr.weight, std=0.02); nn.init.normal_(hi.weight, std=0.02)
    all_p = list(in_r.parameters())+list(in_i.parameters())+list(it.parameters())+list(hr.parameters())+list(hi.parameters())
    opt = torch.optim.AdamW(all_p, lr=1e-3)
    P = sum(p.numel() for p in all_p)

    for ep in range(15):
        for i in range(0, len(tr), 8):
            b = tr[i:i+8]
            if not b: continue
            x = torch.stack([p[0] for p in b])
            yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
            zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
            zo = it(zp)
            pr = hr(zo.real)+hi(zo.imag); pi = hr(zo.imag)-hi(zo.real)
            loss = F.mse_loss(pr, yr) + F.mse_loss(pi, yi)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0); opt.step()

    saved = [l['phase'].ang.data.clone() for l in it.core.layers]
    results = {}
    for mode in ['normal','ablated']:
        if mode == 'ablated':
            for l in it.core.layers: l['phase'].ang.data.zero_()
        ls = n = 0
        it.eval(); hr.eval(); hi.eval()
        with torch.no_grad():
            for i in range(0, len(te), 8):
                b = te[i:i+8]
                if not b: continue
                x = torch.stack([p[0] for p in b])
                yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
                zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
                zo = it(zp)
                pr = hr(zo.real)+hi(zo.imag); pi = hr(zo.imag)-hi(zo.real)
                ls += (F.mse_loss(pr, yr) + F.mse_loss(pi, yi)).item(); n += 1
        results[mode] = ls / max(n, 1)
    for l,s in zip(it.core.layers, saved): l['phase'].ang.data.copy_(s)
    delta = (results['ablated']-results['normal'])/(results['normal']+1e-8)*100
    print(f"d={d} h={h} L={L} cyc={cycles} P={P:>7,} norm={results['normal']:.4f} abl={results['ablated']:.4f} delta={delta:+.1f}% time={time.time()-t0:.0f}s")
    print(f"  PHASE HOPS ACROSS BOUNDARIES" if delta>10 else "  WEAK" if delta>3 else "  PHASE DOESNT HOP")
