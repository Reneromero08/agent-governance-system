"""Train NativeEigenCore on the Feral Resident's vector database.

8904 vectors, 4381 edges, 99 paper sequences.
Trains the Core to follow geodesics through the Feral's semantic space.
Saves weights for use by NativeEigenReasoner.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, random, os
import numpy as np, sqlite3
from pathlib import Path
from collections import defaultdict

torch.manual_seed(42); random.seed(42)

DB = Path(r'THOUGHT/LAB/FERAL_RESIDENT/data/db/feral_eternal.db')
EIGEN_DIR = Path(__file__).parent.parent
WEIGHTS_PATH = EIGEN_DIR / 'weights' / 'feral.pt'

import sys; sys.path.insert(0, str(EIGEN_DIR))
from core import NativeEigenCore

conn = sqlite3.connect(str(DB))
D_emb = 384; D = D_emb // 2  # 192 complex

# ---- Load geodesics from Feral DB ----
print("Loading Feral DB geodesics...")

# Paper sequences (natural geodesics)
seq_rows = conn.execute("""
    SELECT sequence_id, vec_blob, sequence_idx
    FROM vectors WHERE sequence_id IS NOT NULL
    ORDER BY sequence_id, sequence_idx
""").fetchall()

sequences = defaultdict(list)
for sid, blob, idx in seq_rows:
    sequences[sid].append(np.frombuffer(blob, dtype=np.float32))

paper_walks = []
for sid, vecs in sequences.items():
    if len(vecs) >= 17:
        for i in range(0, len(vecs) - 16, 4):
            w = vecs[i:i+17]
            if len(w) == 17:
                paper_walks.append((np.stack(w[:16]), np.stack(w[1:17])))

# Semantic edge walks (concept transitions)
edges = conn.execute("""
    SELECT a.vec_blob, b.vec_blob, e_score
    FROM e_edges e
    JOIN vectors a ON e.vector_id_a = a.vector_id
    JOIN vectors b ON e.vector_id_b = b.vector_id
    WHERE e_score > 0.7 LIMIT 3000
""").fetchall()

edge_walks = []
for a_blob, b_blob, score in edges:
    a = np.frombuffer(a_blob, np.float32)
    b = np.frombuffer(b_blob, np.float32)
    # 8 copies of a -> 8 copies of b (teaches edge transitions)
    inp = np.tile(a, (16, 1)); tgt = np.tile(b, (16, 1))
    edge_walks.append((inp, tgt))

conn.close()

# Build training data
data = []
for inp, tgt in paper_walks + edge_walks[:len(paper_walks)]:
    inp_v = np.array(inp, np.float32); tgt_v = np.array(tgt, np.float32)
    z = torch.complex(torch.tensor(inp_v[:, :D]), torch.tensor(inp_v[:, D:]))
    tr = torch.tensor(tgt_v[:, :D]); ti = torch.tensor(tgt_v[:, D:])
    data.append((z, tr, ti))

random.shuffle(data)
n_tr = len(data) * 3 // 4
tr, te = data[:n_tr], data[n_tr:]
print(f"Walks: {len(paper_walks)} paper + {len(edge_walks)} edge = {len(data)} total")
print(f"Train: {len(tr)}  Test: {len(te)}")

# ---- Train the Core ----
print("\n" + "=" * 55)
print("TRAINING FERAL CORE")
print("=" * 55)

best_delta = 0; best_d = 0; best_h = 0; best_L = 0

for d, h, L in [(32, 4, 4), (64, 4, 4), (48, 6, 4)]:
    t0 = time.time()
    # Projectors
    in_r = nn.Linear(D, d, bias=False); in_i = nn.Linear(D, d, bias=False)
    out_r = nn.Linear(d, D, bias=False); out_i = nn.Linear(d, D, bias=False)
    for w in [in_r, in_i, out_r, out_i]:
        nn.init.normal_(w.weight, std=0.02)

    core = NativeEigenCore(d=d, heads=h, layers=L, merge='concat', geo_init=True)
    all_p = list(in_r.parameters())+list(in_i.parameters())+list(core.parameters())+list(out_r.parameters())+list(out_i.parameters())
    opt = torch.optim.AdamW(all_p, lr=1e-3)
    P = sum(p.numel() for p in all_p)

    # 30 epochs — the Core needs to learn the Feral's space
    for ep in range(30):
        tl = n = 0
        for i in range(0, len(tr), 8):
            b = tr[i:i+8]
            if not b: continue
            x = torch.stack([p[0] for p in b])
            yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
            zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
            z_out, _ = core(zp + zp)  # iterative state injection
            pr = out_r(z_out.real)+out_i(z_out.imag)
            pi = out_r(z_out.imag)-out_i(z_out.real)
            loss = F.mse_loss(pr, yr) + F.mse_loss(pi, yi)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(all_p, 1.0); opt.step()
            tl += loss.item(); n += 1
        if ep % 10 == 0:
            print(f"  d={d} h={h} L={L} E{ep+1}: loss={tl/max(1,n):.6f}")

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
                x = torch.stack([p[0] for p in b])
                yr = torch.stack([p[1] for p in b]); yi = torch.stack([p[2] for p in b])
                zp = torch.complex(in_r(x.real)-in_i(x.imag), in_r(x.imag)+in_i(x.real))
                z_out, _ = core(zp + zp)
                pr = out_r(z_out.real)+out_i(z_out.imag)
                pi = out_r(z_out.imag)-out_i(z_out.real)
                ls += (F.mse_loss(pr, yr) + F.mse_loss(pi, yi)).item(); n += 1
        results[mode] = ls / max(n, 1)
    for l,s in zip(core.layers, saved): l['phase'].ang.data.copy_(s)

    delta = (results['ablated']-results['normal'])/(results['normal']+1e-8)*100
    compr = D / d
    print(f"  d={d} h={h} L={L} compr={compr:.0f}x P={P:>7,} norm={results['normal']:.4f} abl={results['ablated']:.4f} delta={delta:+.1f}% time={time.time()-t0:.0f}s")
    print(f"  {'PHASE CARRIES' if delta>10 else 'WEAK' if delta>3 else 'NOT LOAD-BEARING'}")

    if delta > best_delta:
        best_delta = delta; best_d = d; best_h = h; best_L = L
        # Save best model
        torch.save({
            'core': core.state_dict(),
            'in_r': in_r.state_dict(), 'in_i': in_i.state_dict(),
            'out_r': out_r.state_dict(), 'out_i': out_i.state_dict(),
            'config': {'d': d, 'heads': h, 'layers': L, 'D_emb': D_emb, 'D': D, 'delta': delta},
        }, str(WEIGHTS_PATH))
        print(f"  -> saved to {WEIGHTS_PATH.name}")

print(f"\nBest: d={best_d} h={best_h} L={best_L} delta={best_delta:+.1f}%")
print(f"Weights saved to {WEIGHTS_PATH}")
print("TRAINED — ready for NativeEigenReasoner")
