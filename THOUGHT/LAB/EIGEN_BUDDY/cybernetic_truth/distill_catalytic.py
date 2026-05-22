"""
Catalytic Distillation v6 — ACTIVE CATALYTIC CACHE
===================================================
First layer of each type: FJLT + QR + SVD (cache miss).
All subsequent layers: PURE PROJECTION (cache hit).
  U = orth(W @ V_cached) — two matmuls, zero SVD.
  V = V_cached (borrowed from previous layer, restored).

This IS the catalytic lab: cross-depth active cache,
borrowed eigenbasis, zero recomputation.
"""
import os, sys, math, time, json
import torch, torch.nn.functional as F
from safetensors import safe_open
from pathlib import Path

MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"
INDEX_PATH = os.path.join(MODEL_DIR, "model.safetensors.index.json")
OUTPUT_PATH = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\EIGEN_BUDDY\cybernetic_truth\qwen_27b_catalytic_k256.holo"
RANK_K = 256

# ---- Rust Hadamard (26-qubit core) ----
_RUST = None
try:
    for d in [
        str(Path(__file__).parent / "rust_hadamard" / "target" / "release"),
        str(Path(__file__).parent),
    ]:
        sys.path.insert(0, d)
        try: _RUST = __import__("rust_hadamard"); break
        except: pass
except: pass

# Pre-computed Hadamard perms
_HC = {}
def _hp(d):
    if d not in _HC:
        p, iv = [], []
        for t in range(d):
            td = t+1; perm = [0,td]+[j for j in range(1,d+1) if j!=td]
            inv = [0]*(d+1)
            for j,pe in enumerate(perm): inv[pe] = j
            p.append(perm); iv.append(inv)
        _HC[d] = (p, iv)
    return _HC[d]

def fjlt(A, k, chunk=65536):
    m,n = A.shape; d = math.ceil(math.log2(n)); pad = 2**d - n
    out = torch.empty((m,k))
    H = torch.tensor([[1,1],[1,-1]])/math.sqrt(2); perms, invs = _hp(d)
    for i in range(0,m,chunk):
        c = A[i:i+chunk]
        if pad > 0: c = F.pad(c, (0, pad))
        if _RUST:
            ct = c.reshape(c.shape[0], 2**d).numpy()
            ct = _RUST.hadamard_transform(ct)
            out[i:i+chunk] = torch.from_numpy(ct[:,:k])
        else:
            ct = c.reshape([c.shape[0]]+[2]*d)
            for t in range(d):
                ct = ct.permute(*perms[t]).contiguous().reshape(c.shape[0],2,-1)
                ct = torch.matmul(H, ct)
                ct = ct.reshape([c.shape[0]]+[2]*d).permute(*invs[t]).contiguous()
            out[i:i+chunk] = ct.reshape(c.shape[0], 2**d)[:,:k]
    return out

def wt(k):
    p = k.split(".")
    for i,x in enumerate(p):
        if x in ("layers","blocks") and i+2 < len(p): return ".".join(p[i+2:-1])
    return "unknown"

# ---- CATALYTIC CACHE ----
cache = {}
import threading; _l = threading.Lock()

def compress(tensor, k, tp):
    if tensor.ndim != 2: return tensor
    odt = tensor.dtype; t = tensor.float()
    k = min(k, t.size(1), t.size(0))
    has = tp in cache
    if has and cache[tp][0].shape[0] != t.size(1): has = False
    
    try:
        if has:
            # CATALYTIC CACHE HIT: pure projection, zero SVD
            with _l: Vc = cache[tp][0].float()
            # U = orth(W @ V_cached)
            U = torch.matmul(t, Vc)  # (m, n) @ (n, k) = (m, k)
            U, _ = torch.linalg.qr(U)  # re-orthonormalize
            V = Vc  # borrow V from previous layer, restored unchanged
            # Singular values via projection
            Sd = torch.diag(torch.norm(torch.matmul(torch.matmul(U.t(), t), V[:,:k]), dim=0))
        else:
            # CACHE MISS: full FJLT + QR + SVD
            D = torch.randint(0, 2, (t.size(1),), dtype=torch.float32) * 2 - 1
            Y = fjlt(t * D, k)
            Q, _ = torch.linalg.qr(Y)
            B = torch.matmul(Q.t(), t)
            Us, Sv, Vh = torch.linalg.svd(B, full_matrices=False)
            U = torch.matmul(Q, Us[:,:k])
            V = Vh[:k,:].t()
            Sd = torch.diag(Sv[:k])
        
        with _l: cache[tp] = (V.detach().clone(), U.detach().clone())
        SVh = (Sd @ V.T).to(odt); U = U.to(odt)
        return U, SVh
    except Exception as e:
        print(f"    [Warn] {e}")
        U,S,Vh = torch.linalg.svd(t, full_matrices=False)
        U,S = U[:,:k], S[:k]; V = Vh[:k,:].T
        return U.to(odt), (torch.diag(S) @ V.T).to(odt)

def process(fp, rk):
    t0 = time.perf_counter(); holo = {}; n = 0; misses = 0; hits = 0
    with safe_open(fp, framework="pt", device="cpu") as f:
        for key in f.keys():
            if "vision" in key or "mtp" in key or "embed" in key: continue
            t = f.get_tensor(key)
            if t.ndim != 2: continue
            n += 1; ts = time.perf_counter()
            tp = wt(key)
            kind = "HIT" if tp in cache else "MISS"
            if kind == "HIT": hits += 1
            else: misses += 1
            Uk, SVh = compress(t, rk, tp)
            holo[key+".U"] = Uk; holo[key+".SVh"] = SVh
            dt = time.perf_counter() - ts
            if dt > 2 or n % 10 == 0:
                print(f"    [{n}] {tp}: {kind} {dt:.1f}s")
    return holo, time.perf_counter() - t0, misses, hits

def main():
    print(f"Rank: {RANK_K} | Rust: {'YES' if _RUST else 'torch'} | Mode: CATALYTIC CACHE")
    with open(INDEX_PATH) as f: ix = json.load(f)
    wm = ix.get("weight_map", {}); files = sorted(set(wm.values()))
    print(f"Files: {len(files)}")
    all_holo = {}; tt = 0.0; tm = 0; th = 0
    for i,fn in enumerate(files):
        fp = os.path.join(MODEL_DIR, fn)
        print(f"\n[{i+1}/{len(files)}] {fn}")
        holo, dt, m, h = process(fp, RANK_K)
        all_holo.update(holo); tt += dt; tm += m; th += h
        print(f"  {dt:.1f}s | misses={m} hits={h}")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    print(f"\nSaving {OUTPUT_PATH}...")
    torch.save(all_holo, OUTPUT_PATH)
    gb = os.path.getsize(OUTPUT_PATH)/1024**3
    print(f"Done! {gb:.2f} GB | {tt:.0f}s | misses={tm} hits={th}")

if __name__ == "__main__":
    if os.path.exists(MODEL_DIR): main()
    else:
        N = 2**26; x = torch.randn(1,N)
        t0 = time.perf_counter()
        Y = fjlt(x, 256)
        print(f"26q Hadamard: {N:,}->256 in {time.perf_counter()-t0:.2f}s ({N/256:.0f}x)")
