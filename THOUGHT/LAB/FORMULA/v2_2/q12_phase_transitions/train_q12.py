"""Q12: Train complex-native transformer with checkpoint phase transition tracking."""
import torch, torch.nn as nn, torch.nn.functional as F, math, json, time
import numpy as np
from collections import Counter
from datasets import load_dataset
torch.manual_seed(42)

# ---- Architecture (same as native_eigen.py) ----
class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d)
        self.im = nn.Embedding(V, d)
        nn.init.normal_(self.re.weight, std=0.02)
        nn.init.normal_(self.im.weight, std=0.02)
    def forward(self, x):
        return torch.complex(self.re(x), self.im(x))

class NativeAttention(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.qr = nn.Linear(d, d, bias=False)
        self.qi = nn.Linear(d, d, bias=False)
        self.kr = nn.Linear(d, d, bias=False)
        self.ki = nn.Linear(d, d, bias=False)
        self.vr = nn.Linear(d, d, bias=False)
        self.vi = nn.Linear(d, d, bias=False)
        self.sc = 1.0 / math.sqrt(d)
        for w in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi]:
            nn.init.normal_(w.weight, std=0.02)
    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag)
        qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag)
        ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag)
        vi = self.vr(x.imag) + self.vi(x.real)
        qr, kr, vr = qr.transpose(1,2), kr.transpose(1,2), vr.transpose(1,2)
        qi, ki, vi = qi.transpose(1,2), ki.transpose(1,2), vi.transpose(1,2)
        sr = (qr.transpose(-2,-1) @ kr + qi.transpose(-2,-1) @ ki) * self.sc
        si = (qi.transpose(-2,-1) @ kr - qr.transpose(-2,-1) @ ki) * self.sc
        dtheta = si.diagonal(offset=1, dim1=-2, dim2=-1)
        curv = (dtheta[:, 1:] - dtheta[:, :-1]).abs()
        curv_pad = F.pad(curv, (1, 1)).unsqueeze(1)
        sr = sr + 1.0 * curv_pad
        si = si + 0.5 * curv_pad * torch.sign(si)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        si = si.masked_fill(mask, 0.0)
        attn = F.softmax(sr, dim=-1)
        cp, sp = torch.cos(si), torch.sin(si)
        out_r = (attn*cp)@vr.transpose(-2,-1) - (attn*sp)@vi.transpose(-2,-1)
        out_i = (attn*cp)@vi.transpose(-2,-1) + (attn*sp)@vr.transpose(-2,-1)
        return torch.complex(out_r, out_i)

class PhaseRot(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d)*0.1)
    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class NativeEigen(nn.Module):
    def __init__(self, V=2000, d=2, L=2):
        super().__init__()
        self.emb = ComplexEmbed(V, d)
        self.layers = nn.ModuleList([
            nn.ModuleDict({'a': NativeAttention(d), 'p': PhaseRot(d)}) for _ in range(L)
        ])
        self.out = nn.Linear(d, V)
        nn.init.normal_(self.out.weight, std=0.02)
    def forward(self, x):
        z = self.emb(x)
        for l in self.layers:
            z = l['p'](l['a'](z))
        return self.out(torch.abs(z))
    def get_embedding_weights(self):
        return self.emb.re.weight.detach().cpu(), self.emb.im.weight.detach().cpu()

def load(V=2000, seq=32, N=2000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    c = Counter()
    for ex in ds:
        for w in str(ex["text"]).split():
            c[w] += 1
    voc = ["<pad>","<unk>","<eos>"] + [w for w,_ in c.most_common(V-3)]
    w2i = {w:i for i,w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex["text"]).split():
            toks.append(w2i.get(w, 1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks)-seq, N*seq), seq//2):
        s = toks[i:i+seq+1]
        if len(s) == seq+1:
            data.append((s[:-1], s[1:]))
    return data[:N], len(voc)

# ---- Q12 Phase Transition Metrics ----
def compute_embedding_metrics(re_w, im_w):
    z = re_w.numpy() + 1j * im_w.numpy()  # (V, 2) complex
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-12)
    np.random.seed(0)
    idx = np.random.choice(len(z), min(200, len(z)), replace=False)
    zi = z[idx]
    n = len(zi)
    H = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i, n):
            v = np.conj(zi[i]).dot(zi[j])
            H[i,j] = v; H[j,i] = np.conj(v)
    ev = np.linalg.eigvalsh(H)
    ev = np.maximum(ev, 1e-15)
    ev = ev / ev.sum()
    sigma = 1.0 / max(ev.sum()**2 / (ev**2).sum(), 1e-10)
    nabla = -np.sum(ev * np.log(ev + 1e-15))
    c_sem = np.sqrt(sigma / max(nabla, 1e-10))
    coh = np.abs(np.mean(np.exp(1j * np.angle(H))))
    phase_angs = np.angle(z)
    return {
        "sigma": float(sigma), "nabla_S": float(nabla), "c_sem": float(c_sem),
        "phase_coh": float(coh),
        "phase_mean": float(np.mean(phase_angs)),
        "phase_std": float(np.std(phase_angs)),
    }

# ---- Training with Checkpoints ----
if __name__ == "__main__":
    D = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {D}")
    data, V = load(N=2000)
    model = NativeEigen(V=V, d=2, L=2).to(D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    P = sum(p.numel() for p in model.parameters())
    print(f"Native Eigen Q12: V={V} seqs={len(data)} params={P:,}")

    metrics_history = []
    checkpoints = []
    steps_per_checkpoint = max(1, len(data) // 40)
    global_step = 0

    model.train()
    for ep in range(5):
        tl = 0; batches = 0
        for i in range(0, len(data), 16):
            b = data[i:i+16]
            if not b: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item(); batches += 1; global_step += 1

            if global_step % steps_per_checkpoint == 0 or (ep == 4 and i + 16 >= len(data)):
                re_w, im_w = model.get_embedding_weights()
                met = compute_embedding_metrics(re_w, im_w)
                met["step"] = global_step; met["ppl"] = float(tl / max(batches, 1))
                metrics_history.append(met)
                checkpoints.append({"step": global_step, "metrics": met})
                if len(metrics_history) % 8 == 0:
                    print(f"  step={global_step:4d} ppl={met['ppl']:.0f} sigma={met['sigma']:.4f} nabla_S={met['nabla_S']:.4f} c_sem={met['c_sem']:.4f}", flush=True)

        ppl = math.exp(tl / max(batches, 1))
        print(f"  E{ep+1}: ppl={ppl:.0f}", flush=True)

    # ---- Phase Transition Analysis ----
    print(f"\n{'='*60}")
    print("PHASE TRANSITION ANALYSIS")
    print(f"{'='*60}")
    for key in ["sigma", "nabla_S", "c_sem", "phase_coh"]:
        vals = [m[key] for m in metrics_history]
        start_v = vals[0]; end_v = vals[-1]; delta = end_v - start_v
        print(f"  {key:<12s}: start={start_v:.4f} end={end_v:.4f} delta={delta:+.4f}")
        # Check for jumps: max absolute step-to-step change
        diffs = np.abs(np.diff(vals))
        max_jump = diffs.max()
        max_jump_idx = np.argmax(diffs)
        print(f"            max_jump={max_jump:.4f} at step {max_jump_idx}")

    # Kuramoto check
    sigma_vals = np.array([m["sigma"] for m in metrics_history])
    nabla_vals = np.array([m["nabla_S"] for m in metrics_history])
    crosses = (sigma_vals > 2 * nabla_vals).any()
    print(f"\n  Kuramoto crossing (sigma > 2*nabla_S): {'YES' if crosses else 'NO'}")

    # Save checkpoint data
    json.dump({"checkpoints": checkpoints, "config": {"V": V, "seqs": len(data), "params": P}},
              open("checkpoints_q12.json", "w"))
    print(f"Saved {len(checkpoints)} checkpoints to checkpoints_q12.json")
