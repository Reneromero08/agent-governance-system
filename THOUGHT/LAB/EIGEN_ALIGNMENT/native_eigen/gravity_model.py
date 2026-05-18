"""Semiotic gravity — entropy as mass in C^d complex attention."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
torch.manual_seed(42)

def compute_entropy(emb, window=16):
    B, S, D = emb.shape
    pad = window // 2
    padded = F.pad(emb, (0,0,pad,pad), mode='replicate')
    H = torch.zeros(B, S, device=emb.device)
    for i in range(S):
        local = padded[:, i:i+window, :]
        centered = local - local.mean(dim=1, keepdim=True)
        cov = centered.transpose(1,2) @ centered / window
        try:
            _, sv, _ = torch.linalg.svd(cov)
            p = sv / (sv.sum(dim=-1, keepdim=True) + 1e-8)
            p = torch.clamp(p, 1e-8, 1.0)
            H[:, i] = -(p * torch.log(p)).sum(dim=-1) / math.log(D)
        except:
            H[:, i] = 0.5
    return H

class GravityAttn(nn.Module):
    def __init__(self, d=8, heads=2, mw=0.5):
        super().__init__()
        self.d, self.H, self.dh = d, heads, d // heads
        self.mw = nn.Parameter(torch.tensor(mw))
        self.qr = nn.Linear(d, d, bias=False); self.qi = nn.Linear(d, d, bias=False)
        self.kr = nn.Linear(d, d, bias=False); self.ki = nn.Linear(d, d, bias=False)
        self.vr = nn.Linear(d, d, bias=False); self.vi = nn.Linear(d, d, bias=False)
        self.or_ = nn.Linear(d, d, bias=False); self.oi = nn.Linear(d, d, bias=False)
        self.sc = 1.0 / math.sqrt(self.dh)
        for w in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi, self.or_, self.oi]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        B, S, d = x.shape
        mass = compute_entropy(x.detach())
        qr, qi = self.qr(x), self.qi(x); kr, ki = self.kr(x), self.ki(x); vr, vi = self.vr(x), self.vi(x)
        qr = qr.view(B,S,self.H,self.dh).permute(0,2,3,1); qi = qi.view(B,S,self.H,self.dh).permute(0,2,3,1)
        kr = kr.view(B,S,self.H,self.dh).permute(0,2,3,1); ki = ki.view(B,S,self.H,self.dh).permute(0,2,3,1)
        vr = vr.view(B,S,self.H,self.dh).permute(0,2,3,1); vi = vi.view(B,S,self.H,self.dh).permute(0,2,3,1)
        sr = (qr.transpose(-2,-1)@kr + qi.transpose(-2,-1)@ki) * self.sc
        si = (qi.transpose(-2,-1)@kr - qr.transpose(-2,-1)@ki) * self.sc
        # +++ SEMIOTIC GRAVITY: mass curves the Q-K geodesic +++
        # Q32: Higher mass = shorter geodesic = stronger attention
        # score_new = score * (1 + alpha * mass_q * mass_k)
        # Broadcast: mass_q (B, H, S, 1) * mass_k (B, H, 1, S) -> (B, H, S, S)
        mass_q = mass.unsqueeze(1).unsqueeze(-1)  # (B, 1, S, 1)
        mass_k = mass.unsqueeze(1).unsqueeze(1)   # (B, 1, 1, S)
        gravity_factor = 1.0 + self.mw * mass_q * mass_k
        sr = sr * gravity_factor
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf')); si = si.masked_fill(mask, 0.0)
        am = F.softmax(sr, dim=-1); cp = torch.cos(si); sp = torch.sin(si)
        out_r = (am*cp)@vr.transpose(-2,-1) - (am*sp)@vi.transpose(-2,-1)
        out_i = (am*cp)@vi.transpose(-2,-1) + (am*sp)@vr.transpose(-2,-1)
        out_r = out_r.permute(0,3,1,2).contiguous().view(B,S,d)
        out_i = out_i.permute(0,3,1,2).contiguous().view(B,S,d)
        return self.or_(out_r)-self.oi(out_i), self.or_(out_i)+self.oi(out_r)

class PhaseAcc(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(dim) * 0.1)
    def forward(self, r, i):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return r*c - i*s, r*s + i*c

class GravityModel(nn.Module):
    def __init__(self, V=2000, d=8, H=2, L=2):
        super().__init__()
        self.emb = nn.Embedding(V, d); nn.init.normal_(self.emb.weight, std=0.02)
        self.layers = nn.ModuleList([nn.ModuleDict({'a': GravityAttn(d,H), 'p': PhaseAcc(d)}) for _ in range(L)])
        self.out = nn.Linear(d, V)

    def forward(self, x):
        e = self.emb(x)
        for l in self.layers:
            r, i = l['a'](e); r, i = l['p'](r, i); e = r
        return self.out(e)

def load(V=2000, seq=32, N=1000):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    c = Counter()
    for ex in ds:
        for w in str(ex["text"]).split(): c[w] += 1
    voc = ["<pad>","<unk>","<eos>"] + [w for w,_ in c.most_common(V-3)]
    w2i = {w:i for i,w in enumerate(voc)}
    toks = []
    for ex in ds:
        for w in str(ex["text"]).split(): toks.append(w2i.get(w,1))
        toks.append(2)
    data = []
    for i in range(0, min(len(toks)-seq, N*seq), seq//2):
        s = toks[i:i+seq+1]
        if len(s)==seq+1: data.append((s[:-1], s[1:]))
    return data[:N], len(voc)

if __name__ == "__main__":
    D = "cuda" if torch.cuda.is_available() else "cpu"
    data, V = load(N=1000)
    model = GravityModel(V=V).to(D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print("Gravity model: V={} seqs={} params={:,}".format(V, len(data), sum(p.numel() for p in model.parameters())))
    print("Training...", flush=True)
    model.train()
    for ep in range(3):
        tl = 0
        for i in range(0, len(data), 16):
            b = data[i:i+16]
            if not b: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
            loss = F.cross_entropy(model(x).view(-1, V), y.view(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()
        print("  E{}: ppl={:.0f}  mass_weight={:.3f}".format(ep+1, math.exp(tl/max(1,len(data)//16)), float(model.layers[0]['a'].mw)), flush=True)

    # Mass ablation — zero out mass weight, not phase
    saved_mw = [l['a'].mw.data.clone() for l in model.layers]
    res = {}
    for mode in ["normal", "zero_mass"]:
        if mode == "zero_mass":
            for l in model.layers: l['a'].mw.data.zero_()
        nll, n = 0, 0
        model.eval()
        with torch.no_grad():
            for i in range(0, min(200, len(data)), 16):
                b = data[i:i+16]
                if not b: continue
                x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
                y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
                lo = model(x)
                nll += F.cross_entropy(lo.view(-1, V), y.view(-1)).item() * y.numel()
                n += y.numel()
        res[mode] = math.exp(nll / max(n, 1))
    for l, s in zip(model.layers, saved_mw): l['a'].mw.data.copy_(s)

    delta = (res["zero_mass"] - res["normal"]) / res["normal"] * 100
    print("\nNormal: {:.1f}  Zero mass: {:.1f}  Delta: {:+.1f}%  mass={:.3f}".format(res["normal"], res["zero_mass"], delta, float(model.layers[0]['a'].mw)))
    if delta > 5: print("MASS MATTERS — entropy curves meaning-space")
    elif delta > 1: print("WEAK MASS")
    else: print("MASS NOT USED")
