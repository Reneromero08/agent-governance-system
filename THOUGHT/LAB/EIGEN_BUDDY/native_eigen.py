"""Native Eigen Architecture — assembled. 2D complex, 14K params, WikiText-2."""
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
torch.manual_seed(42)

class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d)
        self.im = nn.Embedding(V, d)
        nn.init.normal_(self.re.weight, std=0.02)
        nn.init.normal_(self.im.weight, std=0.02)
    def forward(self, x): return torch.complex(self.re(x), self.im(x))

class NativeAttention(nn.Module):
    def __init__(self, d=2):
        super().__init__()
        self.qr = nn.Linear(d, d, bias=False); self.qi = nn.Linear(d, d, bias=False)
        self.kr = nn.Linear(d, d, bias=False); self.ki = nn.Linear(d, d, bias=False)
        self.vr = nn.Linear(d, d, bias=False); self.vi = nn.Linear(d, d, bias=False)
        self.sc = 1.0 / math.sqrt(d)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag); qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag); ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag); vi = self.vr(x.imag) + self.vi(x.real)
        qr, kr, vr = qr.transpose(1,2), kr.transpose(1,2), vr.transpose(1,2)
        qi, ki, vi = qi.transpose(1,2), ki.transpose(1,2), vi.transpose(1,2)
        sr = (qr.transpose(-2,-1) @ kr + qi.transpose(-2,-1) @ ki) * self.sc
        si = (qi.transpose(-2,-1) @ kr - qr.transpose(-2,-1) @ ki) * self.sc
        dtheta = si.diagonal(offset=1, dim1=-2, dim2=-1)
        curv = (dtheta[:, 1:] - dtheta[:, :-1]).abs()
        curv_pad = F.pad(curv, (1, 1)).unsqueeze(1)
        # Curvature amplifies both real (attention weight) and imaginary (phase rotation) channels
        sr = sr + 1.0 * curv_pad
        si = si + 0.5 * curv_pad * torch.sign(si)
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf')); si = si.masked_fill(mask, 0.0)
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
        self.layers = nn.ModuleList([nn.ModuleDict({'a': NativeAttention(d), 'p': PhaseRot(d)}) for _ in range(L)])
        self.out = nn.Linear(d, V); nn.init.normal_(self.out.weight, std=0.02)
    def forward(self, x):
        z = self.emb(x)
        for l in self.layers: z = l['p'](l['a'](z))
        return self.out(torch.abs(z))

def load(V=2000, seq=32, N=2000):
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
    data, V = load(N=2000)
    model = NativeEigen(V=V, d=2, L=2).to(D)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    P = sum(p.numel() for p in model.parameters())
    print("Native Eigen: V={} seqs={} params={:,}".format(V, len(data), P))
    model.train()
    for ep in range(5):
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
        print("  E{}: ppl={:.0f}".format(ep+1, math.exp(tl/max(1,len(data)//16))), flush=True)

    saved = [l['p'].ang.data.clone() for l in model.layers]
    res = {}
    for mode in ["normal", "ablated"]:
        if mode == "ablated":
            for l in model.layers: l['p'].ang.data.zero_()
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
    for l, s in zip(model.layers, saved): l['p'].ang.data.copy_(s)

    delta = (res["ablated"] - res["normal"]) / res["normal"] * 100
    print("\nNormal: {:.1f}  Ablated: {:.1f}  Delta: {:+.1f}%".format(res["normal"], res["ablated"], delta))
    if delta > 10: print("PHASE CARRIES SEMANTIC INFORMATION")
    elif delta > 3: print("WEAK PHASE SIGNAL")
    else: print("PHASE NOT LOAD-BEARING")
