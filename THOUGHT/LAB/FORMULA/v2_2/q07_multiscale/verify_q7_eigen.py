"""Q7: R composes across scales — Native Eigen token-level → sequence-level phase_coh.

Uses the actual native_eigen.py architecture (ComplexEmbed → NativeAttention → output).
Measures phase_coh at three scales:
  1. Token-level: phase_coh within each token's complex embedding
  2. Attention-level: phase_coh of Q·K^† attention scores
  3. Sequence-level: phase_coh of output logit distribution

Q7 claim: phase_coh at token level predicts phase_coh at sequence level.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math
from collections import Counter
from datasets import load_dataset
torch.manual_seed(42)

# ---- Native Eigen components (from native_eigen.py) ----
class ComplexEmbed(nn.Module):
    def __init__(self, V, d=2):
        super().__init__()
        self.re = nn.Embedding(V, d); self.im = nn.Embedding(V, d)
        nn.init.normal_(self.re.weight, std=0.02); nn.init.normal_(self.im.weight, std=0.02)
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
        qr = self.qr(x.real)-self.qi(x.imag); qi = self.qr(x.imag)+self.qi(x.real)
        kr = self.kr(x.real)-self.ki(x.imag); ki = self.kr(x.imag)+self.ki(x.real)
        vr = self.vr(x.real)-self.vi(x.imag); vi = self.vr(x.imag)+self.vi(x.real)
        qr,kr,vr = qr.transpose(1,2), kr.transpose(1,2), vr.transpose(1,2)
        qi,ki,vi = qi.transpose(1,2), ki.transpose(1,2), vi.transpose(1,2)
        sr = (qr.transpose(-2,-1)@kr + qi.transpose(-2,-1)@ki) * self.sc
        si = (qi.transpose(-2,-1)@kr - qr.transpose(-2,-1)@ki) * self.sc
        dtheta = si.diagonal(offset=1, dim1=-2, dim2=-1)
        curv = (dtheta[:, 1:] - dtheta[:, :-1]).abs()
        curv_pad = F.pad(curv, (1, 1)).unsqueeze(1)
        sr = sr + 1.0*curv_pad; si = si + 0.5*curv_pad*torch.sign(si)
        mask = torch.triu(torch.ones(S,S,device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf')); si = si.masked_fill(mask, 0.0)
        attn = F.softmax(sr, dim=-1)
        cp, sp = torch.cos(si), torch.sin(si)
        out_r = (attn*cp)@vr.transpose(-2,-1) - (attn*sp)@vi.transpose(-2,-1)
        out_i = (attn*cp)@vi.transpose(-2,-1) + (attn*sp)@vr.transpose(-2,-1)
        return torch.complex(out_r, out_i), si

class PhaseRot(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ang = nn.Parameter(torch.ones(d)*0.1)
    def forward(self, z):
        c, s = torch.cos(self.ang), torch.sin(self.ang)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class Q7Instrumented(nn.Module):
    def __init__(self, V=2000, d=2, L=2):
        super().__init__()
        self.emb = ComplexEmbed(V, d)
        self.layers = nn.ModuleList([nn.ModuleDict({'a':NativeAttention(d), 'p':PhaseRot(d)}) for _ in range(L)])
        self.out = nn.Linear(d, V); nn.init.normal_(self.out.weight, std=0.02)
        self.d = d
        self.L = L

    def forward(self, x):
        z = self.emb(x)
        all_si = []
        for l in self.layers:
            z, si = l['a'](z)
            all_si.append(si)
            z = l['p'](z)
        logits = self.out(torch.abs(z))
        return logits, z, all_si


def phase_coh_tensor(si):
    """Compute phase coherence from complex attention scores matrix.
    si shape: (B, 1, S, S) or (B, S, S)"""
    si_flat = si.reshape(si.shape[0], -1)
    cos_mean = torch.cos(si_flat).mean(dim=-1)
    sin_mean = torch.sin(si_flat).mean(dim=-1)
    return (cos_mean**2 + sin_mean**2).sqrt()


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


print("=" * 70)
print("Q7: MULTI-SCALE PHASE COMPOSITION")
print("Native Eigen on WikiText-2")
print("=" * 70)

D = "cuda" if torch.cuda.is_available() else "cpu"
data, V = load(N=1000)
model = Q7Instrumented(V=V, d=2, L=2).to(D)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
P = sum(p.numel() for p in model.parameters())
print(f"\n  V={V} seqs={len(data)} params={P:,} device={D}")

# Train for 5 epochs
model.train()
for ep in range(5):
    tl = 0
    for i in range(0, len(data), 16):
        b = data[i:i+16]
        if not b: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in b], device=D, dtype=torch.long)
        loss = F.cross_entropy(model(x)[0].view(-1, V), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); tl += loss.item()
    print(f"  E{ep+1}: ppl={math.exp(tl/max(1,len(data)//16)):.0f}", flush=True)

# ---- MULTI-SCALE MEASUREMENT ----
print(f"\n{'='*70}")
print(f"MULTI-SCALE PHASE COHERENCE MEASUREMENT")
print(f"{'='*70}")

model.eval()
all_measures = []
with torch.no_grad():
    for batch_idx in range(0, min(200, len(data)), 16):
        b = data[batch_idx:batch_idx+16]
        if len(b) < 2: continue
        x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)

        logits, z_final, all_si = model(x)

        # Scale 1: Token-level phase_coh (complex embedding)
        z_emb = model.emb(x)
        token_phases = torch.angle(z_emb)
        cos_m = torch.cos(token_phases).mean()
        sin_m = torch.sin(token_phases).mean()
        pc_token = (cos_m**2 + sin_m**2).sqrt().item()

        # Scale 2: Attention-level phase_coh (Q·Kdagger per layer)
        pc_attn = []
        for l_idx, si in enumerate(all_si):
            pc_l = phase_coh_tensor(si)
            pc_attn.append(pc_l.mean().item())

        # Scale 3: Sequence-level phase_coh (output distribution)
        probs = F.softmax(logits.view(-1, V), dim=-1)
        avg_dist = probs.mean(dim=0)
        H_seq = -(avg_dist * torch.log(avg_dist + 1e-8)).sum() / math.log(V)
        pc_seq = 1.0 - H_seq.item()

        all_measures.append({
            'pc_token': pc_token,
            'pc_attn': pc_attn,
            'pc_seq': pc_seq,
        })

# ---- CORRELATION ANALYSIS ----
print(f"\n  {'batch':>6} {'pc_token':>9} {'pc_attn_L1':>10} {'pc_attn_L2':>10} {'pc_seq':>8}")
print("  " + "-" * 50)
for i, m in enumerate(all_measures[:10]):
    print(f"  {i:>6} {m['pc_token']:>9.4f} {m['pc_attn'][0]:>10.4f} {m['pc_attn'][1]:>10.4f} {m['pc_seq']:>8.4f}")

pc_token_vals = torch.tensor([m['pc_token'] for m in all_measures])
pc_attn_L1 = torch.tensor([m['pc_attn'][0] for m in all_measures])
pc_attn_L2 = torch.tensor([m['pc_attn'][1] for m in all_measures])
pc_seq_vals = torch.tensor([m['pc_seq'] for m in all_measures])

# Correlations
r_token_seq = torch.corrcoef(torch.stack([pc_token_vals, pc_seq_vals]))[0,1].item()
r_attn1_seq = torch.corrcoef(torch.stack([pc_attn_L1, pc_seq_vals]))[0,1].item()
r_attn2_seq = torch.corrcoef(torch.stack([pc_attn_L2, pc_seq_vals]))[0,1].item()
r_token_attn = torch.corrcoef(torch.stack([pc_token_vals, pc_attn_L1]))[0,1].item()
r_attn12 = torch.corrcoef(torch.stack([pc_attn_L1, pc_attn_L2]))[0,1].item()

print(f"\n  CORRELATIONS:")
print(f"  Token -> Sequence:       r = {r_token_seq:+.4f}")
print(f"  Attention L1 -> Sequence: r = {r_attn1_seq:+.4f}")
print(f"  Attention L2 -> Sequence: r = {r_attn2_seq:+.4f}")
print(f"  Token -> Attention L1:   r = {r_token_attn:+.4f}")
print(f"  Attention L1 -> L2:      r = {r_attn12:+.4f}")

# Phase propagation: does pc change across scales?
mean_token = pc_token_vals.mean().item()
mean_attn1 = pc_attn_L1.mean().item()
mean_attn2 = pc_attn_L2.mean().item()
mean_seq = pc_seq_vals.mean().item()

print(f"\n  PHASE PROPAGATION:")
print(f"  Token (embedding):  {mean_token:.4f}")
print(f"  Attention L1:       {mean_attn1:.4f}  (delta: {mean_attn1-mean_token:+.4f})")
print(f"  Attention L2:       {mean_attn2:.4f}  (delta: {mean_attn2-mean_attn1:+.4f})")
print(f"  Sequence (output):  {mean_seq:.4f}  (delta: {mean_seq-mean_attn2:+.4f})")

# Phase ablation test: what happens to pc_seq when we ablate phase?
model_abl = Q7Instrumented(V=V, d=2, L=2).to(D)
model_abl.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})
for l in model_abl.layers:
    l['p'].ang.data.zero_()

nll_abl, n = 0, 0
b_abl = data[:100]
abl_seqs = []
model_abl.eval()
with torch.no_grad():
    for i in range(0, 100, 16):
        batch = b_abl[i:i+16]
        if not batch: continue
        x = torch.tensor([p[0] for p in batch], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in batch], device=D, dtype=torch.long)
        lo, _, _ = model_abl(x)
        nll_abl += F.cross_entropy(lo.view(-1, V), y.view(-1)).item() * y.numel()
        n += y.numel()
        probs = F.softmax(lo.view(-1, V), dim=-1)
        avg_d = probs.mean(dim=0)
        H = -(avg_d * torch.log(avg_d + 1e-8)).sum() / math.log(V)
        abl_seqs.append(1.0 - H.item())

ppl_abl = math.exp(nll_abl / max(n, 1))
pc_seq_abl = sum(abl_seqs) / len(abl_seqs)
nll_norm_val, n2 = 0, 0
normal_seqs = []
with torch.no_grad():
    for i in range(0, 100, 16):
        batch = b_abl[i:i+16]
        if not batch: continue
        x = torch.tensor([p[0] for p in batch], device=D, dtype=torch.long)
        y = torch.tensor([p[1] for p in batch], device=D, dtype=torch.long)
        lo, _, _ = model(x)
        nll_norm_val += F.cross_entropy(lo.view(-1, V), y.view(-1)).item() * y.numel()
        n2 += y.numel()
        probs = F.softmax(lo.view(-1, V), dim=-1)
        avg_d = probs.mean(dim=0)
        H = -(avg_d * torch.log(avg_d + 1e-8)).sum() / math.log(V)
        normal_seqs.append(1.0 - H.item())

ppl_norm = math.exp(nll_norm_val / max(n2, 1))
pc_seq_norm = sum(normal_seqs) / len(normal_seqs)

print(f"\n  PHASE ABLATION IMPACT:")
print(f"  Normal:  ppl={ppl_norm:.1f}  pc_seq={pc_seq_norm:.4f}")
print(f"  Ablated: ppl={ppl_abl:.1f}  pc_seq={pc_seq_abl:.4f}")
delta_ppl = (ppl_abl - ppl_norm) / ppl_norm * 100
delta_pc = pc_seq_norm - pc_seq_abl
print(f"  PPL change: {delta_ppl:+.1f}%  pc_seq change: {delta_pc:+.4f}")

print(f"\n  Q7 VERDICT:")
if abs(r_token_seq) > 0.3:
    print(f"  R COMPOSES ACROSS SCALES — token-level pc predicts sequence-level pc (r={r_token_seq:+.3f})")
elif abs(r_token_seq) > 0.15:
    print(f"  WEAK cross-scale composition (r={r_token_seq:+.3f})")
else:
    print(f"  No cross-scale composition — token and sequence phase are independent (r={r_token_seq:+.3f})")
if delta_ppl > 5:
    print(f"  Phase is LOAD-BEARING at sequence scale ({delta_ppl:+.1f}% PPL impact)")
elif delta_ppl > 1:
    print(f"  Phase carries structural information ({delta_ppl:+.1f}% PPL)")
else:
    print(f"  Phase not load-bearing at d=2 scale")

# ============================================================
# C5 BOUNDARY TEST: Complex vs Real manifold cross-scale R
# ============================================================
print(f"\n{'='*70}")
print(f"C5 BOUNDARY TEST: Cross-scale R on Complex vs Real")
print(f"{'='*70}")

# Complex: standard Native Eigen
model_cpx = Q7Instrumented(V=V, d=2, L=2).to(D)
model_cpx.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})

# Real: freeze imaginary channels (qi, ki, PhaseRot angles -> 0)
model_real = Q7Instrumented(V=V, d=2, L=2).to(D)
model_real.load_state_dict({k:v.clone() for k,v in model.state_dict().items()})
for l in model_real.layers:
    l['a'].qi.weight.data.zero_(); l['a'].qi.weight.requires_grad = False
    l['a'].ki.weight.data.zero_(); l['a'].ki.weight.requires_grad = False
    l['a'].vi.weight.data.zero_(); l['a'].vi.weight.requires_grad = False
    l['p'].ang.data.zero_(); l['p'].ang.requires_grad = False
    l['a'].qr.weight.requires_grad = False; l['a'].kr.weight.requires_grad = False
    l['a'].vr.weight.requires_grad = False

for label, mdl in [('COMPLEX', model_cpx), ('REAL', model_real)]:
    measures = []
    with torch.no_grad():
        for batch_idx in range(0, min(200, len(data)), 16):
            b = data[batch_idx:batch_idx+16]
            if len(b) < 2: continue
            x = torch.tensor([p[0] for p in b], device=D, dtype=torch.long)
            logits, z_final, all_si = mdl(x)
            z_emb = mdl.emb(x)
            token_p = torch.angle(z_emb)
            pc_t = (torch.cos(token_p).mean()**2 + torch.sin(token_p).mean()**2).sqrt().item()
            pc_a = []
            for si in all_si:
                pc_l = phase_coh_tensor(si); pc_a.append(pc_l.mean().item())
            probs = F.softmax(logits.view(-1, V), dim=-1)
            avg_d = probs.mean(dim=0)
            H_s = -(avg_d * torch.log(avg_d + 1e-8)).sum() / math.log(V)
            pc_s = 1.0 - H_s.item()
            measures.append((pc_t, pc_a, pc_s))

    pc_t = torch.tensor([m[0] for m in measures]); pc_s = torch.tensor([m[2] for m in measures])
    pc_a1 = torch.tensor([m[1][0] for m in measures]); pc_a2 = torch.tensor([m[1][1] for m in measures])
    r_ts = torch.corrcoef(torch.stack([pc_t, pc_s]))[0,1].item() if pc_t.std()>1e-6 else 0
    r_a1s = torch.corrcoef(torch.stack([pc_a1, pc_s]))[0,1].item() if pc_a1.std()>1e-6 else 0
    print(f"\n  {label}:")
    print(f"    Token->Sequence:  r={r_ts:+.4f}")
    print(f"    AttnL1->Sequence: r={r_a1s:+.4f}")
    print(f"    Token pc: {pc_t.mean().item():.4f}  Attn L1 pc: {pc_a1.mean().item():.4f}  Seq pc: {pc_s.mean().item():.4f}")

cpx_ts = r_token_seq; real_ts = r_ts
print(f"\n  C5 CROSS-SCALE VERDICT:")
if abs(cpx_ts) > abs(real_ts) + 0.1:
    print(f"  C5 CONFIRMED: cross-scale R is stronger on complex (r={cpx_ts:+.3f}) than real (r={real_ts:+.3f})")
elif abs(cpx_ts - real_ts) < 0.1:
    print(f"  C5 NOT DETECTED: cross-scale R similar on both manifolds (cpx={cpx_ts:+.3f}, real={real_ts:+.3f})")
else:
    print(f"  C5 REVERSED: real manifold stronger cross-scale R")
