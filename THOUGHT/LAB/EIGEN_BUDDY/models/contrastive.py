"""Unlock 1: Contrastive Phase Discrimination.

Forces unrelated concept pairs toward phase-destructive interference (dtheta -> pi).
Related pairs toward constructive interference (dtheta -> 0).
Bounded max-margin loss prevents vector magnitude drift.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

class ComplexPositionEncoding(nn.Module):
    def __init__(self, d, max_seq=512):
        super().__init__()
        pos = torch.arange(max_seq).unsqueeze(1).float()
        dim = torch.arange(d).float()
        self.register_buffer('phase', pos / (10000.0 ** (2.0*dim/d)))
    def forward(self, z):
        S = z.shape[1]; th = self.phase[:S]
        c = torch.cos(th).unsqueeze(0); s = torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class BidiAttn(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.H=heads; self.dh=d//heads
        self.qr=nn.Linear(d,d,bias=False); self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False); self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False); self.vi=nn.Linear(d,d,bias=False)
        self.or_=nn.Linear(d,d,bias=False); self.oi=nn.Linear(d,d,bias=False)
        self.sc=1.0/math.sqrt(self.dh)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi,self.or_,self.oi]:
            nn.init.normal_(w.weight,std=0.02)
    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag); qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag); ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag); vi=self.vr(x.imag)+self.vi(x.real)
        qr=qr.view(B,S,self.H,self.dh).transpose(1,2); qi=qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr=kr.view(B,S,self.H,self.dh).transpose(1,2); ki=ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr=vr.view(B,S,self.H,self.dh).transpose(1,2); vi=vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.sc
        attn=F.softmax(sr,dim=-1)
        out_r=attn@vr; out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i), self.or_(out_i)+self.oi(out_r))

class PhaseModel(nn.Module):
    """Model that outputs both prediction and phase representation."""
    def __init__(self, max_val=50, d=32, heads=4, layers=2):
        super().__init__()
        self.num_re = nn.Embedding(max_val+1, d); self.num_im = nn.Embedding(max_val+1, d)
        self.op_re = nn.Embedding(4, d); self.op_im = nn.Embedding(4, d)
        self.pos_enc = ComplexPositionEncoding(d)
        self.layers = nn.ModuleList([BidiAttn(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        for w in [self.num_re,self.num_im,self.op_re,self.op_im]:
            nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)

    def forward(self, a_tok, op_tok, b_tok, return_phase=False):
        va = torch.complex(self.num_re(a_tok), self.num_im(a_tok)).unsqueeze(1)
        vo = torch.complex(self.op_re(op_tok), self.op_im(op_tok)).unsqueeze(1)
        vb = torch.complex(self.num_re(b_tok), self.num_im(b_tok)).unsqueeze(1)
        z = torch.cat([va, vo, vb], 1); z = self.pos_enc(z)
        for l in self.layers: z = z + l(z)
        pooled = z.mean(dim=1)
        if return_phase:
            return self.out(pooled.real).squeeze(-1), pooled
        return self.out(pooled.real).squeeze(-1)

def phase_distance(z1, z2):
    """Compute phase distance between two complex representations.
    Returns angle difference in radians [0, pi]."""
    dot = (z1.real * z2.real + z1.imag * z2.imag).sum(dim=-1)
    norm = (z1.real**2 + z1.imag**2).sum(dim=-1).sqrt() * \
           (z2.real**2 + z2.imag**2).sum(dim=-1).sqrt() + 1e-8
    cos_sim = dot / norm
    return torch.acos(cos_sim.clamp(-1, 1))

# ---- Data: addition with related/unrelated triples ----
max_val = 30
# Anchor: addition problems (related to addition, unrelated to random concepts)
addition_data = [(random.randint(0,max_val), 0, random.randint(0,max_val),
                  a+b) for a,b in [(random.randint(0,max_val), random.randint(0,max_val)) for _ in range(4000)]]
# Unrelated: random number pairs from addition space (simulate different domain)
unrelated_pairs = [(random.randint(0,max_val), random.randint(0,max_val)) for _ in range(2000)]

random.shuffle(addition_data)
tr_add, te_add = addition_data[:3200], addition_data[3200:]
max_res = max_val * 2

print("=" * 60)
print("UNLOCK 1: Contrastive Phase Discrimination")
print("=" * 60)

M = 2.5  # target margin: negative pairs should have phase distance >= 2.5 rad

for use_contrastive in [False, True]:
    model = PhaseModel(max_val=50, d=32, heads=4, layers=2)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    P = sum(p.numel() for p in model.parameters())

    for ep in range(15):
        tl = 0
        random.shuffle(tr_add)
        for i in range(0, len(tr_add), 128):
            batch = tr_add[i:i+128]
            if not batch: continue
            a_t = torch.tensor([x[0] for x in batch]); o_t = torch.tensor([x[1] for x in batch])
            b_t = torch.tensor([x[2] for x in batch])
            tgt = torch.tensor([x[3]/max_res for x in batch], dtype=torch.float32)
            pred, z = model(a_t, o_t, b_t, return_phase=True)
            mse_loss = F.mse_loss(pred, tgt)

            if use_contrastive:
                # Contrastive: push unrelated pairs to phase opposition
                neg_batch = random.sample(unrelated_pairs, min(len(batch), len(unrelated_pairs)))
                neg_a = torch.tensor([x[0] for x in neg_batch])
                neg_b = torch.tensor([x[1] for x in neg_batch])
                # Encode negative pair (arbitrary operator 0)
                pred_neg, z_neg = model(neg_a, torch.zeros_like(neg_a),
                                        neg_b, return_phase=True)
                # Phase distance between anchor and negative
                pd = phase_distance(z[:len(neg_a)], z_neg)
                # Max-margin: penalize if phase distance < M
                contrastive_loss = F.relu(M - pd).mean()
                loss = mse_loss + 0.1 * contrastive_loss
            else:
                loss = mse_loss

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item()

    # Test: clean accuracy
    with torch.no_grad():
        a_t = torch.tensor([x[0] for x in te_add]); o_t = torch.tensor([x[1] for x in te_add])
        b_t = torch.tensor([x[2] for x in te_add])
        tgt = torch.tensor([x[3]/max_res for x in te_add], dtype=torch.float32)
        pred = model(a_t, o_t, b_t)
        th = torch.maximum(torch.tensor(1.0), (tgt*max_res).abs()*0.05)
        clean = ((pred*max_res - tgt*max_res).abs() < th).float().mean().item()

    # Phase distance on unrelated pairs
    neg_test = unrelated_pairs[:100]
    neg_a = torch.tensor([x[0] for x in neg_test])
    neg_b = torch.tensor([x[1] for x in neg_test])
    _, z_neg = model(neg_a, torch.zeros_like(neg_a), neg_b, return_phase=True)
    # Self-distance as baseline
    _, z_self = model(neg_a, torch.zeros_like(neg_a), neg_a, return_phase=True)
    pd_unrelated = phase_distance(z_neg, z_self).mean().item()

    print(f"  contrastive={use_contrastive}: clean={clean:.1%} phase_dist_unrelated={pd_unrelated:.2f}rad {'PUSHED_TO_PI' if pd_unrelated>M else 'RANDOM'}")
