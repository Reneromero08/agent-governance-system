"""Unlock 3: Narrow Boundary — decouple token dim from core dim.

Tokens d_token=16 (narrow sensory boundary).
Core d_model=128 (wide implicate manifold).
Complex projection layers map between them, preserving phase geometry.
Prevents vocabulary structure from diluting phase-routing mechanics.
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
    def __init__(self, d=128, heads=8):
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

class NarrowBoundaryModel(nn.Module):
    """Token embeddings at d_token=16. Core at d_model=128.
    Complex projection maps between them, preserving phase geometry."""
    def __init__(self, max_val=50, d_token=16, d_model=128, heads=8, layers=4):
        super().__init__()
        # Narrow token boundary (explicate order)
        self.num_re = nn.Embedding(max_val+1, d_token)
        self.num_im = nn.Embedding(max_val+1, d_token)
        self.op_re = nn.Embedding(4, d_token)
        self.op_im = nn.Embedding(4, d_token)

        # Complex projection: d_token -> d_model (lift into implicate manifold)
        self.proj_up_r = nn.Linear(d_token, d_model, bias=False)
        self.proj_up_i = nn.Linear(d_token, d_model, bias=False)

        # Wide core manifold (implicate order)
        self.pos_enc = ComplexPositionEncoding(d_model)
        self.layers = nn.ModuleList([BidiAttn(d_model, heads) for _ in range(layers)])

        # Complex projection: d_model -> 1 (explicate output)
        self.out = nn.Linear(d_model, 1)

        for w in [self.num_re, self.num_im, self.op_re, self.op_im,
                  self.proj_up_r, self.proj_up_i]:
            nn.init.normal_(w.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, a_tok, op_tok, b_tok):
        # Narrow embeddings (real-valued discrete boundary)
        va_r = self.num_re(a_tok); va_i = self.num_im(a_tok)
        vo_r = self.op_re(op_tok); vo_i = self.op_im(op_tok)
        vb_r = self.num_re(b_tok); vb_i = self.num_im(b_tok)

        # Complex projection: lift into implicate manifold
        va = torch.complex(self.proj_up_r(va_r) - self.proj_up_i(va_i),
                           self.proj_up_r(va_i) + self.proj_up_i(va_r))
        vo = torch.complex(self.proj_up_r(vo_r) - self.proj_up_i(vo_i),
                           self.proj_up_r(vo_i) + self.proj_up_i(vo_r))
        vb = torch.complex(self.proj_up_r(vb_r) - self.proj_up_i(vb_i),
                           self.proj_up_r(vb_i) + self.proj_up_i(vb_r))

        z = torch.cat([va.unsqueeze(1), vo.unsqueeze(1), vb.unsqueeze(1)], dim=1)
        z = self.pos_enc(z)
        for layer in self.layers:
            z = z + layer(z)
        return self.out(z.mean(dim=1).real).squeeze(-1)

class WideFlatModel(nn.Module):
    """Baseline: d_token = d_model = 128. No boundary separation."""
    def __init__(self, max_val=50, d=128, heads=8, layers=4):
        super().__init__()
        self.num_re = nn.Embedding(max_val+1, d); self.num_im = nn.Embedding(max_val+1, d)
        self.op_re = nn.Embedding(4, d); self.op_im = nn.Embedding(4, d)
        self.pos_enc = ComplexPositionEncoding(d)
        self.layers = nn.ModuleList([BidiAttn(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        for w in [self.num_re,self.num_im,self.op_re,self.op_im]:
            nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)

    def forward(self, a_tok, op_tok, b_tok):
        va = torch.complex(self.num_re(a_tok), self.num_im(a_tok)).unsqueeze(1)
        vo = torch.complex(self.op_re(op_tok), self.op_im(op_tok)).unsqueeze(1)
        vb = torch.complex(self.num_re(b_tok), self.num_im(b_tok)).unsqueeze(1)
        z = torch.cat([va, vo, vb], 1); z = self.pos_enc(z)
        for l in self.layers: z = z + l(z)
        return self.out(z.mean(1).real).squeeze(-1)

# ---- Test: Narrow vs Wide on addition ----
max_val = 30
data = []
for _ in range(12000):
    a = random.randint(0,max_val); b = random.randint(0,max_val)
    data.append((a, 0, b, a + b))
random.shuffle(data); tr, te = data[:6400], data[6400:]
max_res = max_val * 2

print("=" * 60)
print("UNLOCK 3: Narrow Boundary (d_token=16, d_model=128)")
print("=" * 60)

for name, ModelClass, kwargs in [
    ("narrow (16->64)", NarrowBoundaryModel, {'d_model': 64, 'heads': 4, 'layers': 3}),
    ("wide flat (64=64)", WideFlatModel, {'d': 64, 'heads': 4, 'layers': 3}),
]:
    model = ModelClass(max_val=max_val, **kwargs)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
    P = sum(p.numel() for p in model.parameters())

    for ep in range(20):
        tl = n = 0
        random.shuffle(tr)
        for i in range(0, len(tr), 128):
            batch = tr[i:i+128]
            if not batch: continue
            a_t = torch.tensor([x[0] for x in batch]); o_t = torch.tensor([x[1] for x in batch])
            b_t = torch.tensor([x[2] for x in batch])
            tgt = torch.tensor([x[3]/max_res for x in batch], dtype=torch.float32)
            loss = F.mse_loss(model(a_t, o_t, b_t), tgt)
            opt.zero_grad(); loss.backward()
            opt.step(); tl += loss.item(); n += 1
        sched.step()

    # Clean accuracy
    with torch.no_grad():
        a_t = torch.tensor([x[0] for x in te]); o_t = torch.tensor([x[1] for x in te])
        b_t = torch.tensor([x[2] for x in te])
        tgt = torch.tensor([x[3]/max_res for x in te], dtype=torch.float32)
        pred = model(a_t, o_t, b_t)
        th = torch.maximum(torch.tensor(1.0), (tgt*max_res).abs()*0.05)
        clean = ((pred*max_res - tgt*max_res).abs() < th).float().mean().item()

    # Phase ablation: zero out imaginary embeddings (phase channel only)
    for pname, param in model.named_parameters():
        if 'num_im' in pname or 'op_im' in pname:
            param.data.zero_()

    with torch.no_grad():
        pred = model(a_t, o_t, b_t)
        ablated = ((pred*max_res - tgt*max_res).abs() < th).float().mean().item()

    delta = clean - ablated
    print(f"  {name}: P={P:>7,} clean={clean:.1%} ablated={ablated:.1%} delta={delta:+.1%} {'SHARP PHASE' if delta>0.20 else 'WEAK' if delta>0.05 else 'FLAT'}")
