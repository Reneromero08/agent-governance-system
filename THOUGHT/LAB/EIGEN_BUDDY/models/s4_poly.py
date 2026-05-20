"""Section 4: Polynomial Evaluation — composition, operator precedence.

Evaluate a_n*x^n + ... + a_1*x + a_0 given x and coefficients.
Model must learn: multiplication before addition, higher powers before lower.
This is syntactic structure — operations have precedence.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__()
        p=torch.arange(ms).unsqueeze(1).float(); d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1]; th=self.ph[:S]; c=torch.cos(th).unsqueeze(0); s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class BA(nn.Module):
    def __init__(self,d=32,H=4):
        super().__init__()
        self.H=H;self.dh=d//H;self.sc=1.0/math.sqrt(self.dh)
        self.qr=nn.Linear(d,d,bias=False);self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False);self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False);self.vi=nn.Linear(d,d,bias=False)
        self.or_=nn.Linear(d,d,bias=False);self.oi=nn.Linear(d,d,bias=False)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi,self.or_,self.oi]:
            nn.init.normal_(w.weight,std=0.02)
    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag);qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag);ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag);vi=self.vr(x.imag)+self.vi(x.real)
        qr=qr.view(B,S,self.H,self.dh).transpose(1,2);qi=qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr=kr.view(B,S,self.H,self.dh).transpose(1,2);ki=ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr=vr.view(B,S,self.H,self.dh).transpose(1,2);vi=vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.sc
        attn=F.softmax(sr,dim=-1);out_r=attn@vr;out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i),self.or_(out_i)+self.oi(out_r))

class IterPolyModel(nn.Module):
    """Iterative polynomial evaluation. Each cycle builds one term of the polynomial.
    Cycle 1: x^3*a3, Cycle 2: +x^2*a2, Cycle 3: +x*a1, Cycle 4: +a0.
    The spiral trajectory accumulates the result across iterations."""
    def __init__(self,max_val=30,d=32,H=4,L=2,cycles=4):
        super().__init__()
        self.cycles=cycles
        self.emb_r=nn.Embedding(max_val+1,d); self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d)
        self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,1)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)

    def forward(self,tokens):
        z_init=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        z_init=self.pe(z_init)
        z=z_init
        result=0
        for cycle in range(self.cycles):
            # Process through attention with residual from initial state
            z_cycle=z_init+z  # spiral: each cycle adds to previous
            for l in self.attns: z_cycle=z_cycle+l(z_cycle)
            z=z_cycle
            # Accumulate: each cycle contributes one term
            result=result+self.out(z.mean(1).real).squeeze(-1)
        return result

# Data: [x, a_4, a_3, a_2, a_1, a_0] -> polynomial value
def gen_poly(n,max_val=6,deg=3):
    data=[]
    for _ in range(n):
        x=random.randint(0,max_val)
        coeffs=[random.randint(0,max_val) for _ in range(deg+1)]
        y=sum(coeffs[i]*x**(deg-i) for i in range(deg+1))
        toks=[x]+coeffs  # only positive
        data.append((toks,y))
    return data

max_val=6; deg=3
data=gen_poly(12000,max_val,deg)
random.shuffle(data)
n_tr=len(data)*4//5; tr,te=data[:n_tr],data[n_tr:]
max_emb=max_val; max_y=max_val*(max_val**deg)*(deg+1)

print("="*60)
print(f"SECTION 4: POLYNOMIAL EVALUATION — degree {deg}")
print(f"  Train: {len(tr)}  Test: {len(te)}  max_y: {max_y}")
print("="*60)

# Use per-sample normalization instead of global max_y
for d,H,L,cyc in [(48,6,3,4)]:
    m=IterPolyModel(max_emb,d,H,L,cyc)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(60):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b])
            y_t=torch.tensor([x[1] for x in b],dtype=torch.float32)
            # Per-sample normalize by max_y
            tgt=y_t/max_y
            pred=m(x_t)
            loss=F.mse_loss(pred,tgt)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te])
        y_t=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=m(x_t)*max_y
        th=torch.maximum(y_t.abs()*0.05,torch.tensor(5.0))
        acc=((pred-y_t).abs()<th).float().mean().item()
    print(f"  d={d} L={L}: acc={acc:.1%} P={P:>6,} {'PASS' if acc>.85 else ''}")
