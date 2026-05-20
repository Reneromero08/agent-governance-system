"""Section 3: Linear Equations — Inversion. Solve ax + b = c for x.

The model must undo the forward transform. Given (a, b, c), predict x.
Teaches: inversion, backward mapping, the first step toward reasoning.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
from math import pi

class CPE(nn.Module):
    def __init__(self, d, ms=512):
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
        self.qr=nn.Linear(d,d,bias=False); self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False); self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False); self.vi=nn.Linear(d,d,bias=False)
        self.or_=nn.Linear(d,d,bias=False); self.oi=nn.Linear(d,d,bias=False)
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
        attn=F.softmax(sr,dim=-1); out_r=attn@vr; out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i), self.or_(out_i)+self.oi(out_r))

class InvModel(nn.Module):
    def __init__(self,max_val=50,d=32,H=4,L=2):
        super().__init__()
        self.emb_r=nn.Embedding(max_val+1,d); self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d); self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,1)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self, a_t, b_t, c_t):
        va=torch.complex(self.emb_r(a_t),self.emb_i(a_t)).unsqueeze(1)
        vb=torch.complex(self.emb_r(b_t),self.emb_i(b_t)).unsqueeze(1)
        vc=torch.complex(self.emb_r(c_t),self.emb_i(c_t)).unsqueeze(1)
        z=torch.cat([va,vb,vc],1); z=self.pe(z)
        for l in self.attns: z=z+l(z)
        return self.out(z.mean(1).real).squeeze(-1)

# Data: solve ax + b = c for x, where a,b,c,x are integers
def gen_linear(n,max_val=8):
    data=[]
    for _ in range(n):
        a=random.randint(1,max_val); x=random.randint(0,max_val)
        b=random.randint(0,max_val)
        c=a*x+b
        data.append((a,b,c,x))
    return data

max_val=8
data=gen_linear(5000)
random.shuffle(data)
n_tr=len(data)*4//5; tr,te=data[:n_tr],data[n_tr:]
mr=max_val; max_emb=max_val*max_val+max_val  # a*x+b max

print("="*60)
print("SECTION 3: LINEAR EQUATIONS — Inversion (ax+b=c, find x)")
print("="*60)
print(f"  Train: {len(tr)}  Test: {len(te)}  max_emb: {max_emb}")

for d,H,L in [(32,4,2),(48,6,3)]:
    m=InvModel(max_emb,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(25):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            a_t=torch.tensor([x[0] for x in b]); b_t=torch.tensor([x[1] for x in b])
            c_t=torch.tensor([x[2] for x in b])
            tgt=torch.tensor([x[3]/mr for x in b],dtype=torch.float32)
            loss=F.mse_loss(m(a_t,b_t,c_t),tgt)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        a_t=torch.tensor([x[0] for x in te]); b_t=torch.tensor([x[1] for x in te])
        c_t=torch.tensor([x[2] for x in te])
        tgt=torch.tensor([x[3] for x in te],dtype=torch.float32)
        pred=(m(a_t,b_t,c_t)*mr).round().clamp(0,mr)
        acc=(pred==tgt).float().mean().item()
    print(f"  d={d} L={L}: acc={acc:.1%} P={P:>6,} {'PASS' if acc>.90 else ''}")
