"""Section 8: Basic Calculus — derivatives of polynomials, limits.
Derivative = infinitesimal phase difference (power rule).
Model learns: d/dx (a*x^n) = a*n*x^(n-1).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
from math import pi
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
        super().__init__(); self.H=H;self.dh=d//H;self.sc=1.0/math.sqrt(self.dh)
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

class CalcModel(nn.Module):
    def __init__(self,max_val=30,d=32,H=4,L=3):
        super().__init__()
        self.emb_r=nn.Embedding(max_val+1,d); self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d); self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,2)  # predict [derivative_coeff, derivative_power]
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,tokens):
        z=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        z=self.pe(z)
        for l in self.attns: z=z+l(z)
        return self.out(z.mean(1).real)

# Data: [coeff, power] -> derivative [coeff*power, power-1]
def gen_calc(n,max_coeff=10,max_pow=4):
    data=[]
    for _ in range(n):
        a=random.randint(1,max_coeff); n_pow=random.randint(1,max_pow)
        da=a*n_pow; dp=n_pow-1
        data.append(([a,n_pow],[da,dp]))
    return data

data=gen_calc(5000,10,6)
random.shuffle(data)
n_tr=len(data)*4//5; tr,te=data[:n_tr],data[n_tr:]
max_emb=60

print("="*60)
print("SECTION 8: BASIC CALCULUS — derivatives (power rule)")
print("="*60)

for d,H,L in [(32,4,3),(48,6,4)]:
    m=CalcModel(max_emb,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(60):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b])
            tgt=torch.tensor([x[1] for x in b],dtype=torch.float32)/max_emb
            loss=F.mse_loss(m(x_t),tgt)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te])
        tgt=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=(m(x_t)*max_emb).round()
        d_ok=(pred[:,0]==tgt[:,0])&(pred[:,1]==tgt[:,1])
        acc=d_ok.float().mean().item()
    print(f"  d={d} L={L}: deriv={acc:.1%} P={P:>6,} {'PASS' if acc>.80 else ''}")
