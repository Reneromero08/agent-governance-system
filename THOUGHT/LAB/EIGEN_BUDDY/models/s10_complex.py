"""Section 10: Complex multiplication — separate embeddings for operands.
The Core's Q·K† IS complex multiplication: Q(a)·K†(b) = a*conj(b).
With separate embeddings, a and b route through different pathways."""
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

class SepMultModel(nn.Module):
    def __init__(self,max_v=20,d=32,H=4,L=3):
        super().__init__()
        # SEPARATE embeddings for first operand vs second operand
        self.a_r=nn.Embedding(max_v+1,d); self.a_i=nn.Embedding(max_v+1,d)
        self.b_r=nn.Embedding(max_v+1,d); self.b_i=nn.Embedding(max_v+1,d)
        self.pe=CPE(d); self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out_r=nn.Linear(d,1); self.out_i=nn.Linear(d,1)
        for w in [self.a_r,self.a_i,self.b_r,self.b_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out_r.weight,std=0.02); nn.init.normal_(self.out_i.weight,std=0.02)
    def forward(self,a_rt,a_it,b_rt,b_it):
        va=torch.complex(self.a_r(a_rt),self.a_i(a_it)).unsqueeze(1)
        vb=torch.complex(self.b_r(b_rt),self.b_i(b_it)).unsqueeze(1)
        z=torch.cat([va,vb],1); z=self.pe(z)
        for l in self.attns: z=z+l(z)
        z=z.mean(1)
        return self.out_r(z.real).squeeze(-1), self.out_i(z.real).squeeze(-1)

# Data: complex multiplication only
data=[]
for _ in range(20000):
    ar=random.randint(-10,10);ai=random.randint(-10,10)
    br=random.randint(-10,10);bi=random.randint(-10,10)
    tr=ar*br-ai*bi;ti=ar*bi+ai*br
    data.append((ar+10,ai+10,br+10,bi+10,tr,ti))
random.shuffle(data)
n_tr=len(data)*4//5; tr,te=data[:n_tr],data[n_tr:]
print(f"  Train: {len(tr)}  Test: {len(te)}")

print("="*60)
print("SECTION 10: COMPLEX NUMBERS — separate operand embeddings")
print("="*60)

for d,H,L in [(48,6,4),(64,8,4)]:
    m=SepMultModel(30,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(60):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            # Input: (real_a, imag_a, real_b, imag_b) — 4 tokens with separate embeddings
            a_rt=torch.tensor([x[0] for x in b]); a_it=torch.tensor([x[1] for x in b])
            b_rt=torch.tensor([x[2] for x in b]); b_it=torch.tensor([x[3] for x in b])
            tgt_r=torch.tensor([x[4]/50 for x in b],dtype=torch.float32)
            tgt_i=torch.tensor([x[5]/50 for x in b],dtype=torch.float32)
            pr,pi=m(a_rt,a_it,b_rt,b_it)
            loss=F.mse_loss(pr,tgt_r)+F.mse_loss(pi,tgt_i)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        a_rt=torch.tensor([x[0] for x in te]); a_it=torch.tensor([x[1] for x in te])
        b_rt=torch.tensor([x[2] for x in te]); b_it=torch.tensor([x[3] for x in te])
        tgt_r=torch.tensor([x[4] for x in te],dtype=torch.float32)
        tgt_i=torch.tensor([x[5] for x in te],dtype=torch.float32)
        pr,pi=m(a_rt,a_it,b_rt,b_it); pr=pr*50; pi=pi*50
        th_r=torch.maximum(tgt_r.abs()*0.1,torch.tensor(2.0))
        th_i=torch.maximum(tgt_i.abs()*0.1,torch.tensor(2.0))
        ok=((pr-tgt_r).abs()<th_r)&((pi-tgt_i).abs()<th_i)
        acc=ok.float().mean().item()
    print(f"  d={d} L={L}: acc={acc:.1%} P={P:>6,} {'PASS' if acc>.95 else ''}")
