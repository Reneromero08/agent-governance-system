"""Section 6: Sequences — detect pattern, predict next term.
Arithmetic, geometric, Fibonacci sequences. Model must detect the
phase relationship between consecutive terms and extend it.
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

class SeqModel(nn.Module):
    def __init__(self,max_val=50,d=32,H=4,L=3):
        super().__init__()
        self.emb_r=nn.Embedding(max_val+1,d); self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d); self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,1)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,tokens):
        z=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        z=self.pe(z)
        for l in self.attns: z=z+l(z)
        return self.out(z.mean(1).real).squeeze(-1)

# Data: predict next term from prefix of 5 terms
def gen_seq(n,max_val=20):
    data=[]
    for _ in range(n):
        t=random.randint(0,2)
        if t==0:  # arithmetic: a, a+d, a+2d, ...
            a=random.randint(0,max_val//2); d=random.randint(1,5)
            seq=[a+d*i for i in range(6)]
        elif t==1:  # geometric: a, a*r, a*r^2, ...
            a=random.randint(1,5); r=random.randint(2,4)
            seq=[a*(r**i) for i in range(6)]
            seq=[min(s,max_val) for s in seq]
        else:  # Fibonacci-like: a, b, a+b, b+(a+b), ...
            a=random.randint(0,max_val//3); b=random.randint(0,max_val//3)
            seq=[a,b]
            for i in range(4): seq.append(seq[-1]+seq[-2])
            seq=[min(s,max_val) for s in seq]
        data.append((seq[:5],seq[5]))
    return data

max_val=30; max_emb=60  # double for safety
data=gen_seq(8000,max_val)
random.shuffle(data)
n_tr=len(data)*4//5; tr,te=data[:n_tr],data[n_tr:]

print("="*60)
print("SECTION 6: SEQUENCES — predict next term")
print("="*60)

for d,H,L in [(32,4,3),(48,6,4)]:
    m=SeqModel(max_emb,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=60)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(60):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b])
            tgt=torch.tensor([x[1]/max_val for x in b],dtype=torch.float32)
            loss=F.mse_loss(m(x_t),tgt)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te])
        tgt=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=(m(x_t)*max_val).round().clamp(0,max_val)
        acc=(pred==tgt).float().mean().item()
    print(f"  d={d} L={L}: acc={acc:.1%} P={P:>6,} {'PASS' if acc>.85 else ''}")
