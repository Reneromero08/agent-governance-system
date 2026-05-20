"""Hybrid Core — BD attention (no causal mask) + catalytic rounds. Self-contained."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

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

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__()
        p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class HybridCore(nn.Module):
    def __init__(self,d=64,H=8,L_std=3,L_cat=1,cat_r=3):
        super().__init__()
        self.std=nn.ModuleList([BA(d,H) for _ in range(L_std)])
        self.cat=BA(d,H); self.cat_r=cat_r
    def forward(self,z):
        for l in self.std: z=z+l(z)
        for _ in range(self.cat_r): z=z+self.cat(z)
        return z

def test_sys():
    def gen(n,mv=6):
        d=[]
        for _ in range(n):
            x=random.randint(0,mv);y=random.randint(0,mv)
            a1=random.randint(1,mv);b1=random.randint(0,mv)
            a2=random.randint(0,mv);b2=random.randint(1,mv)
            if a1*b2-a2*b1==0:continue
            d.append(([a1,b1,a1*x+b1*y,a2,b2,a2*x+b2*y],[x,y]))
        return d
    mv=6;data=gen(10000,mv);random.shuffle(data);n=len(data)*4//5;tr,te=data[:n],data[n:]
    me=mv*mv*2+mv
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.er=nn.Embedding(me+1,48);self.ei=nn.Embedding(me+1,48)
            self.pe=CPE(48);self.core=HybridCore(48,6,3,1,2);self.out=nn.Linear(48,2)
            for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
            nn.init.normal_(self.out.weight,std=0.02)
        def forward(self,t):
            z=torch.complex(self.er(t),self.ei(t));z=self.pe(z);z=self.core(z)
            return self.out(z.mean(1).real)
    m=M();opt=torch.optim.AdamW(m.parameters(),lr=5e-3);P=sum(p.numel() for p in m.parameters())
    for ep in range(60):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b],dtype=torch.float32)/mv
            loss=F.mse_loss(m(x_t),y_t);opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=(m(x_t)*mv).round().clamp(0,mv)
        xy=((pred[:,0]==y_t[:,0])&(pred[:,1]==y_t[:,1])).float().mean().item()
    print(f"  S7 2x2 systems: xy={xy:.1%} P={P:>6,}")

def test_set():
    data=[]
    for _ in range(8000):
        s=[random.randint(0,10) for _ in range(random.randint(2,6))]
        s=s+[0]*(6-len(s));data.append((s,len(set(s))))
    random.shuffle(data);tr,te=data[:6400],data[6400:]
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.er=nn.Embedding(20,64);self.ei=nn.Embedding(20,64)
            self.pe=CPE(64);self.core=HybridCore(64,8,1,1,4);self.out=nn.Linear(64,1)
            for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
            nn.init.normal_(self.out.weight,std=0.02)
        def forward(self,t):
            z=torch.complex(self.er(t),self.ei(t));z=self.pe(z);z=self.core(z)
            return self.out(z.mean(1).real).squeeze(-1)
    m=M();opt=torch.optim.AdamW(m.parameters(),lr=5e-3);P=sum(p.numel() for p in m.parameters())
    for ep in range(40):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b],dtype=torch.float32)
            loss=F.mse_loss(m(x_t),y_t/8);opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te],dtype=torch.float32)
        acc=((m(x_t)*8).round()==y_t).float().mean().item()
    print(f"  S13 sets: acc={acc:.1%} P={P:>6,}")

print("="*60)
print("HYBRID CORE: BD attention + catalytic rounds")
print("="*60)
test_sys()
test_set()
