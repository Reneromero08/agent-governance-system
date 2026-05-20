"""Close gaps: S7 focal loss + S15 separate node embedding. Self-contained."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__();p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class BA(nn.Module):
    def __init__(self,d=32,H=4):
        super().__init__();self.H=H;self.dh=d//H;self.sc=1.0/math.sqrt(self.dh)
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

# ===== FIX 1: S7 with focal loss =====
print("="*60)
print("FIX 1: Section 7 — focal loss vs standard CE")
print("="*60)

def gen_sys(n,mv=6):
    d=[]
    for _ in range(n):
        x=random.randint(0,mv);y=random.randint(0,mv)
        a1=random.randint(1,mv);b1=random.randint(0,mv)
        a2=random.randint(0,mv);b2=random.randint(1,mv)
        if a1*b2-a2*b1==0:continue
        d.append(([a1,b1,a1*x+b1*y,a2,b2,a2*x+b2*y],[x,y]))
    return d

def focal_loss(logits,target,gamma=2.0):
    ce=F.cross_entropy(logits,target,reduction='none')
    pt=torch.exp(-ce)
    return ((1-pt)**gamma*ce).mean()

class StructAttn(BA):
    def __init__(self,d=32,H=4,cb=None):
        super().__init__(d,H)
        if cb is not None: self.register_buffer('cb',cb)
    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag);qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag);ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag);vi=self.vr(x.imag)+self.vi(x.real)
        qr=qr.view(B,S,self.H,self.dh).transpose(1,2);qi=qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr=kr.view(B,S,self.H,self.dh).transpose(1,2);ki=ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr=vr.view(B,S,self.H,self.dh).transpose(1,2);vi=vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.sc
        if hasattr(self,'cb'): sr=sr+self.cb.unsqueeze(0)
        attn=F.softmax(sr,dim=-1);out_r=attn@vr;out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i),self.or_(out_i)+self.oi(out_r))

def cramer_bias(H=8,S=6,boost=2.0):
    b=torch.zeros(H,S,S)
    pairs=[(0,4),(3,1),(2,4),(5,1)]
    for h,(i,j) in enumerate(pairs): b[h,i,j]=boost; b[h,j,i]=boost
    return b

class S7Model(nn.Module):
    def __init__(self,mv=150,d=64,H=8,L=4,nc=7):
        super().__init__()
        self.er=nn.Embedding(mv+1,d);self.ei=nn.Embedding(mv+1,d)
        self.pe=CPE(d)
        self.attns=nn.ModuleList([StructAttn(d,H,cramer_bias(H,6)) for _ in range(L)])
        self.hx=nn.Linear(d,nc);self.hy=nn.Linear(d,nc)
        for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.hx.weight,std=0.02);nn.init.normal_(self.hy.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.er(t),self.ei(t));z=self.pe(z)
        for l in self.attns:z=z+l(z)
        z=z.real.mean(1);return self.hx(z),self.hy(z)

d=gen_sys(15000,6);random.shuffle(d);n=len(d)*4//5;tr,te=d[:n],d[n:]
for gamma,label in [(0,'standard'),(2,'focal gamma=2'),(3,'focal gamma=3')]:
    m=S7Model()
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    for ep in range(30):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b:continue
            x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b])
            lx,ly=m(x_t)
            if gamma>0:
                loss=focal_loss(lx,y_t[:,0])+focal_loss(ly,y_t[:,1])
            else:
                loss=F.cross_entropy(lx,y_t[:,0])+F.cross_entropy(ly,y_t[:,1])
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te])
        lx,ly=m(x_t);px=lx.argmax(-1);py=ly.argmax(-1)
        xy=((px==y_t[:,0])&(py==y_t[:,1])).float().mean().item()
    print(f"  {label}: xy={xy:.1%}")
