"""Section 7: structural head routing — each head specializes on a Cramer pair.

Head 1: a1<->b2 (det1)  Head 2: a2<->b1 (det2)
Head 3: c1<->b2 (numX1)  Head 4: c2<->b1 (numX2)
Heads 5-8: free routing. Removes Q/K routing burden entirely.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__();p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class StructAttention(nn.Module):
    """Bidirectional attention with structural pair masking per head.
    mask[H,S,S]: 1.0 = allow attention, -inf = block.
    Forces each head to specialize on specific token pairs."""
    def __init__(self,d=48,H=8,bias=None):
        super().__init__()
        self.H=H;self.dh=d//H;self.sc=1.0/math.sqrt(self.dh)
        self.qr=nn.Linear(d,d,bias=False);self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False);self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False);self.vi=nn.Linear(d,d,bias=False)
        self.or_=nn.Linear(d,d,bias=False);self.oi=nn.Linear(d,d,bias=False)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi,self.or_,self.oi]:
            nn.init.normal_(w.weight,std=0.02)
        if bias is not None:
            self.register_buffer('bias',bias)
        else:
            self.bias=None

    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag);qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag);ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag);vi=self.vr(x.imag)+self.vi(x.real)
        qr=qr.view(B,S,self.H,self.dh).transpose(1,2);qi=qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr=kr.view(B,S,self.H,self.dh).transpose(1,2);ki=ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr=vr.view(B,S,self.H,self.dh).transpose(1,2);vi=vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.sc
        if self.bias is not None:
            sr=sr+self.bias.unsqueeze(0)  # bias target pairs (boost, not block)
        attn=F.softmax(sr,dim=-1);out_r=attn@vr;out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i),self.or_(out_i)+self.oi(out_r))

def make_cramer_bias(H=8,S=6,boost=2.0):
    """Bias (not mask) to encourage head specialization without blocking.
    Boost target pairs by 'boost' factor in attention logits.
    Heads 4+ get no bias (free routing)."""
    bias=torch.zeros(H,S,S)
    pairs=[(0,4),(3,1),(2,4),(5,1)]
    for h,(i,j) in enumerate(pairs):
        bias[h,i,j]=boost; bias[h,j,i]=boost
    return bias

class StructSys(nn.Module):
    def __init__(self,mv=30,d=48,H=8,L=4,num_c=7):
        super().__init__()
        self.er=nn.Embedding(mv+1,d);self.ei=nn.Embedding(mv+1,d)
        self.pe=CPE(d)
        bias=make_cramer_bias(H,6)
        self.layers=nn.ModuleList([StructAttention(d,H,bias) for _ in range(L)])
        self.hx=nn.Linear(d,num_c);self.hy=nn.Linear(d,num_c)
        for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.hx.weight,std=0.02);nn.init.normal_(self.hy.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.er(t),self.ei(t));z=self.pe(z)
        for l in self.layers:z=z+l(z)
        z=z.real.mean(1)
        return self.hx(z),self.hy(z)

def gen(n,mv=6):
    d=[]
    for _ in range(n):
        x=random.randint(0,mv);y=random.randint(0,mv)
        a1=random.randint(1,mv);b1=random.randint(0,mv)
        a2=random.randint(0,mv);b2=random.randint(1,mv)
        if a1*b2-a2*b1==0:continue
        d.append(([a1,b1,a1*x+b1*y,a2,b2,a2*x+b2*y],[x,y]))
    return d

mv=6;data=gen(40000,mv)
random.shuffle(data);n=len(data)*4//5;tr,te=data[:n],data[n:]
me=150;num_c=mv+1

print("="*60)
print("SECTION 7: STRUCTURAL HEAD ROUTING (Cramer's rule hardwired)")
print("="*60)

for d,H,L in [(48,8,4),(64,8,4)]:
    m=StructSys(me,d,H,L,num_c)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=30)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(30):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b])
            lx,ly=m(x_t)
            loss=F.cross_entropy(lx,y_t[:,0])+F.cross_entropy(ly,y_t[:,1])
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te])
        lx,ly=m(x_t);px=lx.argmax(-1);py=ly.argmax(-1)
        xy=((px==y_t[:,0])&(py==y_t[:,1])).float().mean().item()
        x_ok=(px==y_t[:,0]).float().mean().item()
        y_ok=(py==y_t[:,1]).float().mean().item()
    print(f"  d={d} L={L}: xy={xy:.1%} x={x_ok:.1%} y={y_ok:.1%} P={P:>6,} {'PASS' if xy>.95 else ''}")
