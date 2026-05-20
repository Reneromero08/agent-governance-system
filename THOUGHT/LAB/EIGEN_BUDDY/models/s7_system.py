"""Section 7: 2x2 Systems — separate embeddings for equation 1 vs 2.
Like Section 10 complex multiplication, bilinear systems need distinct pathways."""
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

class SepSysModel(nn.Module):
    def __init__(self,max_val=30,d=32,H=4,L=3):
        super().__init__()
        self.emb_r=nn.Embedding(max_val+1,d);self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,2)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.emb_r(t),self.emb_i(t))
        z=self.pe(z)
        for l in self.attns:z=z+l(z)
        return self.out(z.mean(1).real)

def gen(n,max_val=6):
    data=[]
    for _ in range(n):
        x=random.randint(0,max_val);y=random.randint(0,max_val)
        a1=random.randint(1,max_val);b1=random.randint(0,max_val)
        a2=random.randint(0,max_val);b2=random.randint(1,max_val)
        if a1*b2-a2*b1==0:continue
        c1=a1*x+b1*y;c2=a2*x+b2*y
        data.append(([a1,b1,c1,a2,b2,c2],[x,y]))
    return data

max_val=8; data=gen(30000,max_val)
random.shuffle(data);n_tr=len(data)*4//5;tr,te=data[:n_tr],data[n_tr:]
max_emb=max_val*max_val*2+max_val

for d,H,L in [(64,8,6)]:
    m=SepSysModel(max_emb,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(80):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b]);tgt=torch.tensor([x[1] for x in b],dtype=torch.float32)/max_val
            loss=F.mse_loss(m(x_t),tgt);opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te]);tgt=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=(m(x_t)*max_val).round().clamp(0,max_val)
        xy=((pred[:,0]==tgt[:,0])&(pred[:,1]==tgt[:,1])).float().mean().item()
        x_ok=(pred[:,0]==tgt[:,0]).float().mean().item()
    print(f"  d={d} L={L}: xy={xy:.1%} x={x_ok:.1%} P={P:>6,} {'PASS' if xy>.90 else ''}")
