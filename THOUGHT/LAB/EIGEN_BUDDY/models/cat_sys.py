"""Quick CatalyticCore test on 2x2 systems."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
import sys; sys.path.insert(0,r'THOUGHT/LAB/EIGEN_BUDDY')
from core.catalytic_core import CatalyticCore

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__();p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)
class CatSysModel(nn.Module):
    def __init__(self,mv=30,d=64,H=8,L=2,r=2):
        super().__init__()
        self.er=nn.Embedding(mv+1,d);self.ei=nn.Embedding(mv+1,d)
        self.pe=CPE(d);self.core=CatalyticCore(d,H,L,r)
        self.out=nn.Linear(d,2)
        for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.er(t),self.ei(t));z=self.pe(z);z=self.core(z)
        return self.out(z.mean(1).real)

def gen(n,mv=6):
    d=[]
    for _ in range(n):
        x=random.randint(0,mv);y=random.randint(0,mv)
        a1=random.randint(1,mv);b1=random.randint(0,mv)
        a2=random.randint(0,mv);b2=random.randint(1,mv)
        if a1*b2-a2*b1==0:continue
        d.append(([a1,b1,a1*x+b1*y,a2,b2,a2*x+b2*y],[x,y]))
    return d

mv=6; data=gen(8000,mv)
random.shuffle(data); n=len(data)*4//5; tr,te=data[:n],data[n:]
me=mv*mv*2+mv
m=CatSysModel(me,64,8,2,2)
opt=torch.optim.AdamW(m.parameters(),lr=5e-3);P=sum(p.numel() for p in m.parameters())
for ep in range(40):
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
print(f"CatalyticCore H=8 L=2 r=2: xy={xy:.1%} P={P:>6,} (prev 85.4%)")
