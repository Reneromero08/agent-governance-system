"""Retest Sections 13, 15 with CatalyticFeistel layer."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
import sys; sys.path.insert(0,r'THOUGHT/LAB/EIGEN_BUDDY')
from core.catalytic import CatalyticFeistel

class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__();p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class CatModel(nn.Module):
    def __init__(self,max_v=30,d=64,H=8,rounds=3,out_d=1):
        super().__init__()
        self.e_r=nn.Embedding(max_v+1,d);self.e_i=nn.Embedding(max_v+1,d)
        self.pe=CPE(d);self.cat=CatalyticFeistel(d,H,rounds)
        self.out=nn.Linear(d,out_d)
        for w in [self.e_r,self.e_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.e_r(t),self.e_i(t));z=self.pe(z)
        z,si=self.cat(z)
        return self.out(z.mean(1).real).squeeze(-1)

# S13: Set cardinality
def gen_set(n,max_val=12):
    data=[]
    for _ in range(n):
        s=[random.randint(0,max_val) for _ in range(random.randint(2,6))]
        s=s+[0]*(6-len(s))
        data.append((s,len(set(s))))
    return data

# S15: Graph degree
def gen_graph(n,max_nodes=6):
    data=[]
    for _ in range(n):
        adj=[random.randint(0,1) for _ in range(max_nodes*max_nodes)]
        node=random.randint(0,max_nodes-1)
        deg=sum(adj[node*max_nodes:(node+1)*max_nodes])
        data.append((adj+[node],deg))
    return data

print("="*60)
print("CATALYTIC FEISTEL: Retest Sets + Graphs")
print("="*60)

for name,gen_fn,out_d,epochs,thresh in [
    ("S13: set cardinality",gen_set,1,40,10),
    ("S15: graph degree",gen_graph,1,40,10),
]:
    data=gen_fn(8000)
    random.shuffle(data);tr,te=data[:6400],data[6400:]
    max_emb=max(max(x[0]) for x in data)
    m=CatModel(max_emb+5,64,8,3,out_d)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(epochs):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b])
            y_t=torch.tensor([x[1] for x in b],dtype=torch.float32)
            loss=F.mse_loss(m(x_t),y_t/thresh)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te])
        y_t=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=(m(x_t)*thresh).round()
        acc=(pred==y_t).float().mean().item()
    prev="(was 55.6%)" if "set" in name else "(was 33.8%)"
    print(f"  {name}: acc={acc:.1%} {prev} P={P:>6,} {'PASS' if acc>.85 else ''}")
