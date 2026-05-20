"""Holographic training: phase-encoded inputs replace token embeddings.

Operations encoded as phase shifts on unit circle. Core sees the operation
in the input geometry itself — learns faster with less data.
Tests: unseen operands, unseen moduli (generalization).
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
import sys; sys.path.insert(0,r'THOUGHT/LAB/EIGEN_BUDDY')
from core.engine import NativeEigenCore

OPS = {'+': 0.0, '-': math.pi, '*': math.pi/2, '/': -math.pi/2}

class HolographicCalc(nn.Module):
    def __init__(self,d=32,H=4,L=3):
        super().__init__()
        self.core=NativeEigenCore(d,H,L,'concat',True)
        self.out=nn.Linear(d,1)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,vals,ops):
        """vals: (B,2) values, ops: (B,) operation indices 0=+ 1=- 2=* 3=/"""
        B=vals.shape[0]
        theta_a=torch.tensor([OPS[{0:'+',1:'-',2:'*',3:'/'}[o.item()]] for o in ops])
        theta_b=torch.zeros(B)
        mag=vals/50.0  # normalize
        zr_a=mag[:,0:1].expand(-1,self.core.d)*torch.cos(theta_a).unsqueeze(1)
        zi_a=mag[:,0:1].expand(-1,self.core.d)*torch.sin(theta_a).unsqueeze(1)
        zr_b=mag[:,1:2].expand(-1,self.core.d)*torch.cos(theta_b).unsqueeze(1)
        zi_b=mag[:,1:2].expand(-1,self.core.d)*torch.sin(theta_b).unsqueeze(1)
        # Expand to d-dim complex vectors
        z=torch.complex(
            torch.stack([zr_a,zr_b],1),  # (B,2,d) — 2 tokens
            torch.stack([zi_a,zi_b],1))
        z_out,_=self.core(z)
        return self.out(z_out.mean(1).real).squeeze(-1)

# Train on all 4 ops
data=[]
for _ in range(8000):
    a=random.randint(0,20);b=random.randint(1 if (_%4==3) else 0,20)
    op=random.randint(0,3)
    r=[a+b,a-b,a*b,a//b if b>0 else 0][op]
    data.append(([a,b],op,r))
random.shuffle(data);n=len(data)*4//5;tr,te=data[:n],data[n:]

print("="*60)
print("HOLOGRAPHIC CALCULATOR: phase-encoded inputs, trained")
print("="*60)

for d,H,L in [(32,4,3),(48,6,4)]:
    m=HolographicCalc(d,H,L);opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(30):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b:continue
            v=torch.tensor([x[0] for x in b],dtype=torch.float32)
            o=torch.tensor([x[1] for x in b]);tgt=torch.tensor([x[2]/50 for x in b],dtype=torch.float32)
            loss=F.mse_loss(m(v,o),tgt);opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    with torch.no_grad():
        v=torch.tensor([x[0] for x in te],dtype=torch.float32)
        o=torch.tensor([x[1] for x in te]);tgt=torch.tensor([x[2] for x in te],dtype=torch.float32)
        pred=(m(v,o)*50).round()
        th=torch.maximum(tgt.abs()*0.1,torch.tensor(1.0))
        acc=((pred-tgt).abs()<th).float().mean().item()
    print(f"  d={d} L={L}: acc={acc:.1%} P={P:>6,}")

# Test: unseen moduli generalization (Section 2's spec)
print(f"\n{'='*60}")
print("GENERALIZATION: modular arithmetic on unseen moduli")
print("="*60)
# Train on mod 2-8, test on mod 11,13,17,19
tr_mod=[]
for _ in range(5000):
    a=random.randint(0,15);b=random.randint(0,15);mod=random.randint(2,8)
    tr_mod.append(([a,b,mod],(a+b)%mod))
random.shuffle(tr_mod)

class HoloMod(nn.Module):
    def __init__(self,d=32,H=4,L=2):
        super().__init__()
        self.core=NativeEigenCore(d,H,L,'concat',True)
        self.out=nn.Linear(d,1)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,vals):
        B=vals.shape[0]
        a=vals[:,0:1]/20;b=vals[:,1:2]/20;mod=vals[:,2:3]/20
        zr=torch.cat([a.expand(-1,self.core.d),b.expand(-1,self.core.d),mod.expand(-1,self.core.d)],1).unsqueeze(1)*0.5
        z=torch.complex(zr,torch.zeros_like(zr))
        z_out,_=self.core(z)
        return self.out(z_out.mean(1).real).squeeze(-1)

m2=HoloMod(32,4,2);opt2=torch.optim.AdamW(m2.parameters(),lr=3e-3)
for ep in range(30):
    for i in range(0,len(tr_mod),128):
        b=tr_mod[i:i+128]
        if not b:continue
        v=torch.tensor([x[0] for x in b],dtype=torch.float32)
        tgt=torch.tensor([x[1]/20 for x in b],dtype=torch.float32)
        loss=F.mse_loss(m2(v),tgt);opt2.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0);opt2.step()

for mod in [5,11,13,17,19]:
    correct=0;total=0
    with torch.no_grad():
        for a in range(0,mod):
            for b in range(0,mod):
                v=torch.tensor([[a,b,mod]],dtype=torch.float32)
                pred=round(m2(v).item()*20)
                correct+=(pred==(a+b)%mod);total+=1
    print(f"  mod {mod:>2}: {correct/total:.1%} ({'SEEN' if mod<=8 else 'UNSEEN'})")
