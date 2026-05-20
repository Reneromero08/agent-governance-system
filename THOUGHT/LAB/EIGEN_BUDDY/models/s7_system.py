"""Section 7: Systems of Equations — progressive approach.
Step A: learn determinant (a1*b2 - a2*b1) — bilinear, should be learnable.
Step B: learn full solution x,y from the system.
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

class FlexModel(nn.Module):
    def __init__(self,max_val=30,d=32,H=4,L=3,out_dim=1):
        super().__init__()
        self.emb_r=nn.Embedding(max_val+1,d); self.emb_i=nn.Embedding(max_val+1,d)
        self.pe=CPE(d); self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,out_dim)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,tokens):
        z=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        z=self.pe(z)
        for l in self.attns: z=z+l(z)
        return self.out(z.mean(1).real).squeeze(-1)  # (B,) for 1D, (B,D) for 2D

# Step A: Learn determinant (a1*b2 - a2*b1)
def gen_det(n,max_val=6):
    data=[]
    for _ in range(n):
        a1=random.randint(1,max_val); b1=random.randint(0,max_val)
        a2=random.randint(0,max_val); b2=random.randint(1,max_val)
        det=a1*b2-a2*b1
        data.append(([a1,b1,a2,b2],det))
    return data

# Step B: Learn full system solution
def gen_system(n,max_val=5):
    data=[]
    for _ in range(n):
        x=random.randint(0,max_val); y=random.randint(0,max_val)
        a1=random.randint(1,max_val); b1=random.randint(0,max_val)
        a2=random.randint(0,max_val); b2=random.randint(1,max_val)
        if a1*b2-a2*b1==0: continue
        c1=a1*x+b1*y; c2=a2*x+b2*y
        data.append(([a1,b1,c1,a2,b2,c2],[x,y]))
    return data

max_val=6
# Step A: determinant
data=gen_det(5000,max_val)
random.shuffle(data); n_tr=len(data)*4//5; tr_d,te_d=data[:n_tr],data[n_tr:]
max_emb=max_val*max_val

print("="*60)
print("SECTION 7: 2x2 SYSTEMS — Step A: determinant")
print("="*60)
m_d=FlexModel(max_emb,32,4,3,1)
opt=torch.optim.AdamW(m_d.parameters(),lr=5e-3)
for ep in range(30):
    for i in range(0,len(tr_d),128):
        b=tr_d[i:i+128]
        if not b: continue
        x_t=torch.tensor([x[0] for x in b]); tgt=torch.tensor([x[1]/max_emb for x in b],dtype=torch.float32)
        loss=F.mse_loss(m_d(x_t),tgt); opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m_d.parameters(),1.0);opt.step()
with torch.no_grad():
    x_t=torch.tensor([x[0] for x in te_d]); tgt=torch.tensor([x[1] for x in te_d],dtype=torch.float32)
    pred=m_d(x_t)*max_emb
    acc=((pred-tgt).abs()<torch.maximum(tgt.abs()*0.1,torch.tensor(1.0))).float().mean().item()
    # Print a few
    for j in range(5):
        print(f"  [{te_d[j][0]}] det={te_d[j][1]} pred={pred[j].item():.1f}")
print(f"  det acc={acc:.1%} {'PASS' if acc>.90 else ''}")

# Step B: full system
print(f"\n{'='*60}")
print("SECTION 7: Step B — full solution x,y")
print("="*60)
data=gen_system(20000,max_val)
random.shuffle(data); n_tr=len(data)*4//5; tr_s,te_s=data[:n_tr],data[n_tr:]
max_emb_s=max_val*max_val*2+max_val

m_s=FlexModel(max_emb_s,48,6,4,2)
opt=torch.optim.AdamW(m_s.parameters(),lr=5e-3)
sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
for ep in range(80):
    for i in range(0,len(tr_s),128):
        b=tr_s[i:i+128]
        if not b: continue
        x_t=torch.tensor([x[0] for x in b]); tgt=torch.tensor([x[1] for x in b],dtype=torch.float32)/max_val
        loss=F.mse_loss(m_s(x_t),tgt); opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m_s.parameters(),1.0);opt.step()
    sched.step()
with torch.no_grad():
    x_t=torch.tensor([x[0] for x in te_s]); tgt=torch.tensor([x[1] for x in te_s],dtype=torch.float32)
    pred=(m_s(x_t)*max_val).round().clamp(0,max_val)
    xy=((pred[:,0]==tgt[:,0])&(pred[:,1]==tgt[:,1])).float().mean().item()
    x_ok=(pred[:,0]==tgt[:,0]).float().mean().item()
    y_ok=(pred[:,1]==tgt[:,1]).float().mean().item()
    for j in range(5):
        print(f"  [{te_s[j][0]}] x={te_s[j][1][0]} y={te_s[j][1][1]} pred=({pred[j,0].item():.0f},{pred[j,1].item():.0f})")
P=sum(p.numel() for p in m_s.parameters())
print(f"  xy={xy:.1%} x={x_ok:.1%} y={y_ok:.1%} P={P:>6,} {'PASS' if xy>.70 else ''}")
