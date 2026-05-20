"""Sections 11-15: Linear Algebra, Logic, Sets, Number Theory, Graph Theory.
Each section uses the same architecture (CPE + BA + token embeddings).
Key learning: separate embeddings for operands that need bilinear routing.
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

class FlexModel(nn.Module):
    def __init__(self,max_v=30,d=32,H=4,L=3,out_d=1):
        super().__init__()
        self.e_r=nn.Embedding(max_v+1,d);self.e_i=nn.Embedding(max_v+1,d)
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,out_d)
        for w in [self.e_r,self.e_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,t):
        z=torch.complex(self.e_r(t),self.e_i(t));z=self.pe(z)
        for l in self.attns:z=z+l(z)
        o=self.out(z.mean(1).real)
        return o.squeeze(-1)  # (B,) for out_d=1, (B,out_d) for out_d>1

def train_section(name,gen_fn,n=10000,out_d=1,epochs=60,threshold=None,acc_metric=None):
    data=gen_fn(n)
    random.shuffle(data);n_tr=len(data)*4//5;tr,te=data[:n_tr],data[n_tr:]
    max_emb=max(max(x[0]) for x in data) if hasattr(data[0][0],'__iter__') else 30
    m=FlexModel(max_emb+5,48,6,4,out_d)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=epochs)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(epochs):
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b: continue
            x_t=torch.tensor([x[0] for x in b])
            y_t=torch.tensor([x[1] for x in b],dtype=torch.float32)
            pred=m(x_t);loss=F.mse_loss(pred,y_t/threshold if threshold else y_t)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    with torch.no_grad():
        x_t=torch.tensor([x[0] for x in te])
        y_t=torch.tensor([x[1] for x in te],dtype=torch.float32)
        pred=m(x_t)*(threshold if threshold else 1)
        if acc_metric: acc=acc_metric(pred,y_t)
        else: acc=((pred.round()-y_t).abs()<1).float().mean().item()
    print(f"  {name}: acc={acc:.1%} P={P:>6,}")

# S11: Linear Algebra - dot product
def gen_dot(n,max_val=10,dim=4):
    data=[]
    for _ in range(n):
        v1=[random.randint(0,max_val) for _ in range(dim)]
        v2=[random.randint(0,max_val) for _ in range(dim)]
        dot=sum(a*b for a,b in zip(v1,v2))
        data.append((v1+v2,dot))
    return data

# S12: Logic - boolean AND, OR, NOT
def gen_logic(n):
    data=[]
    for _ in range(n):
        a=random.randint(0,1);b=random.randint(0,1);op=random.randint(0,2)
        if op==0:r=a&b; op_tok=2
        elif op==1:r=a|b; op_tok=3
        else:r=1-a;b=0;op_tok=4
        data.append(([a,op_tok,b],r))
    return data

# S13: Set Theory - cardinality
def gen_set(n,max_val=15):
    data=[]
    for _ in range(n):
        s=[random.randint(0,max_val) for _ in range(random.randint(3,8))]
        s=s+[0]*(8-len(s))  # pad to 8 tokens
        data.append((s,len(set(s))))
    return data

# S14: Number Theory - GCD
def gen_gcd(n,max_val=30):
    data=[]
    for _ in range(n):
        a=random.randint(1,max_val);b=random.randint(1,max_val)
        r=math.gcd(a,b)
        data.append(([a,b],r))
    return data

# S15: Graph Theory - degree
def gen_degree(n,max_nodes=6):
    data=[]
    for _ in range(n):
        adj=[random.randint(0,1) for _ in range(max_nodes*max_nodes)]
        node=random.randint(0,max_nodes-1)
        deg=sum(adj[node*max_nodes:(node+1)*max_nodes])
        data.append((adj+[node],deg))
    return data

print("="*60)
print("SECTIONS 11-15: Linear Algebra, Logic, Sets, Number Theory, Graph Theory")
print("="*60)

train_section("S11: dot product",gen_dot,8000,out_d=1,epochs=60,threshold=50,
    acc_metric=lambda p,y:((p.squeeze()-y).abs()<torch.maximum(y.abs()*0.1,torch.tensor(3.0))).float().mean().item())
train_section("S12: logic ops",gen_logic,5000,out_d=1,epochs=30,threshold=1)
train_section("S13: set cardinality",gen_set,6000,out_d=1,epochs=30,threshold=8)
train_section("S14: GCD",gen_gcd,6000,out_d=1,epochs=60,threshold=30)
train_section("S15: graph degree",gen_degree,6000,out_d=1,epochs=30,threshold=8)
