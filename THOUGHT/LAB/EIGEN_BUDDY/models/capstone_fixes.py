"""Capstone fixes: sharpen S7 logits, separate S15 node embed, bucketize sin."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time, sys
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

# ===== FIX 1: Section 7 — focal loss to sharpen large-value logits =====
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
    """Focal loss: down-weight easy examples, focus on hard boundary cases."""
    ce=F.cross_entropy(logits,target,reduction='none')
    pt=torch.exp(-ce)
    return ((1-pt)**gamma*ce).mean()

print("FIX 1: Section 7 — focal loss")
mv=6;d1=gen_sys(30000,mv);random.shuffle(d1);n=len(d1)*4//5;tr1,te1=d1[:n],d1[n:]
sys.path.insert(0,r'THOUGHT/LAB/EIGEN_BUDDY')
from models.push_s7 import StructSys,make_cramer_bias,StructAttention
m1=StructSys(150,64,8,4,7)
m1.layers=nn.ModuleList([StructAttention(64,8,make_cramer_bias(8,6)) for _ in range(4)])
opt1=torch.optim.AdamW(m1.parameters(),lr=5e-3)
for ep in range(30):
    for i in range(0,len(tr1),128):
        b=tr1[i:i+128]
        if not b:continue
        x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b])
        lx,ly=m1(x_t)
        loss=focal_loss(lx,y_t[:,0])+focal_loss(ly,y_t[:,1])
        opt1.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m1.parameters(),1.0);opt1.step()
with torch.no_grad():
    x_t=torch.tensor([x[0] for x in te1]);y_t=torch.tensor([x[1] for x in te1])
    lx,ly=m1(x_t);px=lx.argmax(-1);py=ly.argmax(-1)
    xy=((px==y_t[:,0])&(py==y_t[:,1])).float().mean().item()
print(f"  S7 focal loss: xy={xy:.1%} (was 98.5%)")

# ===== FIX 2: Section 15 — separate target node embedding =====
class GraphModel(nn.Module):
    def __init__(self,d=64,H=8,L=3):
        super().__init__()
        self.edge_er=nn.Embedding(3,d);self.edge_ei=nn.Embedding(3,d)  # 0/1 edge values
        self.node_er=nn.Embedding(6,d);self.node_ei=nn.Embedding(6,d)  # target node index
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,7)  # degree 0-6
        for w in [self.edge_er,self.edge_ei,self.node_er,self.node_ei]:
            nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,adj_tok,node_tok):
        """adj_tok: (B,36) adjacency, node_tok: (B,) target node"""
        z_adj=torch.complex(self.edge_er(adj_tok),self.edge_ei(adj_tok))
        z_node=torch.complex(self.node_er(node_tok),self.node_ei(node_tok)).unsqueeze(1)
        z=torch.cat([z_node,z_adj],1);z=self.pe(z)
        for l in self.attns:z=z+l(z)
        return self.out(z.mean(1).real)

def gen_graph(n,mn=6):
    d=[]
    for _ in range(n):
        adj=[random.randint(0,1) for _ in range(mn*mn)]
        node=random.randint(0,mn-1)
        deg=sum(adj[node*mn:(node+1)*mn])
        d.append((adj,node,deg))
    return d

print("\nFIX 2: Section 15 — separate node embedding")
d15=gen_graph(8000,6);random.shuffle(d15);n=len(d15)*4//5;tr15,te15=d15[:n],d15[n:]
m15=GraphModel(64,8,3)
opt15=torch.optim.AdamW(m15.parameters(),lr=5e-3)
for ep in range(40):
    for i in range(0,len(tr15),128):
        b=tr15[i:i+128]
        if not b:continue
        a_t=torch.tensor([x[0] for x in b]);n_t=torch.tensor([x[1] for x in b])
        y_t=torch.tensor([x[2] for x in b])
        loss=F.cross_entropy(m15(a_t,n_t),y_t)
        opt15.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m15.parameters(),1.0);opt15.step()
with torch.no_grad():
    a_t=torch.tensor([x[0] for x in te15]);n_t=torch.tensor([x[1] for x in te15])
    y_t=torch.tensor([x[2] for x in te15])
    pred=m15(a_t,n_t).argmax(-1)
    acc=(pred==y_t).float().mean().item()
print(f"  S15 separate node: acc={acc:.1%} (was 93.7%)")

# ===== FIX 3: Sin — bucketize into classification =====
print("\nFIX 3: Sin — bucketize into 20 discrete classes")
data_sin=[]
for _ in range(8000):
    deg=random.randint(0,359)
    val=math.sin(deg*math.pi/180)
    bucket=min(19,int((val+1)*10))  # [-1,1] -> [0,19]
    data_sin.append((deg,bucket))
random.shuffle(data_sin);n=len(data_sin)*4//5;tr_s,te_s=data_sin[:n],data_sin[n:]
class SinModel(nn.Module):
    def __init__(self,d=32,H=4,L=2):
        super().__init__()
        self.er=nn.Embedding(360,d);self.ei=nn.Embedding(360,d)
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,20)
        for w in [self.er,self.ei]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,deg):
        z=torch.complex(self.er(deg),self.ei(deg)).unsqueeze(1);z=self.pe(z)
        for l in self.attns:z=z+l(z)
        return self.out(z.mean(1).real)
m_s=SinModel(32,4,2);opt_s=torch.optim.AdamW(m_s.parameters(),lr=5e-3)
for ep in range(30):
    for i in range(0,len(tr_s),128):
        b=tr_s[i:i+128]
        if not b:continue
        d_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b])
        loss=F.cross_entropy(m_s(d_t),y_t)
        opt_s.zero_grad();loss.backward();opt_s.step()
with torch.no_grad():
    d_t=torch.tensor([x[0] for x in te_s]);y_t=torch.tensor([x[1] for x in te_s])
    acc=(m_s(d_t).argmax(-1)==y_t).float().mean().item()
print(f"  Sin classification: acc={acc:.1%}")
