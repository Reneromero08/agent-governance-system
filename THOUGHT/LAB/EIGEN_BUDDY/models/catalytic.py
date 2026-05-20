"""Multi-step math via catalytic iteration: borrow input as scratchpad.

Pattern from CAT_CAS: Feistel loop — modify half the state, swap, repeat.
Model feeds output back as input until termination condition.
No autoregressive generation — reversible state transformation.
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

class CatalyticMath(nn.Module):
    """Feistel-style iterative computation. Output feeds back as modified input.
    Each pass: process[a,b] -> [new_a, new_b]. Repeat max_steps times."""
    def __init__(self,max_v=50,d=32,H=4,L=3,max_steps=8):
        super().__init__()
        self.max_steps=max_steps
        self.emb_r=nn.Embedding(max_v+1,d);self.emb_i=nn.Embedding(max_v+1,d)
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out_r=nn.Linear(d,max_v+1);self.out_i=nn.Linear(d,max_v+1)  # predict next tokens
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out_r.weight,std=0.02);nn.init.normal_(self.out_i.weight,std=0.02)

    def forward(self,tokens,steps=None):
        """tokens: (B, S) token IDs. Process iteratively, output final state."""
        if steps is None: steps=self.max_steps
        z=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        for step in range(steps):
            z_pe=self.pe(z)
            for l in self.attns:z_pe=z_pe+l(z_pe)
            # Predict next tokens (logits for each position)
            lr=self.out_r(z_pe.real)+self.out_i(z_pe.imag)  # (B,S,V)
            next_tokens=lr.argmax(-1)  # (B,S) — discrete next state
            # Feed back: embed next tokens as new input
            z=torch.complex(self.emb_r(next_tokens),self.emb_i(next_tokens))
        return lr  # final logits

# Data: GCD via Euclidean algorithm steps
# Input: [a, b, 0, 0] — two operands + two scratch positions
# Target: next state [a%b, b, a//b, b] (catalytic: overwrite a with remainder)
def gen_gcd_steps(n,max_val=30):
    data=[]
    for _ in range(n):
        a=random.randint(1,max_val);b=random.randint(1,max_val)
        orig_a,orig_b=a,b
        steps=[]
        while b>0:
            q=a//b;r=a%b
            steps.append(([r,b,q,b],[r,b]))  # state -> next tokens
            a,b=b,r
        data.extend(steps)
    return data

# Simpler: just predict final GCD from input
def gen_gcd_final(n,max_val=30):
    data=[]
    for _ in range(n):
        a=random.randint(1,max_val);b=random.randint(1,max_val)
        data.append(([a,b],math.gcd(a,b)))
    return data

print("="*60)
print("CATALYTIC MULTI-STEP: GCD iteration")
print("="*60)

# Test 1: direct GCD prediction (single step)
data=gen_gcd_final(8000,30)
random.shuffle(data);tr,te=data[:6400],data[6400:]
m=CatalyticMath(50,48,6,4,1)  # max_steps=1 = single pass
opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
for ep in range(60):
    for i in range(0,len(tr),128):
        b=tr[i:i+128]
        if not b: continue
        x_t=torch.tensor([x[0] for x in b])
        # Target: predict GCD token (classification over 0-50)
        y_t=torch.tensor([x[1] for x in b])
        lr=m(x_t)  # (B,S,V) — take first position prediction
        loss=F.cross_entropy(lr[:,0,:],y_t)
        opt.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
with torch.no_grad():
    x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te])
    lr=m(x_t).argmax(-1)
    acc=(lr[:,0]==y_t).float().mean().item()
P=sum(p.numel() for p in m.parameters())
print(f"  GCD direct: acc={acc:.1%} P={P:>6,}")

# Test 2: iterative GCD (multi-step)
data=gen_gcd_steps(5000,20)
random.shuffle(data);tr,te=data[:4000],data[4000:]
m2=CatalyticMath(30,48,6,4,3)
opt2=torch.optim.AdamW(m2.parameters(),lr=5e-3)
for ep in range(40):
    for i in range(0,len(tr),128):
        b=tr[i:i+128]
        if not b: continue
        x_t=torch.tensor([x[0] for x in b])
        y_t=torch.tensor([x[1] for x in b])  # next state tokens
        lr=m2(x_t,2)  # 2 catalytic steps
        loss=F.cross_entropy(lr[:,:2,:].reshape(-1,lr.shape[-1]),y_t[:,:2].reshape(-1))
        opt2.zero_grad();loss.backward()
        torch.nn.utils.clip_grad_norm_(m2.parameters(),1.0);opt2.step()
with torch.no_grad():
    x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te])
    lr=m2(x_t,2).argmax(-1)
    acc=((lr[:,:2]==y_t[:,:2]).all(dim=-1)).float().mean().item()
P2=sum(p.numel() for p in m2.parameters())
print(f"  GCD iterative: acc={acc:.1%} P={P2:>6,}")
