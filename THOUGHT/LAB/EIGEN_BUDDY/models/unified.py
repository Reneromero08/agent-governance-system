"""Unified Math Model — 15 sections, one architecture, domain-tag routing.

Takes [section_tag, ...problem_tokens] and outputs the answer.
The model learns to route through the appropriate operation based on the tag.
Meta-learner: addition, multiplication, systems, calculus — one model handles all.
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

class UnifiedMath(nn.Module):
    def __init__(self,n_sections=15,max_v=200,d=64,H=8,L=4,out_d=1):
        super().__init__()
        self.tag_emb=nn.Embedding(n_sections,d)
        # A-side embeddings (even positions) and B-side embeddings (odd positions)
        self.val_a_r=nn.ModuleList([nn.Embedding(max_v+1,d) for _ in range(n_sections)])
        self.val_a_i=nn.ModuleList([nn.Embedding(max_v+1,d) for _ in range(n_sections)])
        self.val_b_r=nn.ModuleList([nn.Embedding(max_v+1,d) for _ in range(n_sections)])
        self.val_b_i=nn.ModuleList([nn.Embedding(max_v+1,d) for _ in range(n_sections)])
        self.pe=CPE(d);self.attns=nn.ModuleList([BA(d,H) for _ in range(L)])
        self.out=nn.Linear(d,out_d)
        for emb_list in [self.val_a_r,self.val_a_i,self.val_b_r,self.val_b_i]:
            for emb in emb_list: nn.init.normal_(emb.weight,std=0.02)
        nn.init.normal_(self.tag_emb.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)

    def forward(self,tag,tokens):
        B,S=tokens.shape
        z_tag=torch.complex(self.tag_emb(tag),torch.zeros_like(self.tag_emb(tag))).unsqueeze(1)
        z_tok=torch.zeros(B,S,self.val_a_r[0].weight.shape[1],dtype=torch.cfloat,device=tokens.device)
        for sec in range(len(self.val_a_r)):
            mask=(tag==sec)
            if mask.any():
                # Even positions -> A-side, odd positions -> B-side
                tok=tokens[mask]
                for pos in range(S):
                    if pos%2==0:
                        z_tok[mask,pos]=torch.complex(self.val_a_r[sec](tok[:,pos]),self.val_a_i[sec](tok[:,pos]))
                    else:
                        z_tok[mask,pos]=torch.complex(self.val_b_r[sec](tok[:,pos]),self.val_b_i[sec](tok[:,pos]))
        z=torch.cat([z_tag,z_tok],1);z=self.pe(z)
        for l in self.attns:z=z+l(z)
        return self.out(z.mean(1).real).squeeze(-1)

# ---- Data: mix all sections with domain tags ----
def get_batch(n_per_section=500):
    data=[]
    mv=30
    # Section 1: addition
    for _ in range(n_per_section):
        a=random.randint(0,mv);b=random.randint(0,mv)
        data.append((0,[a,b],a+b))
    # Section 1: multiplication (more data, it's the bottleneck)
    for _ in range(n_per_section*2):
        a=random.randint(0,min(15,mv));b=random.randint(0,min(15,mv))
        data.append((1,[a,b],a*b))
    # Section 3: linear equations ax+b=c find x
    for _ in range(n_per_section):
        a=random.randint(1,8);x=random.randint(0,8);b=random.randint(0,8)
        data.append((2,[a,b,a*x+b],x))
    # Section 6: arithmetic sequences next term
    for _ in range(n_per_section):
        a=random.randint(0,10);d=random.randint(1,5)
        seq=[a+d*i for i in range(5)]
        data.append((3,seq,a+d*5))
    # Section 8: derivative of ax^n -> a*n, n-1
    for _ in range(n_per_section):
        a=random.randint(1,6);n=random.randint(1,4)
        data.append((4,[a,n],a*n))
    # Section 9: sin(degrees)
    for _ in range(n_per_section):
        deg=random.randint(0,359)
        val=math.sin(deg*math.pi/180)
        data.append((5,[deg],int(val*100+100)))  # scaled sin
    # Section 12: boolean AND
    for _ in range(n_per_section):
        a=random.randint(0,1);b=random.randint(0,1)
        data.append((6,[a,b],a&b))
    return data

data=get_batch(2000)
random.shuffle(data);n=len(data)*4//5;tr,te=data[:n],data[n:]

print("="*60)
print("UNIFIED MATH MODEL — A/B embeddings per section")
print(f"  Train: {len(tr)}  Test: {len(te)}  (7 sections)")
print("="*60)

max_per_sec={0:60,1:225,2:8,3:60,4:24,5:200,6:1}

for d,H,L in [(64,8,4)]:
    m=UnifiedMath(7,200,d,H,L)
    opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=80)
    P=sum(p.numel() for p in m.parameters())
    for ep in range(80):
        tl=0
        random.shuffle(tr)
        for i in range(0,len(tr),128):
            b=tr[i:i+128]
            if not b:continue
            tag_t=torch.tensor([x[0] for x in b])
            tok_t=torch.tensor([[min(x[1][j],200) if j<len(x[1]) else 0 for j in range(6)] for x in b])
            y_t=torch.tensor([x[2]/max_per_sec.get(t.item(),60) for x,t in zip(b,tag_t)],dtype=torch.float32)
            pred=m(tag_t,tok_t)
            loss=F.mse_loss(pred,y_t)
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        sched.step()
    sec_names={0:'add',1:'mult',2:'linear',3:'seq',4:'deriv',5:'sin',6:'bool'}
    with torch.no_grad():
        for sec_id in range(7):
            mask=torch.tensor([x[0] for x in te])==sec_id
            if not mask.any():continue
            idx=mask.nonzero(as_tuple=True)[0][:100]
            tag_t=torch.tensor([te[i][0] for i in idx])
            tok_t=torch.tensor([[min(te[i][1][j],200) if j<len(te[i][1]) else 0 for j in range(6)] for i in idx])
            y_t=torch.tensor([te[i][2] for i in idx],dtype=torch.float32)
            pred=(m(tag_t,tok_t)*max_per_sec[sec_id])
            if sec_id in [0,1,2,3,4,6]:
                acc=(pred.round()==y_t).float().mean().item()
                print(f"  {sec_names[sec_id]:>6}: acc={acc:.1%}")
            else:
                mae=(pred-y_t).abs().mean().item()
                print(f"  {sec_names[sec_id]:>6}: mae={mae:.1f}")
    print(f"  params={P:>6,}")
