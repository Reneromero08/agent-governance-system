"""Math Curriculum — Section 1 & 2. Position-encoded bidirectional complex attention."""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

# ---- Shared components ----
class ComplexPositionEncoding(nn.Module):
    def __init__(self, d, max_seq=512):
        super().__init__()
        pos = torch.arange(max_seq).unsqueeze(1).float()
        dim = torch.arange(d).float()
        self.register_buffer('phase', pos / (10000.0 ** (2.0*dim/d)))
    def forward(self, z):
        S = z.shape[1]; th = self.phase[:S]
        c = torch.cos(th).unsqueeze(0); s = torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class BidiAttn(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.H=heads; self.dh=d//heads
        self.qr=nn.Linear(d,d,bias=False); self.qi=nn.Linear(d,d,bias=False)
        self.kr=nn.Linear(d,d,bias=False); self.ki=nn.Linear(d,d,bias=False)
        self.vr=nn.Linear(d,d,bias=False); self.vi=nn.Linear(d,d,bias=False)
        self.or_=nn.Linear(d,d,bias=False); self.oi=nn.Linear(d,d,bias=False)
        self.sc=1.0/math.sqrt(self.dh)
        for w in [self.qr,self.qi,self.kr,self.ki,self.vr,self.vi,self.or_,self.oi]:
            nn.init.normal_(w.weight,std=0.02)
    def forward(self,x):
        B,S,D=x.shape
        qr=self.qr(x.real)-self.qi(x.imag); qi=self.qr(x.imag)+self.qi(x.real)
        kr=self.kr(x.real)-self.ki(x.imag); ki=self.kr(x.imag)+self.ki(x.real)
        vr=self.vr(x.real)-self.vi(x.imag); vi=self.vr(x.imag)+self.vi(x.real)
        qr=qr.view(B,S,self.H,self.dh).transpose(1,2); qi=qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr=kr.view(B,S,self.H,self.dh).transpose(1,2); ki=ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr=vr.view(B,S,self.H,self.dh).transpose(1,2); vi=vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.sc
        attn=F.softmax(sr,dim=-1)
        out_r=attn@vr; out_i=attn@vi
        out_r=out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i=out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r)-self.oi(out_i), self.or_(out_i)+self.oi(out_r))

# ---- Section 1: Arithmetic ----
def s1_arithmetic():
    class ArithModel(nn.Module):
        def __init__(self, max_val=50, d=32, heads=4, layers=2):
            super().__init__()
            self.num_re=nn.Embedding(max_val+1,d); self.num_im=nn.Embedding(max_val+1,d)
            self.op_re=nn.Embedding(4,d); self.op_im=nn.Embedding(4,d)
            self.pos_enc=ComplexPositionEncoding(d)
            self.layers=nn.ModuleList([BidiAttn(d,heads) for _ in range(layers)])
            self.out=nn.Linear(d,1)
            for w in [self.num_re,self.num_im,self.op_re,self.op_im]: nn.init.normal_(w.weight,std=0.02)
            nn.init.normal_(self.out.weight,std=0.02)
        def forward(self,a_t,o_t,b_t):
            va=torch.complex(self.num_re(a_t),self.num_im(a_t)).unsqueeze(1)
            vo=torch.complex(self.op_re(o_t),self.op_im(o_t)).unsqueeze(1)
            vb=torch.complex(self.num_re(b_t),self.num_im(b_t)).unsqueeze(1)
            z=torch.cat([va,vo,vb],1); z=self.pos_enc(z)
            for l in self.layers: z=z+l(z)
            return self.out(z.mean(1).real).squeeze(-1)

    max_val=50
    for op_idx,op_name in [(0,"+"),(1,"-"),(2,"*"),(3,"//")]:
        if op_idx==2:
            data=[(random.randint(0,max_val),op_idx,random.randint(0,max_val),0) for _ in range(8000)]
            data=[(a,op,b,a*b) for a,op,b,_ in data]
            m=ArithModel(max_val,48,6,3); epochs=30
        else:
            data=[(random.randint(0,max_val),op_idx,random.randint(1,max_val) if op_idx==3 else random.randint(0,max_val),0) for _ in range(5000)]
            data=[(a,op,b,a+b if op==0 else a-b if op==1 else a//b) for a,op,b,_ in data]
            m=ArithModel(max_val,32,4,2); epochs=20
        random.shuffle(data); tr,te=data[:int(len(data)*.8)],data[int(len(data)*.8):]
        mr={0:max_val*2,1:max_val,2:max_val*max_val,3:max_val}[op_idx]
        opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
        for ep in range(epochs):
            for i in range(0,len(tr),128):
                b=tr[i:i+128]
                if not b: continue
                a_t=torch.tensor([x[0] for x in b]); o_t=torch.tensor([x[1] for x in b])
                b_t=torch.tensor([x[2] for x in b])
                tgt=torch.tensor([x[3]/mr for x in b],dtype=torch.float32)
                loss=F.mse_loss(m(a_t,o_t,b_t),tgt)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        with torch.no_grad():
            a_t=torch.tensor([x[0] for x in te]); o_t=torch.tensor([x[1] for x in te])
            b_t=torch.tensor([x[2] for x in te])
            tgt=torch.tensor([x[3]/mr for x in te],dtype=torch.float32)
            pred=m(a_t,o_t,b_t)
            th=torch.maximum(torch.tensor(1.0),(tgt*mr).abs()*.05)
            acc=((pred*mr-tgt*mr).abs()<th).float().mean().item()
        P=sum(p.numel() for p in m.parameters())
        print(f"  {op_name}: {acc:.1%} P={P:>6,} {'PASS' if acc>.95 else ''}")

# ---- Section 2: Modular Arithmetic ----
def s2_modular():
    class ModModel(nn.Module):
        def __init__(self, max_val=30, max_mod=13, d=32, heads=4, layers=2):
            super().__init__()
            self.num_re=nn.Embedding(max_val+1,d); self.num_im=nn.Embedding(max_val+1,d)
            self.op_re=nn.Embedding(4,d); self.op_im=nn.Embedding(4,d)
            self.mod_re=nn.Embedding(max_mod+1,d); self.mod_im=nn.Embedding(max_mod+1,d)
            self.pos_enc=ComplexPositionEncoding(d)
            self.layers=nn.ModuleList([BidiAttn(d,heads) for _ in range(layers)])
            self.out=nn.Linear(d,1)
            for w in [self.num_re,self.num_im,self.op_re,self.op_im,self.mod_re,self.mod_im]:
                nn.init.normal_(w.weight,std=0.02)
            nn.init.normal_(self.out.weight,std=0.02)
        def forward(self,a_t,o_t,b_t,m_t):
            va=torch.complex(self.num_re(a_t),self.num_im(a_t)).unsqueeze(1)
            vo=torch.complex(self.op_re(o_t),self.op_im(o_t)).unsqueeze(1)
            vb=torch.complex(self.num_re(b_t),self.num_im(b_t)).unsqueeze(1)
            vm=torch.complex(self.mod_re(m_t),self.mod_im(m_t)).unsqueeze(1)
            z=torch.cat([va,vo,vb,vm],1); z=self.pos_enc(z)
            for l in self.layers: z=z+l(z)
            return self.out(z.mean(1).real).squeeze(-1)

    for mod_n in [2,3,5,7,11,13]:
        data=[]
        for _ in range(200):
            for a in range(min(20,mod_n*3)):
                for b in range(min(20,mod_n*3)):
                    data.append((a,0,b,mod_n,(a+b)%mod_n))
        random.shuffle(data); tr,te=data[:int(len(data)*.8)],data[int(len(data)*.8):]
        m=ModModel(max_val=30,max_mod=mod_n,d=32,heads=4,layers=2)
        opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
        for ep in range(20):
            for i in range(0,len(tr),128):
                b=tr[i:i+128]
                if not b: continue
                a_t=torch.tensor([x[0] for x in b]); o_t=torch.tensor([x[1] for x in b])
                b_t=torch.tensor([x[2] for x in b]); m_t=torch.tensor([x[3] for x in b])
                tgt=torch.tensor([x[4]/mod_n for x in b],dtype=torch.float32)
                loss=F.mse_loss(m(a_t,o_t,b_t,m_t),tgt)
                opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        with torch.no_grad():
            a_t=torch.tensor([x[0] for x in te]); o_t=torch.tensor([x[1] for x in te])
            b_t=torch.tensor([x[2] for x in te]); m_t=torch.tensor([x[3] for x in te])
            tgt=torch.tensor([x[4] for x in te],dtype=torch.float32)
            pred=(m(a_t,o_t,b_t,m_t)*mod_n).round().clamp(0,mod_n-1)
            acc=(pred==tgt).float().mean().item()
        P=sum(p.numel() for p in m.parameters())
        print(f"  + mod {mod_n:>2}: {acc:.1%} P={P:>5,} {'OK' if acc>.90 else ''}")

if __name__ == '__main__':
    print("="*60); print("SECTION 1: ARITHMETIC"); print("="*60)
    s1_arithmetic()
    print("\n"+"="*60); print("SECTION 2: MODULAR ARITHMETIC"); print("="*60)
    s2_modular()
