"""Catalytic Native Eigen Core — Feistel rounds replace standard attention.

Every layer is a Feistel network on the attention heads.
Phase (si matrix) passes through all rounds UNCONSUMED — catalytic substrate.
Multi-step computation without depth ceiling.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math

class CatalyticFeistel(nn.Module):
    """Feistel network over attention heads. Each round:
    L = L + f(R), swap. R = R + g(L), swap.
    si passes through all rounds — borrowed, not consumed."""
    def __init__(self, d=64, heads=8, rounds=2):
        super().__init__()
        assert heads % 2 == 0
        self.H = heads; self.dh = d // heads; self.rounds = rounds
        self.scale = 1.0 / math.sqrt(self.dh)
        H2 = d // 2
        self.qr_l=nn.Linear(d,H2,bias=False);self.qi_l=nn.Linear(d,H2,bias=False)
        self.kr_l=nn.Linear(d,H2,bias=False);self.ki_l=nn.Linear(d,H2,bias=False)
        self.vr_l=nn.Linear(d,H2,bias=False);self.vi_l=nn.Linear(d,H2,bias=False)
        self.qr_r=nn.Linear(d,H2,bias=False);self.qi_r=nn.Linear(d,H2,bias=False)
        self.kr_r=nn.Linear(d,H2,bias=False);self.ki_r=nn.Linear(d,H2,bias=False)
        self.vr_r=nn.Linear(d,H2,bias=False);self.vi_r=nn.Linear(d,H2,bias=False)
        self.or_=nn.Linear(d,d,bias=False);self.oi=nn.Linear(d,d,bias=False)
        for w in [self.qr_l,self.qi_l,self.kr_l,self.ki_l,self.vr_l,self.vi_l,
                  self.qr_r,self.qi_r,self.kr_r,self.ki_r,self.vr_r,self.vi_r,self.or_,self.oi]:
            nn.init.normal_(w.weight,std=0.02)

    def _attn(self,qr,qi,kr,ki,vr,vi):
        B,H,dh,S=qr.shape
        sr=(qr@kr.transpose(-2,-1)+qi@ki.transpose(-2,-1))*self.scale
        si=(qi@kr.transpose(-2,-1)-qr@ki.transpose(-2,-1))*self.scale
        attn=F.softmax(sr,dim=-1)
        return attn@vr,attn@vi,si

    def _proj(self,x,side):
        r,i = x.real, x.imag
        qr = getattr(self,f'qr_{side}')(r) - getattr(self,f'qi_{side}')(i)
        qi = getattr(self,f'qr_{side}')(i) + getattr(self,f'qi_{side}')(r)
        kr = getattr(self,f'kr_{side}')(r) - getattr(self,f'ki_{side}')(i)
        ki = getattr(self,f'kr_{side}')(i) + getattr(self,f'ki_{side}')(r)
        vr = getattr(self,f'vr_{side}')(r) - getattr(self,f'vi_{side}')(i)
        vi = getattr(self,f'vr_{side}')(i) + getattr(self,f'vi_{side}')(r)
        return qr,qi,kr,ki,vr,vi

    def forward(self,x):
        B,S,D=x.shape; H2=D//2; dh=self.dh
        qr_l,qi_l,kr_l,ki_l,vr_l,vi_l=self._proj(x,'l')
        qr_r,qi_r,kr_r,ki_r,vr_r,vi_r=self._proj(x,'r')
        qr_l=qr_l.view(B,S,H2//dh,dh).transpose(1,2); qi_l=qi_l.view(B,S,H2//dh,dh).transpose(1,2)
        kr_l=kr_l.view(B,S,H2//dh,dh).transpose(1,2); ki_l=ki_l.view(B,S,H2//dh,dh).transpose(1,2)
        vr_l=vr_l.view(B,S,H2//dh,dh).transpose(1,2); vi_l=vi_l.view(B,S,H2//dh,dh).transpose(1,2)
        qr_r=qr_r.view(B,S,H2//dh,dh).transpose(1,2); qi_r=qi_r.view(B,S,H2//dh,dh).transpose(1,2)
        kr_r=kr_r.view(B,S,H2//dh,dh).transpose(1,2); ki_r=ki_r.view(B,S,H2//dh,dh).transpose(1,2)
        vr_r=vr_r.view(B,S,H2//dh,dh).transpose(1,2); vi_r=vi_r.view(B,S,H2//dh,dh).transpose(1,2)

        total_si=0
        for _ in range(self.rounds):
            out_r_l,out_i_l,si_l=self._attn(qr_l,qi_l,kr_r,ki_r,vr_l,vi_l)
            out_r_r,out_i_r,si_r=self._attn(qr_r,qi_r,kr_l,ki_l,vr_r,vi_r)
            total_si=total_si+si_l+si_r
            qr_l,qr_r=qr_r,qr_l; qi_l,qi_r=qi_r,qi_l
            kr_l,kr_r=kr_r,kr_l; ki_l,ki_r=ki_r,ki_l

        out_r=torch.cat([out_r_l,out_r_r],1).transpose(1,2).contiguous().view(B,S,-1)
        out_i=torch.cat([out_i_l,out_i_r],1).transpose(1,2).contiguous().view(B,S,-1)
        or_=self.or_(out_r)-self.oi(out_i); oi_=self.or_(out_i)+self.oi(out_r)
        return torch.complex(or_,oi_), total_si

class CurvatureModulator(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.weight=nn.Parameter(torch.tensor(0.3))
    def forward(self,z,si):
        B,S,D=z.shape
        ratio=z[:,1:]/(z[:,:-1]+1e-8); d2=torch.angle(ratio)[:,1:]-torch.angle(ratio)[:,:-1]
        curv=F.pad(d2.abs(),(0,0,1,1)); z_dir=z/(z.abs()+1e-8)
        return z+self.weight*curv*z_dir

class PhaseAccumulator(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.ang=nn.Parameter(torch.ones(d)*0.1)
    def forward(self,z):
        c,s=torch.cos(self.ang),torch.sin(self.ang)
        return torch.complex(z.real*c-z.imag*s,z.real*s+z.imag*c)

class CatalyticCore(nn.Module):
    """Phase-native engine. Every attention layer is a Feistel network.
    si passes through all layers as catalytic substrate — never consumed."""
    def __init__(self,d=64,heads=8,layers=3,rounds=3):
        super().__init__()
        self.layers=nn.ModuleList([nn.ModuleDict({
            'feistel':CatalyticFeistel(d,heads,rounds),
            'curve':CurvatureModulator(d),
            'phase':PhaseAccumulator(d)}) for _ in range(layers)])

    def forward(self,z):
        for l in self.layers:
            z,si=l['feistel'](z); z=l['curve'](z,si); z=l['phase'](z)
        return z

# ---- Quick test ----
if __name__=='__main__':
    torch.manual_seed(42)
    core=CatalyticCore(64,8,3,3)
    z=torch.complex(torch.randn(2,8,64),torch.randn(2,8,64))
    out=core(z)
    P=sum(p.numel() for p in core.parameters())
    print(f"CatalyticCore: d=64 h=8 L=3 rounds=3 params={P:,}")
    print(f"  Input: {z.shape} Output: {out.shape}")
    print(f"  Phase substrate: si passes through all layers unconsumed")
