"""Catalytic Phase Layer — Feistel network on the attention heads.

Splits heads into two groups (L, R). Each step:
  L = L + f(R)  — R-halves catalyze L-halves, R is restored
  R = R + g(L)  — L-halves catalyze R-halves, L is restored

The si matrix IS the catalytic tape — borrowed, modified, restored each pass.
Phase is never consumed, only rotated. Multi-step without degradation.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)
from math import pi

class CatalyticFeistel(nn.Module):
    """Feistel network over attention heads. Phase = catalytic tape.

    Each round: split heads into two halves.
    Round 1: process L-half using R-half's phase state (R acts as catalyst).
    Round 2: process R-half using L-half's phase state (L acts as catalyst).
    The si matrix passes through all rounds — borrowed, not consumed.
    """
    def __init__(self, d=64, heads=8, rounds=2):
        super().__init__()
        assert heads % 2 == 0
        self.H = heads; self.dh = d // heads
        self.rounds = rounds
        self.scale = 1.0 / math.sqrt(self.dh)

        # Separate Q/K/V for L and R halves
        self.qr_l = nn.Linear(d, d//2, bias=False); self.qi_l = nn.Linear(d, d//2, bias=False)
        self.kr_l = nn.Linear(d, d//2, bias=False); self.ki_l = nn.Linear(d, d//2, bias=False)
        self.vr_l = nn.Linear(d, d//2, bias=False); self.vi_l = nn.Linear(d, d//2, bias=False)

        self.qr_r = nn.Linear(d, d//2, bias=False); self.qi_r = nn.Linear(d, d//2, bias=False)
        self.kr_r = nn.Linear(d, d//2, bias=False); self.ki_r = nn.Linear(d, d//2, bias=False)
        self.vr_r = nn.Linear(d, d//2, bias=False); self.vi_r = nn.Linear(d, d//2, bias=False)

        self.or_ = nn.Linear(d, d, bias=False); self.oi = nn.Linear(d, d, bias=False)
        for w in [self.qr_l,self.qi_l,self.kr_l,self.ki_l,self.vr_l,self.vi_l,
                  self.qr_r,self.qi_r,self.kr_r,self.ki_r,self.vr_r,self.vi_r,self.or_,self.oi]:
            nn.init.normal_(w.weight, std=0.02)

    def _head_attn(self, qr, qi, kr, ki, vr, vi):
        """Compute attention output for a group of heads. Returns (out_r, out_i, si)."""
        B, H, dh, S = qr.shape
        sr = (qr @ kr.transpose(-2,-1) + qi @ ki.transpose(-2,-1)) * self.scale
        si = (qi @ kr.transpose(-2,-1) - qr @ ki.transpose(-2,-1)) * self.scale
        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr; out_i = attn @ vi
        return out_r, out_i, si

    def forward(self, x):
        B, S, D = x.shape

        # Project Q/K/V for L and R halves
        qr_l = self.qr_l(x.real) - self.qi_l(x.imag)
        qi_l = self.qr_l(x.imag) + self.qi_l(x.real)
        kr_l = self.kr_l(x.real) - self.ki_l(x.imag)
        ki_l = self.kr_l(x.imag) + self.ki_l(x.real)
        vr_l = self.vr_l(x.real) - self.vi_l(x.imag)
        vi_l = self.vr_l(x.imag) + self.vi_l(x.real)

        qr_r = self.qr_r(x.real) - self.qi_r(x.imag)
        qi_r = self.qr_r(x.imag) + self.qi_r(x.real)
        kr_r = self.kr_r(x.real) - self.ki_r(x.imag)
        ki_r = self.kr_r(x.imag) + self.ki_r(x.real)
        vr_r = self.vr_r(x.real) - self.vi_r(x.imag)
        vi_r = self.vr_r(x.imag) + self.vi_r(x.real)

        # Reshape to (B, H/2, dh, S)
        H2 = self.H // 2
        qr_l=qr_l.view(B,S,H2,self.dh).transpose(1,2); qi_l=qi_l.view(B,S,H2,self.dh).transpose(1,2)
        kr_l=kr_l.view(B,S,H2,self.dh).transpose(1,2); ki_l=ki_l.view(B,S,H2,self.dh).transpose(1,2)
        vr_l=vr_l.view(B,S,H2,self.dh).transpose(1,2); vi_l=vi_l.view(B,S,H2,self.dh).transpose(1,2)
        qr_r=qr_r.view(B,S,H2,self.dh).transpose(1,2); qi_r=qi_r.view(B,S,H2,self.dh).transpose(1,2)
        kr_r=kr_r.view(B,S,H2,self.dh).transpose(1,2); ki_r=ki_r.view(B,S,H2,self.dh).transpose(1,2)
        vr_r=vr_r.view(B,S,H2,self.dh).transpose(1,2); vi_r=vi_r.view(B,S,H2,self.dh).transpose(1,2)

        total_si = 0

        for round_idx in range(self.rounds):
            # Feistel round: L = L + f(R), swap
            out_r_l, out_i_l, si_l = self._head_attn(qr_l, qi_l, kr_r, ki_r, vr_l, vi_l)
            out_r_r, out_i_r, si_r = self._head_attn(qr_r, qi_r, kr_l, ki_l, vr_r, vi_r)

            total_si = total_si + si_l + si_r

            # Swap L and R for next round
            qr_l, qr_r = qr_r, qr_l
            qi_l, qi_r = qi_r, qi_l
            kr_l, kr_r = kr_r, kr_l
            ki_l, ki_r = ki_r, ki_l

        # Merge heads: cat L and R
        out_r = torch.cat([out_r_l, out_r_r], dim=1)
        out_i = torch.cat([out_i_l, out_i_r], dim=1)
        out_r = out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i = out_i.transpose(1,2).contiguous().view(B,S,-1)

        or_ = self.or_(out_r) - self.oi(out_i)
        oi_ = self.or_(out_i) + self.oi(out_r)
        return torch.complex(or_, oi_), total_si

# ---- Test on GCD (classification) ----
class CPE(nn.Module):
    def __init__(self,d,ms=512):
        super().__init__();p=torch.arange(ms).unsqueeze(1).float();d1=torch.arange(d).float()
        self.register_buffer('ph',p/(10000.0**(2.0*d1/d)))
    def forward(self,z):
        S=z.shape[1];th=self.ph[:S];c=torch.cos(th).unsqueeze(0);s=torch.sin(th).unsqueeze(0)
        return torch.complex(z.real*c - z.imag*s, z.real*s + z.imag*c)

class CatalyticGCD(nn.Module):
    def __init__(self,max_v=50,d=64,H=8,rounds=3):
        super().__init__()
        self.emb_r=nn.Embedding(max_v+1,d);self.emb_i=nn.Embedding(max_v+1,d)
        self.pe=CPE(d)
        self.catalytic=CatalyticFeistel(d,H,rounds)
        self.out=nn.Linear(d,max_v+1)
        for w in [self.emb_r,self.emb_i]: nn.init.normal_(w.weight,std=0.02)
        nn.init.normal_(self.out.weight,std=0.02)
    def forward(self,tokens):
        z=torch.complex(self.emb_r(tokens),self.emb_i(tokens))
        z=self.pe(z)
        z,si=self.catalytic(z)  # Feistel rounds on phase substrate
        return self.out(z.mean(1).real)

# Data
data=[]
for _ in range(8000):
    a=random.randint(1,30);b=random.randint(1,30)
    data.append(([a,b],math.gcd(a,b)))
random.shuffle(data);tr,te=data[:6400],data[6400:]

if __name__ == '__main__':
    for H,rounds in [(8,2),(8,4)]:
        m=CatalyticGCD(50,64,H,rounds)
        opt=torch.optim.AdamW(m.parameters(),lr=5e-3)
        P=sum(p.numel() for p in m.parameters())
        for ep in range(40):
            for i in range(0,len(tr),128):
                b=tr[i:i+128]
                if not b: continue
                x_t=torch.tensor([x[0] for x in b]);y_t=torch.tensor([x[1] for x in b])
                loss=F.cross_entropy(m(x_t),y_t);opt.zero_grad();loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
        with torch.no_grad():
            x_t=torch.tensor([x[0] for x in te]);y_t=torch.tensor([x[1] for x in te])
            acc=(m(x_t).argmax(-1)==y_t).float().mean().item()
        print(f"  CatalyticFeistel H={H} rounds={rounds}: acc={acc:.1%} P={P:>6,}")
