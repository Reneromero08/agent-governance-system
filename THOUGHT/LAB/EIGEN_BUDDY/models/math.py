"""Section 1: Arithmetic — with complex position encoding.

Position encoding adds position-dependent phase rotation before attention.
Order becomes phase. a - b ≠ b - a because positions 0 and 2 differ.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, random, time
torch.manual_seed(42); random.seed(42)

class ComplexPositionEncoding(nn.Module):
    """Fixed sinusoidal position encoding as complex phase rotation."""
    def __init__(self, d_model, max_seq=512):
        super().__init__()
        pos = torch.arange(0, max_seq).unsqueeze(1).float()
        dim = torch.arange(0, d_model).float()
        wavelength = 10000.0 ** (2.0 * dim / d_model)
        phase = pos / wavelength
        self.register_buffer('phase', phase)

    def forward(self, z, offset=0):
        seq_len = z.shape[1]
        theta = self.phase[offset:offset + seq_len, :]
        c = torch.cos(theta).unsqueeze(0)
        s = torch.sin(theta).unsqueeze(0)
        zr = z.real * c - z.imag * s
        zi = z.real * s + z.imag * c
        return torch.complex(zr, zi)

class BidirectionalComplexAttention(nn.Module):
    def __init__(self, d=32, heads=4):
        super().__init__()
        self.H = heads; self.dh = d // heads
        self.qr = nn.Linear(d, d, bias=False); self.qi = nn.Linear(d, d, bias=False)
        self.kr = nn.Linear(d, d, bias=False); self.ki = nn.Linear(d, d, bias=False)
        self.vr = nn.Linear(d, d, bias=False); self.vi = nn.Linear(d, d, bias=False)
        self.or_ = nn.Linear(d, d, bias=False); self.oi = nn.Linear(d, d, bias=False)
        self.scale = 1.0 / math.sqrt(self.dh)
        for w in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi, self.or_, self.oi]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        B, S, D = x.shape
        qr = self.qr(x.real) - self.qi(x.imag); qi = self.qr(x.imag) + self.qi(x.real)
        kr = self.kr(x.real) - self.ki(x.imag); ki = self.kr(x.imag) + self.ki(x.real)
        vr = self.vr(x.real) - self.vi(x.imag); vi = self.vr(x.imag) + self.vi(x.real)
        qr = qr.view(B,S,self.H,self.dh).transpose(1,2); qi = qi.view(B,S,self.H,self.dh).transpose(1,2)
        kr = kr.view(B,S,self.H,self.dh).transpose(1,2); ki = ki.view(B,S,self.H,self.dh).transpose(1,2)
        vr = vr.view(B,S,self.H,self.dh).transpose(1,2); vi = vi.view(B,S,self.H,self.dh).transpose(1,2)
        sr = (qr @ kr.transpose(-2,-1) + qi @ ki.transpose(-2,-1)) * self.scale
        attn = F.softmax(sr, dim=-1)
        out_r = attn @ vr; out_i = attn @ vi
        out_r = out_r.transpose(1,2).contiguous().view(B,S,-1)
        out_i = out_i.transpose(1,2).contiguous().view(B,S,-1)
        return torch.complex(self.or_(out_r) - self.oi(out_i), self.or_(out_i) + self.oi(out_r))

def gen_data(n, max_val, op_idx):
    data = []
    for _ in range(n):
        a = random.randint(0, max_val)
        b = random.randint(1, max_val) if op_idx == 3 else random.randint(0, max_val)
        if op_idx == 0:     c = a + b
        elif op_idx == 1:   c = a - b
        elif op_idx == 2:   c = a * b
        else:               c = a // b
        data.append((a, op_idx, b, c))
    return data

class MathModel(nn.Module):
    def __init__(self, max_val=50, d=32, heads=4, layers=2):
        super().__init__()
        self.num_re = nn.Embedding(max_val+1, d); self.num_im = nn.Embedding(max_val+1, d)
        self.op_re = nn.Embedding(4, d); self.op_im = nn.Embedding(4, d)
        self.pos_enc = ComplexPositionEncoding(d)
        self.layers = nn.ModuleList([BidirectionalComplexAttention(d, heads) for _ in range(layers)])
        self.out = nn.Linear(d, 1)
        for w in [self.num_re, self.num_im, self.op_re, self.op_im]:
            nn.init.normal_(w.weight, std=0.02)
        nn.init.normal_(self.out.weight, std=0.02)

    def forward(self, a_tok, op_tok, b_tok):
        va = torch.complex(self.num_re(a_tok), self.num_im(a_tok)).unsqueeze(1)
        vo = torch.complex(self.op_re(op_tok), self.op_im(op_tok)).unsqueeze(1)
        vb = torch.complex(self.num_re(b_tok), self.num_im(b_tok)).unsqueeze(1)
        z = torch.cat([va, vo, vb], dim=1)
        z = self.pos_enc(z)  # position encoding before attention
        for layer in self.layers:
            z = z + layer(z)
        return self.out(z.mean(dim=1).real).squeeze(-1)

max_val = 50; ops = ["+", "-", "*", "//"]
for op_idx in range(4):
    if op_idx == 2:  # multiplication: harder, more data + epochs + wider model
        data = gen_data(8000, max_val, op_idx)
        model = MathModel(max_val=max_val, d=48, heads=6, layers=3)
        epochs = 30
    else:
        data = gen_data(5000, max_val, op_idx)
        model = MathModel(max_val=max_val, d=32, heads=4, layers=2)
        epochs = 20
    random.shuffle(data)
    n_tr = int(len(data) * 0.8)
    tr, te = data[:n_tr], data[n_tr:]
    max_res = {0: max_val*2, 1: max_val, 2: max_val*max_val, 3: max_val}[op_idx]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    P = sum(p.numel() for p in model.parameters())

    for ep in range(epochs):
        tl = n = 0
        random.shuffle(tr)
        for i in range(0, len(tr), 128):
            batch = tr[i:i+128]
            if not batch: continue
            a_t = torch.tensor([x[0] for x in batch]); o_t = torch.tensor([x[1] for x in batch])
            b_t = torch.tensor([x[2] for x in batch])
            tgt = torch.tensor([x[3]/max_res for x in batch], dtype=torch.float32)
            pred = model(a_t, o_t, b_t)
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += loss.item(); n += 1
        sched.step()

    with torch.no_grad():
        a_t = torch.tensor([x[0] for x in te]); o_t = torch.tensor([x[1] for x in te])
        b_t = torch.tensor([x[2] for x in te])
        tgt = torch.tensor([x[3]/max_res for x in te], dtype=torch.float32)
        pred = model(a_t, o_t, b_t)
        th = torch.maximum(torch.tensor(1.0), (tgt*max_res).abs()*0.05)
        acc = ((pred*max_res - tgt*max_res).abs() < th).float().mean().item()
    print(f"  {ops[op_idx]}: {acc:.1%} P={P:>6,} {'PASS' if acc>0.95 else ''}")
