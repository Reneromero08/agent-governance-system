"""Native Eigen Architecture v0 — trainable 2D complex transformer.

Proof: after training, does zeroing the phase accumulator increase perplexity?
If yes, phase carries learned information. 12K params, 500 sequences, 1 epoch.
"""
import torch, torch.nn as nn, torch.nn.functional as F, math, time, json, sys
from pathlib import Path
from collections import Counter

# ============================================================================
# Complex embedding, attention, FFN, phase accumulation (compact)
# ============================================================================
class NativeEigen(nn.Module):
    def __init__(self, vocab_size=2000, embed_dim=2, num_layers=2, num_heads=2, ffn_dim=8):
        super().__init__()
        self.embed_r = nn.Embedding(vocab_size, embed_dim)
        self.embed_i = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embed_r.weight, std=0.02)
        nn.init.normal_(self.embed_i.weight, std=0.02)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(_ComplexLayer(embed_dim, num_heads, ffn_dim))
        
        self.output = nn.Linear(embed_dim, vocab_size, bias=False)
        nn.init.normal_(self.output.weight, std=0.02)

    def forward(self, x):
        x = torch.complex(self.embed_r(x), self.embed_i(x))
        for layer in self.layers:
            x = layer(x)
        return self.output(torch.abs(x))


class _ComplexLayer(nn.Module):
    def __init__(self, dim=2, heads=2, ffn=8):
        super().__init__()
        self.attn = _ComplexAttention(dim, heads)
        self.ffn = _ComplexFFN(dim, ffn)
        self.phase_angle = nn.Parameter(torch.ones(dim) * 0.1)

    def forward(self, x):
        mag = torch.abs(x).mean(dim=-1, keepdim=True) + 1e-8
        x = x + self.attn(x / mag)
        # Phase rotation: e^(i*theta)
        c, s = torch.cos(self.phase_angle), torch.sin(self.phase_angle)
        x = torch.complex(x.real*c - x.imag*s, x.real*s + x.imag*c)
        mag = torch.abs(x).mean(dim=-1, keepdim=True) + 1e-8
        return x + self.ffn(x / mag)


class _ComplexAttention(nn.Module):
    def __init__(self, dim=2, heads=2):
        super().__init__()
        self.H = heads
        self.D = dim
        hd = dim * heads
        self.qr, self.qi = nn.Linear(dim, hd, bias=False), nn.Linear(dim, hd, bias=False)
        self.kr, self.ki = nn.Linear(dim, hd, bias=False), nn.Linear(dim, hd, bias=False)
        self.vr, self.vi = nn.Linear(dim, hd, bias=False), nn.Linear(dim, hd, bias=False)
        self.or_, self.oi = nn.Linear(hd, dim, bias=False), nn.Linear(hd, dim, bias=False)
        self.scale = 1.0 / math.sqrt(dim)
        for w in [self.qr, self.qi, self.kr, self.ki, self.vr, self.vi, self.or_, self.oi]:
            nn.init.normal_(w.weight, std=0.02)

    def _cl(self, x, wr, wi):
        return torch.complex(F.linear(x.real, wr.weight) - F.linear(x.imag, wi.weight),
                             F.linear(x.real, wi.weight) + F.linear(x.imag, wr.weight))

    def forward(self, x):
        B, S, D = x.shape; H = self.H
        q = self._cl(x, self.qr, self.qi).view(B, S, H, D).transpose(1,2)
        k = self._cl(x, self.kr, self.ki).view(B, S, H, D).transpose(1,2)
        v = self._cl(x, self.vr, self.vi).view(B, S, H, D).transpose(1,2)
        # Hermitian inner product: real=similarity, |imag|=phase twist
        sr = (q.real @ k.real.transpose(-2,-1) + q.imag @ k.imag.transpose(-2,-1)) * self.scale
        si = (q.imag @ k.real.transpose(-2,-1) - q.real @ k.imag.transpose(-2,-1)) * self.scale
        mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        sr = sr.masked_fill(mask, float('-inf'))
        # Phase twist reduces attention: large imag score -> less weight
        aw = F.softmax(sr - si.abs(), dim=-1)
        out_r = (aw @ v.real).transpose(1,2).contiguous().view(B, S, H*D)
        out_i = (aw @ v.imag).transpose(1,2).contiguous().view(B, S, H*D)
        o = torch.complex(F.linear(out_r, self.or_.weight) - F.linear(out_i, self.oi.weight),
                          F.linear(out_r, self.oi.weight) + F.linear(out_i, self.or_.weight))
        return o


class _ComplexFFN(nn.Module):
    def __init__(self, dim=2, hidden=8):
        super().__init__()
        self.w1r, self.w1i = nn.Linear(dim, hidden, bias=False), nn.Linear(dim, hidden, bias=False)
        self.w2r, self.w2i = nn.Linear(hidden, dim, bias=False), nn.Linear(hidden, dim, bias=False)
        for w in [self.w1r, self.w1i, self.w2r, self.w2i]:
            nn.init.normal_(w.weight, std=0.02)

    def forward(self, x):
        r1 = F.linear(x.real, self.w1r.weight) - F.linear(x.imag, self.w1i.weight)
        i1 = F.linear(x.real, self.w1i.weight) + F.linear(x.imag, self.w1r.weight)
        mag = torch.sqrt(r1**2 + i1**2 + 1e-8)
        gate = F.relu(mag)
        scale = gate / (mag + 1e-8)
        r2 = F.linear(r1*scale, self.w2r.weight) - F.linear(i1*scale, self.w2i.weight)
        i2 = F.linear(r1*scale, self.w2i.weight) + F.linear(i1*scale, self.w2r.weight)
        return torch.complex(r2, i2)


# ============================================================================
# Data, Training, Evaluation
# ============================================================================
def load_data(vocab_size=2000, seq_len=32, max_seq=500):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    counter = Counter()
    for ex in ds:
        for w in str(ex["text"]).split():
            counter[w] += 1
    vocab = ["<pad>","<unk>","<eos>"] + [w for w,_ in counter.most_common(vocab_size-3)]
    w2i = {w:i for i,w in enumerate(vocab)}
    tokens = []
    for ex in ds:
        for w in str(ex["text"]).split():
            tokens.append(w2i.get(w, w2i["<unk>"]))
        tokens.append(w2i["<eos>"])
    data = []
    for i in range(0, min(len(tokens)-seq_len, max_seq*seq_len), seq_len//2):
        seq = tokens[i:i+seq_len+1]
        if len(seq)==seq_len+1:
            data.append((seq[:-1], seq[1:]))
    return data[:max_seq], len(vocab)


def test_single(device="cpu"):
    print("Native Eigen v0 — Phase Ablation Test")
    print("=" * 50)
    
    data, vocab_size = load_data(max_seq=2000)
    print("Vocab: {}  Sequences: {}".format(vocab_size, len(data)))

    model = NativeEigen(vocab_size, embed_dim=2, num_layers=2, num_heads=2, ffn_dim=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    print("Params: {:,}".format(sum(p.numel() for p in model.parameters())))

    # Train 5 epochs
    print("\nTraining (5 epochs)...", flush=True)
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch_idx in range(0, len(data), 16):
            batch = data[batch_idx:batch_idx+16]
            if not batch: continue
            x = torch.tensor([b[0] for b in batch], device=device, dtype=torch.long)
            y = torch.tensor([b[1] for b in batch], device=device, dtype=torch.long)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        avg = total_loss / max(1, len(data)//16)
        print("  Epoch {}: loss={:.4f} ppl={:.1f}".format(epoch+1, avg, math.exp(avg)), flush=True)
    train_loss = loss.item()

    # Phase ablation (save/restore)
    model.eval()
    saved = [layer.phase_angle.data.clone() for layer in model.layers]
    results = {}
    for mode in ["normal", "ablated"]:
        if mode == "ablated":
            for layer in model.layers:
                layer.phase_angle.data.zero_()
        nll, n = 0, 0
        with torch.no_grad():
            for batch_idx in range(0, min(200, len(data)), 16):
                batch = data[batch_idx:batch_idx+16]
                if not batch: continue
                x = torch.tensor([b[0] for b in batch], device=device, dtype=torch.long)
                y = torch.tensor([b[1] for b in batch], device=device, dtype=torch.long)
                logits = model(x)
                nll += F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1)).item() * y.numel()
                n += y.numel()
        results[mode] = round(math.exp(nll / max(n, 1)), 1)
    # Restore
    for layer, s in zip(model.layers, saved):
        layer.phase_angle.data.copy_(s)

    normal_ppl = results["normal"]
    ablated_ppl = results["ablated"]
    delta = (ablated_ppl - normal_ppl) / normal_ppl * 100 if normal_ppl > 0 else 0

    print("\n" + "=" * 50)
    print("Normal PPL:  {:.1f}".format(normal_ppl))
    print("Ablated PPL: {:.1f}".format(ablated_ppl))
    print("Delta: {:+.1f}%".format(delta))
    if delta > 5:
        print("PASS — Phase carries learned information")
    elif delta > 1:
        print("WEAK — Marginal phase contribution")
    else:
        print("NEGLIGIBLE — Phase not used during training")
    print("=" * 50)


if __name__ == "__main__":
    test_single("cuda" if torch.cuda.is_available() else "cpu")

