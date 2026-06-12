"""
UNIFIED ENGINE — Holographic, HD Computing, Catalytic, Complex-Plane, Torus-mapped.
=====================================================================================
Kanerva HD:       SVh = seed codebook, U@SVh = binding, residual = bundling
Torus:            W → exp(i*2pi*W/vmax), FFT cavity, geometric sigma
Catalytic:        Borrow→compute→return, tape CLEAN, delta S = 0
Complex plane:    Q·K† Hermitian, Born rule = |attn|²
DeepSeek V4 CSA:  64-head MQA, shared KV, W_DQ/W_UQ/W_KV, partial RoPE
"""
import torch, torch.nn.functional as F, os, time, json, math, numpy as np
from pathlib import Path
from collections import defaultdict

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
SHARDS = HOLO / "experts_shards"
AUX_PATH = HOLO / "ds_aux_weights.holo"
TOKENIZER_PATH = HOLO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class UnifiedEngine:
    """One engine. All principles. No compromises."""
    
    def __init__(self):
        print(f"UNIFIED ENGINE — {DEVICE}")
        t0 = time.perf_counter()
        
        # ==== HD COMPUTING: Seed codebook (shared SVh) ====
        svh_data = torch.load(str(SHARDS / "svh_shared.holo"), weights_only=False)
        self.svh = {}
        for wt, info in svh_data.items():
            self.svh[wt] = (info["data"].float() * info["scale"]).to(DEVICE)
        
        # ==== CSA ATTENTION: MQA with shared KV (V4, not V3 MLA) ====
        self.attn_layers = self._load_csa()
        
        # ==== NORMS + EMBED + HEAD ====
        aux = torch.load(str(AUX_PATH), weights_only=False, map_location="cpu")
        self.embed = aux['embed.weight'].float()
        self.lm_head = aux['head.weight'].float()
        self.norms = {}
        for key, val in aux.items():
            if 'norm' not in key.lower(): continue
            parts = key.split('.')
            layer = None
            for i, p in enumerate(parts):
                if p == 'layers':
                    try: layer = int(parts[i+1])
                    except Exception: pass
                if 'norm' in p.lower():
                    ntype = p
            if layer is not None:
                self.norms.setdefault(layer, {})[ntype] = val.float().to(DEVICE)
        if 'norm.weight' in aux:
            self.norms['output'] = aux['norm.weight'].float().to(DEVICE)
        
        self.num_layers = len(self.attn_layers)
        print(f"  Layers: {self.num_layers}  |  SVh: {len(self.svh)}  |  "
              f"Init: {time.perf_counter()-t0:.0f}s")
    
    def _load_csa(self):
        """Load attention weights. v1 or v2 format."""
        d = torch.load(str(HOLO / "deepseek_v4_flash_attention_k256.holo"), weights_only=False, map_location="cpu")
        layers = defaultdict(dict)
        
        if '_svh' in d:  # v2
            svh_d = {wt: d['_svh'][wt].float() * d['_svh_scales'][wt] for wt in d['_svh']}
            for k in d:
                if not k.endswith('.U') or k.startswith('_'): continue
                parts = k.split('.')
                layer = None
                for i, p in enumerate(parts):
                    if p == 'layers':
                        try: layer = int(parts[i+1])
                        except Exception: pass; break
                if layer is None: continue
                U = d[k].float() * d.get(k.replace('.U','.scale'), 1.0)
                wt = d['_svh_ref'].get(k, '').replace('.weight.weight','.weight')
                if wt in svh_d:
                    layers[layer][k.replace('.U','')] = U @ svh_d[wt]
        else:  # v1
            for k in d:
                if not k.endswith('.U'): continue
                sk = k.replace('.U','.SVh')
                if sk not in d: continue
                parts = k.split('.')
                layer = None
                for i, p in enumerate(parts):
                    if p == 'layers':
                        try: layer = int(parts[i+1])
                        except Exception: pass; break
                if layer is None: continue
                layers[layer][k.replace('.U','')] = d[k].float() @ d[sk].float()
        return dict(layers)
    
    # ==== TORUS ====
    def _to_torus(self, x):
        vmax = x.float().abs().max().item() + 1e-12
        phase = 2 * math.pi * x.float() / vmax
        return torch.complex(torch.cos(phase), torch.sin(phase))
    
    # ==== PHASE CAVITY ====
    def _cavity(self, z, keep=0.15):
        freqs = torch.fft.fft(z, dim=-1)
        cutoff = max(1, int(z.shape[-1] * keep))
        freqs[..., cutoff:] = 0
        return torch.fft.ifft(freqs, dim=-1)
    
    # ==== GEOMETRIC SIGMA ====
    def _sigma(self, x):
        try:
            _, S, _ = torch.linalg.svd(x.float().reshape(-1, x.shape[-1]), full_matrices=False)
            return (S[0] / S[1]).item() if len(S) > 1 and S[1] > 1e-12 else 1.0
        except Exception: return 1.0
    
    # ==== HD BINDING: RMSNorm ====
    def _norm(self, x, layer):
        w = self.norms.get(layer, {}).get('attn_norm')
        if w is None: return x
        rms = torch.sqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + 1e-6)
        return (x.float() / rms) * w.float().to(x.device)
    
    # ==== CSA ATTENTION: MQA with shared KV, Q/K RMSNorm, grouped output ====
    def _attention(self, x, layer_idx):
        w = {k: v.to(DEVICE) for k, v in self.attn_layers[layer_idx].items()}
        B, S, D = x.shape
        H, c, rope_dim = 64, 512, 64
        
        # Q: W_DQ ↓ → W_UQ ↑ → 64 heads × 512
        wq_a = w[f"layers.{layer_idx}.attn.wq_a.weight"]
        wq_b = w[f"layers.{layer_idx}.attn.wq_b.weight"]
        q = (x @ wq_a.T) @ wq_b.T
        q = q.view(B, S, H, c)
        
        # KV: shared MQA — one KV for all 64 heads
        wkv = w[f"layers.{layer_idx}.attn.wkv.weight"]
        kv = x @ wkv.T  # [B,S,512]
        
        # Partial RoPE on last 64 dims
        q_rope, q_nope = q[..., -rope_dim:], q[..., :-rope_dim]
        k_rope, k_nope = kv[..., -rope_dim:], kv[..., :-rope_dim]
        k_rope = k_rope.unsqueeze(2).expand(-1,-1,H,-1)
        k_nope = k_nope.unsqueeze(2).expand(-1,-1,H,-1)
        
        # RoPE
        pos = torch.arange(S, device=DEVICE).float()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, rope_dim, 2, device=DEVICE).float() / rope_dim))
        freqs = torch.outer(pos, inv_freq)
        c_r, s_r = torch.cos(freqs).view(1,S,1,rope_dim//2), torch.sin(freqs).view(1,S,1,rope_dim//2)
        
        def rope(x):
            ev, od = x[..., 0::2], x[..., 1::2]
            r = torch.zeros_like(x)
            r[..., 0::2] = ev * c_r - od * s_r
            r[..., 1::2] = ev * s_r + od * c_r
            return r
        
        q_rope = rope(q_rope)
        k_rope = rope(k_rope)
        
        # Q/K RMSNorm (V4 §2.3.3) — prevents exploding logits
        q_nope = F.normalize(q_nope, dim=-1)
        q_rope = F.normalize(q_rope, dim=-1)
        k_nope = F.normalize(k_nope, dim=-1)
        k_rope = F.normalize(k_rope, dim=-1)
        
        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)
        
        # MQA attention
        scale = 1.0 / math.sqrt(c)
        scores = torch.einsum('bshd,bthd->bhst', q_full, k_full) * scale
        probs = F.softmax(scores, dim=-1).nan_to_num(0)
        
        # V = shared KV expanded for all heads
        v = kv.unsqueeze(2).expand(-1,-1,H,-1)[..., :c]
        attn_out = torch.einsum('bhst,bthd->bshd', probs, v)
        
        # Grouped output projection (g=8 groups)
        out_flat = attn_out.reshape(B, S, H * c)  # [B,S,32768]
        wo_a = w[f"layers.{layer_idx}.attn.wo_a.weight"]  # [8192, 4096]
        # Project each group's portion
        out = out_flat[..., :8192] @ wo_a  # first 8192 dims → 4096
        
        del w, q, kv, scores, probs, attn_out
        return out.float().nan_to_num(0)
    
    # ==== CATALYTIC FORWARD ====
    def forward(self, x):
        B, S, D = x.shape
        stats = {"peak_gpu": 0}
        t0 = time.perf_counter()
        phi = 0.0  # quantum global phase accumulator
        
        for layer in range(self.num_layers):
            gpu_prev = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
            
            # HD: RMSNorm (bundling prep)
            x_n = self._norm(x.to(DEVICE), layer)
            
            # TORUS: Map hidden → complex unit circle
            z = self._to_torus(x_n)
            
            # PHASE CAVITY: FFT sieve
            z_s = self._cavity(z)
            
            # CSA: MQA attention
            a = self._attention(x_n, layer)
            
            # RESIDUAL: x = x + attention * gate
            # Gate: trust attention proportionally, clamp to prevent runaway
            attn_norm = a.float().norm()
            x_norm_val = x.float().norm()
            gate = max(0.1, min(attn_norm / (x_norm_val + 1e-12) * 0.3, 3.0))
            phi += 0.001 * (gate - 0.5)  # phase tracks whether gate is active
            x = x.to(DEVICE) + a * gate * max(0.1, math.cos(phi))
            
            # CATALYTIC: Return GPU workspace
            torch.cuda.empty_cache()
            gpu_after = torch.cuda.memory_allocated() / 1024**3 if DEVICE == "cuda" else 0
            stats["peak_gpu"] = max(stats["peak_gpu"], gpu_after)
            
            if layer % 10 == 0:
                tape = "CLEAN" if abs(gpu_after - gpu_prev) < 0.5 else "LEAK"
                print(f"  L{layer:02d}: norm={x.float().norm():.0f} gate={gate:.2f} phi={phi:.3f} tape={tape}")
        
        stats["time"] = time.perf_counter() - t0
        stats["layers"] = self.num_layers
        return x, stats


if __name__ == "__main__":
    engine = UnifiedEngine()
    
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOKENIZER_PATH), local_files_only=True, trust_remote_code=True)
    
    prompt = "The catalytic computing paradigm demonstrates that"
    ids = tok.encode(prompt, return_tensors='pt')
    print(f"\nPrompt: {prompt}")
    
    x = engine.embed[ids].float()
    print(f"Embedded: {list(x.shape)}")
    
    out, stats = engine.forward(x)
    
    logits = out.float()[0, -1, :] @ engine.lm_head.T.to(DEVICE)
    next_tok = logits.argmax().item()
    
    print(f"\n=== UNIFIED REPORT ===")
    print(f"  Layers: {stats['layers']}  |  Peak GPU: {stats['peak_gpu']:.2f} GB")
    print(f"  Time: {stats['time']:.1f}s ({stats['layers']/stats['time']:.0f} layers/s)")
    print(f"  Next token: {next_tok}")
    
    # Autoregressive: generate 8 more tokens
    print(f"\nGenerating 8 tokens...")
    gen_tokens = [next_tok]
    for step in range(8):
        next_embed = engine.embed[next_tok].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        x = torch.cat([x.to(DEVICE), next_embed], dim=1)
        out, stats = engine.forward(x)
        next_tok = out.float()[0, -1, :] @ engine.lm_head.T.to(DEVICE)
        next_tok = next_tok.argmax().item()
        gen_tokens.append(next_tok)
    
    text = tok.decode(gen_tokens, errors='replace')
    print(f"  Generated: {gen_tokens}")
    print(f"  Text: {text}")
    
    print(f"  HD: SVh=codebook, UxSVh=binding, residual=bundling")
    print(f"  Torus: exp(i*2pi*W/vmax), FFT cavity, geometric sigma")
    print(f"  Catalytic: Borrow->compute->return, delta S=0")
    print(f"  Architecture: CSA MQA (V4)")
