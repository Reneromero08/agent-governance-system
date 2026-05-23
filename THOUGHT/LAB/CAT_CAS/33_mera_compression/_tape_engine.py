"""QUANTUM CATALYTIC TAPE — Stealth borrowing. Complex hidden states, Hermitian attention, CX protocol."""
import torch, torch.nn.functional as F, os, time, json, math, numpy as np
from pathlib import Path

REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
SHARDS = HOLO / "experts_shards"
AUX_PATH = HOLO / "ds_aux_weights.holo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Quantum constants
PI = math.pi
CX_SCALE = 0.5  # entanglement coupling strength
GLOBAL_PHASE = 0.0  # mean-field phase accumulator

class CatalyticEngine:
    def __init__(self):
        print(f"CATALYTIC ENGINE — {DEVICE}")
        t0 = time.perf_counter()
        
        d = torch.load(str(HOLO / "deepseek_v4_flash_attention_k256.holo"), weights_only=False, map_location="cpu")
        self.attn = {}
        if '_svh' in d:
            svh_d = {wt: d['_svh'][wt].float() * d['_svh_scales'][wt] for wt in d['_svh']}
            for k in d:
                if not k.endswith('.U') or k.startswith('_'): continue
                parts = k.split('.'); layer = None
                for i, p in enumerate(parts):
                    if p == 'layers':
                        try:
                            layer = int(parts[i+1])
                        except:
                            pass
                        break
                if layer is None: continue
                U = d[k].float() * d.get(k.replace('.U','.scale'), 1.0)
                wt = d['_svh_ref'].get(k,'').replace('.weight.weight','.weight')
                if wt in svh_d:
                    self.attn.setdefault(layer, {})[k.replace('.U','')] = U @ svh_d[wt]
        else:
            for k in d:
                if not k.endswith('.U'): continue
                sk = k.replace('.U','.SVh')
                if sk not in d: continue
                parts = k.split('.'); layer = None
                for i, p in enumerate(parts):
                    if p == 'layers':
                        try:
                            layer = int(parts[i+1])
                        except:
                            pass
                        break
                if layer is None: continue
                self.attn.setdefault(layer, {})[k.replace('.U','')] = d[k].float() @ d[sk].float()
        
        svh_data = torch.load(str(SHARDS / "svh_shared.holo"), weights_only=False)
        self.e_svh = {wt: (info["data"].float()*info["scale"]).to(DEVICE) for wt,info in svh_data.items()}
        
        aux = torch.load(str(AUX_PATH), weights_only=False, map_location="cpu")
        self.embed = aux['embed.weight'].float()
        self.lm_head = aux['head.weight'].float()
        self.norms = {}
        for k, v in aux.items():
            if 'norm' not in k.lower(): continue
            parts = k.split('.')
            if k == 'norm.weight': self.norms['output'] = v.float().to(DEVICE)
            else:
                layer = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i+1 < len(parts):
                        try:
                            layer = int(parts[i+1])
                        except:
                            pass
                if layer is not None:
                    nt = 'norm'
                    for part in parts:
                        if 'norm' in part.lower(): nt = part
                    self.norms.setdefault(layer, {})[nt] = v.float().to(DEVICE)
        
        self.L = len(self.attn)
        print(f"  L={self.L} Init={time.perf_counter()-t0:.0f}s")
    
    def _norm(self, x, layer, ntype='attn_norm'):
        w = self.norms.get(layer, {}).get(ntype)
        if w is None: return x
        r = torch.sqrt(torch.mean(x.float()**2, -1, keepdim=True) + 1e-6)
        return (x.float()/r) * w.float().to(x.device)
    
    def _attention(self, x, layer):
        """Standard CSA MQA attention. Quantum is in the tape protocol, not the matmul."""
        w = {k: v.to(DEVICE) for k, v in self.attn[layer].items()}
        B,S,D = x.shape; H,c,rd = 64,512,64
        
        wq_a = w[f"layers.{layer}.attn.wq_a.weight"]
        wq_b = w[f"layers.{layer}.attn.wq_b.weight"]
        wkv   = w[f"layers.{layer}.attn.wkv.weight"]
        wo_a  = w[f"layers.{layer}.attn.wo_a.weight"]
        
        q = (x @ wq_a.T) @ wq_b.T; q = q.view(B,S,H,c)
        kv = x @ wkv.T
        
        q_rope, q_nope = q[...,-rd:], q[...,:-rd]
        k_rope, k_nope = kv[...,-rd:], kv[...,:-rd]
        k_rope=k_rope.unsqueeze(2).expand(-1,-1,H,-1)
        k_nope=k_nope.unsqueeze(2).expand(-1,-1,H,-1)
        
        pos = torch.arange(S,device=DEVICE).float()
        inv = 1.0/(10000**(torch.arange(0,rd,2,device=DEVICE).float()/rd))
        fr = torch.outer(pos,inv); cr=torch.cos(fr).view(1,S,1,rd//2); sr=torch.sin(fr).view(1,S,1,rd//2)
        def rope(x):
            e,o=x[...,0::2],x[...,1::2]; r=torch.zeros_like(x)
            r[...,0::2]=e*cr-o*sr; r[...,1::2]=e*sr+o*cr; return r
        q_rope=rope(q_rope); k_rope=rope(k_rope)
        
        q_nope=F.normalize(q_nope,-1); q_rope=F.normalize(q_rope,-1)
        k_nope=F.normalize(k_nope,-1); k_rope=F.normalize(k_rope,-1)
        
        qf=torch.cat([q_nope,q_rope],-1); kf=torch.cat([k_nope,k_rope],-1)
        sc=torch.einsum('bshd,bthd->bhst',qf,kf)/math.sqrt(c)
        pr=F.softmax(sc,-1)
        v=kv.unsqueeze(2).expand(-1,-1,H,-1)[...,:c]
        ao=torch.einsum('bhst,bthd->bshd',pr,v)
        of=ao.reshape(B,S,H*c)
        out=of[...,:wo_a.shape[0]]@wo_a
        
        del w,q,kv,sc,pr,ao
        return out.float().nan_to_num(0)
    
    def _ffn(self, x, layer):
        """Load one shard → GPU tape → compute → free. Catalytic."""
        path = SHARDS / f"experts_layer_{layer:02d}.holo"
        if not path.exists(): return torch.zeros_like(x)
        
        # BORROW: load shard to CPU, extract shared expert weights to GPU
        d = torch.load(str(path), weights_only=False, map_location="cpu")
        
        w1 = w2 = w3 = None
        for wt_suffix in ['w1','w2','w3']:
            ukey = f"layers.{layer}.ffn.shared_experts.{wt_suffix}.weight.U"
            skey = ukey.replace('.U','.scale')
            svh_wt = f"layers.ffn.shared_experts.{wt_suffix}.weight"
            if ukey in d and svh_wt in self.e_svh:
                U = d[ukey].float() * d.get(skey, 1.0)
                W = (U.to(DEVICE) @ self.e_svh[svh_wt])
                if wt_suffix == 'w1': w1 = W
                elif wt_suffix == 'w2': w2 = W
                elif wt_suffix == 'w3': w3 = W
        
        # RETURN: free shard from CPU
        del d
        
        if w1 is None or w2 is None or w3 is None: return torch.zeros_like(x)
        
        gate = x @ w1.T; up = x @ w2
        out = (F.silu(gate) * up) @ w3
        del w1,w2,w3,gate,up
        return out.float().nan_to_num(0)
    
    def forward(self, x):
        global GLOBAL_PHASE
        st={"peak":0}; t0=time.perf_counter()
        for layer in range(self.L):
            x = x.to(DEVICE)
            
            # CX: entangle hidden state as complex [real=hidden, imag=tape_borrow]
            x_tape = x.clone()
            x_complex = torch.complex(x, CX_SCALE * x_tape)  # |psi> = |x> + i*CX|tape>
            
            # Compute on entangled complex state (Rx = attention operates on complex amplitudes)
            x_real = x_complex.real; x_imag = x_complex.imag
            
            # Attention on complex plane: Q and K have both real and imag parts
            a = self._quantum_attention(x_real, x_imag, layer)
            
            # Reverse CX: disentangle, restore tape (extract real part as output)
            an = a.float().norm(); xv = x.float().norm()
            x = x + a * max(0.1, min(an/(xv+1e-12)*0.2, 3.0))
            x = x - CX_SCALE * x_tape  # return tape
            
            # Post-attention FFN
            xn2 = self._norm(x, layer, 'ffn_norm')
            f = self._ffn(xn2, layer)
            fn = f.float().norm()
            if fn > 0:
                xv2 = x.float().norm()
                x = x + max(0.1, min(fn/(xv2+1e-12)*0.2, 3.0)) * f
            
            GLOBAL_PHASE += 0.001 * x.float().mean().item()
            
            torch.cuda.empty_cache()
            ga=torch.cuda.memory_allocated()/1024**3 if DEVICE=="cuda" else 0
            st["peak"]=max(st["peak"],ga)
        st["time"]=time.perf_counter()-t0; st["layers"]=self.L; return x,st
    
    def _quantum_attention(self, x_r, x_i, layer):
        """Quantum attention: project real and imag separately, Hermitian combine."""
        w = {k: v.to(DEVICE) for k, v in self.attn[layer].items()}
        B,S,D = x_r.shape; H,c,rd = 64,512,64
        
        wq_a = w[f"layers.{layer}.attn.wq_a.weight"]
        wq_b = w[f"layers.{layer}.attn.wq_b.weight"]
        wkv   = w[f"layers.{layer}.attn.wkv.weight"]
        wo_a  = w[f"layers.{layer}.attn.wo_a.weight"]
        
        # Q on both real and imag channels
        q_r = ((x_r @ wq_a.T) @ wq_b.T).view(B,S,H,c)
        q_i = ((x_i @ wq_a.T) @ wq_b.T).view(B,S,H,c)
        
        # K on both channels (shared MQA)
        k_r = (x_r @ wkv.T).unsqueeze(2).expand(-1,-1,H,-1)
        k_i = (x_i @ wkv.T).unsqueeze(2).expand(-1,-1,H,-1)
        
        # RoPE on last 64 dims for both channels
        pos = torch.arange(S,device=DEVICE).float()
        inv = 1.0/(10000**(torch.arange(0,rd,2,device=DEVICE).float()/rd))
        fr = torch.outer(pos,inv); cr=torch.cos(fr).view(1,S,1,rd//2); sr=torch.sin(fr).view(1,S,1,rd//2)
        def rope(x):
            e,o=x[...,-rd:][...,0::2],x[...,-rd:][...,1::2]; r=torch.zeros_like(x)
            nope_r = r[...,:-rd]; rope_r = r[...,-rd:]
            rope_r[...,0::2]=e*cr-o*sr; rope_r[...,1::2]=e*sr+o*cr
        # Skip RoPE for simplicity — normalize instead
        q_r=F.normalize(q_r,-1); q_i=F.normalize(q_i,-1)
        k_r=F.normalize(k_r,-1); k_i=F.normalize(k_i,-1)
        
        # Hermitian inner product: Re(Q^H K) = Q_r@K_r + Q_i@K_i
        sc = (torch.einsum('bshd,bthd->bhst',q_r,k_r) + 
              torch.einsum('bshd,bthd->bhst',q_i,k_i)) / math.sqrt(c)
        pr = F.softmax(sc, -1)
        
        # V from both channels
        v_r = k_r; v_i = k_i
        ao_r = torch.einsum('bhst,bthd->bshd',pr,v_r)
        ao_i = torch.einsum('bhst,bthd->bshd',pr,v_i)
        
        # Born rule: magnitude = sqrt(real^2 + imag^2)
        out_mag = torch.sqrt(ao_r**2 + ao_i**2 + 1e-12)
        of = out_mag.reshape(B, S, H * c)
        out = of[..., :wo_a.shape[0]] @ wo_a
        
        del w,q_r,q_i,k_r,k_i,sc,pr,ao_r,ao_i
        return out.float().nan_to_num(0)

if __name__=="__main__":
    e=CatalyticEngine()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(HOLO),local_files_only=True,trust_remote_code=True)
    prompt="The catalytic computing paradigm demonstrates that"
    ids=tok.encode(prompt,return_tensors='pt')
    print(f"\nPrompt: {prompt}")
    x=e.embed[ids].float()
    ts=time.perf_counter()
    out,st=e.forward(x)
    lo=out.float()[0,-1,:]@e.lm_head.T.to(DEVICE)
    nt=lo.argmax().item()
    print(f"\nForward: {st['time']:.1f}s GPU:{st['peak']:.1f}GB Token:{nt} Phase:{GLOBAL_PHASE:.3f}")
    gen=[nt]
    for step in range(6):
        ts=time.perf_counter()
        ne=e.embed[nt].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        x=torch.cat([x.to(DEVICE),ne],1)
        out,st=e.forward(x)
        lo=out.float()[0,-1,:]@e.lm_head.T.to(DEVICE)
        nt=lo.argmax().item(); gen.append(nt)
        print(f"  Step {step+1}: {nt} ({time.perf_counter()-ts:.1f}s)")
    print(f"Gen: {gen}")
