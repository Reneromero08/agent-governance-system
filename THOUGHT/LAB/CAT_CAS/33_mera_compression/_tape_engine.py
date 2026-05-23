"""COMPLETE TAPE ENGINE — Attention + Shared Expert FFN + MHC residual. Zero E: drive."""
import torch, torch.nn.functional as F, os, time, json, math, numpy as np
from pathlib import Path

REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
SHARDS = HOLO / "experts_shards"
AUX_PATH = HOLO / "ds_aux_weights.holo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CompleteEngine:
    def __init__(self):
        print(f"COMPLETE ENGINE — {DEVICE}")
        t0 = time.perf_counter()
        
        # ==== Holo attention (K=256, local) ====
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
        
        # ==== Experts SVh (codebook) ====
        svh_data = torch.load(str(SHARDS / "svh_shared.holo"), weights_only=False)
        self.e_svh = {wt: (info["data"].float()*info["scale"]).to(DEVICE) for wt,info in svh_data.items()}
        
        # ==== Embed, head, norms ====
        aux = torch.load(str(AUX_PATH), weights_only=False, map_location="cpu")
        self.embed = aux['embed.weight'].float()
        self.lm_head = aux['head.weight'].float()
        self.norms = {}
        for k, v in aux.items():
            if 'norm' not in k.lower():
                continue
            parts = k.split('.')
            if k == 'norm.weight':
                self.norms['output'] = v.float().to(DEVICE)
            else:
                layer = None
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer = int(parts[i + 1])
                        except:
                            pass
                if layer is not None:
                    nt = 'norm'
                    for part in parts:
                        if 'norm' in part.lower():
                            nt = part
                    self.norms.setdefault(layer, {})[nt] = v.float().to(DEVICE)
        
        self.L = len(self.attn)
        print(f"  L={self.L} Init={time.perf_counter()-t0:.0f}s")
    
    def _norm(self, x, layer, ntype='attn_norm'):
        w = self.norms.get(layer, {}).get(ntype)
        if w is None: return x
        r = torch.sqrt(torch.mean(x.float()**2, -1, keepdim=True) + 1e-6)
        return (x.float()/r) * w.float().to(x.device)
    
    def _attention(self, x, layer):
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
    
    def _shared_ffn(self, x, layer):
        """Shared expert FFN (applies to ALL tokens, no routing)."""
        path = SHARDS / f"experts_layer_{layer:02d}.holo"
        if not path.exists(): return torch.zeros_like(x)
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
        
        if w1 is None or w2 is None or w3 is None: return torch.zeros_like(x)
        
        gate = x @ w1.T
        up = x @ w2
        out = (F.silu(gate) * up) @ w3
        del w1,w2,w3,gate,up
        return out.float().nan_to_num(0)
    
    def forward(self, x):
        st={"peak":0}; t0=time.perf_counter()
        for layer in range(self.L):
            gb=torch.cuda.memory_allocated()/1024**3 if DEVICE=="cuda" else 0
            
            # Pre-attention norm
            xn=self._norm(x.to(DEVICE), layer, 'attn_norm')
            
            # Attention + MHC-style residual (Sinkhorn-constrained gate)
            a=self._attention(xn, layer)
            an=a.float().norm(); xv=x.float().norm()
            g_attn=max(0.1,min(an/(xv+1e-12)*0.2,3.0))
            x=x.to(DEVICE)+a*g_attn
            
            # Post-attention norm
            xn2=self._norm(x, layer, 'ffn_norm')
            
            # Shared expert FFN + residual
            f=self._shared_ffn(xn2, layer)
            fn=f.float().norm()
            if fn>0:
                xv2=x.float().norm()
                g_ffn=max(0.1,min(fn/(xv2+1e-12)*0.2,3.0))
                x=x+g_ffn*f
            
            torch.cuda.empty_cache()
            ga=torch.cuda.memory_allocated()/1024**3 if DEVICE=="cuda" else 0
            st["peak"]=max(st["peak"],ga)
            if layer%10==0: print(f"  L{layer:02d}: norm={x.float().norm():.0f} tape={'CLEAN' if abs(ga-gb)<0.5 else 'LEAK'}")
        st["time"]=time.perf_counter()-t0; st["layers"]=self.L; return x,st

if __name__=="__main__":
    e=CompleteEngine()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(HOLO),local_files_only=True,trust_remote_code=True)
    prompt="The catalytic computing paradigm demonstrates that"
    ids=tok.encode(prompt,return_tensors='pt')
    print(f"\nPrompt: {prompt}")
    x=e.embed[ids].float()
    out,st=e.forward(x)
    lo=out.float()[0,-1,:]@e.lm_head.T.to(DEVICE)
    nt=lo.argmax().item()
    print(f"\nL:{st['layers']} GPU:{st['peak']:.1f}GB Time:{st['time']:.1f}s Token:{nt}")
    gen=[nt]
    for _ in range(6):
        ne=e.embed[nt].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        x=torch.cat([x.to(DEVICE),ne],1)
        out,st=e.forward(x)
        lo=out.float()[0,-1,:]@e.lm_head.T.to(DEVICE)
        nt=lo.argmax().item(); gen.append(nt)
    print(f"Gen: {gen}")
