"""HOLOGRAPHIC BRAIN — CavitatedHoloLinear + Cybernetic Gate + Pre-extracted FFN.
x @ SVh^T @ U^T for wq_a/wq_b/wkv. Fallback to materialized W for wo_a."""
import torch, torch.nn.functional as F, os, time, json, math, numpy as np
from pathlib import Path

REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
HOLO = REPO / "THOUGHT" / "LAB" / "HOLO" / "_models"
AUX_PATH = HOLO / "ds_aux_weights.holo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CavitatedHoloLinear:
    def __init__(self, U, SVh):
        self.U = U.float()
        self.SVh = SVh.float()
        self.W = None
    def forward(self, x):
        if self.SVh.shape[1] == x.shape[-1]:
            h = x.float() @ self.SVh.T.float().to(x.device)
            return h @ self.U.T.float().to(x.device)
        if self.W is None:
            self.W = (self.U.float() @ self.SVh.float()).to(DEVICE)
        return x.float() @ self.W.T

class HolographicBrain:
    def __init__(self):
        print(f"HOLOGRAPHIC BRAIN — {DEVICE}")
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
                    self.attn.setdefault(layer, {})[k.replace('.U','')] = CavitatedHoloLinear(U, svh_d[wt])
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
                self.attn.setdefault(layer, {})[k.replace('.U','')] = CavitatedHoloLinear(d[k].float(), d[sk].float())
        
        ffn_path = HOLO / "ds_shared_ffn.holo"
        self.ffn = torch.load(str(ffn_path), weights_only=False, map_location="cpu") if ffn_path.exists() else {}
        
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
        
        # Pre-materialize wo_a weights (SVh dim mismatch — distiller bug)
        self._wo_a = {}
        for layer in self.attn:
            w = self.attn[layer][f"layers.{layer}.attn.wo_a.weight"]
            self._wo_a[layer] = (w.U @ w.SVh).half()
    
    def _norm(self, x, layer, ntype='attn_norm'):
        w = self.norms.get(layer, {}).get(ntype)
        if w is None: return x
        r = torch.sqrt(torch.mean(x.float()**2, -1, keepdim=True) + 1e-6)
        return (x.float()/r) * w.float().to(x.device)
    
    def _attention(self, x, layer):
        w = self.attn[layer]; B,S,D = x.shape; H,c,rd = 64,512,64
        q = w[f"layers.{layer}.attn.wq_a.weight"].forward(x)
        q = w[f"layers.{layer}.attn.wq_b.weight"].forward(q)
        q = q.view(B,S,H,c)
        kv = w[f"layers.{layer}.attn.wkv.weight"].forward(x)
        
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
        # Use pre-materialized wo_a (CavitatedHoloLinear has SVh dim mismatch)
        out=of[...,:self._wo_a[layer].shape[0]] @ self._wo_a[layer].float().to(DEVICE)
        return out.float().nan_to_num(0)
    
    def _ffn(self, x, layer):
        if not self.ffn: return torch.zeros_like(x)
        wk = f"layers.{layer}.ffn.shared_experts.w1.weight"
        if wk not in self.ffn: return torch.zeros_like(x)
        w1=self.ffn[wk].float().to(DEVICE)
        w2=self.ffn[wk.replace('w1','w2')].float().to(DEVICE)
        w3=self.ffn[wk.replace('w1','w3')].float().to(DEVICE)
        gate=x@w1.T; up=x@w2; out=(F.silu(gate)*up)@w3
        del w1,w2,w3,gate,up; return out.float().nan_to_num(0)
    
    def forward(self, x):
        st={"peak":0}; t0=time.perf_counter()
        for layer in range(self.L):
            x=x.to(DEVICE); xn=self._norm(x,layer,'attn_norm')
            a=self._attention(xn,layer)
            an=a.float().norm(); xv=x.float().norm()
            x=x+a*max(0.1,min(an/(xv+1e-12)*0.2,3.0))
            xn2=self._norm(x,layer,'ffn_norm')
            f=self._ffn(xn2,layer)
            fn=f.float().norm()
            if fn>0:
                xv2=x.float().norm(); x=x+f*max(0.1,min(fn/(xv2+1e-12)*0.2,3.0))
            torch.cuda.empty_cache()
            ga=torch.cuda.memory_allocated()/1024**3 if DEVICE=="cuda" else 0
            st["peak"]=max(st["peak"],ga)
        st["time"]=time.perf_counter()-t0; st["layers"]=self.L; return x,st

if __name__=="__main__":
    brain=HolographicBrain()
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(str(HOLO),local_files_only=True,trust_remote_code=True)
    prompt="The catalytic computing paradigm demonstrates that"
    ids=tok.encode(prompt,return_tensors='pt')
    print(f"\nPrompt: {prompt}")
    x=brain.embed[ids].float()
    ts=time.perf_counter()
    out,st=brain.forward(x)
    lo=out.float()[0,-1,:]@brain.lm_head.T.to(DEVICE)
    nt=lo.argmax().item()
    print(f"\nForward: {st['time']:.1f}s Token:{nt}")
    gen=[nt]
    for step in range(6):
        ts=time.perf_counter()
        ne=brain.embed[nt].float().unsqueeze(0).unsqueeze(0).to(DEVICE)
        x=torch.cat([x.to(DEVICE),ne],1)
        out,st=brain.forward(x)
        lo=out.float()[0,-1,:]@brain.lm_head.T.to(DEVICE)
        nt=lo.argmax().item(); gen.append(nt)
        print(f"  Step {step+1}: {nt} ({time.perf_counter()-ts:.1f}s)")
    print(f"Gen: {gen}")
