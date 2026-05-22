"""Oracle-Integrated Cybernetic Engine — 4.8 implementation.

Replaces R = Tr(rho*C) (NaN-sensitive) with torus_circular_variance from
CAT_CAS 20.10.5's unified holo oracle. Measures phase coherence on the
unit circle S^1 — NaN-immune because all operations are bounded.

Maintains rolling buffer of L hidden states, maps them to complex torus,
computes circular variance as the resonance measure. Low variance = truth
state (deterministic), high variance = exploration. Rewired temperature
modulation via oracle instead of density matrix trace.
"""
import os, math, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

REPO = Path(r'D:\CCC 2.0\AI\agent-governance-system')
MODEL_DIR = str(REPO / 'THOUGHT' / 'LAB' / 'CAT_CAS' / '16_catalytic_27b_inference' / 'gemini_update' / 'qwen_0.5b')
HOLO_PATH = str(REPO / 'THOUGHT' / 'LAB' / 'EIGEN_BUDDY' / 'cybernetic_truth' / 'qwen_0_5b_k128.holo')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================================================
# ORACLE: Torus winding + circular variance (from 20.10.5)
# =====================================================================
class TorusOracle:
    """Measures topological stability of hidden state trajectory.
    NaN-immune: operates on S^1 phase angles, not raw magnitudes."""
    
    def __init__(self, buffer_size=16):
        self.L = buffer_size
        self.buffer = []
        self.step = 0
    
    def push(self, hidden_state):
        """Add hidden state to rolling buffer. Returns torus metrics."""
        # Clean NaN, cast to float32 (torch.polar doesn't support bf16)
        h = torch.nan_to_num(hidden_state.float(), nan=0., posinf=0., neginf=0.)
        self.buffer.append(h)
        if len(self.buffer) > self.L:
            self.buffer.pop(0)
        self.step += 1
        
        if len(self.buffer) < 3:
            return {'circ_var': 0.5, 'winding_stability': 0.5, 'phase_coh': 0.5}
        
        # Build complex observation matrix on S^1
        obs = torch.stack(self.buffer)  # (n, D)
        D = obs.shape[-1]
        
        # Map to unit circle: normalize magnitude, use as phase
        norms = obs.norm(dim=-1, keepdim=True).float()
        max_norm = norms.max()
        if max_norm > 1e-9:
            phases = (norms / max_norm) * math.pi
        else:
            phases = torch.zeros_like(norms)
        phases = phases.squeeze(-1)  # (n,)
        
        # Complex torus embedding: z = exp(i * phase) — per-token scalar phase
        z = torch.polar(torch.ones_like(phases), phases)  # (n,) complex
        
        # Circular variance: 0 = perfectly aligned, 1 = uniform random
        mean_z = z.mean()  # scalar complex
        R_val = mean_z.abs().item()
        circ_var = 1.0 - R_val
        circ_var = max(0.0, min(1.0, circ_var))
        
        # Winding stability: std of cosine similarity between consecutive states
        if len(self.buffer) >= 2:
            diffs = []
            for i in range(1, len(self.buffer)):
                a = self.buffer[i]; b = self.buffer[i-1]
                an = a / (a.norm() + 1e-9); bn = b / (b.norm() + 1e-9)
                cos_sim = (an * bn).sum().item()
                diffs.append(1.0 - max(-1.0, min(1.0, cos_sim)))
            winding_stability = 1.0 - min(1.0, np.std(diffs) * 5)
        else:
            winding_stability = 0.5
        
        # Phase coherence: mean of |z| across the buffer
        phase_coh = R_val
        
        # NaN guard
        if math.isnan(circ_var): circ_var = 0.5
        if math.isnan(winding_stability): winding_stability = 0.5
        if math.isnan(phase_coh): phase_coh = 0.5
        
        return {
            'circ_var': circ_var,
            'winding_stability': winding_stability,
            'phase_coh': phase_coh,
        }

# =====================================================================
# HoloLinear (unchanged)
# =====================================================================
class HoloLinear(nn.Module):
    def __init__(self, U, SVh, bias=None):
        super().__init__()
        self.U = nn.Parameter(U, requires_grad=False)
        self.SVh = nn.Parameter(SVh, requires_grad=False)
        self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None
    def forward(self, x):
        out = torch.matmul(torch.matmul(x, self.SVh.t()), self.U.t())
        if self.bias is not None: out += self.bias
        return out

def patch_holo(model, holo_dict):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            wk = name + ".weight"
            if wk + ".U" in holo_dict:
                U = holo_dict[wk + ".U"].to(device, dtype=torch.bfloat16)
                SVh = holo_dict[wk + ".SVh"].to(device, dtype=torch.bfloat16)
                bk = name + ".bias"
                bias = holo_dict[bk].to(device, dtype=torch.bfloat16) if bk in holo_dict else None
                holo = HoloLinear(U, SVh, bias)
                parent = model.get_submodule(name.rsplit('.', 1)[0])
                setattr(parent, name.rsplit('.', 1)[1], holo)

# =====================================================================
# Oracle-integrated inference
# =====================================================================
def oracle_inference(model, tokenizer, prompt, oracle, max_tokens=30):
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)['input_ids']
    output_tokens = []
    past_key_values = None
    
    print(f"Prompt: '{prompt}'")
    print("-" * 60)
    
    for step in range(max_tokens):
        with torch.no_grad():
            out = model(input_ids, past_key_values=past_key_values,
                       output_hidden_states=True, use_cache=True)
        
        logits = out.logits[:, -1, :]
        h_t = out.hidden_states[-1][:, -1, :].squeeze()
        
        # Oracle: torus metrics from rolling buffer
        metrics = oracle.push(h_t)
        circ_var = metrics['circ_var']
        winding = metrics['winding_stability']
        phase_coh = metrics['phase_coh']
        
        # Control law: circular variance drives temperature
        # Low variance (stable/truth) -> T ~ 0.3 (deterministic)
        # High variance (chaotic/exploring) -> T ~ 2.0 (exploratory)
        T = circ_var * 1.7 + 0.3
        
        # Sample
        safe = torch.nan_to_num(logits / T, nan=0., posinf=10., neginf=-10.)
        probs = torch.softmax(safe.float(), dim=-1)
        probs = torch.nan_to_num(probs, nan=0.)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_tok = torch.multinomial(probs, 1)
        output_tokens.append(next_tok.item())
        
        try:
            word = tokenizer.decode([next_tok.item()]).encode('ascii', errors='replace').decode('ascii')
        except:
            word = '?'
        
        rho = (circ_var < 0.3 and '~' or circ_var < 0.6 and '-' or '*')
        print(f"  {step+1:>2} {rho} var={circ_var:.3f} wind={winding:.3f} coh={phase_coh:.3f} T={T:.3f} | {word}")
        
        input_ids = next_tok
        past_key_values = out.past_key_values
    
    print("-" * 60)
    print(tokenizer.decode(output_tokens))

# =====================================================================
# MAIN
# =====================================================================
print("=" * 78)
print("ORACLE-INTEGRATED CYBERNETIC ENGINE (4.8)")
print("  Replacing R=Tr(rho*C) with Torus Circular Variance")
print("=" * 78)

if not os.path.exists(HOLO_PATH):
    print(f"Error: {HOLO_PATH} not found. Run distill_0_5b_holo.py first.")
    exit(1)

print("\nLoading model...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True)

with torch.device("meta"):
    model = AutoModelForCausalLM.from_config(config)
model.to_empty(device=device)

print("Loading .holo weights...", flush=True)
holo_dict = torch.load(HOLO_PATH, map_location="cpu", weights_only=True)

print("Patching HoloLinear...", flush=True)
patch_holo(model, holo_dict)

# Load remaining params
print("Loading remaining parameters...", flush=True)
original = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True, torch_dtype=torch.bfloat16)
orig_p = dict(original.named_parameters()); orig_b = dict(original.named_buffers())
for nm, p in model.named_parameters():
    if p.device.type == "meta":
        p.data = orig_p[nm].data.to(device) if nm in orig_p else torch.zeros(p.shape, dtype=torch.bfloat16, device=device)
for nm, b in model.named_buffers():
    if b.device.type == "meta" and nm in orig_b:
        b.data = orig_b[nm].data.to(device)
del original, orig_p, orig_b; torch.cuda.empty_cache()
model.eval()
print("Model ready.", flush=True)

# Run oracle inference
oracle = TorusOracle(buffer_size=16)
prompt = "The most interesting thing about artificial intelligence is"
oracle_inference(model, tokenizer, prompt, oracle, max_tokens=25)
