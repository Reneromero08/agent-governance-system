"""
Eigen Buddy Inference — Qwen-distilled phase eigenbasis generating text
=========================================================================
Loads distilled .holo phase eigenbasis, injects into scaled Eigen Buddy
attention, runs autoregressive text generation via Qwen tokenizer.
"""
import sys, json, math, numpy as np, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

MODEL_DIR = r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B"
HOLO_DIR = Path(__file__).parent / "distilled"
D_MODEL = 1024  # smallest distilled dimension
N_HEADS = 8
N_LAYERS = 2

def load_distilled_weights(holo_path):
    """Load phase gratings from .holo file."""
    data = np.load(holo_path)
    weights = {}
    for k in data.files:
        weights[k] = torch.tensor(data[k])  # (k_dim, dim) complex64
    return weights

def inject_attention(attn, weights, layer_name):
    """Inject distilled eigenbasis into one attention module."""
    d_model = attn.qr.weight.shape[1]
    n_heads = attn.H
    dh = d_model // n_heads
    
    # Find matching distilled weights for this layer
    matches = [(k, w) for k, w in weights.items() 
               if layer_name in k and w.shape[1] == d_model]
    
    if not matches:
        return False
    
    with torch.no_grad():
        for h in range(n_heads):
            start, end = h * dh, (h + 1) * dh
            for proj_name in ['qr', 'qi', 'kr', 'ki', 'vr', 'vi']:
                w = getattr(attn, proj_name).weight
                # Use different eigenmodes for different projections
                eigen_idx = (h + (0 if 'q' in proj_name else 1 if 'k' in proj_name else 2)) % len(matches)
                grating = matches[eigen_idx][1]  # (k, dim) complex64
                eigenmode = grating[h % grating.shape[0], :]  # (dim,) complex64
                
                head_angle = 2 * math.pi * h / n_heads
                rot = np.exp(1j * head_angle)
                rotated = (eigenmode * rot).numpy()
                if 'i' in proj_name:
                    w.data[start:end] = torch.tensor(rotated.imag.astype(np.float32)).unsqueeze(0).expand(dh, -1) * 0.1
                else:
                    w.data[start:end] = torch.tensor(rotated.real.astype(np.float32)).unsqueeze(0).expand(dh, -1) * 0.1
    return True

class EigenBuddyLM(torch.nn.Module):
    """Minimal Eigen Buddy language model for inference."""
    def __init__(self, d_model, n_heads, n_layers, vocab_size):
        super().__init__()
        self.embed_r = torch.nn.Embedding(vocab_size, d_model)
        self.embed_i = torch.nn.Embedding(vocab_size, d_model)
        self.layers = torch.nn.ModuleList([
            MultiHeadComplexAttention(d_model, n_heads, geo_init=False)
            for _ in range(n_layers)
        ])
        self.out_r = torch.nn.Linear(d_model, vocab_size)
        self.out_i = torch.nn.Linear(d_model, vocab_size)
    
    def forward(self, input_ids):
        x_r = self.embed_r(input_ids)
        x_i = self.embed_i(input_ids)
        x = torch.complex(x_r, x_i)
        for layer in self.layers:
            x, _ = layer(x)
        logits = self.out_r(x.real) + self.out_i(x.imag)
        return logits

def main():
    print("=" * 78)
    print("EIGEN BUDDY INFERENCE — Qwen-distilled phase eigenbasis")
    print(f"  d_model={D_MODEL} heads={N_HEADS} layers={N_LAYERS}")
    print("=" * 78)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size
    print(f"\n  Tokenizer: {vocab_size} tokens  pad_id={tokenizer.pad_token_id}")
    
    # Load distilled weights
    holo = HOLO_DIR / "eigenbuddy_distilled.holo.npz"
    if not holo.exists():
        holo = HOLO_DIR / "eigenbuddy_qwen27b.holo.npz"
    if not holo.exists():
        print(f"  No .holo file found in {HOLO_DIR}")
        return
    
    weights = load_distilled_weights(holo)
    print(f"  Loaded {len(weights)} phase gratings from {holo.name}")
    
    # Create model
    model = EigenBuddyLM(D_MODEL, N_HEADS, N_LAYERS, vocab_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")
    
    # Inject distilled weights into attention layers
    injected = 0
    for i, layer in enumerate(model.layers):
        if inject_attention(layer, weights, f".{i}."):
            injected += 1
    print(f"  Injected Qwen eigenbasis into {injected}/{N_LAYERS} layers")
    
    # Generate
    prompt = "The meaning of life is"
    print(f"\n  Prompt: '{prompt}'")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    with torch.no_grad():
        for _ in range(10):
            logits = model(input_ids)
            next_token = logits[:, -1, :].argmax(dim=-1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
    
    generated = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"  Generated: '{generated}'")
    print("=" * 78)

if __name__ == "__main__":
    main()
