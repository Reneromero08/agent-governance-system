"""
Complex-Phase KV Cache Compression on GPT-2
Packs 12 attention heads into 1 complex-valued head (6x compression, 83.3% VRAM savings).
Optimizes phase alignments and MLP correction adapters to maximize attention similarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from pathlib import Path
from typing import List, Dict

# Set reproducibility seeds
torch.manual_seed(42)
np.random.seed(42)

def compute_attention_output(q, k, v, num_heads, scale):
    batch, seq, hidden = q.shape
    head_dim = hidden // num_heads
    q_r = q.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    k_r = k.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    v_r = v.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    
    attn = torch.matmul(q_r, k_r.transpose(-2, -1)) * scale
    causal = torch.triu(torch.ones(seq, seq, device=q.device) * float('-inf'), diagonal=1)
    attn = attn + causal
    attn = F.softmax(attn, dim=-1)
    out = torch.matmul(attn, v_r)
    return out.transpose(1, 2).contiguous().view(batch, -1, hidden)

def compute_cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    a_flat = a.reshape(-1, a.shape[-1])
    b_flat = b.reshape(-1, b.shape[-1])
    cos = F.cosine_similarity(a_flat, b_flat, dim=-1)
    return cos.mean().item()

class ComplexPhaseKVCompressor(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Learnable phase offsets initialized to orthogonal spacing
        angles = torch.linspace(0, 2 * math.pi, num_heads + 1)[:num_heads]
        self.phases = nn.Parameter(angles)
        
        # Increased non-linear MLP capacity (12x head_dim bottleneck = 768)
        self.correction_k = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim * 12),
            nn.GELU(),
            nn.Linear(head_dim * 12, head_dim * num_heads, bias=False)
        )
        self.correction_v = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim * 12),
            nn.GELU(),
            nn.Linear(head_dim * 12, head_dim * num_heads, bias=False)
        )
        
        # Initialize MLP weights to be very small initially (residual corrections)
        for layer in [self.correction_k[0], self.correction_k[2], self.correction_v[0], self.correction_v[2]]:
            nn.init.normal_(layer.weight, std=0.001)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
        
    def compress(self, k, v):
        k_c = torch.complex(k, torch.zeros_like(k))
        v_c = torch.complex(v, torch.zeros_like(v))
        
        k_comp = torch.zeros(k.shape[0], k.shape[1], self.head_dim, dtype=torch.complex64, device=k.device)
        v_comp = torch.zeros(v.shape[0], v.shape[1], self.head_dim, dtype=torch.complex64, device=v.device)
        
        for h in range(self.num_heads):
            c, s = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            phase = torch.complex(c, s)
            k_comp = k_comp + k_c[:, :, h] * phase
            v_comp = v_comp + v_c[:, :, h] * phase
            
        return k_comp, v_comp

    def retrieve(self, k_comp, v_comp):
        k_decomp_list = []
        v_decomp_list = []
        
        for h in range(self.num_heads):
            c, s = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            phase_conj = torch.complex(c, -s)
            k_decomp_list.append((k_comp * phase_conj).real)
            v_decomp_list.append((v_comp * phase_conj).real)
            
        k_decomp = torch.stack(k_decomp_list, dim=2)
        v_decomp = torch.stack(v_decomp_list, dim=2)
        
        batch, seq, _ = k_comp.shape
        comp_flat_k = torch.cat([k_comp.real, k_comp.imag], dim=-1)
        comp_flat_v = torch.cat([v_comp.real, v_comp.imag], dim=-1)
        
        corr_k = self.correction_k(comp_flat_k).view(batch, seq, self.num_heads, self.head_dim)
        corr_v = self.correction_v(comp_flat_v).view(batch, seq, self.num_heads, self.head_dim)
        
        return k_decomp + corr_k, v_decomp + corr_v

def run_experiment(device="cpu"):
    print("[1/4] Loading GPT-2 from local model...")
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    
    local_path = str(Path(__file__).parent.parent.parent / "models" / "gpt2")
    model = GPT2LMHeadModel.from_pretrained(local_path).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(local_path)
    model.eval()
    
    n_layers = model.config.n_layer
    num_heads = model.config.n_head
    head_dim = model.config.n_embd // num_heads
    scale = 1.0 / math.sqrt(head_dim)
    
    train_texts = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology every day.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons.",
        "Climate change poses significant challenges for future generations.",
        "Quantum mechanics describes the behavior of matter and light at the atomic scale.",
        "The universe is expanding at an accelerating rate due to dark energy.",
        "DNA stores the genetic instructions for all living organisms.",
        "Photosynthesis converts sunlight, water, and carbon dioxide into oxygen and glucose.",
        "Economic systems distribute resources to satisfy human needs and wants.",
        "Throughout history, empires have risen and fallen in patterns of growth and decay.",
        "Standard languages evolve through cultural integration and geographic separation.",
        "Prime numbers are the building blocks of arithmetic and cryptography.",
        "General relativity explains gravity as the curvature of space and time.",
        "Neural networks simulate the brain to learn from patterns in images.",
        "Computer architectures optimize pipeline execution to execute instructions faster.",
        "Software engineering practices emphasize modularity, testing, and continuous delivery.",
        "The Earth revolves around the Sun in an elliptical orbit once a year.",
        "Viruses represent the boundary between living organisms and non-living matter.",
        "Human languages express complex thoughts through hierarchical grammatical rules."
    ]
    
    test_texts = [
        "Economic systems attempt to explain how resources are allocated in complex societies.",
        "The history of science is a story of ideas evolving through observation.",
        "Electromagnetism governs the interactions of charged particles in fields.",
        "Information theory defines entropy as the average rate of information produced.",
        "Plate tectonics explains the large-scale motion of Earth's lithospheric plates."
    ]
    
    print("\n[2/4] Hooking and collecting activations for training...")
    qkv_data = {i: [] for i in range(n_layers)}
    hooks = []
    
    def make_qkv_hook(layer_idx):
        def hook(module, input, output):
            split_sz = output.shape[-1] // 3
            q = output[..., :split_sz]
            k = output[..., split_sz:2*split_sz]
            v = output[..., 2*split_sz:]
            qkv_data[layer_idx].append((q.detach(), k.detach(), v.detach()))
        return hook
        
    for idx in range(n_layers):
        h = model.transformer.h[idx].attn.c_attn.register_forward_hook(make_qkv_hook(idx))
        hooks.append(h)
        
    with torch.no_grad():
        for text in train_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
            
    # Test data
    qkv_test = {i: [] for i in range(n_layers)}
    for h in hooks:
        h.remove()
        
    hooks = []
    def make_test_hook(layer_idx):
        def hook(module, input, output):
            split_sz = output.shape[-1] // 3
            q = output[..., :split_sz]
            k = output[..., split_sz:2*split_sz]
            v = output[..., 2*split_sz:]
            qkv_test[layer_idx].append((q.detach(), k.detach(), v.detach()))
        return hook
        
    for idx in range(n_layers):
        h = model.transformer.h[idx].attn.c_attn.register_forward_hook(make_test_hook(idx))
        hooks.append(h)
        
    with torch.no_grad():
        for text in test_texts:
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            _ = model(**inputs)
            
    for h in hooks:
        h.remove()
        
    print(f"Collected {len(qkv_data[0])} train samples and {len(qkv_test[0])} test samples per layer.")
    
    print("\n[3/4] Evaluating Zero-Shot and Training Complex Adapters...")
    results = []
    
    for layer_idx in range(n_layers):
        print(f"\n--- Layer {layer_idx} ---")
        
        compressor = ComplexPhaseKVCompressor(num_heads, head_dim).to(device)
        
        # Optimize learning rate and weight decay
        optimizer = torch.optim.AdamW(compressor.parameters(), lr=1.8e-3, weight_decay=1e-5)
        
        epochs = 80
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        # 1. Zero-shot evaluation
        zero_shot_cos_list = []
        for q, k, v in qkv_test[layer_idx]:
            B, S, _ = q.shape
            k_h = k.view(B, S, num_heads, head_dim)
            v_h = v.view(B, S, num_heads, head_dim)
            
            with torch.no_grad():
                k_comp, v_comp = compressor.compress(k_h, v_h)
                k_ret, v_ret = compressor.retrieve(k_comp, v_comp)
                
                k_ret_flat = k_ret.view(B, S, -1)
                v_ret_flat = v_ret.view(B, S, -1)
                
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q, k_ret_flat, v_ret_flat, num_heads, scale)
                
                zero_shot_cos_list.append(compute_cosine_sim(orig_attn, comp_attn))
                
        zero_shot_cos = np.mean(zero_shot_cos_list)
        print(f"  Zero-Shot Attention Cosine Similarity: {zero_shot_cos:.4f}")
        
        # 2. Train the compressor (80 epochs with pure attention loss)
        compressor.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for q, k, v in qkv_data[layer_idx]:
                optimizer.zero_grad()
                B, S, _ = q.shape
                k_h = k.view(B, S, num_heads, head_dim)
                v_h = v.view(B, S, num_heads, head_dim)
                
                k_comp, v_comp = compressor.compress(k_h, v_h)
                k_ret, v_ret = compressor.retrieve(k_comp, v_comp)
                
                k_ret_flat = k_ret.view(B, S, -1)
                v_ret_flat = v_ret.view(B, S, -1)
                
                # Pure attention representation alignment loss
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q, k_ret_flat, v_ret_flat, num_heads, scale)
                loss = F.mse_loss(comp_attn, orig_attn)
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            scheduler.step()
                
        # 3. Post-training evaluation
        compressor.eval()
        trained_cos_list = []
        for q, k, v in qkv_test[layer_idx]:
            B, S, _ = q.shape
            k_h = k.view(B, S, num_heads, head_dim)
            v_h = v.view(B, S, num_heads, head_dim)
            
            with torch.no_grad():
                k_comp, v_comp = compressor.compress(k_h, v_h)
                k_ret, v_ret = compressor.retrieve(k_comp, v_comp)
                
                k_ret_flat = k_ret.view(B, S, -1)
                v_ret_flat = v_ret.view(B, S, -1)
                
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q, k_ret_flat, v_ret_flat, num_heads, scale)
                
                trained_cos_list.append(compute_cosine_sim(orig_attn, comp_attn))
                
        trained_cos = np.mean(trained_cos_list)
        delta = trained_cos - zero_shot_cos
        print(f"  Trained Attention Cosine Similarity:   {trained_cos:.4f} (Delta: {delta:+.4f})")
        
        results.append({
            'layer': layer_idx,
            'zero_shot': zero_shot_cos,
            'trained': trained_cos,
            'delta': delta
        })
        
    print("\n\n" + "=" * 70)
    print("GPT-2 COMPLEX-PHASE COMPRESSION REPORT (6x / 83.3% VRAM Savings)")
    print("=" * 70)
    print("| Layer | Zero-Shot Attn Cos | Trained Attn Cos | Delta |")
    print("|-------|-------------------|------------------|-------|")
    for r in results:
        print(f"| L{r['layer']:<4d} | {r['zero_shot']:<17.4f} | {r['trained']:<16.4f} | {r['delta']:<+5.4f} |")
    
    avg_zs = np.mean([r['zero_shot'] for r in results])
    avg_tr = np.mean([r['trained'] for r in results])
    print(f"| **AVG** | **{avg_zs:.4f}** | **{avg_tr:.4f}** | **{avg_tr - avg_zs:+.4f}** |")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_experiment(device)
