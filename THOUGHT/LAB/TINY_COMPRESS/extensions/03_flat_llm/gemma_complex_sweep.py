"""
Gemma 4 Complex-Phase Query Compression Sweep
Sweeps all 15 non-shared layers of Gemma 4.
Compresses 8 query heads to 1 complex query head (4x compression).
Trains a 2-layer MLP adapter with joint loss over 80 epochs per layer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from pathlib import Path

# Set reproducibility seeds
torch.manual_seed(42)
np.random.seed(42)

def compute_attention_output(q, k, v, num_heads, scale):
    batch, seq, hidden = q.shape
    head_dim = hidden // num_heads
    q_r = q.view(batch, -1, num_heads, head_dim).transpose(1, 2)
    
    # GQA broadcast K and V
    k_r = k.view(batch, -1, k.shape[-1] // head_dim, head_dim).transpose(1, 2)
    v_r = v.view(batch, -1, v.shape[-1] // head_dim, head_dim).transpose(1, 2)
    
    num_query_heads = num_heads
    num_kv_heads = k_r.shape[1]
    num_key_value_groups = num_query_heads // num_kv_heads
    
    if num_key_value_groups > 1:
        k_r = k_r.repeat_interleave(num_key_value_groups, dim=1)
        v_r = v_r.repeat_interleave(num_key_value_groups, dim=1)
        
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

class ComplexPhaseQueryCompressor(nn.Module):
    def __init__(self, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        
        # Learnable phase offsets
        angles = torch.linspace(0, 2 * math.pi, num_heads + 1)[:num_heads]
        self.phases = nn.Parameter(angles)
        
        # 2-layer MLP adapter
        self.correction = nn.Sequential(
            nn.Linear(head_dim * 2, head_dim * 8),
            nn.GELU(),
            nn.Linear(head_dim * 8, head_dim * num_heads, bias=False)
        )
        
        # Initialize MLP weights to be very small initially
        for layer in [self.correction[0], self.correction[2]]:
            nn.init.normal_(layer.weight, std=0.002)
            if hasattr(layer, 'bias') and layer.bias is not None:
                nn.init.constant_(layer.bias, 0.0)
                
    def compress(self, q):
        q_c = torch.complex(q, torch.zeros_like(q))
        q_comp = torch.zeros(q.shape[0], q.shape[1], self.head_dim, dtype=torch.complex64, device=q.device)
        
        for h in range(self.num_heads):
            c, s = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            phase = torch.complex(c, s)
            q_comp = q_comp + q_c[:, :, h] * phase
            
        return q_comp

    def retrieve(self, q_comp):
        q_decomp_list = []
        for h in range(self.num_heads):
            c, s = torch.cos(self.phases[h]), torch.sin(self.phases[h])
            phase_conj = torch.complex(c, -s)
            q_decomp_list.append((q_comp * phase_conj).real)
            
        q_decomp = torch.stack(q_decomp_list, dim=2)
        
        batch, seq, _ = q_comp.shape
        comp_flat = torch.cat([q_comp.real, q_comp.imag], dim=-1)
        corr = self.correction(comp_flat).view(batch, seq, self.num_heads, self.head_dim)
        
        return q_decomp + corr

def run_sweep(device="cpu"):
    print("[1/4] Loading Gemma 4 from local model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    model_id = "google/gemma-4-E2B-it"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    model.eval()
    
    # We sweep the first 15 non-shared layers
    n_layers_to_sweep = 15
    num_heads = model.config.text_config.num_attention_heads
    
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
        "Economic systems distribute resources to satisfy human needs and wants."
    ]
    
    test_texts = [
        "Economic systems attempt to explain how resources are allocated in complex societies.",
        "The history of science is a story of ideas evolving through observation.",
        "Electromagnetism governs the interactions of charged particles in fields."
    ]
    
    results = []
    
    for layer_idx in range(n_layers_to_sweep):
        print(f"\n--- Layer {layer_idx} ---")
        attn_module = model.model.language_model.layers[layer_idx].self_attn
        head_dim = attn_module.head_dim
        scale = 1.0 / math.sqrt(head_dim)
        
        q_data, k_data, v_data = [], [], []
        
        def q_hook(module, input, output):
            q_data.append(output.detach().float().cpu())
        def k_hook(module, input, output):
            k_data.append(output.detach().float().cpu())
        def v_hook(module, input, output):
            v_data.append(output.detach().float().cpu())
            
        hq = attn_module.q_proj.register_forward_hook(q_hook)
        hk = attn_module.k_proj.register_forward_hook(k_hook)
        hv = attn_module.v_proj.register_forward_hook(v_hook)
        
        with torch.no_grad():
            for text in train_texts:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                _ = model(**inputs)
                
        hq.remove()
        hk.remove()
        hv.remove()
        
        num_train = len(q_data)
        
        hq = attn_module.q_proj.register_forward_hook(q_hook)
        hk = attn_module.k_proj.register_forward_hook(k_hook)
        hv = attn_module.v_proj.register_forward_hook(v_hook)
        
        with torch.no_grad():
            for text in test_texts:
                inputs = tokenizer(text, return_tensors='pt').to(device)
                _ = model(**inputs)
                
        hq.remove()
        hk.remove()
        hv.remove()
        
        qkv_data = []
        for i in range(len(q_data)):
            q = q_data[i]
            k = k_data[i]
            v = v_data[i]
            
            with torch.no_grad():
                q_h = q.view(q.shape[0], q.shape[1], num_heads, head_dim)
                q_normed = attn_module.q_norm(q_h.to(device)).cpu()
                q_final = q_normed.view(q.shape[0], q.shape[1], -1)
                
                k_h = k.view(k.shape[0], k.shape[1], -1, head_dim)
                k_normed = attn_module.k_norm(k_h.to(device)).cpu()
                k_final = k_normed.view(k.shape[0], k.shape[1], -1)
                
                v_h = v.view(v.shape[0], v.shape[1], -1, head_dim)
                v_normed = attn_module.v_norm(v_h.to(device)).cpu()
                v_final = v_normed.view(v.shape[0], v.shape[1], -1)
                
                qkv_data.append((q_final, k_final, v_final))
                
        train_data = qkv_data[:num_train]
        test_data = qkv_data[num_train:]
        
        # Setup compressor
        compressor = ComplexPhaseQueryCompressor(num_heads, head_dim).to(device)
        optimizer = torch.optim.AdamW(compressor.parameters(), lr=1.5e-3, weight_decay=1e-5)
        
        # Zero-shot
        zero_shot_cos_list = []
        for q, k, v in test_data:
            q = q.to(device)
            k = k.to(device)
            v = v.to(device)
            B, S, _ = q.shape
            q_h = q.view(B, S, num_heads, head_dim)
            with torch.no_grad():
                q_comp = compressor.compress(q_h)
                q_ret = compressor.retrieve(q_comp)
                q_ret_flat = q_ret.view(B, S, -1)
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q_ret_flat, k, v, num_heads, scale)
                zero_shot_cos_list.append(compute_cosine_sim(orig_attn, comp_attn))
        zero_shot_cos = np.mean(zero_shot_cos_list)
        print(f"  Zero-Shot Attention Cosine Similarity: {zero_shot_cos:.4f}")
        
        # Train
        epochs = 80
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        compressor.train()
        for epoch in range(epochs):
            for q, k, v in train_data:
                q = q.to(device)
                k = k.to(device)
                v = v.to(device)
                optimizer.zero_grad()
                B, S, _ = q.shape
                q_h = q.view(B, S, num_heads, head_dim)
                
                q_comp = compressor.compress(q_h)
                q_ret = compressor.retrieve(q_comp)
                q_ret_flat = q_ret.view(B, S, -1)
                
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q_ret_flat, k, v, num_heads, scale)
                
                loss_attn = F.mse_loss(comp_attn, orig_attn)
                loss_q = F.mse_loss(q_ret, q_h)
                loss = loss_attn + 0.1 * loss_q
                
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        # Eval
        compressor.eval()
        trained_cos_list = []
        for q, k, v in test_data:
            q = q.to(device)
            k = k.to(device)
            v = v.to(device)
            B, S, _ = q.shape
            q_h = q.view(B, S, num_heads, head_dim)
            with torch.no_grad():
                q_comp = compressor.compress(q_h)
                q_ret = compressor.retrieve(q_comp)
                q_ret_flat = q_ret.view(B, S, -1)
                orig_attn = compute_attention_output(q, k, v, num_heads, scale)
                comp_attn = compute_attention_output(q_ret_flat, k, v, num_heads, scale)
                trained_cos_list.append(compute_cosine_sim(orig_attn, comp_attn))
        trained_cos = np.mean(trained_cos_list)
        delta = trained_cos - zero_shot_cos
        print(f"  Trained Attention Cosine Similarity:   {trained_cos:.4f} (Delta: {delta:+.4f})")
        
        results.append({
            'layer': layer_idx,
            'type': "full" if attn_module.layer_type == "full_attention" else "sliding",
            'zero_shot': zero_shot_cos,
            'trained': trained_cos,
            'delta': delta
        })
        
    print("\n\n" + "=" * 70)
    print("GEMMA 4 QUERY COMPLEX-PHASE COMPRESSION REPORT (8 heads -> 1 complex head)")
    print("=" * 70)
    print("| Layer | Type | Zero-Shot Attn Cos | Trained Attn Cos | Delta |")
    print("|-------|------|-------------------|------------------|-------|")
    for r in results:
        print(f"| L{r['layer']:<4d} | {r['type']:<4s} | {r['zero_shot']:<17.4f} | {r['trained']:<16.4f} | {r['delta']:<+5.4f} |")
    
    avg_zs = np.mean([r['zero_shot'] for r in results])
    avg_tr = np.mean([r['trained'] for r in results])
    print(f"| **AVG** | **-**  | **{avg_zs:.4f}** | **{avg_tr:.4f}** | **{avg_tr - avg_zs:+.4f}** |")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_sweep(device)
