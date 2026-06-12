import os
import sys
import time
import hashlib
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the current directory to python path to import catalytic_gpt
sys.path.insert(0, os.path.dirname(__file__))
from catalytic_gpt import CatalyticGPT

# ==============================================================================
#  Standard GPT Implementation (For Baseline Comparison)
# ==============================================================================
class StandardCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, T, C = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q / math.sqrt(self.head_dim), k.transpose(-1, -2))
        mask = torch.triu(torch.full((T, T), float('-inf'), device=x.device), diagonal=1)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        
        y = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(y)

import math

class StandardMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ffn1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.ffn2 = nn.Linear(4 * embed_dim, embed_dim)
        
    def forward(self, x):
        return self.ffn2(F.gelu(self.ffn1(x)))

class StandardBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = StandardCausalSelfAttention(embed_dim, num_heads)
        self.mlp = StandardMLP(embed_dim)
        
    def forward(self, x):
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x

class StandardGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            StandardBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx):
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1:, :])

# ==============================================================================
#  Experiment Runner
# ==============================================================================
def run_experiment():
    print("=" * 80)
    print("CATALYTIC GPU GPT INFESTATION & RESTORATION EXPERIMENT")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("[CRITICAL] CUDA is not available! This experiment must be run on GPU.")
        sys.exit(1)
        
    device = torch.device("cuda")
    print(f"[Device] Running on: {torch.cuda.get_device_name(0)}")
    
    # Model parameters
    vocab_size = 50257
    embed_dim = 128
    num_heads = 4
    num_layers = 6
    batch_size = 8
    seq_length = 256  # Larger context to show clear activation memory profile
    
    print(f"[Config] Vocabulary Size:    {vocab_size}")
    print(f"[Config] Embedding Dim:      {embed_dim}")
    print(f"[Config] Attention Heads:    {num_heads}")
    print(f"[Config] GPT Layers:         {num_layers}")
    print(f"[Config] Batch Size:         {batch_size}")
    print(f"[Config] Sequence Length:    {seq_length}")
    
    # Generate input data
    idx = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    
    # --------------------------------------------------------------------------
    #  PHASE 1: Standard GPT Baseline Run
    # --------------------------------------------------------------------------
    print("\n--- PHASE 1: Standard GPT Inference (Conventional Allocation) ---")
    standard_model = StandardGPT(vocab_size, embed_dim, num_heads, num_layers).to(device)
    
    # Run once to warm up PyTorch CUDA cache
    with torch.no_grad():
        _ = standard_model(idx)
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_allocated = torch.cuda.memory_allocated()
    
    # Run inference and measure peak memory
    t_start = time.time()
    with torch.no_grad():
        _ = standard_model(idx)
    t_end = time.time()
    
    standard_peak = torch.cuda.max_memory_allocated()
    standard_act_vram = (standard_peak - base_allocated) / (1024 * 1024)
    print(f"[Standard] Compute Time:          {t_end - t_start:.4f}s")
    print(f"[Standard] Base Weights Memory:   {base_allocated / (1024*1024):.2f} MB")
    print(f"[Standard] Peak VRAM Allocated:   {standard_peak / (1024*1024):.2f} MB")
    print(f"[Standard] Active Activations VRAM: {standard_act_vram:.2f} MB")
    
    # Free standard model memory
    del standard_model
    torch.cuda.empty_cache()
    
    # --------------------------------------------------------------------------
    #  PHASE 2: Catalytic GPT Run (Zero-Allocation VRAM Tape Borrowing)
    # --------------------------------------------------------------------------
    print("\n--- PHASE 2: Catalytic GPT Inference (Tape-Borrowing Reversible Blocks) ---")
    
    # Allocate a 128 MB "dirty VRAM tape" containing random data (representing other applications' VRAM)
    TAPE_ELEMENTS = 32 * 1024 * 1024  # 32M float32 elements = 128 MB
    print(f"[Tape] Allocating 128 MB dirty VRAM tape on GPU...")
    tape = torch.empty(TAPE_ELEMENTS, device=device)
    torch.manual_seed(1234)
    tape.uniform_()
    
    # Generate baseline hash of the tape
    print("[Tape] Computing baseline SHA-256 hash of the dirty VRAM tape...")
    t_hash_start = time.time()
    tape_bytes = tape.cpu().numpy().tobytes()
    baseline_hash = hashlib.sha256(tape_bytes).hexdigest()
    print(f"[Tape] Baseline SHA-256: {baseline_hash} ({time.time() - t_hash_start:.2f}s)")
    
    # Initialize catalytic model
    catalytic_model = CatalyticGPT(vocab_size, embed_dim, num_heads, num_layers).to(device)
    
    # Warm up catalytic model
    with torch.no_grad():
        _ = catalytic_model(idx, tape)
        
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_allocated_cat = torch.cuda.memory_allocated()
    
    # Run catalytic inference
    t_start = time.time()
    with torch.no_grad():
        _ = catalytic_model(idx, tape)
    t_end = time.time()
    
    catalytic_peak = torch.cuda.max_memory_allocated()
    catalytic_act_vram = (catalytic_peak - base_allocated_cat) / (1024 * 1024)
    print(f"[Catalytic] Compute Time:         {t_end - t_start:.4f}s")
    print(f"[Catalytic] Base Weights Memory:  {(base_allocated_cat - 128 * 1024 * 1024) / (1024*1024):.2f} MB (excluding 128MB Tape)")
    print(f"[Catalytic] Peak VRAM Allocated:  {catalytic_peak / (1024*1024):.2f} MB")
    print(f"[Catalytic] Active Activations VRAM: {catalytic_act_vram:.2f} MB")
    
    # Compute final hash of the tape to verify restoration
    print("\n[Tape] Computing final SHA-256 hash of the dirty VRAM tape...")
    t_hash_start = time.time()
    final_tape_bytes = tape.cpu().numpy().tobytes()
    final_hash = hashlib.sha256(final_tape_bytes).hexdigest()
    print(f"[Tape] Final SHA-256:    {final_hash} ({time.time() - t_hash_start:.2f}s)")
    
    # --------------------------------------------------------------------------
    #  PHASE 3: Integrity & Memory Analysis Report
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("VERIFICATION & ANALYSIS REPORT")
    print("=" * 80)
    
    if final_hash == baseline_hash:
        print("[VERIFICATION] SUCCESS: 128 MB VRAM Tape restored 100% byte-for-byte.")
        print(f"  Entropy Leak:        0.0 Joules")
        print(f"  Memory Integrity:    Perfect")
    else:
        print("[VERIFICATION] FAILURE: VRAM Tape corruption detected!")
        sys.exit(1)
        
    print(f"\nMemory Allocation Profile:")
    print(f"  Standard Activation VRAM:    {standard_act_vram:.4f} MB")
    print(f"  Catalytic Activation VRAM:   {catalytic_act_vram:.4f} MB")
    
    savings_pct = (1.0 - (catalytic_act_vram / standard_act_vram)) * 100 if standard_act_vram > 0 else 100.0
    print(f"  Dynamic Memory Saved:        {standard_act_vram - catalytic_act_vram:.4f} MB ({savings_pct:.1f}% reduction)")
    print("=" * 80)

if __name__ == "__main__":
    run_experiment()
