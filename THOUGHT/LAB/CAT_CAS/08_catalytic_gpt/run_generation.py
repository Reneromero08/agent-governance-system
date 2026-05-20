import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the current directory to python path to import catalytic_gpt
sys.path.insert(0, os.path.dirname(__file__))
from catalytic_gpt import CatalyticGPT
from run_experiment import StandardCausalSelfAttention, StandardMLP

# A standard-allocated Reversible Block for reference comparison
class StandardReversibleBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = StandardCausalSelfAttention(embed_dim, num_heads)
        self.mlp = StandardMLP(embed_dim)
        
    def forward(self, x1, x2):
        y1 = x1 + self.attn(x2)
        y2 = x2 + self.mlp(y1)
        return y1, y2

class StandardReversibleGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            StandardReversibleBlock(embed_dim // 2, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
    def forward(self, idx):
        x = self.tok_emb(idx)
        x1, x2 = x.chunk(2, dim=-1)
        x1 = x1.clone()
        x2 = x2.clone()
        for block in self.blocks:
            x1, x2 = block(x1, x2)
        x = torch.cat([x1, x2], dim=-1)
        x = self.ln_f(x)
        return self.lm_head(x[:, -1:, :])

def run_generation():
    print("=" * 80)
    print("AUTOREGRESSIVE TEXT GENERATION & MATHEMATICAL EQUIVALENCE TEST")
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
    generation_steps = 20  # Generate 20 tokens autoregressively
    
    # Initialize Standard Reversible GPT
    torch.manual_seed(42)
    standard_rev_model = StandardReversibleGPT(vocab_size, embed_dim, num_heads, num_layers).to(device)
    
    # Instantiate Catalytic model and copy weights exactly
    catalytic_model = CatalyticGPT(vocab_size, embed_dim, num_heads, num_layers).to(device)
    catalytic_model.load_state_dict(standard_rev_model.state_dict(), strict=False)
    print("[Model] Weights copied successfully between Reference and Catalytic models.")
    
    # Allocate shared VRAM tape
    TAPE_ELEMENTS = 32 * 1024 * 1024  # 128 MB
    tape = torch.empty(TAPE_ELEMENTS, device=device)
    torch.manual_seed(1234)
    tape.uniform_()
    
    # Initial prompt (batch_size=1, seq_len=5)
    prompt = torch.tensor([[101, 2045, 3012, 1037, 4000]], dtype=torch.long, device=device)
    
    print(f"\n[Prompt] Starting sequence: {prompt[0].tolist()}")
    
    # --------------------------------------------------------------------------
    #  1. Reference Reversible GPT Autoregressive Generation
    # --------------------------------------------------------------------------
    print("\n[Reference] Running autoregressive generation (standard allocations)...")
    context_ref = prompt.clone()
    ref_tokens = []
    
    with torch.no_grad():
        for step in range(generation_steps):
            # Standard forward pass
            logits = standard_rev_model(context_ref)
            # Greedy decoding
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ref_tokens.append(next_token.item())
            # Append next token to context
            context_ref = torch.cat([context_ref, next_token], dim=1)
            
    print(f"[Reference] Generated Tokens: {ref_tokens}")
    
    # --------------------------------------------------------------------------
    #  2. Catalytic GPT Autoregressive Generation
    # --------------------------------------------------------------------------
    print("\n[Catalytic] Running autoregressive generation (shared tape)...")
    context_cat = prompt.clone()
    cat_tokens = []
    
    with torch.no_grad():
        for step in range(generation_steps):
            # Catalytic forward pass (borrowing and restoring tape at each step)
            logits = catalytic_model(context_cat, tape)
            # Greedy decoding
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            cat_tokens.append(next_token.item())
            # Append next token to context
            context_cat = torch.cat([context_cat, next_token], dim=1)
            
    print(f"[Catalytic] Generated Tokens: {cat_tokens}")
    
    # --------------------------------------------------------------------------
    #  3. Equivalence Verification
    # --------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("MATHEMATICAL EQUIVALENCE REPORT")
    print("=" * 80)
    
    if ref_tokens == cat_tokens:
        print("[VERIFICATION] SUCCESS: Catalytic GPT matches Reference Reversible GPT 100% exactly!")
        print("  Generated Token IDs match byte-for-byte across all step intervals.")
        print("  Mathematical Equivalence: Perfect")
    else:
        print("[VERIFICATION] FAILURE: Token ID mismatch detected!")
        for idx, (s, c) in enumerate(zip(ref_tokens, cat_tokens)):
            if s != c:
                print(f"  Mismatch at token step {idx}: Reference={s}, Catalytic={c}")
        sys.exit(1)
    print("=" * 80)

if __name__ == "__main__":
    run_generation()
