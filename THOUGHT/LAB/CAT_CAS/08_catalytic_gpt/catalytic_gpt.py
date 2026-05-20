import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReversibleCausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward_with_tape(self, x, tape, tape_offset, mask):
        """
        Computes attention using the tape as scratch space.
        x: [B, T, C] (clean input)
        tape: [TAPE_SIZE] (dirty VRAM tensor)
        tape_offset: int
        Returns:
            y: [B, T, C] (output)
            next_offset: int
        """
        B, T, C = x.shape
        
        # Calculate tensor sizes in elements
        proj_size = B * T * C
        attn_size = B * self.num_heads * T * T
        
        # Slice views directly from the pre-existing tape (no allocation)
        q_buf = tape[tape_offset : tape_offset + proj_size].view(B, T, C)
        k_buf = tape[tape_offset + proj_size : tape_offset + 2 * proj_size].view(B, T, C)
        v_buf = tape[tape_offset + 2 * proj_size : tape_offset + 3 * proj_size].view(B, T, C)
        s_buf = tape[tape_offset + 3 * proj_size : tape_offset + 3 * proj_size + attn_size].view(B, self.num_heads, T, T)
        y_buf = tape[tape_offset + 3 * proj_size + attn_size : tape_offset + 4 * proj_size + attn_size].view(B, T, C)
        
        # Size of the total borrowed region
        attn_size_total = 4 * proj_size + attn_size
        
        # Perform matrix multiplications directly into the tape views
        # Q = X_2 * W_q + b_q
        torch.addmm(self.q_proj.bias, x.view(-1, C), self.q_proj.weight.t(), out=q_buf.view(-1, C))
        # K = X_2 * W_k + b_k
        torch.addmm(self.k_proj.bias, x.view(-1, C), self.k_proj.weight.t(), out=k_buf.view(-1, C))
        # V = X_2 * W_v + b_v
        torch.addmm(self.v_proj.bias, x.view(-1, C), self.v_proj.weight.t(), out=v_buf.view(-1, C))
        
        # Reshape Q, K, V for multi-head attention
        q = q_buf.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k_buf.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v_buf.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores S = (Q * K^T) / sqrt(head_dim)
        # Scale in-place on the tape view to save allocation
        q.div_(math.sqrt(self.head_dim))
        torch.matmul(q, k.transpose(-1, -2), out=s_buf)
        
        # Apply causal mask (in-place on s_buf from pre-allocated buffer)
        s_buf.add_(mask)
        
        # Softmax in-place
        s_buf.copy_(F.softmax(s_buf, dim=-1))
        
        # Y = S * V
        attn_out = y_buf.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        torch.matmul(s_buf, v, out=attn_out)
        
        # Reshape y_buf back to 2D for linear projection
        y_buf_flat = y_buf.view(-1, C)
        
        # Compute final projection directly to the output of the attention layer
        out = torch.addmm(self.out_proj.bias, y_buf_flat, self.out_proj.weight.t()).view(B, T, C)
        
        return out, attn_size_total

    def restore_tape(self, tape, tape_offset, size):
        """
        Restores the borrowed tape slice back to its original dirty state.
        """
        torch.manual_seed(1234)
        tape[tape_offset : tape_offset + size].uniform_()




class ReversibleMLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.ffn1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.ffn2 = nn.Linear(4 * embed_dim, embed_dim)
        
    def forward_with_tape(self, x, tape, tape_offset):
        B, T, C = x.shape
        intermediate_size = B * T * 4 * C
        
        # Slice intermediate buffer from tape
        mid_buf = tape[tape_offset : tape_offset + intermediate_size].view(B, T, 4 * C)
        
        # First FFN Layer: intermediate = GeLU(X * W1 + b1)
        torch.addmm(self.ffn1.bias, x.view(-1, C), self.ffn1.weight.t(), out=mid_buf.view(-1, 4 * C))
        mid_buf.copy_(F.gelu(mid_buf))
        
        # Second FFN Layer: output = intermediate * W2 + b2
        out = torch.addmm(self.ffn2.bias, mid_buf.view(-1, 4 * C), self.ffn2.weight.t()).view(B, T, C)
        
        return out, intermediate_size

    def restore_tape(self, tape, tape_offset, size):
        torch.manual_seed(1234)
        tape[tape_offset : tape_offset + size].uniform_()



class ReversibleBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = ReversibleCausalSelfAttention(embed_dim // 2, num_heads)
        self.mlp = ReversibleMLP(embed_dim // 2)
        
    def forward(self, x1, x2, tape, tape_offset, mask):
        # Y1 = X1 + Attention(X2)
        attn_out, size_attn = self.attn.forward_with_tape(x2, tape, tape_offset, mask)
        x1.add_(attn_out)
        self.attn.restore_tape(tape, tape_offset, size_attn)
        
        # Y2 = X2 + MLP(Y1)
        mlp_out, size_mlp = self.mlp.forward_with_tape(x1, tape, tape_offset)
        x2.add_(mlp_out)
        self.mlp.restore_tape(tape, tape_offset, size_mlp)
        
        return x1, x2

    def backward(self, y1, y2, tape, tape_offset, mask):
        # Reconstruct X2 = Y2 - MLP(Y1)
        mlp_out, size_mlp = self.mlp.forward_with_tape(y1, tape, tape_offset)
        y2.sub_(mlp_out)
        self.mlp.restore_tape(tape, tape_offset, size_mlp)
        
        # Reconstruct X1 = Y1 - Attention(X2)
        attn_out, size_attn = self.attn.forward_with_tape(y2, tape, tape_offset, mask)
        y1.sub_(attn_out)
        self.attn.restore_tape(tape, tape_offset, size_attn)
        
        return y1, y2



class CatalyticGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([
            ReversibleBlock(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Pre-allocate causal mask buffer (up to 2048 sequence length)
        self.register_buffer("mask", torch.triu(torch.full((2048, 2048), float('-inf')), diagonal=1))
        
    def forward(self, idx, tape):
        B, T = idx.shape
        x = self.tok_emb(idx)
        mask = self.mask[:T, :T]
        
        # Split into two channels for reversible residuals
        x1, x2 = x.chunk(2, dim=-1)
        
        # Clone to avoid in-place modification of leaf variables
        x1 = x1.clone()
        x2 = x2.clone()
        
        # Forward pass through reversible blocks (all sharing the same tape offset)
        for block in self.blocks:
            x1, x2 = block(x1, x2, tape, 0, mask)
            
        # Recombine channels
        x = torch.cat([x1, x2], dim=-1)
        x = self.ln_f(x)
        # Project only the last token's representation to save logit buffer memory (standard generation behavior)
        logits = self.lm_head(x[:, -1:, :])
        return logits
