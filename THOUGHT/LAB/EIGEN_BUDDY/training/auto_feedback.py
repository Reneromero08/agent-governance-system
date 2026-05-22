"""
HOLO 4.5: EigenBuddy Auto-Feedback Training Module
===================================================
Implements the Auto-Feedback Loop and Semiotic Wave Interference Attention.
Teacher: Uncompressed Qwen 0.5B
Student: Holographically Compressed Qwen 0.5B + LowRankPhaseAdapters + Wave Attention.
"""

import sys, math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention

REPO = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = str(REPO / "THOUGHT" / "LAB" / "CAT_CAS" / "16_catalytic_27b_inference" / "gemini_update" / "qwen_0.5b")

class LowRankPhaseAdapter(nn.Module):
    def __init__(self, in_features, out_features, k=64):
        super().__init__()
        # Tiny linear bottleneck adapter
        self.down = nn.Linear(in_features, k, bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(k, out_features, bias=False)
        # Initialize near zero so it starts as identity
        nn.init.normal_(self.down.weight, std=0.01)
        nn.init.zeros_(self.up.weight)
        
    def forward(self, x):
        return self.up(self.act(self.down(x)))

class HolographicAdaptedLinear(nn.Module):
    """
    Replaces a standard Linear layer. Stores the static continuous wave topology (U, S, V)
    and routes the input through the fixed wave + the learnable phase adapter.
    """
    def __init__(self, original_linear, k_compress=128, bottleneck=64):
        super().__init__()
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.bias = original_linear.bias is not None
        
        # Distill to continuous wave
        W = original_linear.weight.data.float()
        max_val = torch.max(torch.abs(W)) + 1e-9
        self.register_buffer("max_val", max_val)
        
        phase_angles = (W / max_val) * math.pi
        grating = torch.complex(torch.cos(phase_angles), torch.sin(phase_angles))
        U, S, Vh = torch.linalg.svd(grating, full_matrices=False)
        
        k_actual = min(k_compress, U.shape[1])
        U_k = U[:, :k_actual]
        S_k = S[:k_actual]
        Vh_k = Vh[:k_actual, :]
        
        grating_recon = (U_k * S_k.unsqueeze(0)) @ Vh_k
        recon_angles = torch.angle(grating_recon)
        W_recon = (recon_angles / math.pi) * max_val
        
        # Store frozen base weight
        self.register_buffer("W_holo", W_recon.to(original_linear.weight.dtype))
        if self.bias:
            self.register_buffer("b_holo", original_linear.bias.data.clone())
            
        # The learnable adapter
        self.adapter = LowRankPhaseAdapter(self.in_features, self.out_features, k=bottleneck).to(
            device=original_linear.weight.device,
            dtype=original_linear.weight.dtype
        )
        
    def forward(self, x):
        # Base wave routing (frozen)
        base_out = F.linear(x, self.W_holo, self.b_holo if self.bias else None)
        # Adaptive phase correction (learnable)
        return base_out + self.adapter(x)

def apply_semiotic_attention_patch(model):
    """
    Monkey-patches Qwen's attention to use Semiotic Wave Interference:
    Attention = |Q|^2 + |K|^2 + 2|Q||K|cos(theta_Q - theta_K)
    Instead of Q dot K^T.
    """
    import transformers.models.qwen2.modeling_qwen2 as qwen2_module
    
    # We store the original forward just in case, though we will override it globally
    # for the Student model.
    # Actually, we will just patch the specific module instances so we don't break Teacher.
    
    def semiotic_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        past_key_values = None,
        **kwargs
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        from transformers.models.qwen2.modeling_qwen2 import repeat_kv
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # ---------------------------------------------------------------------
        # SEMIOTIC WAVE INTERFERENCE ATTENTION
        # ---------------------------------------------------------------------
        q_max = torch.max(torch.abs(query_states), dim=-1, keepdim=True)[0] + 1e-9
        k_max = torch.max(torch.abs(key_states), dim=-1, keepdim=True)[0] + 1e-9
        
        theta_q = (query_states / q_max) * math.pi
        theta_k = (key_states / k_max) * math.pi
        
        t_q = theta_q.unsqueeze(3)
        t_k = theta_k.unsqueeze(2)
        
        attn_weights = torch.sum(2.0 + 2.0 * torch.cos(t_q - t_k), dim=-1)
        attn_weights = attn_weights / self.head_dim
        # ---------------------------------------------------------------------

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
        
    for layer in model.model.layers:
        layer.self_attn.forward = semiotic_forward.__get__(layer.self_attn, Qwen2Attention)

def compress_and_adapt(model, k_compress=128, bottleneck=64):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip very small layers
            if module.weight.shape[0] <= k_compress or module.weight.shape[1] <= k_compress:
                continue
            
            # Replace with Adapted Linear
            # We must set it on the parent
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[-1]
            
            parent = model
            if parent_name != "":
                for p in parent_name.split('.'):
                    parent = getattr(parent, p)
                    
            adapted = HolographicAdaptedLinear(module, k_compress, bottleneck)
            setattr(parent, child_name, adapted)

def auto_feedback_loop():
    print("=" * 78)
    print("HOLO 4.5: AUTO-FEEDBACK TRAINING LOOP")
    print("  Teacher: Qwen 0.5B (Standard Dot-Product)")
    print("  Student: Holographic Phase Adapter + Wave Interference Attention")
    print("=" * 78)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device: {device}")
    
    print("[1] Loading Teacher...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    teacher = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
    teacher.eval()
    teacher.to(device)
    
    print("[2] Loading Student...")
    student = AutoModelForCausalLM.from_pretrained(MODEL_DIR, local_files_only=True)
    student.to(device)
    
    print("[3] Distilling Student and Injecting Phase Adapters...")
    compress_and_adapt(student, k_compress=128, bottleneck=64)
    
    print("[4] Patching Semiotic Wave Interference Attention...")
    apply_semiotic_attention_patch(student)
    
    # Ensure only adapters are trainable
    for p in student.parameters():
        p.requires_grad = False
    for name, p in student.named_parameters():
        if "adapter" in name:
            p.requires_grad = True
            
    # Optimizer
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, student.parameters()), lr=1e-3)
    
    print("[5] Generating Pre-Training Baseline...")
    prompt = "The holographic computing paradigm demonstrates that"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    student.eval()
    with torch.no_grad():
        out = student.generate(**inputs, max_new_tokens=20)
    print(f"Student Zero-Shot Output:\n{tokenizer.decode(out[0], skip_special_tokens=True)}")
    
    print("\n[6] Commencing Auto-Feedback Training (Layer-wise MSE)...")
    student.train()
    
    # Random token dataset for training
    seq_len = 16
    batch_size = 4
    iterations = 50
    
    loss_fn = nn.MSELoss()
    
    for i in range(iterations):
        # Random input tokens
        x = torch.randint(0, tokenizer.vocab_size, (batch_size, seq_len)).to(device)
        
        # Forward Teacher
        with torch.no_grad():
            t_out = teacher(x, output_hidden_states=True)
            t_hiddens = t_out.hidden_states
            
        # Forward Student
        s_out = student(x, output_hidden_states=True)
        s_hiddens = s_out.hidden_states
        
        # Layer-wise Fidelity Loss
        loss = 0
        for l in range(1, len(t_hiddens)):
            loss += loss_fn(s_hiddens[l], t_hiddens[l])
            
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if (i+1) % 10 == 0:
            print(f"  Step {i+1}/{iterations} | Layer-wise MSE Loss: {loss.item():.4f}")
            
    print("\n[7] Generating Post-Training Output...")
    student.eval()
    with torch.no_grad():
        out = student.generate(**inputs, max_new_tokens=20)
    print(f"Student Adapted Output:\n{tokenizer.decode(out[0], skip_special_tokens=True)}")
    print("=" * 78)

if __name__ == "__main__":
    auto_feedback_loop()
