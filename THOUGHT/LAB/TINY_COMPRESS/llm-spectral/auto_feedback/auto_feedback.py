"""Phase 3.5 v2: Generation-Quality Auto-Feedback Loop

Closes the adapter training loop using generation-quality metrics instead of
factual accuracy. The target is the uncompressed model's own generation — the
adapter learns to minimize the divergence between compressed and uncompressed
output on the same prompt.

Metrics:
    - Perplexity ratio: PPL(compressed) / PPL(uncompressed) on same text
    - Output perplexity: PPL of uncompressed text under compressed model
    - Attention MSE: per-layer attention output divergence

Architecture:
    For each prompt:
        1. Uncompressed GPT-2 generates target text
        2. Compressed model computes perplexity of target text
        3. Compressed model extracts attention outputs for target text
        4. Uncompressed model extracts attention outputs (target)
        5. Loss = perplexity_loss + lambda * attention_MSE
        6. One gradient step on adapters only

Usage:
    python auto_feedback.py --model gpt2 --k 9 --passes 3
    python auto_feedback.py --model gpt2-medium --k 9 --passes 3
"""

import argparse, json, math, sys, time, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "extensions" / "03_flat_llm"))
from flat_llm_adapter import LowRankAdapter, EigenProjector, compute_cosine_sim


# ============================================================================
# Prompts — text completion style that GPT-2 handles well
# ============================================================================

TRAIN_PROMPTS = [
    "The capital of France is",
    "The chemical formula for water is",
    "The largest planet in the solar system is",
    "The speed of light is approximately",
    "The human body contains",
    "The theory of relativity was developed by",
    "The boiling point of water is",
    "The Great Wall of China is",
    "Photosynthesis is the process by which",
    "The Pacific Ocean is the",
    "DNA stands for",
    "The square root of 144 is",
    "The Mona Lisa was painted by",
    "Mount Everest is the",
    "The Amazon rainforest is located in",
    "World War II ended in",
    "The first man on the moon was",
    "The fastest land animal is",
    "The element with symbol Fe is",
    "The Earth orbits the Sun in approximately",
]

TEST_PROMPTS = [
    "The capital of Japan is",
    "The chemical symbol for gold is",
    "The largest mammal on Earth is the",
    "The Pythagorean theorem states that",
    "The freezing point of water is",
    "Shakespeare wrote",
    "The currency of Japan is the",
    "Einstein is famous for",
    "The largest desert on Earth is the",
    "Oxygen is produced by",
]


# ============================================================================
# AdapterGPT2 — same as before
# ============================================================================

class AdapterAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, k, original_attn=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.k = k
        if original_attn is not None:
            self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
            self.c_attn.weight.data.copy_(original_attn.c_attn.weight.data.T)
            self.c_attn.bias.data.copy_(original_attn.c_attn.bias.data)
            self.c_proj = nn.Linear(hidden_size, hidden_size)
            self.c_proj.weight.data.copy_(original_attn.c_proj.weight.data.T)
            self.c_proj.bias.data.copy_(original_attn.c_proj.bias.data)
        else:
            self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)
            self.c_proj = nn.Linear(hidden_size, hidden_size)
        self.k_projector = EigenProjector(hidden_size, k)
        self.v_projector = EigenProjector(hidden_size, k)
        # Scale bottleneck with residual dimension: ~12% of (hidden - k)
        bn = max(32, (hidden_size - k) // 8)
        self.adapter_k = LowRankAdapter(k=k, hidden=hidden_size, bottleneck=bn, seed=42, alpha_init=0.1)
        self.adapter_v = LowRankAdapter(k=k, hidden=hidden_size, bottleneck=bn, seed=43, alpha_init=0.1)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def clear_cache(self):
        self.k_cache = None
        self.v_cache = None

    def init_projectors(self, k_data, v_data):
        self.k_projector.init_from_pca(k_data)
        self.v_projector.init_from_pca(v_data)
        self.adapter_k.set_residual_subspace(self.k_projector.get_pca_vectors())
        self.adapter_v.set_residual_subspace(self.v_projector.get_pca_vectors())

    def forward(self, hidden_states, use_cache=False, return_attention_output=False):
        b, s, _ = hidden_states.shape
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        kc = self.k_projector.compress(k)
        vc = self.v_projector.compress(v)
        if use_cache:
            if self.k_cache is not None:
                kc = torch.cat([self.k_cache, kc], dim=1)
                vc = torch.cat([self.v_cache, vc], dim=1)
            self.k_cache = kc
            self.v_cache = vc
        kd = self.adapter_k(kc, self.k_projector.decompress(kc))
        vd = self.adapter_v(vc, self.v_projector.decompress(vc))
        q = q.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kd = kd.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        vd = vd.view(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        aw = torch.matmul(q, kd.transpose(-2, -1)) * self.scale
        sk = kd.shape[2]; sq = q.shape[2]
        cm = torch.triu(torch.ones(sq, sk, device=q.device, dtype=torch.bool), diagonal=sk-sq+1)
        aw = aw.masked_fill(cm, float('-inf'))
        aw = F.softmax(aw, dim=-1)
        ao = torch.matmul(aw, vd).transpose(1,2).contiguous().view(b, -1, self.hidden_size)
        out = self.c_proj(ao)
        return (out, out.clone()) if return_attention_output else out


class AdapterBlock(nn.Module):
    def __init__(self, config, k, original_block=None):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        if original_block is not None:
            self.attn = AdapterAttention(config.n_embd, config.n_head, k, original_block.attn)
            self.ln_1.weight.data.copy_(original_block.ln_1.weight.data)
            self.ln_1.bias.data.copy_(original_block.ln_1.bias.data)
            self.ln_2.weight.data.copy_(original_block.ln_2.weight.data)
            self.ln_2.bias.data.copy_(original_block.ln_2.bias.data)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd*4), nn.GELU(),
                nn.Linear(config.n_embd*4, config.n_embd), nn.Dropout(config.resid_pdrop))
            self.mlp[0].weight.data.copy_(original_block.mlp.c_fc.weight.data.T)
            self.mlp[0].bias.data.copy_(original_block.mlp.c_fc.bias.data)
            self.mlp[2].weight.data.copy_(original_block.mlp.c_proj.weight.data.T)
            self.mlp[2].bias.data.copy_(original_block.mlp.c_proj.bias.data)
        else:
            self.attn = AdapterAttention(config.n_embd, config.n_head, k)
            self.mlp = nn.Sequential(
                nn.Linear(config.n_embd, config.n_embd*4), nn.GELU(),
                nn.Linear(config.n_embd*4, config.n_embd), nn.Dropout(config.resid_pdrop))

    def forward(self, x, use_cache=False, return_attn_out=False):
        if return_attn_out:
            ao, raw = self.attn(self.ln_1(x), use_cache=use_cache, return_attention_output=True)
            x = x + ao; x = x + self.mlp(self.ln_2(x)); return x, raw
        else:
            x = x + self.attn(self.ln_1(x), use_cache=use_cache)
            x = x + self.mlp(self.ln_2(x)); return x


class AdapterGPT2(nn.Module):
    def __init__(self, config, k=9):
        super().__init__()
        self.config = config; self.k = k
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList()
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, model_name="gpt2", k=9, device="cpu"):
        from transformers import GPT2LMHeadModel
        print(f"Loading {model_name}...", flush=True)
        original = GPT2LMHeadModel.from_pretrained(model_name).to(device)
        config = original.config
        model = cls(config, k=k).to(device)
        model.wte.weight.data.copy_(original.transformer.wte.weight.data)
        model.wpe.weight.data.copy_(original.transformer.wpe.weight.data)
        model.ln_f.weight.data.copy_(original.transformer.ln_f.weight.data)
        model.ln_f.bias.data.copy_(original.transformer.ln_f.bias.data)
        model.lm_head.weight = model.wte.weight
        for block in original.transformer.h:
            model.h.append(AdapterBlock(config, k, block))
        print(f"  {config.n_layer} layers, hidden={config.n_embd}, k={k}", flush=True)
        return model, original

    def init_projectors(self, tokenizer, sample_texts, device="cpu"):
        print("Initializing PCA projectors...", flush=True)
        self.eval()
        lk = [[] for _ in range(len(self.h))]
        lv = [[] for _ in range(len(self.h))]
        with torch.no_grad():
            for text in sample_texts[:20]:
                inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                ids = inp['input_ids'].to(device)
                h = self.wte(ids) + self.wpe(torch.arange(ids.shape[1], device=device))
                h = self.drop(h)
                for li, b in enumerate(self.h):
                    n = b.ln_1(h)
                    qkv = b.attn.c_attn(n)
                    q, k, v = qkv.chunk(3, dim=-1)
                    lk[li].append(k.reshape(-1, k.shape[-1]))
                    lv[li].append(v.reshape(-1, v.shape[-1]))
                    h = b(h, use_cache=False)
        for li, b in enumerate(self.h):
            b.attn.init_projectors(torch.cat(lk[li], dim=0), torch.cat(lv[li], dim=0))
        print(f"  {len(sample_texts[:20])} texts, {len(self.h)} layers", flush=True)

    def clear_cache(self):
        for b in self.h: b.attn.clear_cache()

    def forward(self, input_ids, use_cache=False, position_ids=None,
                return_attention_outputs=False, labels=None):
        b, s = input_ids.shape; dev = input_ids.device
        if position_ids is None:
            position_ids = torch.arange(s, device=dev).unsqueeze(0)
        h = self.wte(input_ids) + self.wpe(position_ids); h = self.drop(h)
        attn_outs = []
        for block in self.h:
            if return_attention_outputs:
                h, ao = block(h, use_cache=use_cache, return_attn_out=True)
                attn_outs.append(ao)
            else:
                h = block(h, use_cache=use_cache)
        h = self.ln_f(h); logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1))
        if return_attention_outputs:
            return (loss, logits, attn_outs) if loss is not None else (logits, attn_outs)
        return (loss, logits) if loss is not None else logits

    def generate(self, input_ids, max_new_tokens=40, temperature=0.7, top_p=0.9):
        self.clear_cache(); self.eval()
        dev = input_ids.device; cp = input_ids.shape[1]
        with torch.no_grad():
            pp = torch.arange(cp, device=dev).unsqueeze(0)
            logits = self.forward(input_ids, use_cache=True, position_ids=pp)
            if isinstance(logits, tuple): logits = logits[1] if len(logits) > 1 else logits[0]
            for _ in range(max_new_tokens):
                nl = logits[:, -1, :] / temperature
                sl, si = torch.sort(nl, descending=True)
                cp_ = torch.cumsum(F.softmax(sl, dim=-1), dim=-1)
                s2r = cp_ > top_p; s2r[:, 1:] = s2r[:, :-1].clone(); s2r[:, 0] = 0
                nl = nl.masked_fill(s2r.scatter(1, si, s2r), float('-inf'))
                nt = torch.multinomial(F.softmax(nl, dim=-1), 1)
                input_ids = torch.cat([input_ids, nt], dim=1)
                np_ = torch.tensor([[cp]], device=dev)
                logits = self.forward(nt, use_cache=True, position_ids=np_)
                if isinstance(logits, tuple): logits = logits[1] if len(logits) > 1 else logits[0]
                cp += 1
        return input_ids

    def adapter_parameters(self):
        params = []
        for b in self.h:
            params.extend(list(b.attn.adapter_k.parameters()))
            params.extend(list(b.attn.adapter_v.parameters()))
        return params


# ============================================================================
# Attention Extraction (Uncompressed)
# ============================================================================

@torch.no_grad()
def extract_uncompressed_attention(model, tokenizer, text, device):
    inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    ids = inp['input_ids'].to(device)
    seq = ids.shape[1]
    n_heads = model.config.n_head; hd = model.config.n_embd // n_heads
    scale = 1.0 / math.sqrt(hd)
    outs = []
    h = model.transformer.wte(ids) + model.transformer.wpe(torch.arange(seq, device=device))
    h = model.transformer.drop(h)
    for blk in model.transformer.h:
        n = blk.ln_1(h)
        qkv = blk.attn.c_attn(n); q, k, v = qkv.chunk(3, dim=-1)
        b, s, _ = q.shape
        qr = q.view(b, -1, n_heads, hd).transpose(1, 2)
        kr = k.view(b, -1, n_heads, hd).transpose(1, 2)
        vr = v.view(b, -1, n_heads, hd).transpose(1, 2)
        a = torch.matmul(qr, kr.transpose(-2,-1)) * scale
        a = a + torch.triu(torch.ones(s, s, device=device)*float('-inf'), diagonal=1)
        a = F.softmax(a, dim=-1)
        o = torch.matmul(a, vr).transpose(1,2).contiguous().view(b, -1, hd*n_heads)
        ao = blk.attn.c_proj(o)
        outs.append(ao)
        h = h + ao; h = h + blk.mlp(blk.ln_2(h))
    return outs


# ============================================================================
# Perplexity computation
# ============================================================================

@torch.no_grad()
def compute_perplexity(model, tokenizer, text, device):
    inp = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    ids = inp['input_ids'].to(device)
    out = model.forward(ids, labels=ids)
    if hasattr(out, 'loss'):
        loss_val = out.loss.item()
    elif isinstance(out, tuple):
        loss_val = out[0].item()
    else:
        loss_val = out.item()
    return math.exp(loss_val)


# ============================================================================
# Generation-quality metrics
# ============================================================================

def compute_generation_metrics(compressed_model, original_model, tokenizer,
                                prompt, device, max_tokens=40):
    """Generate with both models and compute divergence metrics."""
    # Generate target text with uncompressed model
    inp = tokenizer(prompt, return_tensors='pt')
    ids = inp['input_ids'].to(device)
    with torch.no_grad():
        unc_out = original_model.generate(ids, max_new_tokens=max_tokens,
                                          temperature=0.7, top_p=0.9, do_sample=True,
                                          pad_token_id=tokenizer.eos_token_id)
    target_text = tokenizer.decode(unc_out[0], skip_special_tokens=True)

    # Perplexity of target text under compressed model
    compressed_ppl = compute_perplexity(compressed_model, tokenizer, target_text, device)
    original_ppl = compute_perplexity(original_model, tokenizer, target_text, device)

    # Attention cosine on target text
    with torch.no_grad():
        target_attns = extract_uncompressed_attention(original_model, tokenizer, target_text, device)
    t_inp = tokenizer(target_text, return_tensors='pt', truncation=True, max_length=128)
    t_ids = t_inp['input_ids'].to(device)
    _, adapter_attns = compressed_model.forward(t_ids, return_attention_outputs=True)

    attn_cos = float(np.mean([
        compute_cosine_sim(ta, aa) for ta, aa in zip(target_attns, adapter_attns)
    ]))

    return {
        "target_text": target_text[:200],
        "compressed_ppl": round(compressed_ppl, 2),
        "original_ppl": round(original_ppl, 2),
        "ppl_ratio": round(compressed_ppl / max(original_ppl, 0.01), 2),
        "attention_cosine": round(attn_cos, 4),
    }


# ============================================================================
# Auto-Feedback Loop v2
# ============================================================================

class AutoFeedbackLoop:
    def __init__(self, model, original, tokenizer, device="cpu", lr=1e-4):
        self.model = model
        self.original = original
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.adapter_parameters(), lr=lr)
        self.steps = []

    def evaluate(self, prompts, name="eval", max_tokens=40):
        self.model.eval()
        self.original.eval()
        metrics = []
        for prompt in prompts:
            m = compute_generation_metrics(
                self.model, self.original, self.tokenizer, prompt, self.device, max_tokens)
            metrics.append(m)
        avg_ppl_r = float(np.mean([m["ppl_ratio"] for m in metrics]))
        avg_cos = float(np.mean([m["attention_cosine"] for m in metrics]))
        print(f"  [{name}] PPL ratio: {avg_ppl_r:.2f}  Attn cos: {avg_cos:.4f}  (n={len(prompts)})", flush=True)
        return {"ppl_ratio": round(avg_ppl_r, 2), "attention_cosine": round(avg_cos, 4), "per_prompt": metrics}

    def run_feedback(self, prompts, max_passes=3, max_tokens=40, attn_lambda=0.1,
                      batch_size=4, layer_gamma=0.85, kl_lambda=0.0, cortex=None):
        """Run the auto-feedback loop.

        Improvements over v1:
        - Per-layer attention weighting: early layers get more gradient (gamma^layer)
        - Mini-batch gradient accumulation: accumulate over batch_size prompts before stepping
        - Optional KL loss: compare compressed vs uncompressed logits at each generation step
        - Optional cortex cache: caches uncompressed attention targets in resident memory
        """
        n_layers = len(self.model.h)
        layer_weights = torch.tensor(
            [layer_gamma ** i for i in range(n_layers)], device=self.device)
        layer_weights = layer_weights / layer_weights.sum() * n_layers

        # Initialize cortex cache if available
        cache = None
        if cortex is not None:
            from cortex_recovery import CortexFeedbackCache
            cache = CortexFeedbackCache(cortex)
            print(f"  Cortex cache enabled (agent={cortex.agent_id})", flush=True)

        for pass_idx in range(max_passes):
            print(f"\n--- Pass {pass_idx+1}/{max_passes} "
                  f"(batch={batch_size}, gamma={layer_gamma}, kl={kl_lambda}) ---", flush=True)
            total_loss = 0.0
            n_updates = 0
            n_cache_hits = 0
            self.optimizer.zero_grad()

            for i, prompt in enumerate(prompts):
                # Check cortex cache for pre-computed attention targets
                cached = None
                if cache is not None:
                    cached = cache.get(prompt)

                if cached is not None:
                    target_text, target_attns = cached
                    n_cache_hits += 1
                else:
                    # 1. Generate target with uncompressed model
                    inp = self.tokenizer(prompt, return_tensors='pt')
                    ids = inp['input_ids'].to(self.device)
                    with torch.no_grad():
                        unc_out = self.original.generate(
                            ids, max_new_tokens=max_tokens, temperature=0.7,
                            top_p=0.9, do_sample=True,
                            pad_token_id=self.tokenizer.eos_token_id)
                    target_text = self.tokenizer.decode(unc_out[0], skip_special_tokens=True)

                    # 2. Extract uncompressed attention (target)
                    with torch.no_grad():
                        target_attns = extract_uncompressed_attention(
                            self.original, self.tokenizer, target_text, self.device)

                    # Cache for future passes
                    if cache is not None:
                        cache.put(prompt, target_text, target_attns)

                # 3. Forward compressed model on target text
                t_inp = self.tokenizer(target_text, return_tensors='pt', truncation=True, max_length=128)
                t_ids = t_inp['input_ids'].to(self.device)
                ce_loss, _, adapter_attns = self.model.forward(
                    t_ids, labels=t_ids, return_attention_outputs=True)

                # 4. Per-layer weighted attention MSE
                attn_loss = 0.0
                for li, (ta, aa) in enumerate(zip(target_attns, adapter_attns)):
                    attn_loss += layer_weights[li] * F.mse_loss(aa, ta)
                attn_loss = attn_loss / n_layers

                # 5. Optional KL loss: generate with compressed model, compare logits
                kl_loss_val = 0.0
                if kl_lambda > 0:
                    kl_loss_val = self._compute_generation_kl(prompt, target_text, max_tokens)

                # 6. Combined loss
                loss = (ce_loss + attn_lambda * attn_loss + kl_lambda * kl_loss_val) / batch_size
                loss.backward()

                total_loss += loss.item() * batch_size
                n_updates += 1

                # Gradient accumulation: step every batch_size prompts
                if (i + 1) % batch_size == 0 or i == len(prompts) - 1:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if i < 3 or i % 10 == 0:
                    print(f"  [{i+1:3d}/{len(prompts)}] loss={loss.item()*batch_size:.4f} "
                          f"(ce={ce_loss.item():.3f} attn={attn_loss.item():.4f} "
                          f"kl={kl_loss_val:.4f}) "
                          f"tgt_len={len(target_text)}", flush=True)

            avg_loss = total_loss / max(n_updates, 1)
            cache_str = f"  cache_hits={n_cache_hits}" if cache else ""
            print(f"  Pass {pass_idx+1} avg loss: {avg_loss:.4f}  "
                  f"updates={n_updates//batch_size + (1 if n_updates%batch_size else 0)} steps{cache_str}", flush=True)

    def _compute_generation_kl(self, prompt, target_text, max_tokens):
        """Compute KL divergence between compressed and uncompressed logits during generation."""
        self.model.eval()  # Need eval mode for generation but train mode for gradients? Use train
        self.model.train()
        self.original.eval()

        inp = self.tokenizer(prompt, return_tensors='pt')
        ids = inp['input_ids'].to(self.device)
        kl_total = 0.0
        n_steps = 0

        self.model.clear_cache()
        # Process prompt through both models to get initial state
        with torch.no_grad():
            pp = torch.arange(ids.shape[1], device=self.device).unsqueeze(0)
            orig_logits = self.original(ids).logits

        comp_logits = self.model.forward(ids, use_cache=True, position_ids=pp)
        if isinstance(comp_logits, tuple):
            comp_logits = comp_logits[1] if len(comp_logits) > 1 else comp_logits[0]

        # First token KL
        kl_step = self._kl_divergence(comp_logits[:, -1, :], orig_logits[:, -1, :])
        kl_total += kl_step
        n_steps += 1

        # Generate with compressed model, compute KL at each step
        current_ids = ids.clone()
        current_pos = ids.shape[1]
        for _ in range(min(max_tokens, 20)):  # Cap KL computation at 20 steps
            # Sample next token from compressed model
            with torch.no_grad():
                nl = comp_logits[:, -1, :] / 0.7
                probs = F.softmax(nl, dim=-1)
                next_token = torch.multinomial(probs, 1)

            # Get uncompressed logits for this next token
            full_ids = torch.cat([current_ids, next_token], dim=1)
            with torch.no_grad():
                orig_out = self.original(full_ids)
                orig_step_logits = orig_out.logits[:, -1, :]

            # Get compressed logits (with gradients)
            np_ = torch.tensor([[current_pos]], device=self.device)
            comp_logits = self.model.forward(next_token, use_cache=True, position_ids=np_)
            if isinstance(comp_logits, tuple):
                comp_logits = comp_logits[1] if len(comp_logits) > 1 else comp_logits[0]

            kl_step = self._kl_divergence(comp_logits[:, -1, :], orig_step_logits)
            kl_total += kl_step
            n_steps += 1

            current_ids = full_ids
            current_pos += 1

        self.model.clear_cache()
        return kl_total / max(n_steps, 1)

    @staticmethod
    def _kl_divergence(logits_p, logits_q):
        """KL(P || Q) where P=compressed logits, Q=uncompressed logits."""
        p = F.log_softmax(logits_p, dim=-1)
        q = F.softmax(logits_q, dim=-1)
        return F.kl_div(p, q, reduction='batchmean', log_target=False)


# ============================================================================
# Sample texts for PCA calibration
# ============================================================================

SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process information through neural networks.",
    "The meaning of life is a philosophical question that has puzzled humanity.",
    "Artificial intelligence is transforming technology and society.",
    "Deep learning enables complex pattern recognition in data.",
    "The weather today is sunny with a chance of rain later.",
    "Scientists discovered a new species in the Amazon rainforest.",
    "Music has the power to evoke strong emotional responses.",
    "In mathematics, prime numbers have fascinated researchers for centuries.",
    "The ocean covers more than seventy percent of Earth's surface.",
    "Historical events shape our understanding of the present day.",
    "Technology companies continue to innovate at rapid pace.",
    "Language is the foundation of human communication and thought.",
    "Climate change poses significant challenges for future generations.",
    "The human brain contains approximately eighty-six billion neurons.",
    "Economic theories attempt to explain market behavior patterns.",
    "Art reflects the culture and values of its time period.",
    "Space exploration has led to many technological breakthroughs.",
    "Education is fundamental to personal and societal development.",
    "The internet has revolutionized how we access information.",
]


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3.5 v2: Generation-Quality Auto-Feedback")
    parser.add_argument("--model", default="gpt2", help="Model name (gpt2, gpt2-medium)")
    parser.add_argument("--k", type=int, default=9, help="Compression dimension")
    parser.add_argument("--passes", type=int, default=3, help="Feedback passes")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device")
    parser.add_argument("--max-tokens", type=int, default=40, help="Max generation tokens")
    parser.add_argument("--attn-lambda", type=float, default=0.1, help="Attention loss weight")
    parser.add_argument("--kl-lambda", type=float, default=0.0, help="KL divergence loss weight")
    parser.add_argument("--batch-size", type=int, default=4, help="Gradient accumulation batch size")
    parser.add_argument("--layer-gamma", type=float, default=0.85, help="Per-layer weight decay (0-1, lower = early layers matter more)")
    args = parser.parse_args()

    device = args.device
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    k = args.k

    print("=" * 60, flush=True)
    print(f"Phase 3.5 v2: Generation-Quality Auto-Feedback", flush=True)
    print(f"Model: {args.model}  k={k}  passes={args.passes}  lr={args.lr}", flush=True)
    print("=" * 60, flush=True)

    # Build
    print("\n[1] Building AdapterGPT2...", flush=True)
    t0 = time.time()
    model, original = AdapterGPT2.from_pretrained(args.model, k=k, device=device)
    model.init_projectors(tokenizer, SAMPLE_TEXTS, device)

    # Load pre-trained adapters if available
    ckpt_path = Path(__file__).resolve().parent.parent.parent / "extensions" / "03_flat_llm" / "trained_adapters.pt"
    if ckpt_path.exists():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        if str(k) in ckpt:
            loaded = 0
            skipped = 0
            for li in range(len(model.h)):
                li_str = str(li)
                if li_str in ckpt[str(k)]:
                    try:
                        if 'adapter_k' in ckpt[str(k)][li_str]:
                            model.h[li].attn.adapter_k.load_state_dict(ckpt[str(k)][li_str]['adapter_k'])
                            loaded += 1
                        if 'adapter_v' in ckpt[str(k)][li_str]:
                            model.h[li].attn.adapter_v.load_state_dict(ckpt[str(k)][li_str]['adapter_v'])
                            loaded += 1
                    except RuntimeError:
                        skipped += 1
            if loaded:
                print(f"  Loaded {loaded} adapter state dicts from checkpoint", flush=True)
            if skipped:
                print(f"  Skipped {skipped} adapters (dimension mismatch - random init)", flush=True)
    print(f"  Build complete ({time.time()-t0:.1f}s)", flush=True)

    # Init loop
    loop = AutoFeedbackLoop(model, original, tokenizer, device=device, lr=args.lr)

    # Pre-feedback evaluation
    print("\n[2] Pre-feedback evaluation...", flush=True)
    pre_eval = loop.evaluate(TEST_PROMPTS, "pre-feedback", max_tokens=args.max_tokens)

    # Feedback loop
    print("\n[3] Running feedback loop...", flush=True)
    loop.run_feedback(TRAIN_PROMPTS, max_passes=args.passes,
                      max_tokens=args.max_tokens, attn_lambda=args.attn_lambda,
                      batch_size=args.batch_size, layer_gamma=args.layer_gamma,
                      kl_lambda=args.kl_lambda)

    # Post-feedback evaluation
    print("\n[4] Post-feedback evaluation...", flush=True)
    post_eval = loop.evaluate(TEST_PROMPTS, "post-feedback", max_tokens=args.max_tokens)

    # Save
    output = {
        "phase": "3.5",
        "version": "v2-generation-quality",
        "model": args.model,
        "k": k, "passes": args.passes, "lr": args.lr,
        "pre_feedback": pre_eval,
        "post_feedback": post_eval,
        "delta": {
            "ppl_ratio": round(post_eval["ppl_ratio"] - pre_eval["ppl_ratio"], 2),
            "attention_cosine": round(post_eval["attention_cosine"] - pre_eval["attention_cosine"], 4),
        },
    }
    out_path = RESULTS / "feedback_results_v2.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}", flush=True)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  PPL ratio:    {pre_eval['ppl_ratio']:.2f} -> {post_eval['ppl_ratio']:.2f} "
          f"({post_eval['ppl_ratio'] - pre_eval['ppl_ratio']:+.2f})", flush=True)
    print(f"  Attn cosine:  {pre_eval['attention_cosine']:.4f} -> {post_eval['attention_cosine']:.4f} "
          f"({post_eval['attention_cosine'] - pre_eval['attention_cosine']:+.4f})", flush=True)


if __name__ == "__main__":
    main()
