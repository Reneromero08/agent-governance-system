"""Fix 2: repetition penalty + Fix 3: KV cache memory profiling."""
import sys, torch, math, time
import torch.nn.functional as F
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback")
from transformers import GPT2Tokenizer
from auto_feedback import AdapterGPT2, SAMPLE_TEXTS

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

# ---- Fix 2: Add repetition_penalty to generate ----
# Patched into AdapterGPT2 dynamically
_orig_generate = AdapterGPT2.generate

def generate_with_penalty(self, input_ids, max_new_tokens=30, temperature=0.7, top_p=0.9, repetition_penalty=1.2):
    self.clear_cache()
    self.eval()
    dev = input_ids.device
    cp = input_ids.shape[1]
    with torch.no_grad():
        pp = torch.arange(cp, device=dev).unsqueeze(0)
        logits = self.forward(input_ids, use_cache=True, position_ids=pp)
        if isinstance(logits, tuple): logits = logits[1] if len(logits) > 1 else logits[0]
        for _ in range(max_new_tokens):
            nl = logits[:, -1, :] / temperature
            # Repetition penalty: penalize tokens already generated
            if repetition_penalty != 1.0:
                for tid in set(input_ids[0].tolist()[-50:]):
                    if nl[0, tid] < 0:
                        nl[0, tid] *= repetition_penalty
                    else:
                        nl[0, tid] /= repetition_penalty
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

AdapterGPT2.generate = generate_with_penalty

# ---- Test Fix 2: regeneration with penalty ----
print("=" * 60)
print("FIX 2: Repetition penalty on stuck prompts")
print("=" * 60)

model, original = AdapterGPT2.from_pretrained("gpt2", k=50)
model.init_projectors(tokenizer, SAMPLE_TEXTS, "cpu")

stuck_prompts = [
    "The speed of light is approximately",
    "The human body contains",
    "The capital of France is",
]

def self_ppl(m, text):
    inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    ids = inp["input_ids"]
    out = m.forward(ids, labels=ids)
    l = out[0].item() if isinstance(out, tuple) else out.loss.item()
    return math.exp(l)

print("\nNO PENALTY:")
for p in stuck_prompts:
    inp = tokenizer(p, return_tensors="pt")
    out = model.generate(inp["input_ids"], max_new_tokens=30, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.0)
    t = tokenizer.decode(out[0], skip_special_tokens=True)
    s = self_ppl(model, t)
    flag = " STUCK" if s > 100 else ""
    print(f"  PPL={s:8.1f}{flag}  {t[:100]}")

print("\nWITH REPETITION_PENALTY=1.2:")
for p in stuck_prompts:
    inp = tokenizer(p, return_tensors="pt")
    out = model.generate(inp["input_ids"], max_new_tokens=30, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.2)
    t = tokenizer.decode(out[0], skip_special_tokens=True)
    s = self_ppl(model, t)
    flag = " STUCK" if s > 100 else ""
    print(f"  PPL={s:8.1f}{flag}  {t[:100]}")

# ---- Fix 3: KV cache memory profiling ----
print()
print("=" * 60)
print("FIX 3: KV cache memory profiling")
print("=" * 60)

for k in [9, 15, 25, 50]:
    print(f"\n--- k={k} ({768//k}x theoretical) ---")
    m, _ = AdapterGPT2.from_pretrained("gpt2", k=k)
    m.init_projectors(tokenizer, SAMPLE_TEXTS, "cpu")
    m.clear_cache()

    prompt = "The history of artificial intelligence begins with"
    inp = tokenizer(prompt, return_tensors="pt")
    ids = inp["input_ids"]

    t0 = time.time()
    out_ids = m.generate(ids, max_new_tokens=200, temperature=0.7, top_p=0.9,
                         repetition_penalty=1.2)
    dt = time.time() - t0

    # Measure cached KV memory
    total_compressed_bytes = 0
    total_full_equivalent = 0
    for block in m.h:
        attn = block.attn
        if attn.k_cache is not None:
            k_bytes = attn.k_cache.numel() * attn.k_cache.element_size()
            v_bytes = attn.v_cache.numel() * attn.v_cache.element_size()
            total_compressed_bytes += k_bytes + v_bytes

            # Full equivalent: same sequence length but full dims
            seq_len = attn.k_cache.shape[1]
            full_k = seq_len * attn.hidden_size * attn.k_cache.element_size()
            full_v = seq_len * attn.hidden_size * attn.v_cache.element_size()
            total_full_equivalent += full_k + full_v

    ratio = total_full_equivalent / max(total_compressed_bytes, 1)
    compressed_mb = total_compressed_bytes / (1024 * 1024)
    full_mb = total_full_equivalent / (1024 * 1024)
    tokens = out_ids.shape[1] - ids.shape[1]

    # Also measure adapter params
    adapter_params = sum(
        sum(p.numel() for p in block.attn.adapter_k.parameters()) +
        sum(p.numel() for p in block.attn.adapter_v.parameters())
        for block in m.h
    )
    adapter_mb = adapter_params * 4 / (1024 * 1024)  # float32

    text = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(f"  Tokens: {tokens}  Time: {dt:.1f}s")
    print(f"  KV cache: {compressed_mb:.2f} MB (would be {full_mb:.2f} MB uncompressed)")
    print(f"  Actual compression: {ratio:.1f}x (theoretical: {768//k}x)")
    print(f"  Adapter params: {adapter_mb:.2f} MB ({adapter_params:,} params)")
    print(f"  Total adapter + cache: {compressed_mb + adapter_mb:.2f} MB")
    print(f"  Output: {text[:150]}...")
