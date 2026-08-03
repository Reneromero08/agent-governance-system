"""Port PCA KV cache compression to Gemma 4B. Calibration only.

Collects K,V activations from Gemma 4B, computes PCA per layer,
measures reconstruction quality at k_K=8, k_V=36 (116x compression).
"""

import json, math, sys, time
from pathlib import Path
import torch, numpy as np

OUT_DIR = Path(__file__).resolve().parent

def load_gemma(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it", quantization_config=quant, device_map="auto",
        dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()
    return model, tokenizer


def collect_kv_activations(model, tokenizer, texts, device="cuda"):
    """Collect K,V from all attention layers using hooks."""
    n_layers = model.config.text_config.num_hidden_layers
    hidden_dim = model.config.text_config.hidden_size
    num_heads = model.config.text_config.num_attention_heads
    num_kv_heads = model.config.text_config.num_key_value_heads
    head_dim = model.config.text_config.head_dim

    print(f"  Layers: {n_layers}, Hidden: {hidden_dim}, Heads: {num_heads}/{num_kv_heads}, Head dim: {head_dim}")

    # Gemma4 structure: model.model.language_model.layers[i].self_attn
    # k_proj and v_proj are separate linear layers
    layers = model.model.language_model.layers
    layer_data = {i: {'k': [], 'v': []} for i in range(n_layers)}

    valid_layers = []
    for i in range(n_layers):
        # Only layers with separate k_proj (0-23) — layers 24-41 share KV
        submods = [name for name, _ in layers[i].self_attn.named_modules() if name]
        if 'k_proj' not in submods:
            continue
        valid_layers.append(i)

        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj':
                def make_k_hook(li):
                    def hook(m, inp, out): layer_data[li]['k'].append(out.cpu())
                    return hook
                mod.register_forward_hook(make_k_hook(i))
            elif name == 'v_proj':
                def make_v_hook(li):
                    def hook(m, inp, out): layer_data[li]['v'].append(out.cpu())
                    return hook
                mod.register_forward_hook(make_v_hook(i))

    print(f"  Collecting from {len(valid_layers)} layers with separate K,V ({valid_layers[0]}-{valid_layers[-1]})")

    print(f"  Running {len(texts)} calibration texts...")
    t0 = time.time()
    with torch.no_grad():
        for j, text in enumerate(texts):
            messages = [{"role": "user", "content": text}]
            chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=256).to(device)
            _ = model(**inputs)
            if (j+1) % 5 == 0: print(f"    {j+1}/{len(texts)} done")

    print(f"  Collected in {time.time()-t0:.1f}s")

    # Concatenate per valid layer
    result = {}
    for i in valid_layers:
        k_cat = torch.cat([k.reshape(-1, k.shape[-1]) for k in layer_data[i]['k']], dim=0)
        v_cat = torch.cat([v.reshape(-1, v.shape[-1]) for v in layer_data[i]['v']], dim=0)
        result[i] = {'k': k_cat, 'v': v_cat, 'k_dim': k_cat.shape[-1], 'v_dim': v_cat.shape[-1]}
    return result, valid_layers, hidden_dim


def compute_pca_quality(k_data, v_data, k_K, k_V):
    """Compute PCA reconstruction quality (cosine similarity)."""
    # K: [n_samples, kv_dim]
    k_centered = k_data - k_data.mean(dim=0, keepdim=True)
    v_centered = v_data - v_data.mean(dim=0, keepdim=True)

    # SVD
    _, _, Vt_k = torch.linalg.svd(k_centered, full_matrices=False)
    _, _, Vt_v = torch.linalg.svd(v_centered, full_matrices=False)

    # PCA basis
    k_proj = Vt_k[:k_K]  # [k_K, kv_dim]
    v_proj = Vt_v[:k_V]

    # Project and reconstruct
    k_compressed = k_centered @ k_proj.T
    v_compressed = v_centered @ v_proj.T
    k_recon = k_compressed @ k_proj + k_data.mean(dim=0, keepdim=True)
    v_recon = v_compressed @ v_proj + v_data.mean(dim=0, keepdim=True)

    # Cosine similarity
    k_cos = torch.nn.functional.cosine_similarity(
        k_data.reshape(-1), k_recon.reshape(-1), dim=0)
    v_cos = torch.nn.functional.cosine_similarity(
        v_data.reshape(-1), v_recon.reshape(-1), dim=0)

    # Compression ratio
    kv_dim = k_data.shape[-1]
    original = 2 * kv_dim
    compressed = k_K + k_V
    ratio = original / compressed

    return float(k_cos), float(v_cos), ratio, k_proj, v_proj


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print("GEMMA 4B PCA KV CACHE CALIBRATION")
    print("=" * 60)

    # Load model
    print("\n[1/3] Loading Gemma 4B E4B-it...")
    model, tokenizer = load_gemma(device)
    print(f"  Device: {next(model.parameters()).device}")

    # Calibration texts
    texts = [
        "The meaning of life is a philosophical question that has puzzled humanity for centuries.",
        "Artificial intelligence is transforming the way we interact with technology every day.",
        "Deep learning enables complex pattern recognition in vast amounts of data.",
        "The human brain contains approximately eighty-six billion neurons.",
        "Climate change poses significant challenges for future generations.",
        "In mathematics, prime numbers have fascinated researchers for millennia.",
        "The ocean covers more than seventy percent of Earth's surface.",
        "Music has the power to evoke strong emotional responses across cultures.",
        "Space exploration has led to many technological breakthroughs.",
        "Language is the foundation of human communication and collective knowledge.",
        "Quantum mechanics describes behavior at the smallest scales of existence.",
        "Evolution by natural selection explains the diversity of life.",
        "The internet has revolutionized how information is shared globally.",
        "Mathematics provides language for describing universal patterns.",
        "Democracy depends on informed citizenry and free exchange of ideas.",
        "Technology advances through accumulation of knowledge across generations.",
        "The scientific method relies on observation and experimentation.",
        "Art reflects cultural values and emotional landscapes of society.",
        "Economic systems explain resource allocation in complex societies.",
        "The history of science is ideas evolving through observation.",
    ]

    # Collect
    print(f"\n[2/3] Collecting K,V activations from {len(texts)} texts...")
    acts, valid_layers, hidden_dim = collect_kv_activations(model, tokenizer, texts, device)

    # PCA quality at various k
    n_layers_active = len(valid_layers)
    print(f"\n[3/3] PCA reconstruction quality ({n_layers_active} layers)")
    print(f"  {'k_K':>5s} {'k_V':>5s} {'Compress':>10s} {'K_cos':>8s} {'V_cos':>8s}")
    print(f"  {'-'*45}")

    k_configs = [(8, 36), (4, 18), (2, 9), (16, 72), (8, 8)]
    results = {}

    for k_K, k_V in k_configs:
        k_cos_list, v_cos_list = [], []
        for l in valid_layers:
            kc, vc, ratio, _, _ = compute_pca_quality(
                acts[l]['k'].float(), acts[l]['v'].float(), k_K, k_V)
            k_cos_list.append(kc)
            v_cos_list.append(vc)

        avg_k = np.mean(k_cos_list)
        avg_v = np.mean(v_cos_list)
        avg_ratio = hidden_dim * 2 / (k_K + k_V)
        print(f"  {k_K:5d} {k_V:5d} {avg_ratio:10.1f}x {avg_k:8.4f} {avg_v:8.4f}")
        results[f"k{k_K}_v{k_V}"] = {"k_cos": float(avg_k), "v_cos": float(avg_v),
                                       "compression": float(avg_ratio)}

    # Compare to GPT-2 baseline
    print(f"\n  GPT-2 baseline (k=8,v=36, 35x): K_cos~0.95, V_cos~0.92")
    print(f"  Gemma at k=8,v=36: K_cos={results.get('k8_v36',{}).get('k_cos','?'):.4f}, V_cos={results.get('k8_v36',{}).get('v_cos','?'):.4f}")

    json.dump({"n_layers_active": n_layers_active, "hidden_dim": hidden_dim, "results": results},
              open(OUT_DIR / "gemma_pca_calibration.json", 'w'), indent=2)
    print(f"\nSaved gemma_pca_calibration.json")


if __name__ == '__main__':
    main()
