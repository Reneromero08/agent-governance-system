"""Train Gemma 4B KV cache adapters. Ported from GPT-2 Phase 3.5 blueprint."""
import json, math, sys, time
from pathlib import Path
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np

OUT_DIR = Path(__file__).resolve().parent

# Import adapter classes from GPT-2
sys.path.insert(0, str(OUT_DIR.parent.parent.parent / "extensions" / "03_flat_llm"))
from flat_llm_adapter import LowRankAdapter, EigenProjector, compute_attention_output, compute_cosine_sim, compute_attention_output_train


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


def get_active_layers(model):
    """Return list of layer indices that have k_proj/v_proj (0-23)."""
    layers = model.model.language_model.layers
    active = []
    for i in range(len(layers)):
        submods = [n for n, _ in layers[i].self_attn.named_modules() if n]
        if 'k_proj' in submods:
            active.append(i)
    return active


def collect_kv_activations(model, tokenizer, texts, device="cuda"):
    """Collect K,V from active layers using hooks."""
    layers = model.model.language_model.layers
    active = get_active_layers(model)
    layer_data = {i: {'k': [], 'v': []} for i in active}

    def make_hook(li, buf, key):
        def hook(m, inp, out): buf[li][key].append(out.cpu())
        return hook

    hooks = []
    for i in active:
        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj':
                hooks.append(mod.register_forward_hook(make_hook(i, layer_data, 'k')))
            elif name == 'v_proj':
                hooks.append(mod.register_forward_hook(make_hook(i, layer_data, 'v')))

    for text in texts:
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad(): _ = model(**inputs)

    for h in hooks: h.remove()

    # Concatenate and compute PCA
    acts = {}
    for i in active:
        k_cat = torch.cat([k.reshape(-1, k.shape[-1]) for k in layer_data[i]['k']], dim=0)
        v_cat = torch.cat([v.reshape(-1, v.shape[-1]) for v in layer_data[i]['v']], dim=0)
        acts[i] = {'k': k_cat.float(), 'v': v_cat.float(), 'k_dim': k_cat.shape[-1], 'v_dim': v_cat.shape[-1]}
    return acts, active


def train_one_layer(model, tokenizer, layer_idx, acts, k_K, k_V, bn, epochs, lr, device,
                    train_texts, test_texts):
    """Train K and V adapters for a single Gemma layer. Returns (pca_cos, ada_cos) OOS."""
    layers = model.model.language_model.layers
    k_dim = acts[layer_idx]['k_dim']; v_dim = acts[layer_idx]['v_dim']
    num_heads = model.config.text_config.num_attention_heads
    num_kv_heads = model.config.text_config.num_key_value_heads
    head_dim = model.config.text_config.head_dim
    scale = 1.0 / math.sqrt(head_dim)

    # Init PCA projectors
    pk = EigenProjector(k_dim, k_K).to(device); pk.init_from_pca(acts[layer_idx]['k'])
    pv = EigenProjector(v_dim, k_V).to(device); pv.init_from_pca(acts[layer_idx]['v'])

    # Create adapters
    ak = LowRankAdapter(k=k_K, hidden=k_dim, bottleneck=bn, seed=42).to(device)
    av = LowRankAdapter(k=k_V, hidden=v_dim, bottleneck=bn, seed=43).to(device)
    ak.set_residual_subspace(pk.get_pca_vectors())
    av.set_residual_subspace(pv.get_pca_vectors())

    # Collect train Q,K,V
    train_qkv = []
    def qkv_hook(m, inp, out):
        sz = out.shape[-1] // 3
        train_qkv.append((out[..., :sz].detach(), out[..., sz:2*sz].detach(), out[..., 2*sz:].detach()))
    # For Gemma, Q,K,V are from separate projections — use the main attention forward
    # Actually hook the self_attn output and reconstruct
    # Simpler: hook the q_proj and k_proj outputs separately
    # We already have K,V from the calibration. Need Q. Hook q_proj too.
    q_data = []
    def q_hook(m, inp, out): q_data.append(out.detach())
    hq = layers[layer_idx].self_attn.q_proj.register_forward_hook(q_hook)
    hk_train = []
    def k_hook(m, inp, out): pass  # we use collect separately
    # Actually simplest: run model forward, capture Q,K,V from attention input
    # Use the k_proj/v_proj hooks we already have
    k_train = []; v_train = []
    def kt_hook(m,i,o): k_train.append(o.detach())
    def vt_hook(m,i,o): v_train.append(o.detach())
    for name, mod in layers[layer_idx].self_attn.named_modules():
        if name == 'k_proj': hk_train.append(mod.register_forward_hook(kt_hook))
        elif name == 'v_proj': hk_train.append(mod.register_forward_hook(vt_hook))
        elif name == 'q_proj': hk_train.append(mod.register_forward_hook(q_hook))

    for text in train_texts:
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad(): _ = model(**inputs)
    for h in hk_train: h.remove(); hq.remove()

    if not q_data or not k_train or not v_train: return None, None

    # Train
    opt = torch.optim.Adam(list(ak.parameters()) + list(av.parameters()), lr=lr)
    # Q has dim [seq, q_dim], K has [seq, k_dim], V has [seq, v_dim]
    # Gemma uses GQA: Q has num_heads*head_dim, K/V have num_kv_heads*head_dim
    # Restructure for attention computation: expand K,V heads to match Q
    for _ in range(epochs):
        for qi, ki, vi in zip(q_data, k_train, v_train):
            opt.zero_grad()
            # Expand K,V for GQA: repeat each KV head for the corresponding Q heads
            n_groups = num_heads // num_kv_heads
            q_r = qi.reshape(-1, num_heads, head_dim).transpose(0, 1)  # [H, seq, d]
            k_r = ki.reshape(-1, num_kv_heads, head_dim).transpose(0, 1).repeat_interleave(n_groups, dim=0)  # [H, seq, d]
            v_r = vi.reshape(-1, num_kv_heads, head_dim).transpose(0, 1).repeat_interleave(n_groups, dim=0)

            # Compress + decompress (projectors operate in float32)
            kc = pk.compress(ki.float()); kp = pk.decompress(kc)
            vc = pv.compress(vi.float()); vp = pv.decompress(vc)
            ka = ak(kc, kp); va = av(vc, vp)

            # Attention with original vs adapted
            ka_r = ka.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            va_r = va.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            # Attention with original vs adapted (match half precision)
            dtype = q_r.dtype
            k_r_h = k_r.to(dtype); v_r_h = v_r.to(dtype)
            orig_attn = torch.matmul(q_r, k_r_h.transpose(-2,-1)) * scale
            orig_attn = F.softmax(orig_attn, dim=-1) @ v_r_h
            ka_r_h = ka_r.to(dtype); va_r_h = va_r.to(dtype)
            ada_attn = torch.matmul(q_r, ka_r_h.transpose(-2,-1)) * scale
            ada_attn = F.softmax(ada_attn, dim=-1) @ va_r_h

            loss = F.mse_loss(ada_attn.float(), orig_attn.float())
            loss.backward(); opt.step()

    # Evaluate on test
    te_q, te_k, te_v = [], [], []
    def te_q_hook(m,i,o): te_q.append(o.detach())
    def te_k_hook(m,i,o): te_k.append(o.detach())
    def te_v_hook(m,i,o): te_v.append(o.detach())
    te_hooks = []
    for name, mod in layers[layer_idx].self_attn.named_modules():
        if name == 'q_proj': te_hooks.append(mod.register_forward_hook(te_q_hook))
        elif name == 'k_proj': te_hooks.append(mod.register_forward_hook(te_k_hook))
        elif name == 'v_proj': te_hooks.append(mod.register_forward_hook(te_v_hook))
    for text in test_texts:
        messages = [{"role": "user", "content": text}]
        chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad(): _ = model(**inputs)
    for h in te_hooks: h.remove()

    pca_cos_list, ada_cos_list = [], []
    for qi, ki, vi in zip(te_q, te_k, te_v):
        with torch.no_grad():
            n_groups = num_heads // num_kv_heads
            q_r = qi.reshape(-1, num_heads, head_dim).transpose(0,1)
            k_r = ki.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            v_r = vi.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            kc = pk.compress(ki.float()); kp = pk.decompress(kc)
            vc = pv.compress(vi.float()); vp = pv.decompress(vc)
            ka = ak(kc, kp); va = av(vc, vp)
            ka_r = ka.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            va_r = va.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            dtype = q_r.dtype
            k_r_h = k_r.to(dtype); v_r_h = v_r.to(dtype)
            orig_attn = F.softmax(torch.matmul(q_r,k_r_h.transpose(-2,-1))*scale, dim=-1) @ v_r_h
            kp_r = kp.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            vp_r = vp.reshape(-1, num_kv_heads, head_dim).transpose(0,1).repeat_interleave(n_groups, dim=0)
            kp_r_h = kp_r.to(dtype); vp_r_h = vp_r.to(dtype)
            pca_attn = F.softmax(torch.matmul(q_r,kp_r_h.transpose(-2,-1))*scale, dim=-1) @ vp_r_h
            ka_r_h = ka_r.to(dtype); va_r_h = va_r.to(dtype)
            ada_attn = F.softmax(torch.matmul(q_r,ka_r_h.transpose(-2,-1))*scale, dim=-1) @ va_r_h
            pca_cos_list.append(compute_cosine_sim(orig_attn, pca_attn))
            ada_cos_list.append(compute_cosine_sim(orig_attn, ada_attn))
    return float(np.mean(pca_cos_list)), float(np.mean(ada_cos_list))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60); print("GEMMA 4B ADAPTER TRAINING"); print("="*60)

    print("\nLoading Gemma 4B...")
    model, tokenizer = load_gemma(device)
    active = get_active_layers(model)
    print(f"Active layers: {len(active)} ({active[0]}-{active[-1]})")

    texts = [
        "The meaning of life is a philosophical question.",
        "Artificial intelligence is transforming technology every day.",
        "Deep learning enables complex pattern recognition.",
        "The human brain contains billions of neurons.",
        "Climate change poses significant challenges.",
        "Prime numbers have fascinated mathematicians for millennia.",
        "The ocean covers most of Earth's surface.",
        "Music evokes strong emotional responses across cultures.",
    ]
    test_texts = [
        "Space exploration has led to technological breakthroughs.",
        "Language is the foundation of human communication.",
    ]

    print(f"\nCollecting activations from {len(texts)} texts...")
    acts, active = collect_kv_activations(model, tokenizer, texts, device)
    print(f"  Collected from {len(active)} layers")

    # Train a few layers as proof of concept
    k_K, k_V, bn = 8, 36, 64
    n_train = len(active)  # all 24 layers

    print(f"\nTraining adapters at k_K={k_K}, k_V={k_V}, bn={bn} on {n_train} layers...")
    t0 = time.time()
    results = []
    for i in range(n_train):
        li = active[i]
        try:
            pc, ac = train_one_layer(model, tokenizer, li, acts, k_K, k_V, bn, 10, 1e-3, device, texts, test_texts)
        except Exception as e:
            print(f"  L{li:2d}: ERROR: {e}")
            continue
        if pc is not None and not (math.isnan(pc) or math.isnan(ac)):
            results.append((li, pc, ac))
            print(f"  L{li:2d}: PCA={pc:.4f}  Ada={ac:.4f}  delta={ac-pc:+.4f}")
        else:
            print(f"  L{li:2d}: NaN — skipping")

    dt = time.time() - t0
    avg_p = np.mean([r[1] for r in results])
    avg_a = np.mean([r[2] for r in results])
    print(f"\n  Avg ({n_train} layers): PCA={avg_p:.4f}  Ada={avg_a:.4f}  delta={avg_a-avg_p:+.4f}  dt={dt:.1f}s")

    json.dump({"k_K": k_K, "k_V": k_V, "bn": bn, "layers_trained": n_train,
               "avg_pca": float(avg_p), "avg_ada": float(avg_a), "results": [[int(r[0]), float(r[1]), float(r[2])] for r in results]},
              open(OUT_DIR / "adapter_results.json", 'w'), indent=2)
    print(f"\nSaved adapter_results.json")


if __name__ == '__main__':
    main()
