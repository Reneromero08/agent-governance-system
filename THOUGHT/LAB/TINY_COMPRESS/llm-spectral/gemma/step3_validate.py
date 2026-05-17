"""Step 3: PCA-compressed generation on Gemma 4B."""
import json, math, sys, time
from pathlib import Path
import torch, numpy as np

OUT_DIR = Path(__file__).resolve().parent
# gemma/llm-spectral/TINY_COMPRESS/LAB/FORMULA/v4/phase4a = 5 parents up
sys.path.insert(0, str(OUT_DIR.parent.parent.parent / "FORMULA" / "v4" / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

def load_model(device="cuda"):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", quantization_config=quant,
              device_map="auto", dtype=torch.float16, trust_remote_code=True)
    t = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
    if t.pad_token_id is None: t.pad_token_id = t.eos_token_id
    m.eval(); return m, t

def calibrate_pca(model, tokenizer, k_K=8, k_V=36, device="cuda"):
    """Collect K,V from one text, compute PCA, return per-layer projectors."""
    layers = model.model.language_model.layers
    n_layers = len(layers)
    text = "The meaning of life is a philosophical question that has puzzled humanity."
    messages = [{"role":"user","content":text}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=256).to(device)

    # Collect K,V from all layers
    layer_k, layer_v = {i:[] for i in range(n_layers)}, {i:[] for i in range(n_layers)}
    def mk(li):
        def h(m,i,o): layer_k[li].append(o.detach())
        return h
    def mv(li):
        def h(m,i,o): layer_v[li].append(o.detach())
        return h

    hooks = []
    for i in range(n_layers):
        submods = [n for n,_ in layers[i].self_attn.named_modules() if n]
        if 'k_proj' not in submods: continue
        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj': hooks.append(mod.register_forward_hook(mk(i)))
            elif name == 'v_proj': hooks.append(mod.register_forward_hook(mv(i)))

    with torch.no_grad(): _ = model(**inputs)
    for h in hooks: h.remove()

    # Compute PCA per layer
    proj_k, proj_v, mean_k, mean_v = {}, {}, {}, {}
    for i in range(n_layers):
        if not layer_k[i] or not layer_v[i]: continue
        kd = layer_k[i][0].reshape(-1, layer_k[i][0].shape[-1]).float()
        vd = layer_v[i][0].reshape(-1, layer_v[i][0].shape[-1]).float()
        km, vm = kd.mean(dim=0,keepdim=True), vd.mean(dim=0,keepdim=True)
        _,_,Vk = torch.linalg.svd(kd-km, full_matrices=False); _,_,Vv = torch.linalg.svd(vd-vm, full_matrices=False)
        k_use = min(k_K, Vk.shape[0]); v_use = min(k_V, Vv.shape[0])
        proj_k[i] = Vk[:k_use]; proj_v[i] = Vv[:v_use]; mean_k[i] = km; mean_v[i] = vm
    return proj_k, proj_v, mean_k, mean_v

def make_compression_hooks(model, proj_k, proj_v, mean_k, mean_v):
    """Install hooks that replace K,V with PCA compressed->decompressed versions."""
    layers = model.model.language_model.layers
    hooks = []
    def mk(li):
        pk, mk = proj_k[li], mean_k[li]
        def h(mod, inp, out):
            c = (out - mk) @ pk.T; out.copy_(c @ pk + mk)
        return h
    def mv(li):
        pv, mv = proj_v[li], mean_v[li]
        def h(mod, inp, out):
            c = (out - mv) @ pv.T; out.copy_(c @ pv + mv)
        return h
    for i in proj_k:
        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj': hooks.append(mod.register_forward_hook(mk(i)))
            elif name == 'v_proj': hooks.append(mod.register_forward_hook(mv(i)))
    return hooks

def generate(model, tokenizer, prompt, T=0.7, max_t=150, device="cuda"):
    messages = [{"role":"user","content":prompt}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
    ilen = inp.input_ids.shape[1]
    with torch.no_grad():
        out = model.generate(**inp, max_new_tokens=max_t, do_sample=True, temperature=T, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
    ids = out.sequences[0] if hasattr(out,'sequences') else out[0]
    return tokenizer.decode(ids[ilen:], skip_special_tokens=True)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60); print("STEP 3: PCA-Compressed Gemma 4B Validation"); print("="*60)

    model, tokenizer = load_model(device)
    print(f"Model loaded on {next(model.parameters()).device}")

    # Calibrate PCA
    print("Calibrating PCA from single text...")
    t0 = time.time()
    proj_k, proj_v, mean_k, mean_v = calibrate_pca(model, tokenizer, 8, 36, device)
    n_active = len(proj_k)
    print(f"  {n_active} layers with PCA projectors ({time.time()-t0:.1f}s)")

    # Baseline (no compression)
    prompts = [p for p in TEST_PROMPTS if p["id"] in ["F1","F2","F3","F5","F6","R1","R2","R3","R5","R6"]]
    print(f"\n--- BASELINE (uncompressed) ---")
    base_correct = 0; base_total = 0
    for entry in prompts:
        t0 = time.time()
        text = generate(model, tokenizer, entry["prompt"])
        vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
        v,s = verify_answer(text, entry) if vt!="none" and gt else (None,None)
        if v is not None: base_total += 1
        if v: base_correct += 1
        print(f"  {entry['id']}: verified={v} dt={time.time()-t0:.1f}s")

    # Compressed
    print(f"\n--- COMPRESSED (k_K=8, k_V=36, 116x) ---")
    hooks = make_compression_hooks(model, proj_k, proj_v, mean_k, mean_v)
    print(f"  {len(hooks)} compression hooks installed")

    comp_correct = 0; comp_total = 0
    for entry in prompts:
        t0 = time.time()
        text = generate(model, tokenizer, entry["prompt"])
        vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
        v,s = verify_answer(text, entry) if vt!="none" and gt else (None,None)
        if v is not None: comp_total += 1
        if v: comp_correct += 1
        print(f"  {entry['id']}: verified={v} dt={time.time()-t0:.1f}s")

    for h in hooks: h.remove()

    ba = base_correct/max(base_total,1); ca = comp_correct/max(comp_total,1)
    print(f"\nBaseline: {base_correct}/{base_total} = {ba:.3f}")
    print(f"Compressed: {comp_correct}/{comp_total} = {ca:.3f}")
    print(f"Delta: {ca-ba:+.3f}")
    json.dump({"baseline_acc":ba,"compressed_acc":ca,"delta":ca-ba}, open(OUT_DIR/"step3_results.json","w"), indent=2)

if __name__=='__main__': main()
