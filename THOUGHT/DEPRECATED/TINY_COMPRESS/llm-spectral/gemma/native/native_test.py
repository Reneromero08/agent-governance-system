"""Native PCA integration test: return new tensors from hooks instead of in-place copy."""
import json, math, sys, time
from pathlib import Path
import torch, numpy as np

OUT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(OUT_DIR.parent.parent.parent / "FORMULA" / "v4" / "phase4a"))
from phase4a_prompts import TEST_PROMPTS, verify_answer

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def calibrate_pca_from_model(model, tokenizer, k_K=8, k_V=36, device="cuda"):
    """Quick PCA calibration from a single text."""
    layers = model.model.language_model.layers
    text = "The meaning of life is a philosophical question that has puzzled humanity."
    messages = [{"role":"user","content":text}]
    chat = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt", truncation=True, max_length=256).to(device)

    layer_k, layer_v = {}, {}
    def mk(li):
        def h(m,i,o): layer_k.setdefault(li,[]).append(o.clone())
        return h
    def mv(li):
        def h(m,i,o): layer_v.setdefault(li,[]).append(o.clone())
        return h
    hooks = []
    for i in range(len(layers)):
        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj' and 'k_proj' in [n for n,_ in layers[i].self_attn.named_modules() if n]:
                hooks.append(mod.register_forward_hook(mk(i)))
            elif name == 'v_proj':
                hooks.append(mod.register_forward_hook(mv(i)))
    with torch.no_grad(): _ = model(**inputs)
    for h in hooks: h.remove()

    proj_k, proj_v, mean_k, mean_v = {}, {}, {}, {}
    for i in layer_k:
        kd = torch.cat(layer_k[i], dim=0).reshape(-1, layer_k[i][0].shape[-1]).float()
        vd = torch.cat(layer_v[i], dim=0).reshape(-1, layer_v[i][0].shape[-1]).float()
        km, vm = kd.mean(dim=0,keepdim=True), vd.mean(dim=0,keepdim=True)
        _,_,Vk = torch.linalg.svd(kd-km, full_matrices=False); _,_,Vv = torch.linalg.svd(vd-vm, full_matrices=False)
        proj_k[i] = Vk[:min(k_K, Vk.shape[0])]; proj_v[i] = Vv[:min(k_V, Vv.shape[0])]
        mean_k[i] = km; mean_v[i] = vm
    return proj_k, proj_v, mean_k, mean_v


def install_native_compression(model, proj_k, proj_v, mean_k, mean_v, device="cuda"):
    """Replace k_proj/v_proj forward hooks with RETURNABLE compression hooks.
    Instead of out.copy_(), we return a new tensor. This prevents in-place side effects.
    """
    layers = model.model.language_model.layers
    hooks = []

    for i in proj_k:
        pk, mk = proj_k[i].to(device), mean_k[i].to(device)
        pv, mv = proj_v[i].to(device), mean_v[i].to(device)

        def make_k_hook(li, pk_local, mk_local):
            def hook(module, input, output):
                c = (output.float() - mk_local) @ pk_local.T
                result = (c @ pk_local + mk_local).to(output.dtype)
                return result
            return hook

        def make_v_hook(li, pv_local, mv_local):
            def hook(module, input, output):
                c = (output.float() - mv_local) @ pv_local.T
                result = (c @ pv_local + mv_local).to(output.dtype)
                return result
            return hook

        for name, mod in layers[i].self_attn.named_modules():
            if name == 'k_proj':
                hooks.append(mod.register_forward_hook(make_k_hook(i, pk.clone(), mk.clone())))
            elif name == 'v_proj':
                hooks.append(mod.register_forward_hook(make_v_hook(i, pv.clone(), mv.clone())))

    return hooks


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*60)
    print("NATIVE PCA INTEGRATION — Return-Based Hooks")
    print("="*60)

    # Load
    print("Loading Gemma 4B...")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B-it", quantization_config=quant, device_map="auto",
        dtype=torch.float16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it", trust_remote_code=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token_id = tokenizer.eos_token_id
    model.eval()

    # Test multiple compression ratios
    configs = [(8, 36, 116), (16, 72, 58), (32, 144, 25), (64, 288, 12)]
    prompts = [p for p in TEST_PROMPTS if p["id"] in ["F1","F2","F3","F5","R1","R2","R3","R5"]]

    # Baseline
    print("\n--- BASELINE ---")
    correct = 0; total = 0
    for entry in prompts:
        msg = [{"role":"user","content":entry["prompt"]}]
        chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
        ilen = inp.input_ids.shape[1]
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=150, do_sample=True,
                                 temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
        ids = out.sequences[0] if hasattr(out,'sequences') else out[0]
        text = tokenizer.decode(ids[ilen:], skip_special_tokens=True)
        vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
        v,s = verify_answer(text, entry) if vt!="none" and gt else (None,None)
        if v is not None: total += 1
        if v: correct += 1
        print(f"  {entry['id']}: v={v}  {text[:60]}")
    print(f"  BASELINE: {correct}/{total} = {correct/max(total,1):.3f}")

    for k_K, k_V, ratio in configs:
        print(f"\n--- k_K={k_K}, k_V={k_V} ({ratio}x) ---")
        proj_k, proj_v, mean_k, mean_v = calibrate_pca_from_model(model, tokenizer, k_K, k_V, device)
        hooks = install_native_compression(model, proj_k, proj_v, mean_k, mean_v, device)
        correct = 0; total = 0
        for entry in prompts:
            msg = [{"role":"user","content":entry["prompt"]}]
            chat = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            inp = tokenizer(chat, return_tensors="pt", truncation=True, max_length=4096).to(device)
            ilen = inp.input_ids.shape[1]
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=100, do_sample=True,
                                     temperature=0.7, top_p=0.9, pad_token_id=tokenizer.eos_token_id)
            ids = out.sequences[0] if hasattr(out,'sequences') else out[0]
            text = tokenizer.decode(ids[ilen:], skip_special_tokens=True)
            vt = entry.get("verification_type","none"); gt = entry.get("ground_truth")
            v,s = verify_answer(text, entry) if vt!="none" and gt else (None,None)
            if v is not None: total += 1
            if v: correct += 1
            print(f"  {entry['id']}: v={v}  {text[:60]}")
        for h in hooks: h.remove()
        acc = correct/max(total,1)
        print(f"  COMPRESSED: {correct}/{total} = {acc:.3f}")

    json.dump({"baseline_acc": correct/max(total,1)}, open(OUT_DIR/"native_test.json","w"), indent=2)


if __name__ == '__main__':
    main()
