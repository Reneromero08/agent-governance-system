"""eval_corrected.py — Full 64-layer evaluation of the .holo engine with
analytic corrections vs the exact reference. Reports hidden-state cosine per
layer, final logit agreement, and a short generation sample.
"""
import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen35_holo_engine import Qwen35HoloEngine, load_holo, load_original, DEFAULT_MODEL

PROMPT = "The theory of catalytic computation is"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holo", default="output/qwen_27b_k256.holo")
    ap.add_argument("--corrections", default="output/qwen_corrections_r128.pt")
    ap.add_argument("--model-dir", default=str(DEFAULT_MODEL))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-layers", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=32)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    holo = load_holo(args.holo)
    original = load_original(args.model_dir)
    eng = Qwen35HoloEngine(holo, original, device=args.device, verbose=False, num_layers=args.num_layers)
    eng_exact = Qwen35HoloEngine(None, original, exact=True, device=args.device, verbose=False, num_layers=args.num_layers)

    corr = torch.load(args.corrections, map_location="cpu", weights_only=False)
    eng.corrections = {int(k): v for k, v in corr["corrections"].items()}
    n_mix = sum(1 for c in eng.corrections.values() if "mix" in c)
    n_mlp = sum(1 for c in eng.corrections.values() if "mlp" in c)
    mb = sum(
        (a.numel() + b.numel()) * a.element_size()
        for c in eng.corrections.values()
        for a, b in (c.get("mix"), c.get("mlp")) if a is not None
    ) / 1024**2
    print(f"corrections: {n_mix} mixer + {n_mlp} mlp, {mb:.0f} MB")

    ids = tokenizer(PROMPT, return_tensors="pt")["input_ids"][0]

    # exact reference
    t0 = time.time()
    with torch.no_grad():
        eng_exact.prefill(ids, capture_hidden=True)
    exact_rms = list(eng_exact.last_hidden_rms)
    exact_hidden = [h for h in eng_exact.last_hidden_states]
    exact_logits = None
    exact_time = time.time() - t0
    print(f"exact done in {exact_time:.0f}s")

    # holo + corrections
    t0 = time.time()
    with torch.no_grad():
        eng.prefill(ids, capture_hidden=True)
    holo_rms = list(eng.last_hidden_rms)
    holo_hidden = [h for h in eng.last_hidden_states]
    holo_time = time.time() - t0
    print(f"holo done in {holo_time:.0f}s")

    # per-layer hidden cosine
    coses = []
    for l in range(min(len(exact_hidden), len(holo_hidden))):
        a = exact_hidden[l].float().view(1, -1)
        b = holo_hidden[l].float().view(1, -1)
        coses.append(torch.nn.functional.cosine_similarity(a, b).item())
    print(f"\nhidden cosine: first={coses[0]:.4f} mid={coses[len(coses)//2]:.4f} last={coses[-1]:.4f} mean={sum(coses)/len(coses):.4f}")
    print("decay per quarter:", [f"{sum(coses[q*len(coses)//4:(q+1)*len(coses)//4])/(len(coses)//4):.3f}" for q in range(4)])

    # final logits comparison
    with torch.no_grad():
        exact_logits = eng_exact._lm_head(exact_hidden[-1].to(eng_exact.device)) if exact_logits is None else exact_logits
        holo_logits = eng._lm_head(holo_hidden[-1].to(eng.device))
    el, hl = exact_logits[0].float().cpu(), holo_logits[0].float().cpu()
    cos = torch.nn.functional.cosine_similarity(el.view(1, -1), hl.view(1, -1))
    top_e = el.topk(10).indices.tolist()
    top_h = hl.topk(10).indices.tolist()
    overlap = len(set(top_e) & set(top_h))
    print(f"\nfinal logit cosine: {cos.item():.4f}")
    print(f"top-10 overlap: {overlap}/10")
    print(f"argmax exact={el.argmax().item()} holo={hl.argmax().item()} match={el.argmax().item() == hl.argmax().item()}")

    # generation with corrections
    print(f"\n--- generation ({args.max_new} tokens) ---")
    with torch.no_grad():
        out = eng.generate(ids, args.max_new, temperature=0.0)
    print("OUTPUT:", tokenizer.decode(out, skip_special_tokens=True))


if __name__ == "__main__":
    main()
