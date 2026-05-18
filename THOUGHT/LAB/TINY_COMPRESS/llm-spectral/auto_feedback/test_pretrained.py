"""Test pre-trained k=50 adapters through feedback loop."""
import sys, math, torch
from pathlib import Path
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback")
from transformers import GPT2Tokenizer
from auto_feedback import AutoFeedbackLoop, AdapterGPT2, SAMPLE_TEXTS, TRAIN_PROMPTS, TEST_PROMPTS
from facts_cassette import FactsCassette

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
fc = FactsCassette()

model, original = AdapterGPT2.from_pretrained("gpt2", k=50)
model.init_projectors(tokenizer, SAMPLE_TEXTS, "cpu")

# Load pre-trained k=50 adapters
ckpt_path = Path(__file__).resolve().parent.parent.parent / "extensions" / "03_flat_llm" / "trained_adapters.pt"
if ckpt_path.exists():
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    loaded = 0
    for li in range(len(model.h)):
        li_str = str(li)
        if li_str in ckpt.get("50", {}):
            try:
                if "adapter_k" in ckpt["50"][li_str]:
                    model.h[li].attn.adapter_k.load_state_dict(ckpt["50"][li_str]["adapter_k"])
                    loaded += 1
                if "adapter_v" in ckpt["50"][li_str]:
                    model.h[li].attn.adapter_v.load_state_dict(ckpt["50"][li_str]["adapter_v"])
                    loaded += 1
            except RuntimeError:
                pass
    print("Loaded {} pre-trained adapter state dicts".format(loaded), flush=True)

loop = AutoFeedbackLoop(model, original, tokenizer, lr=3e-4)

# Pre-feedback PPL (attention cosine is embedded in evaluate output)
pre_eval = loop.evaluate(TEST_PROMPTS, "pre-feedback")
print()

# Run feedback with facts cassette
print("Running feedback loop...", flush=True)
loop.run_feedback(TRAIN_PROMPTS, max_passes=5, max_tokens=40, attn_lambda=0.5,
                  batch_size=1, layer_gamma=1.0, facts_cassette=fc)

post_eval = loop.evaluate(TEST_PROMPTS, "post-feedback")
print()
print("PPL: {:.1f} -> {:.1f}".format(pre_eval["ppl_ratio"], post_eval["ppl_ratio"]))
print("Attn: {:.4f} -> {:.4f}".format(pre_eval["attention_cosine"], post_eval["attention_cosine"]))
