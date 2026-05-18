"""Extended feedback run: 10 passes, 40 prompts, k=50."""
import sys, time, json
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback")
from transformers import GPT2Tokenizer
from auto_feedback import AutoFeedbackLoop, AdapterGPT2, SAMPLE_TEXTS, TRAIN_PROMPTS, TEST_PROMPTS

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

MORE_PROMPTS = TRAIN_PROMPTS + [
    "The invention of the printing press",
    "The theory of evolution was proposed by",
    "Quantum mechanics describes the behavior of",
    "The Roman Empire fell due to",
    "The Industrial Revolution began in",
    "Gravity was first described mathematically by",
    "The longest river in the world is the",
    "The human heart has",
    "The periodic table was created by",
    "Mitochondria are the",
    "The French Revolution began in",
    "The speed of sound is approximately",
    "Black holes are formed when",
    "The Renaissance was a period of",
    "The deepest ocean trench is the",
    "The light bulb was invented by",
    "Antibiotics are used to treat",
    "The Berlin Wall fell in",
    "The first artificial satellite was",
    "Neutrons were discovered by",
]

print(f"Training prompts: {len(MORE_PROMPTS)}", flush=True)

model, original = AdapterGPT2.from_pretrained("gpt2", k=50)
model.init_projectors(tokenizer, SAMPLE_TEXTS, "cpu")
loop = AutoFeedbackLoop(model, original, tokenizer, lr=3e-4)

pre_eval = loop.evaluate(TEST_PROMPTS, "pre")
loop.run_feedback(MORE_PROMPTS, max_passes=10, max_tokens=40,
                  attn_lambda=0.5, batch_size=1, layer_gamma=1.0)
post_eval = loop.evaluate(TEST_PROMPTS, "post")

print(f"\nPPL: {pre_eval['ppl_ratio']:.1f} -> {post_eval['ppl_ratio']:.1f} "
      f"(-{(pre_eval['ppl_ratio'] - post_eval['ppl_ratio']) / pre_eval['ppl_ratio'] * 100:.0f}%)", flush=True)
print(f"Attn cos: {pre_eval['attention_cosine']:.4f} -> {post_eval['attention_cosine']:.4f}", flush=True)
