"""Test facts cassette retrieval against GPT-2 generation."""
import sys, torch
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback")
from transformers import GPT2Tokenizer
from auto_feedback import AdapterGPT2, SAMPLE_TEXTS
from facts_cassette import FactsCassette

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
fc = FactsCassette()

test_prompts = [
    "What is the capital of France?",
    "What is the chemical formula for water?",
    "How many bones are in the adult human body?",
    "Who wrote the novel 1984?",
    "What is the speed of light?",
    "What is the largest organ in the human body?",
    "What is 17 times 24?",
    "Who developed the theory of relativity?",
    "In what year did World War II end?",
    "What is the chemical symbol for gold?",
]

model, original = AdapterGPT2.from_pretrained("gpt2", k=50)
model.init_projectors(tokenizer, SAMPLE_TEXTS, "cpu")

print("BASELINE (uncompressed GPT-2, no retrieval):")
correct_base = 0
for prompt in test_prompts:
    inp = tokenizer(prompt, return_tensors="pt")
    out = original.generate(inp["input_ids"], max_new_tokens=30, temperature=0.7,
                            top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    fact = fc.correct(prompt)
    has = fact and fact.lower() in text.lower()
    correct_base += int(has)
    print(f"  {'OK' if has else 'XX'} {prompt[:45]:45s} fact={fact or '?':20s}")

print(f"  Baseline accuracy: {correct_base}/{len(test_prompts)}")

print()
print("CORTEX-AUGMENTED (fact injected into prompt):")
correct_aug = 0
for prompt in test_prompts:
    fact = fc.correct(prompt)
    if fact:
        aug = prompt + " The answer is " + fact + "."
    else:
        aug = prompt
    inp = tokenizer(aug, return_tensors="pt")
    out = original.generate(inp["input_ids"], max_new_tokens=30, temperature=0.7,
                            top_p=0.9, do_sample=True, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(out[0], skip_special_tokens=True)
    has = fact and fact.lower() in text.lower()
    correct_aug += int(has)
    print(f"  {'OK' if has else 'XX'} {prompt[:45]:45s} fact={fact or '?':20s} text={text[:80]}")

print(f"  Augmented accuracy: {correct_aug}/{len(test_prompts)}")
if correct_base > 0:
    print(f"  Recovery: {correct_aug - correct_base} new facts recovered")
