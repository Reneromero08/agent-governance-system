#!/usr/bin/env python3
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "LiquidAI/LFM2-2.6B-Exp"

def run_model(prompt):
    print(f"Helper: Loading {MODEL_ID}... (this may take time on first run)", file=sys.stderr)
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, 
            trust_remote_code=True,
            torch_dtype=torch.float32, # Safe for CPU
            # device_map="auto" # Caused meta device error
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print("Helper: Generating...", file=sys.stderr)
    outputs = model.generate(
        **inputs, 
        max_new_tokens=200, 
        do_sample=True, 
        temperature=0.7
    )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <prompt>")
        sys.exit(1)
    
    prompt = sys.argv[1]
    print(run_model(prompt))
