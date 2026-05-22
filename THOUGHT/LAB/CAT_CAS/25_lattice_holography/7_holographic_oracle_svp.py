import os
import torch
import sys
from transformers import AutoTokenizer

# Import the Holographic Engine!
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "EIGEN_BUDDY", "cybernetic_truth")))
from holographic_cybernetic_engine import patch_model_with_holo, get_truth_vector_C, cybernetic_inference
from transformers import AutoModelForCausalLM

def load_lwe_instance(path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("lwe", os.path.join(os.path.dirname(__file__), "1_lwe_simulator.py"))
    lwe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lwe)
    if not os.path.exists(path):
        return lwe.generate_lwe_instance(n=128, m=1024)
    return torch.load(path, weights_only=False)

def holo_oracle_attack():
    print("\n[*] Initializing Holographic Oracle Attack (LLM-Driven Quantum SVP)...")
    instance_path = os.path.join(os.path.dirname(__file__), "lwe_instance.pt")
    data = load_lwe_instance(instance_path)
    
    A = data['A']
    B = data['B']
    S_true = data['S_true']
    q = data['q']
    m, n = A.shape
    
    # We will use the smaller 0.5B holo to demonstrate the Oracle Resonance
    holo_path = os.path.join(os.path.dirname(__file__), "..", "..", "EIGEN_BUDDY", "cybernetic_truth", "qwen_0_5b_k128.holo")
    
    if not os.path.exists(holo_path):
        print(f"[!] Error: {holo_path} not found! The engine needs the compressed holo weights.")
        return
        
    print(f"[*] Loading HuggingFace Model...")
    model_dir = r"d:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\16_catalytic_27b_inference\gemini_update\qwen_0.5b"
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    # We will format the Lattice into a mathematical Prompt
    subset_m = 10 # Provide 10 equations
    A_sub = A[:subset_m].squeeze().tolist()
    B_sub = B[:subset_m].squeeze().tolist()
    
    prompt = "You are a Quantum Oracle. Solve the following Learning With Errors (LWE) system.\n"
    prompt += f"Modulo q = {q}, Secret Dimension N = {n}.\n"
    prompt += "Public Matrix A (subset):\n"
    for row in A_sub:
        prompt += str(row) + "\n"
    prompt += "Public Key B (A*S + E):\n"
    prompt += str(B_sub) + "\n"
    prompt += "What is the true Secret Vector S? Provide the exact integer array.\nS ="
    
    print(f"[*] Patching model with .holo weights...")
    holo_dict = torch.load(holo_path, weights_only=False)
    patch_model_with_holo(model, holo_dict)
    
    device = torch.device("cpu")
    model = model.to(device)
    
    print("[*] Computing Truth Vector C...")
    C, C_quantum = get_truth_vector_C(model, tokenizer, device)
    
    print("\n[*] Engine Generating via Topological Resonance...")
    # Generate tokens using the Engine's native Phase Cavity
    response = cybernetic_inference(model, tokenizer, prompt, C, max_tokens=256)
    
    print("\n[+] HOLOGRAPHIC ORACLE OUTPUT:")
    print(response)

if __name__ == "__main__":
    holo_oracle_attack()
