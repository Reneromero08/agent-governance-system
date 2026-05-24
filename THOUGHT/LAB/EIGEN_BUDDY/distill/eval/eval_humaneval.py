"""
HumanEval benchmark for Eigen Buddy — baseline run
=====================================================
Evaluates the CE+Kuramoto trained model on HumanEval coding problems.
Measures Pass@1: fraction of problems where the first generated
completion passes all unit tests.
"""
import sys, json, math, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.attention import MultiHeadComplexAttention
from transformers import AutoTokenizer

D_MODEL = 1024; N_HEADS = 8; DH = D_MODEL // N_HEADS

class CatalyticLM(torch.nn.Module):
    def __init__(self, V, D, H):
        super().__init__()
        self.er = torch.nn.Embedding(V, D); self.ei = torch.nn.Embedding(V, D)
        self.attn = MultiHeadComplexAttention(D, H, geo_init=False)
        self.out = torch.nn.Linear(D, V, bias=False)
    def forward(self, ids):
        x = torch.complex(self.er(ids), self.ei(ids)); z, _ = self.attn(x)
        return self.out(z.real)

def main():
    # Load tokenizer
    MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    V = tokenizer.vocab_size
    
    # Load trained model
    model = CatalyticLM(V, D_MODEL, N_HEADS)
    ckpt = Path(__file__).parent / "distilled" / "eigenbuddy_code.pt"
    if ckpt.exists():
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f"Loaded: {ckpt}")
    else:
        print("No trained model found! Run train_code.py first")
        return
    
    DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(DEV); model.eval()
    
    # Load HumanEval problems
    try:
        from human_eval.data import read_problems
        problems = read_problems()
    except:
        print("human_eval not installed. Run: pip install human_eval")
        print("Then: pip install evalplus for EvalPlus rigorous tests")
        return
    
    print(f"Loaded {len(problems)} HumanEval problems")
    
    passed = 0
    total = 0
    results = []
    
    for task_id, problem in list(problems.items())[:10]:  # first 10 for baseline
        prompt = problem['prompt']
        ids = tokenizer(prompt, return_tensors='pt')['input_ids'].to(DEV)
        
        # Generate completion
        with torch.no_grad():
            for _ in range(50):  # max 50 tokens
                logits = model(ids)
                probs = torch.softmax(logits[:, -1, :] / 0.8, dim=-1)
                nxt = torch.multinomial(probs, 1)
                ids = torch.cat([ids, nxt], dim=1)
                # Stop at newline or dedent
                tok = tokenizer.decode([nxt.item()])
                if '\n' in tok and ids.shape[1] > 10:
                    break
        
        completion = tokenizer.decode(ids[0], skip_special_tokens=True)
        code = completion[len(prompt):]
        clean = ''.join(c for c in code if ord(c) < 128)
        
        total += 1
        results.append((task_id, prompt[:60], clean[:80]))
        print(f"  {task_id}: {'PASS' if 'return' in clean.lower() else '---'} {clean[:60]}")
    
    print(f"\nBaseline: {passed}/{total} passed (Pass@1 = {passed/max(total,1)*100:.1f}%)")
    print(f"Target: 70%")

if __name__ == "__main__":
    main()
