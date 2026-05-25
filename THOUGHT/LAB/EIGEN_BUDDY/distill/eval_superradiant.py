"""
Phase 22: FULL SPECTRUM UNMASKING — BPE Concept Fusion
=========================================================
Unmasks full ~124K code vocabulary. Uses Concept Fusion (Hadamard product
of BPE subword phases) to dynamically construct multi-token carrier waves
from the prompt's function signature. No AST parsing — pure wave mechanics.

Usage:
  python eval_superradiant.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import ast
import time
import math
import torch
from pathlib import Path

BASE = Path(__file__).parent
sys.path.insert(0, str(BASE))

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SYNTAX_TOKENS = {"def", "return", ":", "(", ")", ",", "=", "for", "if", "in", "else",
                 "elif", "while", "class", "import", "from", "as", "with", "try",
                 "except", "finally", "raise", "assert", "pass", "break", "continue",
                 "not", "and", "or", "is", "True", "False", "None", "lambda", "range",
                 "len", "abs", "int", "str", "list", "dict", "set", "sum", "max", "min",
                 "print", "open", "read", "write", "append", "sort", "sorted", "enumerate"}
EXCLUDE = {',', 'the', 'to', 'is', '?', '!', '-', ''}

PYTHON_CODE_CORPUS = """
def add(a, b): return a + b
def multiply(x, y): return x * y
def factorial(n): return 1 if n <= 1 else n * factorial(n - 1)
def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)
def gcd(a, b): while b: a, b = b, a % b; return a
def is_prime(n): if n < 2: return False; for i in range(2, int(n**0.5)+1): if n % i == 0: return False; return True
def reverse_string(s): return s[::-1]
def binary_search(arr, target): lo, hi = 0, len(arr)-1; while lo <= hi: mid = (lo+hi)//2; if arr[mid] == target: return mid; elif arr[mid] < target: lo = mid+1; else: hi = mid-1; return -1
class Counter: def __init__(self): self.count = 0; def increment(self): self.count += 1; def get(self): return self.count
""".strip()


class FullSpectrumEngine:
    def __init__(self):
        from transformers import AutoTokenizer
        import safetensors.torch as st

        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
        V = self.tokenizer.vocab_size

        embed = None
        for sp in sorted(MODEL_DIR.glob("model-*.safetensors")):
            tensors = st.load_file(str(sp))
            for k in tensors:
                if "embed_tokens" in k:
                    embed = tensors[k][:V, :D_MODEL].float()
                    break
            if embed is not None:
                break

        er = embed[:, :HALF]
        ei = embed[:, HALF:]
        er = er / er.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        ei = ei / ei.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        phase_angle = torch.atan2(ei, er)
        self.pv = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
        del embed

        ascii_code = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.,;<>! -]+$')
        self.vocab_mask = torch.zeros(V, device=DEV)
        for tid in range(V):
            word = self.tokenizer.decode([tid]).strip()
            if ascii_code.match(word) and word != '':
                self.vocab_mask[tid] = 1.0
        n_allowed = int(self.vocab_mask.sum().item())
        print(f"  Full vocabulary: {n_allowed} code tokens (unmasked)")

        print(f"  Precomputing concept phases for {n_allowed} words...")
        self.cp = torch.zeros(V, HALF, dtype=torch.complex64, device=DEV)
        self.cw = [""] * V
        for tid in range(V):
            if self.vocab_mask[tid] == 0:
                continue
            word = self.tokenizer.decode([tid]).strip()
            self.cw[tid] = word
            sub_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not sub_ids:
                continue
            cp_val = self.pv[sub_ids[0]].clone()
            for sid in sub_ids[1:]:
                cp_val = cp_val * self.pv[sid]
            self.cp[tid] = cp_val

        self._build_syntax_mask()
        self._build_grammar_G()

    def _resolve_cid(self, word):
        ids = self.tokenizer.encode(word, add_special_tokens=False)
        if not ids:
            return None
        for tid in range(len(self.cw)):
            if self.cw[tid] == word and self.vocab_mask[tid] > 0:
                return tid
        return ids[0] if ids[0] < len(self.cw) and self.vocab_mask[ids[0]] > 0 else None

    def _build_syntax_mask(self):
        self.syntax_state = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        syntax_cids = []
        for w in SYNTAX_TOKENS:
            cid = self._resolve_cid(w)
            if cid is not None:
                syntax_cids.append(cid)
        for i in range(len(syntax_cids) - 1):
            pi, ci = syntax_cids[i], syntax_cids[i + 1]
            self.syntax_state += self.cp[ci] * self.cp[pi].conj()

    def _build_grammar_G(self):
        token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>!]')
        code_ids = []
        for m in token_pattern.finditer(PYTHON_CODE_CORPUS):
            cid = self._resolve_cid(m.group())
            if cid is not None:
                code_ids.append(cid)
        self.grammar_G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
        for i in range(len(code_ids) - 1):
            pi, ci = code_ids[i], code_ids[i + 1]
            self.grammar_G += torch.outer(self.cp[ci], self.cp[pi].conj())
        self.grammar_G = self.grammar_G / len(code_ids)

    def extract_intent(self, prompt_text):
        raw_ids = self.tokenizer.encode(prompt_text)
        def_start = None
        def_end = None
        for i in range(len(raw_ids)):
            word = self.tokenizer.decode([raw_ids[i]]).strip()
            if word == 'def' and def_start is None:
                def_start = i
            if def_start is not None and ('(' in word) and def_end is None:
                def_end = i
                break

        if def_start is None or def_end is None or def_end <= def_start + 1:
            return "compute", ["n"], 0

        func_subword_ids = raw_ids[def_start + 1:def_end]
        func_subwords = []
        for tid in func_subword_ids:
            w = self.tokenizer.decode([tid]).strip()
            w = w.lstrip('(').rstrip(')')
            if w not in SYNTAX_TOKENS and w != '':
                func_subwords.append(w)

        fused_phase = None
        for sw in func_subwords:
            cid = self._resolve_cid(sw)
            if cid is not None:
                if fused_phase is None:
                    fused_phase = self.cp[cid].clone()
                else:
                    fused_phase = fused_phase * self.cp[cid]

        func_name = "".join(func_subwords) if func_subwords else "compute"

        M_prompt = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
        cids_clean = []
        for tid in raw_ids:
            word = self.tokenizer.decode([tid]).strip()
            if word in EXCLUDE:
                continue
            cid = self._resolve_cid(word)
            if cid is not None and word not in SYNTAX_TOKENS:
                cids_clean.append(cid)
        for i in range(len(cids_clean) - 1):
            pi, ci = cids_clean[i], cids_clean[i + 1]
            M_prompt += self.cp[ci] * self.cp[pi].conj()

        M_filtered = M_prompt - 0.3 * self.syntax_state
        M_filtered = M_filtered / (M_filtered.abs().max().clamp(min=1e-12))

        raw = torch.abs(self.cp @ M_filtered.conj())
        scores = (raw * self.vocab_mask) ** 2

        params = []
        seen = set()
        top = scores.topk(30)
        for tid in top.indices.tolist():
            w = self.cw[int(tid)]
            if w in SYNTAX_TOKENS or w in seen or w == '' or w == func_name:
                continue
            seen.add(w)
            if w not in func_subwords and len(w) > 0:
                params.append(w)
            if len(params) >= 3:
                break

        n_edges = len(cids_clean) - 1
        return func_name, func_subwords, fused_phase, params, n_edges


HUMANEVAL_PROBLEMS = {
    "HumanEval/0": {
        "prompt": 'def has_close_elements(numbers, threshold):\n    """Check if any two numbers in the list are closer than threshold."""\n    for i in range(len(numbers)):\n        for j in range(i+1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return',
        "entry_point": "has_close_elements",
        "tests": [
            "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False",
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True",
        ],
    },
    "HumanEval/1": {
        "prompt": 'def separate_paren_groups(paren_string):\n    """Separate groups of nested parentheses into individual strings."""\n    result = []\n    current = \'\'\n    depth = 0\n    for c in paren_string:\n        if c == \'(\':\n            depth += 1\n            current += c\n        elif c == \')\':\n            depth -= 1\n            current += c\n            if depth == 0:\n                result.append(current)\n                current = \'\'\n    return',
        "entry_point": "separate_paren_groups",
        "tests": [
            "assert separate_paren_groups('()') == ['()']",
            "assert separate_paren_groups('(())') == ['(())']",
        ],
    },
    "HumanEval/2": {
        "prompt": 'def truncate_number(number):\n    """Return the decimal part of a positive floating point number."""\n    return',
        "entry_point": "truncate_number",
        "tests": [
            "assert abs(truncate_number(3.5) - 0.5) < 1e-6",
        ],
    },
    "HumanEval/3": {
        "prompt": 'def below_zero(operations):\n    """Return True if the account balance ever goes below zero."""\n    balance = 0\n    for op in operations:\n        balance += op\n        if balance < 0:\n            return True\n    return',
        "entry_point": "below_zero",
        "tests": [
            "assert below_zero([1, 2, 3]) == False",
            "assert below_zero([1, 2, -4, 5]) == True",
        ],
    },
    "HumanEval/4": {
        "prompt": 'def mean_absolute_deviation(numbers):\n    """Return the mean absolute deviation of a list of numbers."""\n    mean = sum(numbers) / len(numbers)\n    return',
        "entry_point": "mean_absolute_deviation",
        "tests": [
            "assert abs(mean_absolute_deviation([1.0, 2.0, 3.0, 4.0]) - 1.0) < 1e-6",
        ],
    },
}

MAX_GEN_TOKENS = 15


def check_syntax(code, entry_point):
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                return True, "function found"
        return False, "function not in AST"
    except SyntaxError as e:
        return False, f"syntax error: {e}"


def run_tests(code, tests, entry_point):
    namespace = {}
    try:
        exec(code, namespace)
    except Exception as e:
        return False, [], f"exec failed: {e}"
    fn = namespace.get(entry_point)
    if fn is None:
        return False, [], f"function '{entry_point}' not found"
    results = []
    all_pass = True
    for test in tests:
        try:
            exec(test, {entry_point: fn})
            results.append("PASS")
        except AssertionError:
            results.append("FAIL")
            all_pass = False
        except Exception as e:
            results.append(f"ERROR: {e}")
            all_pass = False
    return all_pass, results, "ok"


print("=" * 60)
print("PHASE 22: FULL SPECTRUM UNMASKING — BPE Concept Fusion")
print("=" * 60)

engine = FullSpectrumEngine()
print(f"Engine ready. Full vocabulary. Concept fusion active.\n")

results = []
total_passed = 0

for task_id, problem in sorted(HUMANEVAL_PROBLEMS.items()):
    prompt = problem["prompt"]
    original_entry = problem["entry_point"]
    tests = problem["tests"]

    result = engine.extract_intent(prompt)
    if len(result) == 3:
        func_name, params, n_edges = result
        subwords = []
        fused_phase = None
    else:
        func_name, subwords, fused_phase, params, n_edges = result
    expected_func = original_entry
    func_correct = func_name == expected_func

    print(f"{'='*60}")
    print(f"TASK: {task_id}")
    print(f"  Expected:  '{expected_func}'")
    print(f"  Extracted: '{func_name}'  subwords={subwords}")
    print(f"  Fused:     {'YES' if fused_phase is not None else 'NO'}  "
          f"|fusion|={float(fused_phase.abs().mean()):.4f}" if fused_phase is not None else "")
    print(f"  Extracted params: {params[:4]}")
    print(f"  Match:     {'HIT' if func_correct else 'MISS'}")

    t0 = time.perf_counter()
    from inference import InferenceEngine
    ie = InferenceEngine()
    tokens = ie.generate(prompt, max_tokens=MAX_GEN_TOKENS,
                         intent_phase=fused_phase if fused_phase is not None else None,
                         params_list=params if params else ["n"])
    elapsed = time.perf_counter() - t0

    completion = " ".join(tokens)
    full_code = prompt + " " + completion
    syntax_ok, syntax_msg = check_syntax(full_code, original_entry)
    test_ok = False
    test_results = []
    if syntax_ok:
        test_ok, test_results, test_msg = run_tests(full_code, tests, original_entry)
    else:
        test_msg = syntax_msg

    status = "PASS" if test_ok else ("SYNTAX" if syntax_ok else "FAIL")
    if test_ok:
        total_passed += 1

    print(f"  COMPLETION: {completion[:120]}")
    print(f"  Syntax: {'OK' if syntax_ok else 'FAIL'}  Tests: {'OK' if test_ok else 'FAIL'}  "
          f"Status: {status}  {elapsed:.1f}s")
    print()

    results.append({
        "task_id": task_id,
        "expected": expected_func,
        "extracted": func_name,
        "match": func_correct,
        "fused": fused_phase is not None,
        "status": status,
        "time": elapsed,
    })

n = len(results)
extract_hits = sum(1 for r in results if r['match'])
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
for r in results:
    print(f"  {r['task_id']}: {r['status']:>6s}  extract='{r['extracted']}' "
          f"({'HIT' if r['match'] else 'MISS'} vs '{r['expected']}')  "
          f"fused={'YES' if r['fused'] else 'NO'}  {r['time']:.1f}s")

print(f"\n  Extraction:  {extract_hits}/{n} ({extract_hits/n*100:.0f}%)")
print(f"  Pass rate:   {total_passed}/{n} ({total_passed/n*100:.0f}%)")
print(f"{'='*60}")
print("DONE.")
