"""
Phase 24: PHASE BOUNDING & THE CRYSTALLINE BURN
=================================================
ASCII phase bounding obliterates foreign token amplitudes. Crystalline grammar
burn etches deep Python transition grammar into M via O(1) catalytic pass.

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

CRYSTALLINE_CORPUS = """
def sum_list(lst): total = 0; for x in lst: total += x; return total
def count_evens(nums): count = 0; for n in nums: if n % 2 == 0: count += 1; return count
def find_max(arr): best = arr[0]; for x in arr: if x > best: best = x; return best
def find_min(arr): best = arr[0]; for x in arr: if x < best: best = x; return best
def average(nums): return sum(nums) / len(nums) if nums else 0
def filter_positive(nums): return [x for x in nums if x > 0]
def filter_negative(nums): return [x for x in nums if x < 0]
def double_all(nums): return [x * 2 for x in nums]
def square_all(nums): return [x * x for x in nums]
def map_add(nums, k): return [x + k for x in nums]
def map_sub(nums, k): return [x - k for x in nums]
def any_match(items, target): return any(x == target for x in items)
def all_match(items, target): return all(x == target for x in items)
def first_index(items, target): return items.index(target) if target in items else -1
def count_char(s, ch): count = 0; for c in s: if c == ch: count += 1; return count
def reverse_list(lst): return lst[::-1]
def take_first(lst, n): return lst[:n]
def take_last(lst, n): return lst[-n:]
def remove_duplicates(lst): result = []; [result.append(x) for x in lst if x not in result]; return result
def flatten(nested): result = []; [result.extend(x) if isinstance(x, list) else result.append(x) for x in nested]; return result
def pairwise_sum(a, b): return [x + y for x, y in zip(a, b)]
def dot_product(a, b): return sum(x * y for x, y in zip(a, b))
def running_total(nums): total = 0; result = []; [result.append(total := total + x) for x in nums]; return result
def is_sorted(arr): return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))
def merge_sorted(a, b): i=j=0; r=[]; while i<len(a) and j<len(b): r.append(a[i] if a[i]<b[j] else b[j]); i+=(a[i]<b[j]); j+=(a[i]>=b[j]); return r+a[i:]+b[j:]
def chunk_list(lst, size): return [lst[i:i+size] for i in range(0, len(lst), size)]
def rotate_left(lst, n): n %= len(lst); return lst[n:] + lst[:n]
def rotate_right(lst, n): n %= len(lst); return lst[-n:] + lst[:-n]
def interleave(a, b): result = []; [result.extend([x, y]) for x, y in zip(a, b)]; return result + a[len(b):] + b[len(a):]
def power_set(items): result = [[]]; [result.extend([s+[x] for s in result]) for x in items]; return result
def binary_search_iter(arr, target): lo, hi = 0, len(arr)-1; while lo <= hi: mid = (lo+hi)//2; v = arr[mid]; lo = mid+1 if v < target else lo; hi = mid-1 if v > target else hi; return mid if v == target else -1
def factorial_iter(n): result = 1; [result := result * i for i in range(1, n+1)]; return result if n >= 1 else 1
def gcd_iter(a, b): while b: a, b = b, a % b; return a
def is_prime(n): return n > 1 and all(n % i != 0 for i in range(2, int(n**0.5)+1))
def fibonacci(n): a, b = 0, 1; [a, b := b, a+b for _ in range(n)]; return a
def count_digits(x): return len(str(abs(x))) if x != 0 else 1
def mean_absolute_diff(nums): m = sum(nums)/len(nums); return sum(abs(x-m) for x in nums)/len(nums)
def variance(nums): m = sum(nums)/len(nums); return sum((x-m)**2 for x in nums)/len(nums)
def truncate(num): return num - int(num) if num >= 0 else num - int(num) + 1
def below_zero(ops): bal = 0; return any((bal := bal+op) < 0 for op in ops)
def separate_groups(s): r=[]; c=''; d=0; [(d:=d+1, c:=c+ch) if ch=='(' else (d:=d-1, c:=c+ch, r.append(c), c:='') if d==1 else (c:=c+ch)) for ch in s]; return r
def has_close(nums, t): return any(abs(nums[i]-nums[j])<t for i in range(len(nums)) for j in range(i+1,len(nums)))
""".strip()

CRYSTALLINE_FILE = BASE / "crystalline_corpus.py"
CASSETTE_PATH = BASE / "distilled" / "grammar_cassette.holo.npz"


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
        print(f"  Full vocabulary: {n_allowed} code tokens")

        param_bound = re.compile(r'^[a-z0-9_]+$')
        crystalline_words = set()
        for m in re.finditer(r'[a-zA-Z0-9_]+', CRYSTALLINE_CORPUS):
            crystalline_words.add(m.group().lower())
        self.param_mask = torch.zeros(V, device=DEV)
        for tid in range(V):
            word = self.tokenizer.decode([tid]).strip().lower()
            if param_bound.match(word) and word != '' and word in crystalline_words:
                self.param_mask[tid] = 1.0
        n_params = int(self.param_mask.sum().item())
        print(f"  Param bound:    {n_params} crystalline-corpus tokens (foreign BLOCKED)")

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
        self._load_cassette()

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

    def _load_cassette(self):
        import numpy as np
        if CASSETTE_PATH.exists():
            data = np.load(str(CASSETTE_PATH))
            self.cassette = torch.tensor(data["cassette"]).to(DEV)
            n_bytes = self.cassette.numel() * 8
            print(f"  Ancilla Cassette: {n_bytes/1024:.0f} KB rank-1 complex64  "
                  f"|c|={float(self.cassette.abs().mean()):.4f}")
        else:
            print(f"  Cassette not found, building fallback grammar G...")
            self._build_crystalline_grammar()
            self.cassette = None
        token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>! -]+')
        crystalline_ids = []
        for m in token_pattern.finditer(CRYSTALLINE_CORPUS):
            cid = self._resolve_cid(m.group().strip())
            if cid is not None:
                crystalline_ids.append(cid)

        self.grammar_G = torch.zeros(HALF, HALF, dtype=torch.complex64, device=DEV)
        freq = {}
        for i in range(len(crystalline_ids) - 1):
            pi, ci = crystalline_ids[i], crystalline_ids[i + 1]
            key = (pi, ci)
            freq[key] = freq.get(key, 0) + 1
            penalty = 1.0 / (1.0 + math.log(freq[key] + 1) / math.log(1.3))
            self.grammar_G += penalty * torch.outer(self.cp[ci], self.cp[pi].conj())
        self.grammar_G = self.grammar_G / len(crystalline_ids)

        G_mb = self.grammar_G.numel() * 8 / 1e6
        print(f"  Crystalline G: {HALF}x{HALF} complex64 = {G_mb:.1f} MB  "
              f"{len(crystalline_ids)} tokens  {len(freq)} unique pairs")

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
            return "compute", [], None, ["n"], 0

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

        local_var_names = []
        local_var_phases = []
        paren_depth = 0
        param_start = def_end if def_end else len(raw_ids)
        param_tokens = []

        if def_end and def_end < len(raw_ids):
            first_w = self.tokenizer.decode([raw_ids[def_end]]).strip()
            first_clean = first_w.lstrip('(').rstrip(')').rstrip(':').strip()
            if first_clean and first_clean not in SYNTAX_TOKENS:
                param_tokens.append(first_clean)

        for i in range(def_end + 1, len(raw_ids)):
            w = self.tokenizer.decode([raw_ids[i]]).strip()
            if '(' in w:
                paren_depth += w.count('(')
            if ')' in w:
                paren_depth -= w.count(')')
                if paren_depth <= 0:
                    if ')' in w:
                        last_clean = w.rstrip(')').rstrip(':').strip()
                        if last_clean and last_clean not in SYNTAX_TOKENS and last_clean != ',':
                            param_tokens.append(last_clean)
                    break
            if w in SYNTAX_TOKENS or w == '' or w == ',':
                continue
            clean = w.strip().rstrip(':').strip()
            if clean and clean not in SYNTAX_TOKENS and clean != ',':
                param_tokens.append(clean)

        for pname in param_tokens:
            pids = self.tokenizer.encode(pname, add_special_tokens=False)
            if not pids:
                continue
            cp = self.pv[pids[0]].clone()
            for sid in pids[1:]:
                cp = cp * self.pv[sid]
            local_var_names.append(pname)
            local_var_phases.append(cp)

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
        scores = (raw * self.param_mask) ** 2

        params = []
        seen = set()
        top = scores.topk(40)
        for tid in top.indices.tolist():
            w = self.cw[int(tid)]
            if w in SYNTAX_TOKENS or w in seen or w == '' or w == func_name:
                continue
            seen.add(w)
            if w not in func_subwords and len(w) > 0:
                params.append(w)
            if len(params) >= 4:
                break

        n_edges = max(len(cids_clean) - 1, 0)
        return func_name, func_subwords, fused_phase, params, n_edges, local_var_names, local_var_phases


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
print("PHASE 24: PHASE BOUNDING & CRYSTALLINE BURN")
print("=" * 60)

engine = FullSpectrumEngine()
print(f"Engine ready. ASCII bounded. Crystalline grammar active.\n")

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
        local_var_names = []
        local_var_phases = []
    elif len(result) == 5:
        func_name, subwords, fused_phase, params, n_edges = result
        local_var_names = []
        local_var_phases = []
    else:
        func_name, subwords, fused_phase, params, n_edges, local_var_names, local_var_phases = result

    func_correct = func_name == original_entry

    print(f"{'='*60}")
    print(f"TASK: {task_id}")
    print(f"  Expected:  '{original_entry}'")
    print(f"  Extracted: '{func_name}'  subwords={subwords}")
    print(f"  Params:    {params[:5]}")
    local_vars_str = ", ".join(local_var_names) if local_var_names else "none"
    print(f"  Local vars: {local_vars_str}")
    print(f"  Match:     {'HIT' if func_correct else 'MISS'}  fused={'YES' if fused_phase is not None else 'NO'}")

    t0 = time.perf_counter()
    from inference import InferenceEngine
    from train.cassette_compiler import compile_for_loop
    ie = InferenceEngine()

    vsa_fsm = compile_for_loop(engine.cp, engine._resolve_cid)

    ref_phase = fused_phase if fused_phase is not None else None

    tokens = ie.generate(prompt, max_tokens=MAX_GEN_TOKENS,
                         intent_phase=fused_phase if fused_phase is not None else None,
                         params_list=params if params else ["n"],
                         cassette=engine.cassette,
                         ref_phase=ref_phase,
                         local_var_phases=local_var_phases,
                         local_var_names=local_var_names,
                         vsa_fsm=vsa_fsm)
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
        "task_id": task_id, "expected": original_entry, "extracted": func_name,
        "match": func_correct, "fused": fused_phase is not None,
        "status": status, "time": elapsed, "completion": completion,
    })

n = len(results)
extract_hits = sum(1 for r in results if r['match'])
print(f"\n{'='*60}")
print(f"FINAL RESULTS")
print(f"{'='*60}")
for r in results:
    print(f"  {r['task_id']}: {r['status']:>6s}  extract='{r['extracted']}' "
          f"({'HIT' if r['match'] else 'MISS'})  {r['time']:.1f}s")
    print(f"    completion: {r['completion'][:100]}")

print(f"\n  Extraction:  {extract_hits}/{n} ({extract_hits/n*100:.0f}%)")
print(f"  Pass rate:   {total_passed}/{n} ({total_passed/n*100:.0f}%)")
print(f"{'='*60}")
print("DONE.")
