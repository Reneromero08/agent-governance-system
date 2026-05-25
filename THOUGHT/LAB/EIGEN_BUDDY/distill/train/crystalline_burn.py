"""
Phase 24: CRYSTALLINE BURN — Ancilla Cassette Builder
=======================================================
Builds a rank-1 complex phase cassette (grammar_cassette.holo) from a curated
Python algorithmic corpus. The cassette encodes grammar transitions via complex
superposition. O(1) memory, catalytic (borrow→compute→restore).

Usage:
  python crystalline_burn.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import time
import torch
from pathlib import Path

BASE = Path(__file__).parent.parent
MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = 512
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_PATH = BASE / "distilled" / "grammar_cassette.holo.npz"

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
def has_close(nums, t): return any(abs(nums[i]-nums[j])<t for i in range(len(nums)) for j in range(i+1,len(nums)))
def power_set(items): result = [[]]; [result.extend([s+[x] for s in result]) for x in items]; return result
def separate_groups(s): r=[]; c=''; d=0; [(d:=d+1, c:=c+ch) if ch=='(' else (d:=d-1, c:=c+ch, r.append(c), c:='') if d==1 else (c:=c+ch)) for ch in s]; return r
""".strip()

print("=" * 60)
print("PHASE 24: CRYSTALLINE BURN — Ancilla Cassette Builder")
print("=" * 60)

print("Loading Qwen 27B embeddings...")
from transformers import AutoTokenizer
import safetensors.torch as st

tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
V = tokenizer.vocab_size
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
pv = torch.complex(torch.cos(torch.atan2(ei, er)), torch.sin(torch.atan2(ei, er))).to(DEV)
del embed

ascii_code = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.,;<>! -]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_code.match(word) and word != '':
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  ASCII bounding: {n_allowed} tokens (non-ASCII amplitude = 0.0)")

def resolve_cid(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        return None
    return ids[0] if ids[0] < V and vocab_mask[ids[0]] > 0 else None

print(f"\nStreaming crystalline corpus into cassette...")
token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>! -]+')
corpus_ids = []
for m in token_pattern.finditer(CRYSTALLINE_CORPUS):
    w = m.group().strip()
    if not w:
        continue
    cid = resolve_cid(w)
    if cid is not None:
        corpus_ids.append(cid)

cassette = torch.zeros(HALF, dtype=torch.complex64, device=DEV)
freq = {}
n_transitions = 0
for i in range(len(corpus_ids) - 1):
    pi, ci = corpus_ids[i], corpus_ids[i + 1]
    key = (pi, ci)
    freq[key] = freq.get(key, 0) + 1
    penalty = 1.0 / (1.0 + math.log(freq[key] + 1))
    cassette += penalty * pv[ci] * pv[pi].conj()
    n_transitions += 1

cassette = cassette / (cassette.abs().max().clamp(min=1e-12))
cassette_mb = cassette.numel() * 8 / 1024
print(f"  Transitions: {n_transitions}")
print(f"  Unique pairs: {len(freq)}")
print(f"  Cassette size: {cassette_mb:.1f} KB (rank-1, O(1))")
print(f"  |cassette|: {cassette.abs().mean().item():.4f}")

import numpy as np
np.savez_compressed(str(OUT_PATH), cassette=cassette.cpu().numpy())
print(f"\n  Saved: {OUT_PATH} ({OUT_PATH.stat().st_size / 1024:.0f} KB)")
print(f"{'='*60}")
print("DONE.")
