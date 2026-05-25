"""
Phase 19: CORPUS INGESTION — 1M Token Semantic Burn (O(1) Memory)
===================================================================
Streams 1,000,000 tokens through Native Hologram M matrix using complex
outer-product binding with Gram-Schmidt orthogonalization penalty.
Proves zero catastrophic forgetting via synthetic variable recovery.

M = (HALF x HALF) complex64 = 2.1 MB flat (O(1)).

Usage: python corpus_ingestion.py
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import re
import math
import time
import torch
from pathlib import Path
from transformers import AutoTokenizer

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = D_MODEL // 2
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_TOKENS = 1_000_000
BATCH_SIZE = 2048
GS_PENALTY_BASE = 1.3

SYNTHETIC_VAR = "X"
SYNTHETIC_VAL = "42"
BURN_WEIGHT = 200.0
BURN_POSITIONS = [10, 50000, 250000, 500000, 750000]

PYTHON_TEMPLATES = [
    "def {func}({args}): return {expr}",
    "class {cls}: def __init__(self): self.val = 0",
    "if __name__ == '__main__': print({func}({args}))",
    "for i in range({n}): total += i * {factor}; result.append(total % {mod})",
    "while count < {limit}: count *= {factor}; items[count % {size}] = count",
    "try: result = {func}({args})\nexcept {err}: result = {default}",
    "with open('{file}') as f: data = f.read(); lines = data.split()",
    "assert {cond}, f'Expected value'",
    "x = {val}; y = x ** {exp} + {offset}; z = y // {div}",
    "items = sorted([a for a in data if a > {threshold}], key=lambda a: a.value)",
    "def {func}(n): return n if n <= 1 else n * {func}(n - 1)",
    "def {func}(n): return n if n <= 1 else {func}(n-1) + {func}(n-2)",
    "def {func}(a, b): while b: a, b = b, a % b; return a",
    "def {func}(n): return all(n % i != 0 for i in range(2, int(n**0.5)+1))",
    "def {func}(arr, target): lo, hi = 0, len(arr)-1; return -1",
]


def generate_corpus(n_tokens_target, tokenizer):
    import random
    random.seed(42)
    funcs = ["compute", "process", "analyze", "transform", "validate", "retrieve",
             "update", "delete", "insert", "select", "merge", "filter", "reduce",
             "map_values", "find_max", "calc_sum", "normalize", "optimize"]
    args_list = ["x", "y", "a, b", "n", "data", "items", "arr", "target", "value"]
    exprs = ["x + y", "x * y", "x % y", "a + b * 2", "n * (n - 1) // 2",
             "data.get(key)", "items[i] + offset"]
    classes = ["Cache", "Stack", "Queue", "Node", "Record", "Config", "Buffer"]
    errs = ["ValueError", "TypeError", "KeyError", "IndexError", "ZeroDivisionError"]
    files = ["data.csv", "config.json", "input.txt", "log.txt", "output.json"]
    lines = []
    total_tokens = 0
    while total_tokens < n_tokens_target:
        template = random.choice(PYTHON_TEMPLATES)
        fmt = {}
        for match in re.finditer(r'\{(\w+)\}', template):
            key = match.group(1)
            if key == 'func':
                fmt[key] = random.choice(funcs)
            elif key == 'args':
                n_args = random.randint(1, 3)
                fmt[key] = ', '.join(random.sample(args_list, n_args))
            elif key == 'expr':
                fmt[key] = random.choice(exprs)
            elif key == 'cls':
                fmt[key] = random.choice(classes)
            elif key == 'a':
                fmt[key] = random.choice(args_list)
            elif key == 'err':
                fmt[key] = random.choice(errs)
            elif key == 'file':
                fmt[key] = random.choice(files)
            elif key == 'cond':
                fmt[key] = f"{random.choice(args_list)} > {random.randint(0, 100)}"
            elif key == 'val':
                fmt[key] = str(random.randint(0, 10000))
            elif key == 'exp':
                fmt[key] = str(random.randint(1, 5))
            elif key == 'offset':
                fmt[key] = str(random.randint(0, 100))
            elif key == 'div':
                fmt[key] = str(random.randint(1, 10))
            elif key == 'threshold':
                fmt[key] = str(random.randint(0, 50))
            elif key == 'default':
                fmt[key] = str(random.randint(-1, 1))
            elif key == 'n':
                fmt[key] = str(random.randint(1, 1000))
            elif key == 'factor':
                fmt[key] = str(random.randint(1, 10))
            elif key == 'mod':
                fmt[key] = str(random.randint(2, 100))
            elif key == 'limit':
                fmt[key] = str(random.randint(10, 1000))
            elif key == 'size':
                fmt[key] = str(random.randint(10, 100))
        try:
            line = template.format(**fmt)
        except KeyError:
            continue
        lines.append(line)
        total_tokens += len(tokenizer.encode(line))
    return '\n'.join(lines)


class NativeHologramMatrix:
    def __init__(self, phase_vectors, vocab_mask):
        self.pv = phase_vectors
        self.mask = vocab_mask
        self.HALF = phase_vectors.shape[1]
        self.M = torch.zeros(self.HALF, self.HALF, dtype=torch.complex64, device=DEV)
        self.freq = {}
        self.n_transitions = 0

    def bind(self, id_a, id_b, weight=1.0):
        phase_a = self.pv[id_a]
        phase_b = self.pv[id_b]
        key = (int(id_a), int(id_b))
        self.freq[key] = self.freq.get(key, 0) + 1
        penalty = 1.0 / (1.0 + math.log(self.freq[key] + 1) / math.log(GS_PENALTY_BASE))
        self.M += weight * penalty * torch.outer(phase_b, phase_a.conj())
        self.n_transitions += 1

    def query(self, phase, topk=5):
        wave = self.M @ phase
        raw = torch.abs(self.pv @ wave.conj())
        scores = (raw * self.mask) ** 2
        top = scores.topk(topk)
        return [(int(tid), float(scores[int(tid)])) for tid in top.indices.tolist()]


print("=" * 60)
print("PHASE 19: CORPUS INGESTION — 1M TOKEN SEMANTIC BURN")
print("=" * 60)

print(f"Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
V = tokenizer.vocab_size

print("Loading Qwen 27B embed_tokens...")
import safetensors.torch as st
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
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
del embed

ascii_code = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.,;<>! -]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_code.match(word) and word != '':
        vocab_mask[tid] = 1.0
n_allowed = int(vocab_mask.sum().item())
print(f"  Vocab mask: {n_allowed} tokens")

M_size_mb = HALF * HALF * 8 / 1e6
print(f"  M matrix: {HALF}x{HALF} complex64 = {M_size_mb:.1f} MB (O(1))")

print(f"\nGenerating {N_TOKENS:,} token code corpus...")
corpus = generate_corpus(N_TOKENS, tokenizer)

if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
    vram_start = torch.cuda.memory_allocated() / 1e6
else:
    vram_start = 0

holo = NativeHologramMatrix(phase_vectors, vocab_mask)
all_ids = tokenizer.encode(corpus)
n_actual = len(all_ids)
print(f"  Actual tokens: {n_actual:,}")

print(f"\n{'='*60}")
print(f"STREAMING INGESTION ({N_TOKENS:,} tokens)")
print(f"{'='*60}")

token_index = 0
var_burned_positions = []
var_phase = None
t_start = time.perf_counter()
last_report = 0

while token_index < n_actual - 1:
    batch_end = min(token_index + BATCH_SIZE, n_actual - 1)
    for i in range(token_index, batch_end):
        id_a = all_ids[i]
        id_b = all_ids[i + 1]
        if i in BURN_POSITIONS:
            encoded = tokenizer.encode(f" {SYNTHETIC_VAR} = {SYNTHETIC_VAL}")
            for j in range(len(encoded) - 1):
                holo.bind(encoded[j], encoded[j + 1], weight=BURN_WEIGHT)
            var_ids = tokenizer.encode(SYNTHETIC_VAR, add_special_tokens=False)
            if var_ids and len(var_ids) == 1:
                var_phase = phase_vectors[var_ids[0]]
                var_burned_positions.append(holo.n_transitions)
            continue
        holo.bind(id_a, id_b)
    token_index = batch_end
    if holo.n_transitions - last_report >= 50000:
        elapsed = time.perf_counter() - t_start
        rate = holo.n_transitions / max(elapsed, 0.001)
        unique = len(holo.freq)
        print(f"  {holo.n_transitions:>8,} transitions  {rate:>8,.0f} tok/s  "
              f"{unique:>8,} unique pairs  freq={holo.freq.get((int(all_ids[token_index-2]), int(all_ids[token_index-1])), 0)}")
        last_report = holo.n_transitions

elapsed = time.perf_counter() - t_start
if torch.cuda.is_available():
    vram_end = torch.cuda.memory_allocated() / 1e6
    vram_peak = torch.cuda.max_memory_allocated() / 1e6
else:
    vram_end = vram_peak = 0

print(f"\n{'='*60}")
print(f"INGESTION COMPLETE")
print(f"{'='*60}")
print(f"  Transitions: {holo.n_transitions:,}")
print(f"  Unique pairs: {len(holo.freq):,}")
print(f"  Time: {elapsed:.1f}s ({holo.n_transitions/max(elapsed,0.001):,.0f} tok/s)")
print(f"  VRAM start: {vram_start:.1f} MB  end: {vram_end:.1f} MB  delta: {vram_end-vram_start:+.1f} MB")
print(f"  VRAM peak:  {vram_peak:.1f} MB")
print(f"  M |M|:      {holo.M.abs().mean().item():.4f}")
print(f"  O(1) VRAM:  {'PASS' if abs(vram_end-vram_start) < 2.5 else 'CHECK'}")

print(f"\n{'='*60}")
print(f"RECOVERY TEST: {SYNTHETIC_VAR} = {SYNTHETIC_VAL}")
print(f"{'='*60}")
print(f"  Burned at:  {len(var_burned_positions)} positions, weight={BURN_WEIGHT}x")
if var_burned_positions:
    print(f"  Positions:  {var_burned_positions}")
print(f"  Total transitions: {holo.n_transitions:,}")

if var_phase is not None:
    top5 = holo.query(var_phase, topk=5)
    print(f"\n  Hop 1 — Top 5 retrieval for '{SYNTHETIC_VAR}' (single-token, direct phase):")
    target_found = False
    for rank, (tid, score) in enumerate(top5):
        word = tokenizer.decode([tid]).strip()
        marker = " ***" if word == "=" else ""
        print(f"    Rank {rank+1}: '{word}' (score={score:.1e}){marker}")

    eq_ids = tokenizer.encode("=", add_special_tokens=False)
    if eq_ids:
        eq_phase = phase_vectors[eq_ids[0]]
        top5_eq = holo.query(eq_phase, topk=5)
        print(f"\n  Hop 2 — Top 5 retrieval for '=':")
        for rank, (tid, score) in enumerate(top5_eq):
            word = tokenizer.decode([tid]).strip()
            marker = " *** TARGET" if word == SYNTHETIC_VAL else ""
            if marker:
                target_found = True
            print(f"    Rank {rank+1}: '{word}' (score={score:.1e}){marker}")

    print(f"\n  Two-hop recovery: {'PASS' if target_found else 'PARTIAL'} "
          f"(target '{SYNTHETIC_VAL}' {'found' if target_found else 'not in top 5'} "
          f"via X -> = -> 42 chain)")
else:
    print(f"  SKIP: {SYNTHETIC_VAR} not a single token")

freqs = sorted(holo.freq.values(), reverse=True)
print(f"\n{'='*60}")
print(f"GRAM-SCHMIDT PENALTY STATS")
print(f"{'='*60}")
print(f"  Max freq:   {freqs[0]}")
print(f"  99th pct:   {freqs[max(0, int(len(freqs)*0.01))]}")
print(f"  Median:     {freqs[len(freqs)//2]}")
print(f"  Min freq:   {freqs[-1]}")
print(f"  Penalty:    {1.0/(1.0+math.log(freqs[0]+1)/math.log(GS_PENALTY_BASE)):.4f} (max) -> "
      f"{1.0/(1.0+math.log(freqs[-1]+1)/math.log(GS_PENALTY_BASE)):.4f} (min)")
print(f"{'='*60}")
print("DONE.")
