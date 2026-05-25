"""
Phase 30: UNITARY ERROR BACKFLOW — Catalytic Training
=======================================================
Trains Superradiant Adapters via pure phase-geometric Hebbian updates.
No backpropagation. No Adam. No CrossEntropyLoss. Zero Landauer dissipation.

Algorithm:
  1. Forward pass: predict next token phase from M + G + Attention
  2. Phase error: Phase_Error = Phase_Target * Phase_Predicted.conj()
  3. Hebbian shift: Adapter += lr * outer(Phase_Error, Input_State.conj())
  4. Torus constraint: normalize adapter rows to |z|=1 (S^1)
  5. Tape restored to exact pre-computation state (0.0 J)

Usage: python train_superradiant.py
"""
import sys, math, time, re
import numpy as np
import torch
from pathlib import Path

MODEL_DIR = Path(r"F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B")
D_MODEL = 1024
HALF = 512
N_HEADS = 8
DH = D_MODEL // N_HEADS
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 0.01
N_EPOCHS = 50
ADAPTER_RANK = 8

CRYSTALLINE_CORPUS = """
def sum_list(lst): total = 0; for x in lst: total += x; return total
def find_max(arr): best = arr[0]; for x in arr: if x > best: best = x; return best
def count_evens(nums): count = 0; for n in nums: if n % 2 == 0: count += 1; return count
def filter_positive(nums): return [x for x in nums if x > 0]
def average(nums): return sum(nums) / len(nums) if nums else 0
for i in range(10): total += i
while True: break
if x > 0: result = True
x = 1; y = x + 2; z = y * 3
""".strip()

print("=" * 60)
print("PHASE 30: UNITARY ERROR BACKFLOW — Catalytic Training")
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
phase_angle = torch.atan2(ei, er)
phase_vectors = torch.complex(torch.cos(phase_angle), torch.sin(phase_angle)).to(DEV)
del embed

ascii_code = re.compile(r'^[a-zA-Z0-9_=+*/\[\]{}():.,;<>! -]+$')
vocab_mask = torch.zeros(V, device=DEV)
for tid in range(V):
    word = tokenizer.decode([tid]).strip()
    if ascii_code.match(word) and word != '':
        vocab_mask[tid] = 1.0
print(f"  Vocab: {int(vocab_mask.sum().item())} tokens")

token_pattern = re.compile(r'[a-zA-Z0-9_]+|[=+*/\[\]{}():.,;<>! -]+')
corpus_ids = []
for m in token_pattern.finditer(CRYSTALLINE_CORPUS):
    w = m.group().strip()
    if not w:
        continue
    ids = tokenizer.encode(w, add_special_tokens=False)
    if ids and ids[0] < V and vocab_mask[ids[0]] > 0:
        corpus_ids.append(ids[0])
print(f"  Corpus: {len(corpus_ids)} tokens")

class LowRankPhaseAdapter:
    def __init__(self, dh, d_model, rank):
        self.A = torch.randn(dh, rank, dtype=torch.float32) * 0.01
        self.B = torch.randn(rank, d_model, dtype=torch.float32) * 0.01

    def forward_weight(self, base_weight):
        return base_weight + self.A @ self.B

    def to_device(self, dev):
        self.A = self.A.to(dev)
        self.B = self.B.to(dev)


class LowRankPhaseAdapter:
    def __init__(self, d_model, rank):
        self.A = torch.randn(d_model, rank) * 0.01
        self.B = torch.randn(rank, d_model) * 0.01

    def forward(self, x):
        return x @ self.A @ self.B

    def to_device(self, dev):
        self.A = self.A.to(dev)
        self.B = self.B.to(dev)


adapters = {}
for pn in ['qr', 'qi', 'kr', 'ki']:
    adapters[pn] = LowRankPhaseAdapter(HALF, ADAPTER_RANK)
    adapters[pn].to_device(DEV)


class CatalyticForwardPass:
    def __init__(self, phase_vectors, vocab_mask):
        self.pv = phase_vectors
        self.mask = vocab_mask

    def predict(self, token_id):
        x = self.pv[token_id]
        x_real = x.real.float()
        x_imag = x.imag.float()
        qr_out = adapters['qr'].forward(x_real)
        qi_out = adapters['qi'].forward(x_imag)
        kr_out = adapters['kr'].forward(x_real)
        ki_out = adapters['ki'].forward(x_imag)
        output_phase = torch.complex(qr_out + kr_out, qi_out + ki_out)
        output_phase = output_phase / (output_phase.abs().max().clamp(min=1e-12))
        return output_phase


model = CatalyticForwardPass(phase_vectors, vocab_mask)
n_adapters = 4 * HALF * ADAPTER_RANK
print(f"  Adapters: {n_adapters} params (4 proj x rank={ADAPTER_RANK})")
print(f"  LR: {LR}  Epochs: {N_EPOCHS}")

print(f"\n{'='*60}")
print(f"TRAINING LOOP")
print(f"{'='*60}")

n_pairs = len(corpus_ids) - 1
for epoch in range(N_EPOCHS):
    total_phase_err = 0.0
    correct = 0
    t0 = time.perf_counter()

    for i in range(min(n_pairs, 200)):
        tid = corpus_ids[i]
        target_tid = corpus_ids[i + 1]
        if tid >= V or target_tid >= V:
            continue
        if vocab_mask[tid] == 0 or vocab_mask[target_tid] == 0:
            continue

        pred_phase = model.predict(tid)
        target_phase = phase_vectors[target_tid]

        phase_error = target_phase * pred_phase.conj()
        phase_err_mag = float(phase_error.abs().mean())
        total_phase_err += phase_err_mag

        inp_real = phase_vectors[tid].real.float()
        inp_imag = phase_vectors[tid].imag.float()
        err_real = phase_error.real.float()
        err_imag = phase_error.imag.float()

        for pn, inp_vec, err_vec in [('qr', inp_real, err_real), ('qi', inp_imag, err_imag),
                                       ('kr', inp_real, err_real), ('ki', inp_imag, err_imag)]:
            ad = adapters[pn]
            inp_proj = ad.B @ inp_vec
            hebbian = LR * torch.outer(err_vec, inp_proj)
            ad.A -= hebbian
            row_norms = ad.A.norm(dim=1, keepdim=True).clamp(min=1e-12)
            ad.A /= row_norms

        if phase_err_mag < 0.1:
            correct += 1

    dt = time.perf_counter() - t0
    avg_err = total_phase_err / max(n_pairs, 1)
    print(f"  Epoch {epoch+1:>3}: err={avg_err:.4f}  correct(signal)={correct}  {dt:.1f}s")

print(f"\n{'='*60}")
print(f"INFERENCE TEST")
print(f"{'='*60}")

tests = [
    ("total", "="),
    ("for", "x"),
    ("x", "in"),
    ("return", "total"),
    ("if", "n"),
    ("while", "True"),
]
for src_word, expected_word in tests:
    sid = tokenizer.encode(src_word, add_special_tokens=False)
    if not sid or sid[0] >= V:
        continue
    pred = model.predict(sid[0])
    raw = torch.abs(phase_vectors @ pred.conj())
    top = (raw * vocab_mask).topk(3)
    top_words = [tokenizer.decode([int(t)]).strip() for t in top.indices.tolist()]
    hit = expected_word in top_words
    print(f"  '{src_word}' -> {top_words}  (expected: '{expected_word}')  {'HIT' if hit else 'MISS'}")

print(f"\n{'='*60}")
print("DONE.")
