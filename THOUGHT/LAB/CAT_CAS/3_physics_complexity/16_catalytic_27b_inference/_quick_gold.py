"""Quick gold data: Qwen oracle (CUDA) + catalytic engine (first 20 tokens)."""
import sys, os, time, numpy as np, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

CAT_CAS = Path(__file__).resolve().parent
EIGEN = next(p for p in Path(__file__).resolve().parents if p.name == "CAT_CAS").parent / 'EIGEN_BUDDY'
MODEL_DIR = CAT_CAS / 'gemini_update' / 'qwen_0.5b'

sys.path.insert(0, str(CAT_CAS))
sys.path.insert(0, str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
os.chdir(str(EIGEN / 'core' / 'rust_ffi' / 'target' / 'release'))
import catalytic_ffi
from experiment import CatalyticInferenceRuntime, COMPLEX_DIM, F32_DIM, NUM_LAYERS, HIDDEN_DIM

# Phase 1: Qwen oracle (CUDA, fast)
print('=== Phase 1: Qwen Oracle ===', flush=True)
print('Loading Qwen...', flush=True)
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), local_files_only=True, dtype=torch.float16, device_map='cuda')
model.eval()
print(f'Done: {time.perf_counter()-t0:.1f}s', flush=True)

prompt = "The catalytic computing paradigm demonstrates that"
print(f'Prompt: {prompt}', flush=True)
inputs = tok(prompt, return_tensors='pt').to(model.device)
with torch.no_grad():
    out = model.generate(inputs['input_ids'], max_new_tokens=20, do_sample=False, pad_token_id=tok.eos_token_id)
gold_tokens = out[0].tolist()[inputs['input_ids'].shape[1]:]
print(f'Gold tokens ({len(gold_tokens)}): {gold_tokens}')
gold_text = tok.decode(gold_tokens)
print(f'Gold text: {gold_text.encode("ascii", errors="replace").decode("ascii")}')

# Phase 2: Catalytic engine (slow, collects hidden states)
print('\n=== Phase 2: Catalytic Verifier ===', flush=True)
print('Init runtime...', flush=True)
t0 = time.perf_counter()
rt = CatalyticInferenceRuntime()
print(f'Done: {time.perf_counter()-t0:.1f}s', flush=True)

hidden_real = []
hidden_imag = []
targets = []

prompt_ids = rt.tokenizer.tokenize(prompt)
all_ids = prompt_ids + gold_tokens
print(f'Running {len(all_ids)} tokens...', flush=True)

for i, tid in enumerate(all_ids):
    emb = rt.tokenizer.embed(tid)
    tape = bytes(rt.tape)
    t0 = time.perf_counter()
    result = catalytic_ffi.catalytic_inference_step(tape, emb, NUM_LAYERS, rt.streamer.scrambled_weights, i)
    elapsed = time.perf_counter() - t0
    if 'working_region' in result:
        rt.tape[:len(result['working_region'])] = bytearray(result['working_region'])
    warm = result.get('warm_hit', False)
    hs = bytes(result.get('hidden_state', bytes(rt.tape[:COMPLEX_DIM])))
    hr = np.frombuffer(hs[:F32_DIM], dtype=np.float32).copy()
    hi = np.frombuffer(hs[F32_DIM:COMPLEX_DIM], dtype=np.float32).copy()
    hr = np.nan_to_num(hr, nan=0, posinf=0, neginf=0)
    hi = np.nan_to_num(hi, nan=0, posinf=0, neginf=0)
    hidden_real.append(hr)
    hidden_imag.append(hi)
    gold_next = all_ids[i+1] if i+1 < len(all_ids) else tid
    targets.append(gold_next)
    label = 'WARM' if warm else 'COLD'
    gold_text = tok.decode([gold_next])
    print(f'  [{i:>3}] tid={tid:>5} -> gold={gold_next:>5} {gold_text.encode("ascii",errors="replace").decode("ascii"):20s} {label} {elapsed*1000:.0f}ms', flush=True)

rt.cleanup()

# Save
states_real = torch.from_numpy(np.stack(hidden_real))
states_imag = torch.from_numpy(np.stack(hidden_imag))
targets_t = torch.tensor(targets, dtype=torch.long)

N = len(targets)
print(f'\nCollected {N} pairs, {len(set(targets))} unique gold tokens')
save_path = CAT_CAS / 'gold_training_data' / 'gold_pairs_quick.pt'
save_path.parent.mkdir(exist_ok=True)
torch.save({
    'states_real': states_real, 'states_imag': states_imag,
    'targets': targets_t, 'hidden_dim': HIDDEN_DIM,
    'num_pairs': N, 'unique_tokens': len(set(targets)),
    'gold_text': gold_text,
}, save_path)
print(f'Saved to {save_path}')
print(f'\nNext: .venv\\Scripts\\python.exe THOUGHT\\LAB\\EIGEN_BUDDY\\eigen_buddy_tokenizer.py --data {save_path}')
