"""Benchmark: Wormhole 27B — safetensors for norms, no 53 GB load."""
import sys, io, torch, torch.nn as nn, time, json, gc, importlib.util
from collections import defaultdict
from safetensors import safe_open
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'THOUGHT/LAB/CAT_CAS/33_mera_compression')
import _paths

# Load patcher for parse_wormhole + patch_model_with_wormhole
spec = importlib.util.spec_from_file_location('patcher', 'THOUGHT/LAB/CAT_CAS/33_mera_compression/13_patch_model.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
wormhole_paths = _paths.MODULE_PATHS
dev = torch.device('cuda')

# Step 1-2: Config + meta architecture
print('Loading config...', flush=True)
config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
with torch.device('meta'):
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

# Step 3: Parse wormhole cassettes
all_groups = {}; all_svh = {}
for mod_name, path in wormhole_paths.items():
    if not path.exists(): continue
    groups, svh = patcher.parse_wormhole(path)
    all_groups.update(groups); all_svh.update(svh)
    print(f'  {mod_name}: {len(groups)} types, {len(svh)} SVh', flush=True)

# Step 4: Patch on meta
print('Patching...', flush=True)
model, stats = patcher.patch_model_with_wormhole(model, all_groups, all_svh, device='meta')
print(f'  {stats["patched"]} layers, {stats["compression"]:.1f}x compression', flush=True)

# Step 5: Materialize compressed model to CUDA
print(f'Materializing to {dev}...', flush=True)
model.to_empty(device=dev)

# Step 6: Load norms/embeddings from SAFETENSORS (not full model!)
print('Loading norms/embeddings from safetensors...', flush=True)
with open(f'{MODEL_DIR}/model.safetensors.index.json') as f:
    idx = json.load(f)
wm = idx['weight_map']
sf_by_shard = defaultdict(list)
# Track which params are already patched (HoloLinear)
hl_modules = set()
for n, m in model.named_modules():
    if isinstance(m, patcher.WormholeLinear):
        hl_modules.add(n)
for name, param in model.named_parameters():
    if param.device.type != 'meta': continue
    mod_name = name.rsplit('.', 1)[0]
    if mod_name in hl_modules: continue
    sf_key = name.replace('model.', 'model.language_model.', 1)
    if sf_key in wm:
        sf_by_shard[wm[sf_key]].append((name, sf_key))

for shard, keys in sf_by_shard.items():
    with safe_open(f'{MODEL_DIR}/{shard}', framework='pt', device='cpu') as sf:
        for name, sf_key in keys:
            val = sf.get_tensor(sf_key)
            try:
                parts = name.rsplit('.', 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    m = getattr(parent, parts[1])
                    if isinstance(m, nn.Parameter):
                        m.data = val.to(dev, dtype=m.dtype)
            except Exception:
                pass

model.eval(); gc.collect(); torch.cuda.empty_cache()
mem = torch.cuda.memory_allocated() / 1e9
meta = sum(1 for p in model.parameters() if p.device.type == 'meta')
print(f'GPU VRAM: {mem:.1f} GB | Meta: {meta}', flush=True)

# ---- BENCHMARK ----
t0 = time.perf_counter()
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
prompts = [
    "The meaning of life is",
    "Artificial intelligence will",
    "The capital of France is",
    "In the year 2050, humans will",
    "The most important scientific discovery was",
]
print(f'\n=== BENCHMARKS ===')
results = []
for p in prompts:
    ids = tok(p, return_tensors='pt').to(dev)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=30, do_sample=True, temperature=0.7)
    torch.cuda.synchronize(); dt = time.perf_counter() - t0
    new_tokens = out.shape[1] - ids['input_ids'].shape[1]
    tps = new_tokens / dt
    text = tok.decode(out[0], skip_special_tokens=True)
    try: text_ascii = text.encode('ascii', errors='replace').decode('ascii')
    except: text_ascii = text
    print(f'  {tps:.1f} tok/s | {repr(text_ascii[:80])}', flush=True)
    results.append({'tps': tps, 'text': text_ascii, 'tokens': new_tokens})

avg_tps = sum(r['tps'] for r in results) / len(results)
print(f'\n=== SUMMARY ===')
print(f'GPU VRAM: {mem:.1f} GB')
print(f'Average: {avg_tps:.1f} tok/s')
print(f'Patched: {stats["patched"]} layers')
print(f'Compression: {stats["compression"]:.1f}x')
print(f'Output: {results[0]["text"][:60]}...')
