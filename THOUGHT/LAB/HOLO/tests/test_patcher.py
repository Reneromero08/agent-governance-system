"""Benchmark: Wormhole 27B using 13_patch_model.py"""
import sys, io, torch, time, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'THOUGHT/LAB/CAT_CAS/33_mera_compression')
import _paths

spec = importlib.util.spec_from_file_location('patcher', 'THOUGHT/LAB/CAT_CAS/33_mera_compression/13_patch_model.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

MODEL_DIR = r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
wormhole_paths = _paths.MODULE_PATHS

print('Loading + patching 27B model...', flush=True)
t0 = time.perf_counter()
model, stats = patcher.load_model_patched(wormhole_paths, model_id=MODEL_DIR, device='cuda')
dt = time.perf_counter() - t0
print(f'Load time: {dt:.1f}s', flush=True)
p = stats.get('patched', 0)
c = stats.get('compression', 0)
print(f'Patched: {p} layers, Compression: {c:.1f}x', flush=True)
print(f'GPU VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB', flush=True)
