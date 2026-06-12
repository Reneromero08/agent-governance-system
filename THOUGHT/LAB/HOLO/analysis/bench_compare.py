"""Compare raw vs calibrated wormhole output."""
import sys, io, torch, time, gc, importlib.util
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, 'THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression')
import _paths

spec = importlib.util.spec_from_file_location('patcher', 'THOUGHT/LAB/CAT_CAS/4_holographic/33_mera_compression/13_patch_model.py')
patcher = importlib.util.module_from_spec(spec); spec.loader.exec_module(patcher)

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from collections import defaultdict
import json

MODEL_DIR = r'F:\LLM_Models\lmstudio-models\Qwen\Qwen3.6-27B'
dev = torch.device('cuda')

def build_and_test(wormhole_paths, label):
    config = AutoConfig.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    with torch.device('meta'): model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    all_groups = {}; all_svh = {}
    for mod_name, path in wormhole_paths.items():
        if not path.exists(): continue
        groups, svh = patcher.parse_wormhole(path)
        all_groups.update(groups); all_svh.update(svh)
    model, stats = patcher.patch_model_with_wormhole(model, all_groups, all_svh, device='meta')
    model.to_empty(device=dev)
    with open(f'{MODEL_DIR}/model.safetensors.index.json') as f: idx = json.load(f)
    wm = idx['weight_map']
    sf_by_shard = defaultdict(list)
    hl = set()
    for n,m in model.named_modules():
        if isinstance(m, patcher.WormholeLinear): hl.add(n)
    for name, param in model.named_parameters():
        if param.device.type != 'meta': continue
        mod_name = name.rsplit('.',1)[0]
        if mod_name in hl: continue
        sf_key = name.replace('model.','model.language_model.',1)
        if sf_key in wm: sf_by_shard[wm[sf_key]].append((name, sf_key))
    for shard, keys in sf_by_shard.items():
        with safe_open(f'{MODEL_DIR}/{shard}', framework='pt', device='cpu') as sf:
            for name, sf_key in keys:
                val = sf.get_tensor(sf_key)
                try:
                    parts = name.rsplit('.',1)
                    if len(parts)==2:
                        parent = model.get_submodule(parts[0]); m = getattr(parent, parts[1])
                        if isinstance(m, torch.nn.Parameter): m.data = val.to(dev, dtype=m.dtype)
                except: pass
    model.eval(); gc.collect(); torch.cuda.empty_cache()
    
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=True)
    prompts = ["The meaning of life is", "Artificial intelligence will", "The capital of France is"]
    print(f'\n-- {label} ({stats["patched"]} layers, {stats["compression"]:.1f}x, {torch.cuda.memory_allocated()/1e9:.1f} GB) --')
    for p in prompts:
        ids = tok(p, return_tensors='pt').to(dev)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.no_grad(): out = model.generate(**ids, max_new_tokens=20, do_sample=True, temperature=0.7)
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        text = tok.decode(out[0], skip_special_tokens=True)
        try: t = text.encode('ascii',errors='replace').decode('ascii')
        except: t = text
        print(f'  {20/dt:.1f} tok/s | {repr(t[:100])}', flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()

# RAW wormhole
build_and_test(_paths.MODULE_PATHS, "RAW CAVITY WORMHOLE")

# CALIBRATED wormhole  
calibrated_paths = {
    "llm": _paths.HOLO_MODELS / "qwen_27b_llm_calibrated.holo",
    "visual": _paths.VISUAL_WORMHOLE,
    "aux": _paths.AUX_WORMHOLE,
}
build_and_test(calibrated_paths, "CALIBRATED WORMHOLE")
