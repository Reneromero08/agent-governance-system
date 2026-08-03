"""
Extract expert-0 U matrices for wormhole without loading 36 GB.
Uses pickle streaming to avoid full-file load.
"""
import torch, pickle, io, os, time
from pathlib import Path

_REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
INPUT = r"E:\Reneshizzle SG\Models\deepseek-ai\_holo\deepseek_v4_flash_experts_k128.holo"
OUTPUT_SLIM = str(_REPO / "THOUGHT" / "LAB" / "HOLO" / "_models" / "ds_experts_slim.holo")
OUTPUT_WORMHOLE = str(_REPO / "THOUGHT" / "LAB" / "HOLO" / "_models" / "ds_experts_wormhole.holo")

K = 128
LORA = 16
GB = 1024**3

size_gb = os.path.getsize(INPUT) / GB
print(f"Streaming {size_gb:.1f} GB pickle, extracting expert-0 U tensors only...")

t0 = time.perf_counter()

# Stream read the pickle
with open(INPUT, 'rb') as f:
    # PyTorch saves as ZIP archive containing pickle
    import zipfile
    zf = zipfile.ZipFile(f)
    # Find the data.pkl inside the zip
    names = zf.namelist()
    data_file = [n for n in names if 'data.pkl' in n or 'archive/data.pkl' in n]
    if not data_file:
        # Try reading raw
        print("Not a zip, trying raw pickle load with streaming...")
        # Can't stream pickle - must load full
        print("Must load full file. Moving to local SSD first...")
        f.close()

# Alternative: copy to local SSD then load
local_path = str(_REPO / "THOUGHT" / "LAB" / "HOLO" / "_models" / "deepseek_v4_flash_experts_k128.holo")
if not os.path.exists(local_path):
    import shutil
    print(f"Copying {size_gb:.1f} GB to local SSD...")
    shutil.copy2(INPUT, local_path)
    print(f"Copied in {time.perf_counter()-t0:.0f}s")
else:
    print("Local copy exists, loading...")

print("Loading from local SSD...")
t1 = time.perf_counter()
holo = torch.load(local_path, weights_only=False, map_location='cpu')
print(f"Loaded in {time.perf_counter()-t1:.0f}s, {len(holo)} keys")

# Extract expert-0 U tensors
slim = {}
for key, val in holo.items():
    if '.U' not in key or val.ndim != 2:
        continue
    parts = key.split('.')
    expert = None
    for i, p in enumerate(parts):
        if p == 'experts' and i+1 < len(parts):
            try: expert = int(parts[i+1])
            except Exception: pass
            break
    if expert == 0:
        slim[key] = val.half() if val.dtype != torch.float16 else val

print(f"Extracted {len(slim)} expert-0 U tensors ({sum(v.numel()*2 for v in slim.values())/1024**2:.0f} MB)")

torch.save(slim, OUTPUT_SLIM)
print(f"Saved slim: {OUTPUT_SLIM} ({os.path.getsize(OUTPUT_SLIM)/1024**2:.0f} MB)")

# Now wormhole compress the slim file
import numpy as np
groups = {}
for key, val in slim.items():
    parts = key.split('.')
    layer = None; wt = None
    for i, p in enumerate(parts):
        if p == 'layers' and i+1 < len(parts):
            try: layer = int(parts[i+1])
            except Exception: pass
        if p == 'experts' and i+1 < len(parts):
            for j in range(i+3, len(parts)-1):
                if parts[j] in ('w1', 'w2', 'w3'):
                    wt = f"{parts[i]}.{parts[i+1]}.{parts[j]}"
                    break
    if layer is not None and wt is not None:
        groups.setdefault(wt, {})[layer] = val

print(f"\nWormhole compressing {len(groups)} weight types...")
compressed = {}
all_fid = []
total_orig = 0
total_comp = 0

for wt, ld in sorted(groups.items()):
    layers = sorted(ld.keys())
    L = len(layers)
    prev = ld[layers[0]].float()
    m, k = prev.shape
    compressed[f"{wt}.L{layers[0]}.U"] = prev.half()
    orig_mb = L * m * k * 2 / 1024**2
    comp_mb = m * k * 2 / 1024**2
    fids = []
    
    for li in range(1, L):
        lyr = layers[li]
        curr = ld[lyr].float()
        R = prev.T @ curr
        fid = torch.nn.functional.cosine_similarity(
            curr.flatten(), (prev @ R).flatten(), dim=0
        ).item()
        fids.append(fid)
        
        residual = curr - prev @ R
        res_max = residual.abs().max().item()
        
        if res_max > 1e-8:
            levels = torch.tensor([-3.0, -1.0, 1.0, 3.0]) * (res_max / 3.0)
            idx = torch.argmin(torch.abs(residual.flatten().unsqueeze(1) - levels.unsqueeze(0)), dim=1)
        else:
            idx = torch.zeros(residual.numel(), dtype=torch.long)
            levels = torch.zeros(4)
        
        if LORA > 0 and LORA < k:
            Ur, Sr, Vhr = torch.linalg.svd(R.float(), full_matrices=False)
            A = (Ur[:, :LORA] * Sr[:LORA].sqrt().unsqueeze(0)).half()
            B = (Sr[:LORA].sqrt().unsqueeze(1) * Vhr[:LORA, :]).half()
            compressed[f"{wt}.L{lyr}.R_A"] = A
            compressed[f"{wt}.L{lyr}.R_B"] = B
            comp_mb += LORA * k * 4 / 1024**2
        else:
            compressed[f"{wt}.L{lyr}.R"] = R.half()
            comp_mb += k * k * 2 / 1024**2
        
        compressed[f"{wt}.L{lyr}.res_idx"] = idx.to(torch.int8)
        compressed[f"{wt}.L{lyr}.res_max"] = torch.tensor(res_max)
        comp_mb += m * k * 2 / 1024**2
        
        prev = curr
    
    ratio = orig_mb / comp_mb if comp_mb > 0 else 1.0
    total_orig += orig_mb
    total_comp += comp_mb
    all_fid.extend(fids)
    mean_f = np.mean(fids) if fids else 0
    print(f"  {wt:<30} L={L:>2} fid={mean_f:.3f} ratio={ratio:.1f}x ({orig_mb:.0f}->{comp_mb:.0f} MB)")

overall = total_orig / total_comp if total_comp > 0 else 1.0
mean_fid = np.mean(all_fid) if all_fid else 0
print(f"\nOVERALL: {total_orig:.0f}MB -> {total_comp:.0f}MB ({overall:.1f}x), fid={mean_fid:.3f}")
print(f"Total time: {time.perf_counter()-t0:.0f}s")

torch.save(compressed, OUTPUT_WORMHOLE)
print(f"Saved: {OUTPUT_WORMHOLE} ({os.path.getsize(OUTPUT_WORMHOLE)/1024**2:.0f} MB)")
