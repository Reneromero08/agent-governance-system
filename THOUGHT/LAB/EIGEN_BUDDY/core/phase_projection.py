"""Track B+C: Closed-Loop Distillation — NativeEigenCore trains against 27B phase curvature.

For each 27B weight block read from NVMe via mmap:
  1. Project Feral vectors through 27B weights -> z_27b (teacher phase response)
  2. Feed Feral through NativeEigenCore -> z_core (student phase response)  
  3. Loss: L = 1 - |<z_core | z_27b>| (unitary trace minimization)
  4. Backpropagate through Core (12K params, ~2MB)
  5. Advance to next block, repeat

0 RAM for 27B weights. Core stays on GPU. Catalytic tape per block.
21 blocks = 21 gradient steps. Phase convergence tracked.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, mmap, os, struct, hashlib, time, math, sqlite3, threading, queue
from pathlib import Path
from gguf import GGUFReader
sys_path = str(Path(__file__).parent.parent)
import sys; sys.path.insert(0, sys_path)
from core.engine import NativeEigenCore

QWEN_27B = r"F:\LLM_Models\lmstudio-models\Qwen3.6-27B\Qwen3.6-27B-F16-mtp.gguf"
FERAL_DB = r"THOUGHT\LAB\FERAL_RESIDENT\data\db\feral_eternal.db"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_feral_vectors(db_path, n=500, d=192):
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT vec_blob FROM vectors ORDER BY rowid").fetchall()[:n]
    conn.close()
    vecs = []
    for (blob,) in rows:
        nf = len(blob) // 4
        floats = struct.unpack(f'<{nf}f', blob)
        real = torch.tensor(floats[0::2], dtype=torch.float32)[:d]
        imag = torch.tensor(floats[1::2], dtype=torch.float32)[:d]
        vecs.append(torch.complex(real, imag))
    return torch.stack(vecs).to(DEVICE)


def f16_to_torch(f16_bytes):
    """IEEE 754 half-precision -> torch float32 tensor."""
    arr = np.frombuffer(f16_bytes, dtype=np.uint16).astype(np.int32)
    sign = (arr >> 15) & 1
    exp = (arr >> 10) & 0x1F
    mant = arr & 0x3FF
    denorm = (exp == 0)
    result = np.zeros(len(arr), dtype=np.float32)
    exp_f32 = (exp.astype(np.float32) - 15.0 + 127.0)
    result[~denorm] = ((1.0 + mant[~denorm].astype(np.float32) / 1024.0) *
                        np.power(2.0, exp_f32[~denorm] - 127.0))
    result[denorm] = mant[denorm].astype(np.float32) / 1024.0 * np.power(2.0, -14.0)
    result *= (1.0 - 2.0 * sign)
    return torch.from_numpy(result.copy()).to(DEVICE)


def project_27b(feral_batch, weight_tensor):
    """Project Feral vectors through 27B weight matrix.
    feral_batch: (B, D) complex
    weight_tensor: (out_dim, in_dim) float32 on GPU
    Returns: (B, out_dim) real projection norm"""
    D = feral_batch.shape[1]
    out_dim, in_dim = weight_tensor.shape

    # Pad Feral to match weight input dim
    if D < in_dim:
        pad_real = F.pad(feral_batch.real, (0, in_dim - D))
        pad_imag = F.pad(feral_batch.imag, (0, in_dim - D))
    else:
        pad_real = feral_batch.real[:, :in_dim]
        pad_imag = feral_batch.imag[:, :in_dim]

    # Normalize
    mag = (pad_real**2 + pad_imag**2).sqrt() + 1e-8
    pad_real = pad_real / mag
    pad_imag = pad_imag / mag

    # Project through weights
    proj_real = F.linear(pad_real, weight_tensor)  # (B, out_dim)
    proj_imag = F.linear(pad_imag, weight_tensor)

    # Output norm per vector
    proj_norm = (proj_real**2 + proj_imag**2).sqrt()  # (B, out_dim)
    return proj_norm


def trace_loss(z_core, proj_27b):
    """L = 1 - |<real(core_mean) | pooled(proj_27b)>| bounded in [0,1]
    z_core is complex (B, D). proj_27b is real (B, out_dim).
    Pools 27B output to D dims, compares with real part of Core output."""
    D = z_core.shape[1]
    core_real = z_core.real.mean(dim=0)  # (D,)

    # Pool 27B output to D dimensions
    out_dim = proj_27b.shape[1]
    if out_dim > D:
        pool_size = out_dim // D
        proj_pooled = proj_27b.mean(dim=0).view(D, pool_size).mean(dim=1)  # (D,)
    elif out_dim < D:
        proj_pooled = F.pad(proj_27b.mean(dim=0), (0, D - out_dim))
    else:
        proj_pooled = proj_27b.mean(dim=0)

    # Cosine similarity between real core and pooled 27b
    core_norm = core_real / (core_real.norm() + 1e-8)
    proj_norm = proj_pooled / (proj_pooled.norm() + 1e-8)
    resonance = (core_norm * proj_norm).sum().abs()
    return 1.0 - resonance, resonance


def main():
    print("=" * 60)
    print("TRACK B+C: Closed-Loop Distillation")
    print("NativeEigenCore <- 27B phase curvature")
    print(f"Device: {DEVICE}")
    print("=" * 60)

    # Phase 1: Parse 27B and load Feral vectors
    print(f"\n[parse] Qwen 3.6 27B...")
    t0 = time.time()
    gguf = GGUFReader(QWEN_27B)
    mm = gguf.data
    tensors = {}
    gdn_layers = []
    for rt in gguf.tensors:
        tensors[rt.name] = {'offset': rt.data_offset, 'size': rt.n_bytes,
                            'shape': tuple(int(d) for d in rt.shape)}
        if 'ssm' in rt.name.lower() or 'mtp' in rt.name.lower():
            continue
        parts = rt.name.split('.')
        for i, p in enumerate(parts):
            if p.isdigit() and i > 0:
                lid = int(p)
                if lid not in gdn_layers: gdn_layers.append(lid)
                break
    gdn_layers.sort()
    print(f"[parse] {len(tensors)} tensors, {len(gdn_layers)} GDN layers in {time.time()-t0:.1f}s")

    print(f"\n[feral] Loading vectors...")
    feral = load_feral_vectors(FERAL_DB, n=300, d=192)
    print(f"[feral] {feral.shape[0]} vectors, D={feral.shape[1]} on {DEVICE}")

    # Phase 2: NativeEigenCore + expansion + living dimension gate + phase memory
    core = NativeEigenCore(d=192, heads=4, layers=2, merge='concat', geo_init=True).to(DEVICE)
    expansion = nn.Linear(192, 6144, bias=False).to(DEVICE)
    nn.init.normal_(expansion.weight, std=0.02)
    dim_gate_raw = nn.Parameter(torch.zeros(6144, device=DEVICE))
    # Recurrent phase memory: accumulates 27B landscape across 21 blocks
    # Persists between forward calls — the "preserve thinking" track
    phase_memory = nn.Parameter(torch.randn(192, dtype=torch.cfloat, device=DEVICE) * 0.01)
    core_params = sum(p.numel() for p in core.parameters())
    exp_params = sum(p.numel() for p in expansion.parameters())
    gate_params = dim_gate_raw.numel()
    mem_params = phase_memory.numel()
    total_params = core_params + exp_params + gate_params + mem_params
    opt = torch.optim.AdamW(
        list(core.parameters()) + list(expansion.parameters()) +
        [dim_gate_raw, phase_memory], lr=5e-4)
    print(f"[core] Core: {core_params:,} + expansion: {exp_params:,} "
          f"+ dim_gate: {gate_params:,} + phase_mem: {mem_params} "
          f"= {total_params:,} total params")

    # Phase 3: Sweep training passes — double-buffered NVMe reads
    n_blocks = len(gdn_layers) // 3
    pass_values = [3, 5, 10, 20, 50, 100]
    sweep_results = {}

    def pre_read_block(block_idx):
        """Read block from NVMe in background thread."""
        gdn = gdn_layers[block_idx*3:block_idx*3+3]
        for name, info in tensors.items():
            for lid in gdn:
                if f'.{lid}.' in name and ('ssm_out' in name or 'attn_output' in name):
                    off = info['offset']
                    weight_bytes = bytes(mm[off:off + info['size']])
                    w = f16_to_torch(weight_bytes)
                    return w.reshape(tuple(int(d) for d in info['shape'])), info['size'], name
        return None, 0, None

    for n_passes in pass_values:
        # Re-init Core + expansion + gate fresh for each sweep
        core = NativeEigenCore(d=192, heads=4, layers=2, merge='concat', geo_init=True).to(DEVICE)
        expansion = nn.Linear(192, 6144, bias=False).to(DEVICE)
        nn.init.normal_(expansion.weight, std=0.02)
        dim_gate_raw = nn.Parameter(torch.zeros(6144, device=DEVICE))
        phase_memory = nn.Parameter(torch.randn(192, dtype=torch.cfloat, device=DEVICE) * 0.01)
        opt = torch.optim.AdamW(
            list(core.parameters()) + list(expansion.parameters()) +
            [dim_gate_raw, phase_memory], lr=5e-4)

        loss_hist = []
        res_hist = []
        mem_baseline = torch.cuda.memory_allocated() // 1024**2 if DEVICE.type == 'cuda' else 0

        # Pipeline: background thread continuously pre-reads + decodes blocks
        # GPU trains from queue without ever waiting for NVMe
        buf = queue.Queue(maxsize=4)  # deep buffer keeps NVMe saturated

        def pre_read_all():
            for bi in range(n_blocks):
                w, sz, name = pre_read_block(bi)
                if w is not None:
                    buf.put((w, sz, name, bi))
            buf.put(None)  # sentinel

        reader = threading.Thread(target=pre_read_all, daemon=True)
        reader.start()

        block_idx = 0
        while True:
            item = buf.get()
            if item is None: break
            weight_27b, sz, block_name, bi = item
            stride_mb = sz / 1e6

            block_losses = []
            for p in range(n_passes):
                # Large batch — saturate GPU compute
                idx = torch.randperm(feral.shape[0], device=DEVICE)[:256]
                if feral.shape[0] < 256:
                    # Repeat vectors if pool is small
                    reps = (256 + feral.shape[0] - 1) // feral.shape[0]
                    idx = torch.cat([torch.randperm(feral.shape[0], device=DEVICE)
                                     for _ in range(reps)])[:256]
                feral_batch = feral[idx]
                B = feral_batch.shape[0]
                with torch.no_grad():
                    proj_27b = project_27b(feral_batch, weight_27b)

                mem_token = phase_memory.unsqueeze(0).expand(B, 1, 192)
                z_in = torch.cat([feral_batch.unsqueeze(1), mem_token], dim=1)
                z_out, coh = core(z_in)
                z_feral_out = z_out[:, 0, :]
                z_mem_out = z_out[:, 1, :]

                # Update memory with fixed gate
                with torch.no_grad():
                    phase_memory.data = 0.9 * phase_memory.data + \
                                        0.1 * z_mem_out.mean(dim=0).detach()

                z_expanded = expansion(z_feral_out.real)
                dim_gate = torch.sigmoid(dim_gate_raw)
                z_gated = z_expanded * dim_gate
                core_norm = z_gated / (z_gated.norm(dim=1, keepdim=True) + 1e-8)
                proj_norm = proj_27b / (proj_27b.norm(dim=1, keepdim=True) + 1e-8)
                resonance = (core_norm * proj_norm).sum(dim=1).mean()
                loss = 1.0 - resonance + 0.001 * dim_gate.mean()
                block_losses.append(loss.item())
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(core.parameters()) + list(expansion.parameters()) + [dim_gate_raw, phase_memory], 1.0)
                opt.step()

            loss_hist.append(sum(block_losses)/len(block_losses))
            res_hist.append(resonance.item())
            del weight_27b

        final_loss = loss_hist[-1] if loss_hist else 0
        final_res = res_hist[-1] if res_hist else 0
        sweep_results[n_passes] = {'final_loss': final_loss, 'final_res': final_res,
                                    'mem_mag': phase_memory.abs().mean().item()}
        final_loss = loss_hist[-1] if loss_hist else 0
        final_res = res_hist[-1] if res_hist else 0
        final_mem = torch.cuda.memory_allocated() // 1024**2 if DEVICE.type == 'cuda' else 0
        sweep_results[n_passes] = {'final_loss': final_loss, 'final_res': final_res,
                                    'mem_mag': phase_memory.abs().mean().item()}
        catalytic = abs(final_mem - mem_baseline) < 10  # within 10MB = catalytic
        print(f"  passes={n_passes:3d}: L={final_loss:.4f} res={final_res:.4f} "
              f"|mem|={sweep_results[n_passes]['mem_mag']:.4f} "
              f"mem={final_mem}MB (baseline={mem_baseline}MB) "
              f"{'CATALYTIC' if catalytic else 'LEAK'}",
              flush=True)

    # Find optimal
    best = max(sweep_results, key=lambda g: sweep_results[g]['final_res'])
    print(f"\n[sweep] Best passes: {best} (resonance={sweep_results[best]['final_res']:.4f})")
    res_str = " | ".join(f"p={p}:res={v['final_res']:.4f}" for p,v in sweep_results.items())
    print(f"[sweep] Full: {res_str}")


if __name__ == '__main__':
    main()
