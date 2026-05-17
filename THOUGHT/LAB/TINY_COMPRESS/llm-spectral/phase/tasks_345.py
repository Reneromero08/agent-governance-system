"""Tasks 3-5: Phase-Aware Adapter Training, Budget, and Monitor."""
import json, math, sys, time, argparse
from pathlib import Path
import torch, torch.nn.functional as F, numpy as np
from scipy.signal import hilbert

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent.parent / "extensions" / "03_flat_llm"))
from flat_llm_adapter import (
    LowRankAdapter, EigenProjector, collect_activations_all_layers,
    compute_attention_output, compute_cosine_sim, compute_attention_output_train)

def load_gpt2(device="cuda"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    p = str(OUT_DIR.parent.parent / "models" / "gpt2")
    m = GPT2LMHeadModel.from_pretrained(p).to(device); m.eval()
    t = GPT2Tokenizer.from_pretrained(p)
    return m, t

def get_data(model, tokenizer, texts, device):
    """Collect activations + training QKV for all layers."""
    acts = collect_activations_all_layers(model, tokenizer, texts, device)
    qkv_per_layer = {}
    for l in range(12):
        data = []  # needs to be a list for closure
        def hk(m,i,o): sz=o.shape[-1]//3; data.append((o[...,:sz].detach(),o[...,sz:2*sz].detach(),o[...,2*sz:].detach()))
        h = model.transformer.h[l].attn.c_attn.register_forward_hook(hk)
        with torch.no_grad():
            for t in texts:
                inp = {k:v.to(device) for k,v in tokenizer(t, return_tensors='pt', truncation=True, max_length=128).items()}
                _ = model(**inp)
        h.remove()
        qkv_per_layer[l] = data
    return acts, qkv_per_layer

def train_adapter(qkv_data, acts, layer_idx, k_K, k_V, bn, epochs, lr, device, test_qkv=None, phase_lambda=0.0, ref_phases=None):
    """Train adapter. Optional phase loss term."""
    n_h, hd = 12, 64; sc = 1.0/math.sqrt(hd)
    pk = EigenProjector(768,k_K).to(device); pk.init_from_pca(acts[layer_idx]['k'])
    pv = EigenProjector(768,k_V).to(device); pv.init_from_pca(acts[layer_idx]['v'])
    ak = LowRankAdapter(k=k_K,hidden=768,bottleneck=bn,seed=42).to(device); ak.set_residual_subspace(pk.get_pca_vectors())
    av = LowRankAdapter(k=k_V,hidden=768,bottleneck=bn,seed=43).to(device); av.set_residual_subspace(pv.get_pca_vectors())
    opt = torch.optim.Adam(list(ak.parameters())+list(av.parameters()), lr=lr)
    for _ in range(epochs):
        for q,k,v in qkv_data:
            opt.zero_grad()
            kc=pk.compress(k);kp=pk.decompress(kc);vc=pv.compress(v);vp=pv.decompress(vc)
            ka=ak(kc,kp);va=av(vc,vp)
            o=compute_attention_output_train(q,k,v,n_h,sc)
            a=compute_attention_output_train(q,ka,va,n_h,sc)
            loss = F.mse_loss(a, o)
            if phase_lambda > 0 and ref_phases is not None:
                # Phase loss: preserve attention pattern on dominant cluster heads
                # ref_phases: [n_heads, head_dim] reference attention from uncompressed
                a_heads = a.view(a.shape[1], n_h, hd)  # [seq, heads, dim]
                o_heads = o.view(o.shape[1], n_h, hd)
                # Compare per-head norms as proxy for phase
                phase_loss = F.mse_loss(a_heads.norm(dim=-1), o_heads.norm(dim=-1))
                loss = loss + phase_lambda * phase_loss
            loss.backward(); opt.step()
    # Eval
    if test_qkv:
        pcos, acos = [], []
        for q,k,v in test_qkv:
            with torch.no_grad():
                kc=pk.compress(k);kp=pk.decompress(kc);vc=pv.compress(v);vp=pv.decompress(vc)
                ka=ak(kc,kp);va=av(vc,vp)
                o=compute_attention_output(q,k,v,n_h,sc)
                pcos.append(compute_cosine_sim(o, compute_attention_output(q,kp,vp,n_h,sc)))
                acos.append(compute_cosine_sim(o, compute_attention_output(q,ka,va,n_h,sc)))
        return float(np.mean(pcos)), float(np.mean(acos))
    return None, None


def task3_phase_loss(device="cuda"):
    print("="*60); print("TASK 3: Phase Coherence Loss"); print("="*60)
    model, tokenizer = load_gpt2(device)
    train_t = ["The meaning of life is a philosophical question.",
               "Deep learning enables complex pattern recognition.",
               "Artificial intelligence is transforming technology."]
    test_t = ["Space exploration has led to technological breakthroughs."]
    acts, qkv_all = get_data(model, tokenizer, train_t + test_t, device)
    train_qkv = {l: qkv_all[l][:len(train_t)] for l in range(12)}
    test_qkv = {l: qkv_all[l][len(train_t):] for l in range(12)}

    results = {}
    for k_dim in [9, 3]:
        print(f"\n--- k={k_dim} ---")
        for lam in [0.0, 0.1, 0.5, 1.0]:
            label = "baseline" if lam == 0 else f"phase_lambda={lam}"
            pcos_list, acos_list = [], []
            t0 = time.time()
            for l in range(12):
                pc, ac = train_adapter(train_qkv[l], acts, l, k_dim, k_dim, 64, 10, 1e-3, device,
                                       test_qkv[l] if l in test_qkv else None, lam, None)
                if pc is not None: pcos_list.append(pc); acos_list.append(ac)
            dt = time.time()-t0
            avg_p = np.mean(pcos_list); avg_a = np.mean(acos_list)
            print(f"  {label:20s}: PCA={avg_p:.4f} Ada={avg_a:.4f} delta={avg_a-avg_p:+.4f} dt={dt:.1f}s")
            results[f"k{k_dim}_lam{lam}"] = {"pca":float(avg_p), "ada":float(avg_a)}
    json.dump(results, open(OUT_DIR/"task3_results.json","w"), indent=2)
    print("\nPhase loss summary: check if lam>0 improves delta over baseline (lam=0)")


def task4_budget(device="cuda"):
    print("="*60); print("TASK 4: Phase-Guided Budget Allocation"); print("="*60)
    model, tokenizer = load_gpt2(device)
    train_t = ["The meaning of life is a philosophical question.",
               "Deep learning enables complex pattern recognition.",
               "Artificial intelligence is transforming technology."]
    test_t = ["Space exploration has led to technological breakthroughs."]
    acts, qkv_all = get_data(model, tokenizer, train_t + test_t, device)
    train_qkv = {l: qkv_all[l][:len(train_t)] for l in range(12)}
    test_qkv = {l: qkv_all[l][len(train_t):] for l in range(12)}

    # Phase data from Task 1: per-layer PLV
    per_layer_plv = {0:0.956,1:0.917,2:0.976,3:0.980,4:0.986,5:0.987,6:0.986,7:0.983,8:0.979,9:0.981,10:0.970,11:0.750}
    baseline_bn = 64
    baseline_params = sum((baseline_bn*9 + 768*baseline_bn + 1)*2 for _ in range(12))  # ~1.19M

    for k_dim in [9, 3]:
        print(f"\n--- k={k_dim} ---")
        strategies = {
            "uniform": {l: baseline_bn for l in range(12)},
            "phase_weighted": {},
        }
        # Phase-weighted: bottleneck proportional to PLV (inverse: low PLV = more params)
        inv_plv = {l: 1.0/(v+1e-6) for l,v in per_layer_plv.items()}
        total_inv = sum(inv_plv.values())
        param_per_layer = baseline_params / 12
        for l in range(12):
            weight = inv_plv[l] / total_inv
            target_params = weight * baseline_params
            # Solve: (bn * k + 768 * bn + 1) * 2 = target_params
            # bn = target_params / (2*(k+768)) approximately
            bn = max(8, min(256, int(target_params / (2 * (k_dim + 768)))))
            strategies["phase_weighted"][l] = bn

        for name, bns in strategies.items():
            pcos_list, acos_list = [], []
            t0 = time.time()
            for l in range(12):
                bn = bns[l]
                pc, ac = train_adapter(train_qkv[l], acts, l, k_dim, k_dim, bn, 10, 1e-3, device,
                                       test_qkv[l] if l in test_qkv else None)
                if pc is not None: pcos_list.append(pc); acos_list.append(ac)
            dt = time.time()-t0
            avg_p = np.mean(pcos_list); avg_a = np.mean(acos_list)
            total_p = sum((bns[l]*k_dim + 768*bns[l] + 1)*2 for l in range(12))
            print(f"  {name:20s}: PCA={avg_p:.4f} Ada={avg_a:.4f} delta={avg_a-avg_p:+.4f} params={total_p//1000}K dt={dt:.1f}s")
            for l in [0,5,11]:
                print(f"    L{l}: bn={bns[l]}")

    print("\nPhase-weighted budget: check if low-PLV layers (L11) benefit from larger budget")


def task5_monitor(device="cuda"):
    print("="*60); print("TASK 5: Phase Dispersion Monitor"); print("="*60)
    model, tokenizer = load_gpt2(device)
    train_t = ["The meaning of life is a philosophical question.",
               "Deep learning enables complex pattern recognition."]
    acts, qkv_all = get_data(model, tokenizer, train_t + train_t, device)
    train_qkv = {l: qkv_all[l][:len(train_t)] for l in range(12)}
    # Train adapter at k=3
    k_dim = 3
    adapters = {}
    projectors = {}
    for l in range(12):
        pk = EigenProjector(768,k_dim).to(device); pk.init_from_pca(acts[l]['k'])
        pv = EigenProjector(768,k_dim).to(device); pv.init_from_pca(acts[l]['v'])
        ak = LowRankAdapter(k=k_dim,hidden=768,bottleneck=64,seed=42).to(device); ak.set_residual_subspace(pk.get_pca_vectors())
        av = LowRankAdapter(k=k_dim,hidden=768,bottleneck=64,seed=43).to(device); av.set_residual_subspace(pv.get_pca_vectors())
        opt = torch.optim.Adam(list(ak.parameters())+list(av.parameters()), lr=1e-3)
        for _ in range(10):
            for q,k,v in train_qkv[l]:
                opt.zero_grad()
                kc=pk.compress(k);kp=pk.decompress(kc);vc=pv.compress(v);vp=pv.decompress(vc)
                ka=ak(kc,kp);va=av(vc,vp)
                o=compute_attention_output_train(q,k,v,12,1.0/math.sqrt(64))
                a=compute_attention_output_train(q,ka,va,12,1.0/math.sqrt(64))
                F.mse_loss(a,o).backward();opt.step()
        adapters[l] = (ak, av); projectors[l] = (pk, pv)

    # Monitor generation with phase dispersion tracking
    test_text = "Space exploration has led to many technological breakthroughs that benefit life on Earth."
    inp = tokenizer(test_text, return_tensors='pt', truncation=True, max_length=128)
    inp = {k:v.to(device) for k,v in inp.items()}
    seq_len = inp['input_ids'].shape[1]

    # Run with hooks to capture per-head attention at dominant layer
    dom_l = 5
    head_sigs = {h:[] for h in range(12)}
    def ahook(m,i,o):
        attn = o[0] if isinstance(o,tuple) else o
        ph = attn[0,:,:].view(-1,12,64)
        for h in range(12): head_sigs[h].extend(ph[:,h,:].norm(dim=-1).cpu().tolist())
    hh = model.transformer.h[dom_l].attn.register_forward_hook(ahook)
    with torch.no_grad(): _ = model(**inp)
    hh.remove()

    # Compute phase per head via Hilbert
    phases = {}
    for h in range(12):
        s = np.array(head_sigs[h])
        phases[h] = np.angle(hilbert(s)) if len(s)>3 else np.zeros(len(s))

    # Compute baseline mean and std of phase dispersion (from tokens 5-15)
    baseline_disp = []
    for t in range(5, min(15, seq_len)):
        pts = [phases[h][t] for h in range(12) if t<len(phases[h])]
        baseline_disp.append(float(np.std(pts)) if pts else 0)
    mean_d = np.mean(baseline_disp); std_d = np.std(baseline_disp)
    warn_t = mean_d + 2*std_d; crit_t = mean_d + 3*std_d

    # Now run through all tokens and flag warnings
    flags = []
    for t in range(seq_len):
        pts = [phases[h][t] for h in range(12) if t<len(phases[h])]
        disp = float(np.std(pts)) if pts else 0
        flag = "OK" if disp < warn_t else ("WARN" if disp < crit_t else "CRIT")
        flags.append({"token":t, "dispersion":disp, "flag":flag})

    n_warn = sum(1 for f in flags if f["flag"] in ("WARN","CRIT"))
    n_crit = sum(1 for f in flags if f["flag"]=="CRIT")
    print(f"  Baseline dispersion: {mean_d:.4f} +/- {std_d:.4f}")
    print(f"  Warning threshold: {warn_t:.4f}  Critical: {crit_t:.4f}")
    print(f"  Warnings: {n_warn}/{len(flags)}  Critical: {n_crit}/{len(flags)}")

    json.dump({"baseline_mean":float(mean_d),"baseline_std":float(std_d),
               "warn_threshold":float(warn_t),"crit_threshold":float(crit_t),
               "flags":flags}, open(OUT_DIR/"task5_monitor.json","w"), indent=2)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument("--task",type=int,choices=[3,4,5],required=True)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    {3:task3_phase_loss,4:task4_budget,5:task5_monitor}[a.task](a.device)
