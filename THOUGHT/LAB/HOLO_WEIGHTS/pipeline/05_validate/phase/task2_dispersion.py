# Task 2: Phase Dispersion Early-Warning
import json, math, sys
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT_DIR.parent.parent / "extensions" / "03_flat_llm"))
import torch, torch.nn.functional as F, numpy as np
from scipy.signal import hilbert
from flat_llm_adapter import (
    LowRankAdapter, EigenProjector, collect_activations_all_layers,
    compute_attention_output, compute_cosine_sim, compute_attention_output_train)

def load_gpt2(device="cpu"):
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    p = str(OUT_DIR.parent.parent / "models" / "gpt2")
    return GPT2LMHeadModel.from_pretrained(p).to(device), GPT2Tokenizer.from_pretrained(p)

def task2(device="cpu"):
    print("="*60); print("TASK 2: Phase Dispersion Early-Warning"); print("="*60)
    model, tokenizer = load_gpt2(device); model.eval()
    n_heads, n_layers, hd = 12, model.config.n_layer, model.config.n_embd//12
    dom_l = 5; scale = 1.0/math.sqrt(hd)
    texts = ["The meaning of life is a philosophical question.",
             "Artificial intelligence is transforming technology.",
             "Deep learning enables complex pattern recognition."]
    acts = collect_activations_all_layers(model, tokenizer, texts, device)

    for k_dim in [9, 3]:
        comp = 768/k_dim
        print(f"\n--- k={k_dim} ({comp:.1f}x) ---")
        # Train adapter
        pk = EigenProjector(768,k_dim).to(device); pk.init_from_pca(acts[dom_l]['k'])
        pv = EigenProjector(768,k_dim).to(device); pv.init_from_pca(acts[dom_l]['v'])
        ak = LowRankAdapter(k=k_dim,hidden=768,bottleneck=64,seed=42).to(device); ak.set_residual_subspace(pk.get_pca_vectors())
        av = LowRankAdapter(k=k_dim,hidden=768,bottleneck=64,seed=43).to(device); av.set_residual_subspace(pv.get_pca_vectors())

        tq = []  # training QKV
        def hk(m,i,o): sz=o.shape[-1]//3; tq.append((o[...,:sz].detach(),o[...,sz:2*sz].detach(),o[...,2*sz:].detach()))
        hh = model.transformer.h[dom_l].attn.c_attn.register_forward_hook(hk)
        with torch.no_grad():
            for t in texts: _ = model(**{k:v.to(device) for k,v in tokenizer(t, return_tensors='pt', truncation=True, max_length=128).items()})
        hh.remove()

        opt = torch.optim.Adam(list(ak.parameters())+list(av.parameters()), lr=1e-3)
        for _ in range(10):
            for q,k,v in tq:
                opt.zero_grad()
                kc=pk.compress(k);kp=pk.decompress(kc);vc=pv.compress(v);vp=pv.decompress(vc)
                ka=ak(kc,kp);va=av(vc,vp)
                o=compute_attention_output_train(q,k,v,n_heads,scale)
                a=compute_attention_output_train(q,ka,va,n_heads,scale)
                F.mse_loss(a,o).backward();opt.step()

        # Per-token measurement with Hilbert
        inp = tokenizer(texts[0], return_tensors='pt', truncation=True, max_length=128).to(device)
        seq_len = inp['input_ids'].shape[1]

        # Collect per-head signals across all tokens
        head_sigs = {l:{h:[] for h in range(n_heads)} for l in range(n_layers)}
        def ahook(l):
            def hook(m,i,o):
                attn = o[0] if isinstance(o,tuple) else o
                ph = attn[0,:,:].view(-1,n_heads,hd)  # [seq,heads,dim]
                for h in range(n_heads): head_sigs[l][h].extend(ph[:,h,:].norm(dim=-1).cpu().tolist())
            return hook
        hooks = [model.transformer.h[l].attn.register_forward_hook(ahook(l)) for l in range(n_layers)]
        dqkv_data = []  # mutable container
        def qkvhook(m,i,o): sz=o.shape[-1]//3; dqkv_data.append((o[...,:sz].detach(),o[...,sz:2*sz].detach(),o[...,2*sz:].detach()))
        dh = model.transformer.h[dom_l].attn.c_attn.register_forward_hook(qkvhook)
        with torch.no_grad(): _ = model(**inp)
        for h in hooks: h.remove(); dh.remove()
        dqkv = dqkv_data[0] if dqkv_data else None

        if dqkv is None: continue
        qf,kf,vf = dqkv; seq = qf.shape[1]

        # Hilbert per head
        phases = {l:{} for l in range(n_layers)}
        for l in range(n_layers):
            for h in range(n_heads):
                s = np.array(head_sigs[l][h])
                phases[l][h] = np.angle(hilbert(s)) if len(s)>3 else np.zeros(len(s))

        per_token = []
        for t in range(5, seq):
            qt=qf[:,t:t+1,:]; kt=kf[:,:t+1,:]; vt=vf[:,:t+1,:]
            with torch.no_grad():
                kc=pk.compress(kt);kp=pk.decompress(kc);vc=pv.compress(vt);vp=pv.decompress(vc)
                ka=ak(kc,kp);va=av(vc,vp)
                ac=compute_cosine_sim(compute_attention_output(qt,kt,vt,n_heads,scale),
                                       compute_attention_output(qt,ka,va,n_heads,scale))
            # Phase dispersion at this token across all heads
            pts = [phases[l][h][t] for l in range(n_layers) for h in range(n_heads) if t<len(phases[l][h])]
            pd = float(np.std(pts)) if pts else 0.0
            per_token.append({"token":t,"attn_cos":float(ac),"phase_disp":pd})

        if len(per_token)>3:
            a_seq = np.array([p["attn_cos"] for p in per_token])
            d_seq = np.array([p["phase_disp"] for p in per_token])
            an = (a_seq-np.mean(a_seq))/max(np.std(a_seq),1e-6)
            dn = (d_seq-np.mean(d_seq))/max(np.std(d_seq),1e-6)
            cc = np.correlate(dn, an, mode='full'); mid=len(cc)//2
            leads = cc[mid-1]>cc[mid] if mid>=1 else False
            print(f"  attn_cos mean={np.mean(a_seq):.4f}  phase_disp mean={np.mean(d_seq):.4f}")
            print(f"  CC lag 0: {cc[mid]:.3f}  lag -1: {cc[mid-1]:.3f}  {'PHASE LEADS' if leads else 'NO LEAD'}")

        json.dump({"k":k_dim,"per_token":per_token}, open(OUT_DIR/f"task2_k{k_dim}.json",'w'), indent=2)
    return None

if __name__=='__main__': task2()
