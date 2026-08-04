"""B4c: cavity-sieve test on attention projections (the lab's compression claim).

The k-curve proved the MLP channel needs near-full rank. The lab's sieve claim
was never about MLP: "cavity-sieved K=49 optimal for standard transformers;
K/V projections Df 160-460." Test: does the ATTENTION output channel survive
selective mode deletion (participation-based sieve, data-free) where arbitrary
top-k truncation fails? This defines the minimal non-collapse architecture:
sieved attention + full-rank MLP.

Selection criteria (both data-free):
  top-k:   by singular value (the collapsed baseline)
  sieve-k: by participation ratio - modes kept in order of their fractional
           energy contribution to the STRUCTURED spectrum (the lab's Df law),
           i.e. the Df-ranked selection: keep modes until cumulative
           participation reaches a threshold, preferring structured modes.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_original, _configure_from_dir  # noqa: E402


def truncate_topk(w: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    return ((u[:, :k] @ torch.diag(s[:k])) @ vh[:k, :]).to(w.dtype)


def truncate_sieve(w: torch.Tensor, k: int) -> torch.Tensor:
    """Participation-selected truncation: keep the k modes with the largest
    per-mode participation share of the total structured energy. Data-free."""
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    p = (s**2) / (s**2).sum()
    idx = torch.argsort(p, descending=True)[:k]
    uu = u[:, idx]
    ss = s[idx]
    vv = vh[idx, :]
    return ((uu @ torch.diag(ss)) @ vv).to(w.dtype)


def attn_forward(exact, x, l, wq, wo):
    """Faithful GQA full-attention through the given q/o projections."""
    from qwen35_holo_engine import FULL_Q_HEADS, FULL_KV_HEADS, FULL_HEAD_DIM
    prefix = f"model.language_model.layers.{l}.self_attn"
    batch, seq, _ = x.shape
    q_gate = (x @ wq.transpose(0, 1)).view(batch, seq, FULL_Q_HEADS, 2 * FULL_HEAD_DIM)
    q, gate = q_gate.chunk(2, dim=-1)
    gate = gate.reshape(batch, seq, -1)
    k = exact._linear(x, f"{prefix}.k_proj.weight", factorized=False, adapters=None).view(
        batch, seq, FULL_KV_HEADS, FULL_HEAD_DIM)
    v = exact._linear(x, f"{prefix}.v_proj.weight", factorized=False, adapters=None).view(
        batch, seq, FULL_KV_HEADS, FULL_HEAD_DIM)
    qn = exact._exact_weight(f"{prefix}.q_norm.weight").to(x.device)
    kn = exact._exact_weight(f"{prefix}.k_norm.weight").to(x.device)
    q = exact._rms_offset(q, qn)
    k = exact._rms_offset(k, kn)
    positions = torch.arange(seq, device=x.device)
    q, k = exact._rope(q.float(), k.float(), positions)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2).repeat_interleave(FULL_Q_HEADS // FULL_KV_HEADS, 1)
    v = v.float().transpose(1, 2).repeat_interleave(FULL_Q_HEADS // FULL_KV_HEADS, 1)
    scores = (q @ k.transpose(-2, -1)) * (FULL_HEAD_DIM**-0.5)
    causal = torch.ones(seq, seq, dtype=torch.bool, device=x.device).triu(1)
    scores = scores.masked_fill(causal, float("-inf"))
    prob = torch.softmax(scores, dim=-1, dtype=torch.float32)
    att = (prob @ v).transpose(1, 2).contiguous()
    att = att.reshape(batch, seq, -1) * torch.sigmoid(gate.float())
    return (att.to(wo.dtype) @ wo.transpose(0, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--stride", type=int, default=4, help="full-attention stride")
    args = ap.parse_args()
    from transformers import AutoTokenizer

    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "corpus.txt").read_text().splitlines() if l.strip()]
    lines = lines[: args.prompts]

    # capture exact per-layer normed inputs at full-attention layers
    caps = {}
    with torch.no_grad():
        for ln in lines:
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            hidden = exact._embed(ids.unsqueeze(0))
            for l in range(args.layers):
                prefix = f"model.language_model.layers.{l}"
                norm = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
                n_in = exact._rms_offset(hidden, norm)
                if l % args.stride == 3:
                    caps.setdefault(l, []).append(n_in[0].cpu())
                mixed = exact._prefill_layer_mixer(l, n_in, None)
                hidden = hidden + mixed.to(hidden.dtype)
                post = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
                mlp_in = exact._rms_offset(hidden, post)
                hidden = hidden + exact._mlp(mlp_in, l, None).to(hidden.dtype)

    print("B4c: attention mixer output cosine vs exact - top-k vs sieve-k")
    print("     (exact inputs; q/o projections truncated by each criterion)")
    print("=" * 78)
    ranks = [64, 128, 256, 512, 1024, 2048, 0]
    acc_top = {k: [] for k in ranks}
    acc_sie = {k: [] for k in ranks}
    for l in sorted(caps):
        prefix = f"model.language_model.layers.{l}.self_attn"
        wq = orig.get(f"{prefix}.q_proj.weight").to("cuda")
        wo = orig.get(f"{prefix}.o_proj.weight").to("cuda")
        x = torch.cat(caps[l]).to("cuda").unsqueeze(1)
        y = attn_forward(exact, x, l, wq, wo)
        row_t, row_s = [], []
        for k in ranks:
            yt = attn_forward(exact, x, l, truncate_topk(wq, k), truncate_topk(wo, k))
            ys = attn_forward(exact, x, l, truncate_sieve(wq, k), truncate_sieve(wo, k))
            ct = torch.nn.functional.cosine_similarity(y.view(1, -1), yt.view(1, -1)).item()
            cs = torch.nn.functional.cosine_similarity(y.view(1, -1), ys.view(1, -1)).item()
            acc_top[k].append(ct); acc_sie[k].append(cs)
            row_t.append(ct); row_s.append(cs)
        print(f"L{l:02d}: top-k " + " ".join(f"{v:.3f}" for v in row_t))
        print(f"       sieve " + " ".join(f"{v:.3f}" for v in row_s))
    print("-" * 78)
    for k in ranks:
        print(f"MEAN k={k or 'full'}: top-k={sum(acc_top[k])/len(acc_top[k]):.4f}  "
              f"sieve={sum(acc_sie[k])/len(acc_sie[k]):.4f}")


if __name__ == "__main__":
    main()
