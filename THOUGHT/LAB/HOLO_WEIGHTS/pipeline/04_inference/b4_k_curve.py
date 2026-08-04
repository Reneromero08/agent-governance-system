"""B4a: the k-curve of the per-layer MLP - where does the ranking channel die?

For each layer, truncate the exact gate/up/down weights to rank k via SVD,
compute the MLP output on the EXACT input trajectory, measure cosine vs the
exact MLP output. Full rank must approach 1.0 (SVD is lossless); the curve
shows how much rank the OUTPUT DIRECTION actually needs - vs the 165-modes
needed for error energy. This maps the non-collapse representation space:
if the channel needs near-full rank, no low-rank compression can hold it
and B4c (cavity sieve) is the only candidate.
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_original, _configure_from_dir  # noqa: E402

KS = [64, 128, 256, 512, 1024, 2048, 0]  # 0 = full rank


def truncate(w: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    return ((u[:, :k] @ torch.diag(s[:k])) @ vh[:k, :]).to(w.dtype)


def mlp_out(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    gate = x @ wg.transpose(0, 1)
    up = x @ wu.transpose(0, 1)
    return (torch.nn.functional.silu(gate) * up) @ wd.transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", type=int, default=8)
    ap.add_argument("--layers", type=int, default=32)
    args = ap.parse_args()
    from transformers import AutoTokenizer

    MD = Path(args.model_dir)
    _configure_from_dir(MD)
    orig = load_original(MD)
    tok = AutoTokenizer.from_pretrained(MD, trust_remote_code=True)
    exact = Qwen35HoloEngine(None, orig, exact=True, device="cuda", verbose=False)

    lines = [l.strip() for l in (REPO / "config" / "corpus.txt").read_text().splitlines() if l.strip()]
    lines = lines[: args.prompts]

    caps = {l: {"x": []} for l in range(args.layers)}
    with torch.no_grad():
        for ln in lines:
            ids = tok(ln, return_tensors="pt")["input_ids"][0]
            hidden = exact._embed(ids.unsqueeze(0))
            for l in range(args.layers):
                prefix = f"model.language_model.layers.{l}"
                norm = exact._exact_weight(f"{prefix}.input_layernorm.weight").to("cuda")
                n_in = exact._rms_offset(hidden, norm)
                mixed = exact._prefill_layer_mixer(l, n_in, None)
                hidden = hidden + mixed.to(hidden.dtype)
                post = exact._exact_weight(f"{prefix}.post_attention_layernorm.weight").to("cuda")
                mlp_in = exact._rms_offset(hidden, post)
                caps[l]["x"].append(mlp_in[0].cpu())
                hidden = hidden + exact._mlp(mlp_in, l, None).to(hidden.dtype)

    print("B4a: per-layer MLP output cosine vs exact, as a function of rank k")
    print("     (exact inputs; SVD-truncated exact weights)")
    print("=" * 78)
    acc = {k: [] for k in KS}
    for l in range(args.layers):
        prefix = f"model.language_model.layers.{l}.mlp"
        wg = orig.get(f"{prefix}.gate_proj.weight").to("cuda")
        wu = orig.get(f"{prefix}.up_proj.weight").to("cuda")
        wd = orig.get(f"{prefix}.down_proj.weight").to("cuda")
        x = torch.cat(caps[l]["x"]).to("cuda")
        y = mlp_out(x, wg, wu, wd)
        row = []
        for k in KS:
            yo = mlp_out(x, truncate(wg, k), truncate(wu, k), truncate(wd, k))
            c = torch.nn.functional.cosine_similarity(y.view(1, -1), yo.view(1, -1)).item()
            acc[k].append(c)
            row.append(c)
        print(f"L{l:02d}: " + "  ".join(f"k={k or 'full'}={v:.3f}" for k, v in zip(KS, row)))
    print("-" * 78)
    for k in KS:
        print(f"MEAN k={k or 'full'}: {sum(acc[k])/len(acc[k]):.4f}")


if __name__ == "__main__":
    main()
