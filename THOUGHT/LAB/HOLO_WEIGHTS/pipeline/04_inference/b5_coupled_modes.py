"""B5: coupled cross-matrix mode selection (Sol's unmeasured direction).

Even when no single matrix has privileged modes, the CHANNEL may live in a
JOINT subspace: gate/up multiply (MLP product), Q/K pair (attention scores),
residual addition, GDN recurrence = joint invariants. A weight-only JOINT
criterion could select coupled modes that beat independent top-k.

Criterion (data-free): the joint gram matrix over the pair -
    C = Wg^T Wg + Wu^T Wu        (input-space joint second moment)
Top-k eigenvectors of C = the joint input directions carrying the most
energy across BOTH maps. Project both maps onto that joint subspace.

Measure: MLP output cosine at joint-k vs independent top-k at the SAME
parameter budget (k modes in each map).
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_original, _configure_from_dir  # noqa: E402


def joint_project(w: torch.Tensor, vk: torch.Tensor) -> torch.Tensor:
    """Project the map onto the joint top-k input subspace V_k."""
    return (w.float() @ (vk @ vk.transpose(0, 1))).to(w.dtype)


def topk(w: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    return ((u[:, :k] @ torch.diag(s[:k])) @ vh[:k, :]).to(w.dtype)


def mlp_out(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    return (torch.nn.functional.silu(xf @ wg.float().transpose(0, 1)) * (xf @ wu.float().transpose(0, 1))) @ wd.float().transpose(0, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/run/media/reneshizzle/860_1/Reneshizzle/Apps/LM Studio/Qwen/Qwen3.5-4B")
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--k", type=int, default=512)
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

    print("B5: coupled gate/up joint-subspace selection vs independent top-k (k=%d)" % args.k)
    print("=" * 78)
    k = args.k
    agg_ind, agg_joint = [], []
    for l in range(args.layers):
        prefix = f"model.language_model.layers.{l}.mlp"
        wg = orig.get(f"{prefix}.gate_proj.weight").to("cuda").float()
        wu = orig.get(f"{prefix}.up_proj.weight").to("cuda").float()
        wd = orig.get(f"{prefix}.down_proj.weight").to("cuda").float()
        x = torch.cat(caps[l]["x"]).to("cuda")
        y = mlp_out(x, wg, wu, wd)
        # independent top-k
        yi = mlp_out(x, topk(wg, k), topk(wu, k), wd)
        ci = torch.nn.functional.cosine_similarity(y.view(1, -1), yi.view(1, -1)).item()
        # joint gram: C = Wg^T Wg + Wu^T Wu
        cg = wg.transpose(0, 1) @ wg
        cu = wu.transpose(0, 1) @ wu
        evals, evecs = torch.linalg.eigh(cg + cu)
        vk = evecs[:, -k:]  # top-k joint input directions
        wg_j = joint_project(wg, vk)
        wu_j = joint_project(wu, vk)
        yj = mlp_out(x, wg_j, wu_j, wd)
        cj = torch.nn.functional.cosine_similarity(y.view(1, -1), yj.view(1, -1)).item()
        agg_ind.append(ci); agg_joint.append(cj)
        # joint energy share
        eg = (wg_j.float() ** 2).sum().item() / (wg**2).sum().item()
        eu = (wu_j.float() ** 2).sum().item() / (wu**2).sum().item()
        print(f"L{l:02d}: independent={ci:.3f}  joint={cj:.3f}  joint_energy g={eg:.3f} u={eu:.3f}")
    print("-" * 78)
    print(f"MEAN: independent={sum(agg_ind)/len(agg_ind):.4f}  joint={sum(agg_joint)/len(agg_joint):.4f}")


if __name__ == "__main__":
    main()
