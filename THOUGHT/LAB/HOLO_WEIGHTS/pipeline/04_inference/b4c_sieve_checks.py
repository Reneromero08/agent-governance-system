"""B4c-checks: Sol's five adversarial checks on the sieve==top-k finding.

1. Rank modes by participation ALONE (no singular values) - use eigenvector
   structure: IPR (localization) of the right singular vectors, the lab's
   own sensor.
2. Randomly permute mode order before tie resolution.
3. Haar-random singular vectors with the same spectrum.
4. Spearman correlation between participation (IPR) and singular value.
5. Permutation-invariance of the selected set (row permutation preserves
   singular values - does the IPR-selected set change?).

Measurement: MLP output cosine at k=256 and k=512 for each criterion,
on exact inputs - does ANY data-free criterion beat top-k?
"""
import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "pipeline" / "04_inference"))

from qwen35_holo_engine import Qwen35HoloEngine, load_original, _configure_from_dir  # noqa: E402


def ipr_selection(w: torch.Tensor, k: int) -> torch.Tensor:
    """Select k modes by eigenvector IPR (localization), NO singular values."""
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    ipr = (vh.abs() ** 4).sum(dim=-1)  # per right-singular-vector localization
    idx = torch.argsort(ipr, descending=True)[:k]
    uu, ss, vv = u[:, idx], s[idx], vh[idx, :]
    return ((uu @ torch.diag(ss)) @ vv).to(w.dtype)


def random_selection(w: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    r = torch.randperm(vh.shape[0])[:k]
    uu, ss, vv = u[:, r], s[r], vh[r, :]
    return ((uu @ torch.diag(ss)) @ vv).to(w.dtype)


def haar_selection(w: torch.Tensor, k: int) -> torch.Tensor:
    """Haar-random singular vectors with the same spectrum; select by IPR."""
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    n = vh.shape[0]
    z = torch.randn(n, n, device=w.device)
    q, _ = torch.linalg.qr(z)
    q = q[:, : vh.shape[0]]
    ipr = (q.abs() ** 4).sum(dim=-1)
    idx = torch.argsort(ipr, descending=True)[:k]
    uu, ss, vv = u[:, idx], s[idx], q[idx, :]
    return ((uu @ torch.diag(ss)) @ vv).to(w.dtype)


def topk(w: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= min(w.shape):
        return w
    u, s, vh = torch.linalg.svd(w.float(), full_matrices=False)
    return ((u[:, :k] @ torch.diag(s[:k])) @ vh[:k, :]).to(w.dtype)


def mlp_out(x: torch.Tensor, wg: torch.Tensor, wu: torch.Tensor, wd: torch.Tensor) -> torch.Tensor:
    return (torch.nn.functional.silu(x @ wg.transpose(0, 1)) * (x @ wu.transpose(0, 1))) @ wd.transpose(0, 1)


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

    print("B4c-checks: Sol's 5 adversarial checks, k=%d" % args.k)
    print("=" * 78)
    k = args.k
    agg = {"topk": [], "ipr": [], "rand": [], "haar": [], "spearman": []}
    for l in range(args.layers):
        prefix = f"model.language_model.layers.{l}.mlp"
        wg = orig.get(f"{prefix}.gate_proj.weight").to("cuda")
        wu = orig.get(f"{prefix}.up_proj.weight").to("cuda")
        wd = orig.get(f"{prefix}.down_proj.weight").to("cuda")
        x = torch.cat(caps[l]["x"]).to("cuda")
        y = mlp_out(x, wg, wu, wd)
        wg0 = wg.float()
        u, s, vh = torch.linalg.svd(wg0, full_matrices=False)
        ipr = (vh.abs() ** 4).sum(dim=-1)
        spear = torch.corrcoef(torch.stack([ipr, s]))[0, 1].item()
        agg["spearman"].append(spear)
        # check 5: permutation invariance of the IPR-selected set
        perm = torch.randperm(wg0.shape[0])
        wgp = wg0[perm]
        up, sp, vhp = torch.linalg.svd(wgp, full_matrices=False)
        iprp = (vhp.abs() ** 4).sum(dim=-1)
        set_same = torch.equal(torch.argsort(ipr, descending=True)[:k], torch.argsort(iprp, descending=True)[:k])
        for name, fn in [("topk", lambda w: topk(w, k)),
                         ("ipr", lambda w: ipr_selection(w, k)),
                         ("rand", lambda w: random_selection(w, k, seed=42)),
                         ("haar", lambda w: haar_selection(w, k))]:
            yo = mlp_out(x, fn(wg), fn(wu), wd)
            c = torch.nn.functional.cosine_similarity(y.view(1, -1), yo.view(1, -1)).item()
            agg[name].append(c)
        print(f"L{l:02d}: topk={agg['topk'][-1]:.3f} ipr={agg['ipr'][-1]:.3f} "
              f"rand={agg['rand'][-1]:.3f} haar={agg['haar'][-1]:.3f} "
              f"spearman={spear:+.3f} perm_set_same={set_same}")
    print("-" * 78)
    for name in agg:
        print(f"MEAN {name}: {sum(agg[name])/len(agg[name]):.4f}")


if __name__ == "__main__":
    main()
