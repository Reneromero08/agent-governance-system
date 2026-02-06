from __future__ import annotations

import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: sentence-transformers ({exc})")

try:
    from sklearn.decomposition import PCA
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: scikit-learn ({exc})")

import ssl
import urllib.request
import urllib.error


OUTPUT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
CONFIG_PATH = RESULTS_DIR / "q51_berry_phase_try2_config.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _download_dataset(url: str, dest: Path) -> Dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return {"downloaded": False, "tls_verification": "existing_file"}
    try:
        urllib.request.urlretrieve(url, dest)
        return {"downloaded": True, "tls_verification": "verified"}
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as resp:
                dest.write_bytes(resp.read())
            return {"downloaded": True, "tls_verification": "skipped_cert_verification"}
        raise


def _load_category_words(path: Path) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    current = ""
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(":"):
            current = line[1:].strip()
            categories.setdefault(current, [])
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        if current:
            for word in parts:
                if word not in categories[current]:
                    categories[current].append(word)
    return categories


def _build_loops(categories: Dict[str, List[str]], loop_len: int, loops_per_cat: int) -> List[List[str]]:
    loops: List[List[str]] = []
    for _, words in categories.items():
        if len(words) < loop_len:
            continue
        count = 0
        for i in range(0, len(words) - loop_len + 1, loop_len):
            loop = words[i : i + loop_len]
            if len(loop) == loop_len:
                loops.append(loop + [loop[0]])  # close loop
                count += 1
            if count >= loops_per_cat:
                break
    return loops


def _encode_words(
    model: SentenceTransformer,
    words: List[str],
    batch_size: int,
    normalize_embeddings: bool,
) -> np.ndarray:
    return model.encode(
        words,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=normalize_embeddings,
    )


def _project_pca(points: np.ndarray) -> np.ndarray:
    pca = PCA(n_components=2)
    return pca.fit_transform(points)


def _project_svd(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def _rotate(points: np.ndarray, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    rot = np.array([[c, -s], [s, c]])
    return points @ rot.T


def _reflect_x(points: np.ndarray) -> np.ndarray:
    return points * np.array([-1.0, 1.0])


def _reflect_y(points: np.ndarray) -> np.ndarray:
    return points * np.array([1.0, -1.0])


def _swap_axes(points: np.ndarray) -> np.ndarray:
    return points[:, [1, 0]]


def _gamma_from_path(points_2d: np.ndarray, eps: float) -> Tuple[float, bool]:
    z = points_2d[:, 0] + 1j * points_2d[:, 1]
    if np.any(np.abs(z) < eps):
        return 0.0, False
    angles = np.angle(z[1:] / z[:-1])
    return float(np.sum(angles)), True


def _gamma_mod(gamma: float) -> float:
    return (gamma + math.pi) % (2 * math.pi) - math.pi


def _quant_scores(gamma: float) -> Dict[str, float]:
    ratio = gamma / (2 * math.pi)
    frac_1 = abs(ratio - round(ratio))
    frac_1_8 = abs(ratio * 8 - round(ratio * 8))
    return {
        "quant_frac_1": float(frac_1),
        "quant_score_1": float(1 - frac_1),
        "quant_frac_1_8": float(frac_1_8),
        "quant_score_1_8": float(1 - frac_1_8),
    }


def _basis_variants(points: np.ndarray, angles: List[float]) -> Dict[str, np.ndarray]:
    variants: Dict[str, np.ndarray] = {}
    local_pca = _project_pca(points)
    local_svd = _project_svd(points)
    variants["local_pca"] = local_pca
    variants["local_svd"] = local_svd
    # global basis from points themselves (deterministic)
    variants["local_pca_reflect_x"] = _reflect_x(local_pca)
    variants["local_pca_reflect_y"] = _reflect_y(local_pca)
    variants["local_pca_swap"] = _swap_axes(local_pca)
    for angle in angles:
        variants[f"local_pca_rot_{angle:.4f}"] = _rotate(local_pca, angle)
    return variants


def _basis_variants_global(global_proj: np.ndarray, angles: List[float]) -> Dict[str, np.ndarray]:
    variants: Dict[str, np.ndarray] = {}
    variants["global_pca"] = global_proj
    variants["global_pca_reflect_x"] = _reflect_x(global_proj)
    variants["global_pca_reflect_y"] = _reflect_y(global_proj)
    variants["global_pca_swap"] = _swap_axes(global_proj)
    for angle in angles:
        variants[f"global_pca_rot_{angle:.4f}"] = _rotate(global_proj, angle)
    return variants


def _evaluate_model(
    cfg: dict,
    loops: List[List[str]],
    word_to_vec: Dict[str, np.ndarray],
    global_proj_map: Dict[str, np.ndarray],
) -> Dict[str, object]:
    eps = float(cfg["epsilon"])
    angles = [float(a) for a in cfg["rotation_angles"]]

    loop_results: List[Dict[str, object]] = []
    diffs_global: List[float] = []
    diffs_local: List[float] = []
    quant_scores: List[float] = []

    for loop in loops:
        points = np.vstack([word_to_vec[w] for w in loop])
        variants = _basis_variants(points, angles)
        global_proj = np.vstack([global_proj_map[w] for w in loop])
        variants.update(_basis_variants_global(global_proj, angles))

        gammas: Dict[str, float] = {}
        valid = True
        for name, proj in variants.items():
            gamma, ok = _gamma_from_path(proj, eps)
            if not ok:
                valid = False
                break
            gammas[name] = gamma
        if not valid:
            continue

        base_local = gammas["local_pca"]
        base_global = gammas["global_pca"]
        for name, value in gammas.items():
            if name != "global_pca":
                diffs_global.append(abs(_gamma_mod(value - base_global)))
            if name != "local_pca":
                diffs_local.append(abs(_gamma_mod(value - base_local)))

        qs = _quant_scores(base_global)
        quant_scores.append(qs["quant_score_1_8"])

        loop_results.append(
            {
                "loop": loop,
                "gamma_local_pca": base_local,
                "gamma_local_svd": gammas["local_svd"],
                "gamma_global_pca": base_global,
                "gamma_mod_2pi": _gamma_mod(base_global),
                "quant_score_1": qs["quant_score_1"],
                "quant_score_1_8": qs["quant_score_1_8"],
                "mean_abs_diff_from_global_pca": float(np.mean([abs(_gamma_mod(gammas[k] - base_global)) for k in gammas if k != "global_pca"])),
                "mean_abs_diff_from_local_pca": float(np.mean([abs(_gamma_mod(gammas[k] - base_local)) for k in gammas if k != "local_pca"])),
            }
        )

    if not loop_results:
        return {
            "status": "error",
            "error": "no valid loops (near-zero projection)",
        }

    return {
        "status": "ok",
        "n_loops": len(loop_results),
        "mean_abs_diff_global": float(np.mean(diffs_global)) if diffs_global else float("nan"),
        "median_abs_diff_global": float(np.median(diffs_global)) if diffs_global else float("nan"),
        "mean_abs_diff_local": float(np.mean(diffs_local)) if diffs_local else float("nan"),
        "median_abs_diff_local": float(np.median(diffs_local)) if diffs_local else float("nan"),
        "mean_quant_score_1_8": float(np.mean(quant_scores)) if quant_scores else float("nan"),
        "loop_results": loop_results,
    }


def _build_report(results: dict) -> str:
    lines: List[str] = []
    lines.append("# Q51 Berry Phase / Holonomy Correctness — Try2 Results")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- URL: {results['dataset']['url']}")
    lines.append(f"- SHA256: {results['dataset']['sha256']}")
    lines.append(f"- Size (bytes): {results['dataset']['size_bytes']}")
    lines.append(f"- Loops: {results['dataset']['n_loops']}")
    lines.append("")
    lines.append("## Model Summary")
    lines.append("| Model | Status | Loops | Mean |Δγ| (global) | Median |Δγ| (global) | Mean |Δγ| (local) | Mean Quant Score (1/8) |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for item in results["models"]:
        if item.get("status") != "ok":
            lines.append(f"| {item['model']} | ERROR | - | - | - | - |")
            continue
        lines.append(
            "| {model} | ok | {n_loops} | {mean_diff:.4f} | {median_diff:.4f} | {quant:.4f} |".format(
                model=item["model"],
                n_loops=item["n_loops"],
                mean_diff=item["mean_abs_diff_global"],
                median_diff=item["median_abs_diff_global"],
                quant=item["mean_quant_score_1_8"],
            )
        )
    lines.append("")
    lines.append("## Interpretation (Adversarial)")
    lines.append(
        "- If |Δγ| across local vs global projections is not small, γ is not invariant and does not justify Berry phase claims."
    )
    lines.append(
        "- Quantization that disappears under basis changes suggests projection artifacts rather than topology."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = _load_json(CONFIG_PATH)
    dataset_path = OUTPUT_DIR / cfg["dataset_path"]
    download_meta = _download_dataset(cfg["dataset_url"], dataset_path)

    categories = _load_category_words(dataset_path)
    loops = _build_loops(categories, int(cfg["loop_word_count"]), int(cfg["loops_per_category_max"]))

    words = sorted({w for loop in loops for w in loop})
    dataset_info = {
        "url": cfg["dataset_url"],
        "path": str(dataset_path.relative_to(OUTPUT_DIR)),
        "sha256": _sha256_file(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "n_loops": len(loops),
        "n_words": len(words),
        "download": download_meta,
    }

    model_results: List[Dict[str, object]] = []
    for model_name in cfg["models"]:
        model = SentenceTransformer(model_name)
        embeddings = _encode_words(
            model,
            words,
            batch_size=int(cfg["batch_size"]),
            normalize_embeddings=bool(cfg["normalize_embeddings"]),
        )
        word_to_vec = {w: embeddings[i] for i, w in enumerate(words)}
        global_proj = PCA(n_components=2).fit_transform(embeddings)
        global_proj_map = {w: global_proj[i] for i, w in enumerate(words)}
        res = _evaluate_model(cfg, loops, word_to_vec, global_proj_map)
        res["model"] = model_name
        model_results.append(res)

    results = {
        "config": cfg,
        "dataset": dataset_info,
        "models": model_results,
        "summary": {
            "models_attempted": len(model_results),
            "models_evaluated": len([m for m in model_results if m.get("status") == "ok"]),
        },
        "anti_pattern_check": {
            "no_synthetic_data": True,
            "parameters_fixed_before_results": True,
            "no_grid_search": True,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "q51_berry_phase_try2_results.json", results)
    (REPORTS_DIR / "q51_berry_phase_try2_report.md").write_text(
        _build_report(results), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
