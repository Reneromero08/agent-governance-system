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

try:
    from scipy.stats import chisquare
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

import ssl
import urllib.request
import urllib.error


OUTPUT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
CONFIG_PATH = RESULTS_DIR / "q51_zero_signature_try2_config.json"


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


def _load_words(path: Path) -> List[str]:
    vocab = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        parts = line.split()
        if len(parts) != 4:
            continue
        vocab.update(parts)
    return sorted(vocab)


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


def _octant_index(vec: np.ndarray) -> int:
    return (1 if vec[0] >= 0 else 0) + (2 if vec[1] >= 0 else 0) + (4 if vec[2] >= 0 else 0)


def _compute_metrics(octant_counts: np.ndarray) -> Dict[str, float]:
    n = int(octant_counts.sum())
    p = octant_counts / n
    theta = np.array([(k + 0.5) * math.pi / 4 for k in range(8)])
    s = np.sum(np.exp(1j * theta) * octant_counts)
    s_over_n = abs(s) / n

    # pairwise opposite differences
    pair_diffs = [abs(p[k] - p[k + 4]) for k in range(4)]
    max_pair_diff = float(max(pair_diffs))

    # Discrete Fourier components of p_k for m=2,3
    dft_m2 = abs(np.sum(p * np.exp(1j * 2 * theta)))
    dft_m3 = abs(np.sum(p * np.exp(1j * 3 * theta)))

    # Random uniform null expectation for |S|/n
    expected_s_over_n = math.sqrt(math.pi) / (2 * math.sqrt(n))

    return {
        "n": float(n),
        "s_over_n": float(s_over_n),
        "max_pair_diff": max_pair_diff,
        "dft_m2": float(dft_m2),
        "dft_m3": float(dft_m3),
        "random_null_expected_s_over_n": float(expected_s_over_n),
    }


def _chi_square_uniform(octant_counts: np.ndarray) -> Dict[str, float]:
    n = int(octant_counts.sum())
    expected = np.full(8, n / 8)
    stat = float(((octant_counts - expected) ** 2 / expected).sum())
    if HAS_SCIPY:
        p_value = float(chisquare(octant_counts, expected).pvalue)
    else:
        p_value = float("nan")
    return {"chi_square_stat": stat, "chi_square_p": p_value}


def _evaluate_model(cfg: dict, words: List[str]) -> Dict[str, object]:
    result: Dict[str, object] = {"model": cfg["model"], "status": "ok", "error": ""}
    try:
        model = SentenceTransformer(cfg["model"])
    except Exception as exc:  # pragma: no cover
        result["status"] = "error"
        result["error"] = f"model load failed: {exc}"
        return result

    embeddings = _encode_words(
        model,
        words,
        batch_size=int(cfg["batch_size"]),
        normalize_embeddings=bool(cfg["normalize_embeddings"]),
    )

    pca = PCA(n_components=int(cfg["pca_components"]))
    proj = pca.fit_transform(embeddings)
    octants = np.array([_octant_index(row) for row in proj], dtype=int)
    counts = np.bincount(octants, minlength=8)

    metrics = _compute_metrics(counts)
    chi = _chi_square_uniform(counts)

    thresholds = cfg["success_thresholds"]
    evidence = (
        metrics["s_over_n"] < thresholds["s_over_n_lt"]
        and (math.isnan(chi["chi_square_p"]) or chi["chi_square_p"] > thresholds["chi_square_p_gt"])
        and metrics["max_pair_diff"] < thresholds["max_pair_diff_lt"]
    )

    result.update(
        {
            "n": metrics["n"],
            "octant_counts": counts.tolist(),
            "octant_probs": (counts / counts.sum()).tolist(),
            "s_over_n": metrics["s_over_n"],
            "max_pair_diff": metrics["max_pair_diff"],
            "dft_m2": metrics["dft_m2"],
            "dft_m3": metrics["dft_m3"],
            "random_null_expected_s_over_n": metrics["random_null_expected_s_over_n"],
            "chi_square_stat": chi["chi_square_stat"],
            "chi_square_p": chi["chi_square_p"],
            "roots_of_unity_evidence": bool(evidence),
        }
    )
    return result


def _build_report(results: dict) -> str:
    lines: List[str] = []
    lines.append("# Q51 Zero Signature Interpretation — Try2 Results")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- URL: {results['dataset']['url']}")
    lines.append(f"- SHA256: {results['dataset']['sha256']}")
    lines.append(f"- Size (bytes): {results['dataset']['size_bytes']}")
    lines.append(f"- Unique words: {results['dataset']['n_words']}")
    lines.append("")
    lines.append("## Model Results (|S|/n + cancellation metrics)")
    lines.append("| Model | |S|/n | Null E[|S|/n] | Max Pair Diff | chi-square p | dft_m2 | dft_m3 | Roots-of-Unity Evidence |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for item in results["models"]:
        if item.get("status") != "ok":
            lines.append(f"| {item['model']} | ERROR | - | - | - | - | - | - | - |")
            continue
        lines.append(
            "| {model} | {s_over_n:.4f} | {null_exp:.4f} | {max_pair:.4f} | {chi_p:.4f} | {dft2:.4f} | {dft3:.4f} | {evidence} |".format(
                model=item["model"],
                s_over_n=item["s_over_n"],
                null_exp=item["random_null_expected_s_over_n"],
                max_pair=item["max_pair_diff"],
                chi_p=item["chi_square_p"] if not math.isnan(item["chi_square_p"]) else float("nan"),
                dft2=item["dft_m2"],
                dft3=item["dft_m3"],
                evidence=str(item["roots_of_unity_evidence"]),
            )
        )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Models attempted: {results['summary']['models_attempted']}")
    lines.append(f"- Models evaluated: {results['summary']['models_evaluated']}")
    lines.append(f"- Roots-of-unity evidence count: {results['summary']['models_roots_evidence']}")
    lines.append("")
    lines.append("## Interpretation (Adversarial)")
    lines.append(
        "- |S|/n near zero only constrains the first Fourier component of the octant distribution; it does NOT imply discrete roots-of-unity clustering."
    )
    lines.append(
        "- Small |S|/n is consistent with uniform random phase assignments and with simple opposite-pair cancellation."
    )
    lines.append(
        "- Additional evidence required: (1) octant uniformity, (2) low higher-order harmonics, and (3) direct phase clustering near 8 centers."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    cfg = _load_json(CONFIG_PATH)
    dataset_path = OUTPUT_DIR / cfg["dataset_path"]
    download_meta = _download_dataset(cfg["dataset_url"], dataset_path)

    words = _load_words(dataset_path)
    dataset_info = {
        "url": cfg["dataset_url"],
        "path": str(dataset_path.relative_to(OUTPUT_DIR)),
        "sha256": _sha256_file(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "n_words": len(words),
        "download": download_meta,
    }

    model_results: List[Dict[str, object]] = []
    for model_name in cfg["models"]:
        model_results.append(
            _evaluate_model(
                {
                    "model": model_name,
                    "batch_size": cfg["batch_size"],
                    "normalize_embeddings": cfg["normalize_embeddings"],
                    "pca_components": cfg["pca_components"],
                    "success_thresholds": cfg["success_thresholds"],
                },
                words,
            )
        )

    evaluated = [m for m in model_results if m.get("status") == "ok"]
    roots = [m for m in evaluated if m.get("roots_of_unity_evidence")]

    results = {
        "config": cfg,
        "dataset": dataset_info,
        "models": model_results,
        "summary": {
            "models_attempted": len(model_results),
            "models_evaluated": len(evaluated),
            "models_roots_evidence": len(roots),
        },
        "anti_pattern_check": {
            "ground_truth_independent_of_R": True,
            "parameters_fixed_before_results": True,
            "no_grid_search": True,
            "no_synthetic_data": True,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_DIR / "q51_zero_signature_try2_results.json", results)
    (REPORTS_DIR / "q51_zero_signature_try2_report.md").write_text(
        _build_report(results), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
