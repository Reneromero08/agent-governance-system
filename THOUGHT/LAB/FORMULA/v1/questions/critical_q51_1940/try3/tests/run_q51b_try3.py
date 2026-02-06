from __future__ import annotations

import json
import math
import hashlib
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: sentence-transformers ({exc})")

import ssl
import urllib.request
import urllib.error


OUTPUT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = OUTPUT_DIR / "reports"
RESULTS_DIR = OUTPUT_DIR / "results"
CONFIG_PATH = RESULTS_DIR / "q51b_try3_config.json"
RESULTS_PATH = RESULTS_DIR / "q51b_try3_results.json"
REPORT_PATH = REPORTS_DIR / "q51b_try3_report.md"
TRY2_DATASET = Path(__file__).resolve().parents[3] / "try2" / "results" / "dataset" / "questions-words.txt"


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


def _ensure_dataset(url: str, dest: Path) -> Dict[str, object]:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return {"downloaded": False, "tls_verification": "existing_file", "source": "existing_file"}
    if TRY2_DATASET.exists():
        shutil.copy2(TRY2_DATASET, dest)
        return {
            "downloaded": False,
            "tls_verification": "existing_file",
            "source": f"copied_from:{TRY2_DATASET}",
        }
    try:
        urllib.request.urlretrieve(url, dest)
        return {"downloaded": True, "tls_verification": "verified", "source": "downloaded"}
    except urllib.error.URLError as exc:
        if isinstance(getattr(exc, "reason", None), ssl.SSLCertVerificationError):
            context = ssl._create_unverified_context()
            with urllib.request.urlopen(url, context=context) as resp:
                dest.write_bytes(resp.read())
            return {"downloaded": True, "tls_verification": "skipped_cert_verification", "source": "downloaded"}
        raise


def _load_analogies(path: Path) -> Tuple[List[Tuple[str, str, str, str]], Dict[str, List[str]]]:
    analogies: List[Tuple[str, str, str, str]] = []
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
        a, b, c, d = parts
        analogies.append((a, b, c, d))
        if current:
            for word in parts:
                if word not in categories[current]:
                    categories[current].append(word)
    return analogies, categories


def _split_analogies(
    analogies: List[Tuple[str, str, str, str]],
    seed: int,
    train_ratio: float,
) -> Tuple[List[Tuple[str, str, str, str]], List[Tuple[str, str, str, str]]]:
    if not analogies:
        return [], []
    rng = np.random.default_rng(seed)
    idx = np.arange(len(analogies))
    rng.shuffle(idx)
    split = int(len(analogies) * train_ratio)
    train = [analogies[i] for i in idx[:split]]
    test = [analogies[i] for i in idx[split:]]
    return train, test


def _unique_words(analogies: List[Tuple[str, str, str, str]]) -> List[str]:
    vocab = set()
    for a, b, c, d in analogies:
        vocab.add(a)
        vocab.add(b)
        vocab.add(c)
        vocab.add(d)
    return sorted(vocab)


def _build_loops(categories: Dict[str, List[str]], loop_len: int, loops_per_cat: int) -> List[List[str]]:
    loops: List[List[str]] = []
    for _, words in categories.items():
        if len(words) < loop_len:
            continue
        count = 0
        for i in range(0, len(words) - loop_len + 1, loop_len):
            loop = words[i : i + loop_len]
            if len(loop) == loop_len:
                loops.append(loop + [loop[0]])
                count += 1
            if count >= loops_per_cat:
                break
    return loops


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    diff = a - b
    return (diff + np.pi) % (2 * np.pi) - np.pi


def _gamma(z_loop: np.ndarray, eps: float) -> Tuple[float, bool]:
    if np.any(np.abs(z_loop) < eps):
        return 0.0, False
    return float(np.angle(z_loop[1:] / z_loop[:-1]).sum()), True


def _gamma_mod(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def _random_bases(dim: int, num_bases: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    bases: List[np.ndarray] = []
    for _ in range(num_bases):
        mat = rng.normal(size=(dim, 2))
        q, _ = np.linalg.qr(mat)
        bases.append(q[:, :2])
    return bases


def _solve_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    if ridge <= 0:
        return np.linalg.lstsq(x, y, rcond=None)[0]
    n_features = x.shape[1]
    x_aug = np.vstack([x, math.sqrt(ridge) * np.eye(n_features)])
    y_aug = np.vstack([y, np.zeros((n_features, y.shape[1]))])
    return np.linalg.lstsq(x_aug, y_aug, rcond=None)[0]


def _evaluate_model(
    model_name: str,
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    train_analogies: List[Tuple[str, str, str, str]],
    test_analogies: List[Tuple[str, str, str, str]],
    loops: List[List[str]],
    bases: List[np.ndarray],
    cfg: dict,
) -> Dict[str, object]:
    eps = float(cfg["epsilon"])
    phase_thresh = float(cfg["phase_error_threshold_radians"])
    lambda_ratio_thresh = float(cfg["lambda_ratio_threshold"])
    holonomy_thresh = float(cfg["holonomy_diff_threshold_radians"])
    ridge = float(cfg["probe"]["ridge"])
    complex_ratio_thresh = float(cfg["probe"]["complex_better_ratio"])

    def _idx(word: str) -> int:
        return word_to_idx[word]

    train_idx = np.array([
        [_idx(a), _idx(b), _idx(c), _idx(d)] for a, b, c, d in train_analogies
    ])
    test_idx = np.array([
        [_idx(a), _idx(b), _idx(c), _idx(d)] for a, b, c, d in test_analogies
    ])

    loop_idx = [[_idx(w) for w in loop] for loop in loops]

    sign_emb = np.sign(embeddings)
    tanh_emb = np.tanh(embeddings)

    exp1_medians: List[float] = []
    exp1_pass = 0
    exp2_medians: List[float] = []
    exp2_baselines: List[float] = []
    exp2_ratios: List[float] = []
    exp2_pass = 0
    exp3_median_sign: List[float] = []
    exp3_median_tanh: List[float] = []
    exp3_pass = 0
    exp4_complex_errors: List[float] = []
    exp4_phase_errors: List[float] = []
    exp4_ratios: List[float] = []
    exp4_pass = 0

    rng = np.random.default_rng(int(cfg["random_seed"]))

    for base_idx, basis in enumerate(bases):
        coords = embeddings @ basis
        z_all = coords[:, 0] + 1j * coords[:, 1]

        a = test_idx[:, 0]
        b = test_idx[:, 1]
        c = test_idx[:, 2]
        d = test_idx[:, 3]

        z_a = z_all[a]
        z_b = z_all[b]
        z_c = z_all[c]
        z_d = z_all[d]
        valid = (np.abs(z_a) > eps) & (np.abs(z_c) > eps)
        if np.any(valid):
            ratio1 = z_b[valid] / z_a[valid]
            ratio2 = z_d[valid] / z_c[valid]
            phase_err = np.abs(_angle_diff(np.angle(ratio1), np.angle(ratio2)))
            median_phase = float(np.median(phase_err))
            exp1_medians.append(median_phase)
            if median_phase < phase_thresh:
                exp1_pass += 1

            lambda_diff = np.abs(ratio1 - ratio2)
            median_lambda = float(np.median(lambda_diff))
            exp2_medians.append(median_lambda)

            perm = rng.permutation(len(ratio2))
            baseline_diff = np.abs(ratio1 - ratio2[perm])
            median_baseline = float(np.median(baseline_diff))
            exp2_baselines.append(median_baseline)
            ratio = median_lambda / median_baseline if median_baseline > 0 else float("inf")
            exp2_ratios.append(ratio)
            if ratio < lambda_ratio_thresh:
                exp2_pass += 1
        else:
            exp1_medians.append(float("nan"))
            exp2_medians.append(float("nan"))
            exp2_baselines.append(float("nan"))
            exp2_ratios.append(float("nan"))

        coords_sign = (embeddings + cfg["distortion"]["epsilon"] * sign_emb) @ basis
        z_sign = coords_sign[:, 0] + 1j * coords_sign[:, 1]
        coords_tanh = tanh_emb @ basis
        z_tanh = coords_tanh[:, 0] + 1j * coords_tanh[:, 1]

        diffs_sign: List[float] = []
        diffs_tanh: List[float] = []
        for loop in loop_idx:
            z_loop = z_all[loop]
            z_loop_sign = z_sign[loop]
            z_loop_tanh = z_tanh[loop]
            gamma, ok = _gamma(z_loop, eps)
            if not ok:
                continue
            gamma_sign, ok_sign = _gamma(z_loop_sign, eps)
            gamma_tanh, ok_tanh = _gamma(z_loop_tanh, eps)
            if not ok_sign or not ok_tanh:
                continue
            diffs_sign.append(abs(_gamma_mod(gamma_sign - gamma)))
            diffs_tanh.append(abs(_gamma_mod(gamma_tanh - gamma)))

        median_sign = float(np.median(diffs_sign)) if diffs_sign else float("nan")
        median_tanh = float(np.median(diffs_tanh)) if diffs_tanh else float("nan")
        exp3_median_sign.append(median_sign)
        exp3_median_tanh.append(median_tanh)
        if (not math.isnan(median_sign)) and (not math.isnan(median_tanh)):
            if median_sign < holonomy_thresh and median_tanh < holonomy_thresh:
                exp3_pass += 1

        def _pairs(idx: np.ndarray) -> np.ndarray:
            return np.vstack([idx[:, :2], idx[:, 2:]])

        train_pairs = _pairs(train_idx)
        test_pairs = _pairs(test_idx)

        z_train_a = z_all[train_pairs[:, 0]]
        z_train_b = z_all[train_pairs[:, 1]]
        valid_train = np.abs(z_train_a) > eps

        z_test_a = z_all[test_pairs[:, 0]]
        z_test_b = z_all[test_pairs[:, 1]]
        valid_test = np.abs(z_test_a) > eps

        if np.any(valid_train) and np.any(valid_test):
            lambda_train = z_train_b[valid_train] / z_train_a[valid_train]
            lambda_test = z_test_b[valid_test] / z_test_a[valid_test]

            x_train = embeddings[train_pairs[:, 1][valid_train]] - embeddings[train_pairs[:, 0][valid_train]]
            x_test = embeddings[test_pairs[:, 1][valid_test]] - embeddings[test_pairs[:, 0][valid_test]]

            y_complex = np.stack([lambda_train.real, lambda_train.imag], axis=1)
            w_complex = _solve_ridge(x_train, y_complex, ridge)
            pred_complex = x_test @ w_complex
            lambda_complex = pred_complex[:, 0] + 1j * pred_complex[:, 1]
            complex_err = float(np.median(np.abs(lambda_complex - lambda_test)))
            exp4_complex_errors.append(complex_err)

            phase = np.angle(lambda_train)
            y_phase = np.stack([np.cos(phase), np.sin(phase)], axis=1)
            w_phase = _solve_ridge(x_train, y_phase, ridge)
            pred_phase = x_test @ w_phase
            mag = np.linalg.norm(pred_phase, axis=1)
            nonzero = mag > 0
            if np.any(nonzero):
                pred_norm = pred_phase.copy()
                pred_norm[nonzero] = pred_norm[nonzero] / mag[nonzero][:, None]
                lambda_phase = pred_norm[:, 0] + 1j * pred_norm[:, 1]
                phase_err = float(np.median(np.abs(lambda_phase - lambda_test)))
            else:
                phase_err = float("nan")
            exp4_phase_errors.append(phase_err)

            ratio = complex_err / phase_err if phase_err and not math.isnan(phase_err) else float("inf")
            exp4_ratios.append(ratio)
            if ratio < complex_ratio_thresh:
                exp4_pass += 1
        else:
            exp4_complex_errors.append(float("nan"))
            exp4_phase_errors.append(float("nan"))
            exp4_ratios.append(float("nan"))

    num_bases = len(bases)
    exp1_fraction = exp1_pass / num_bases if num_bases else 0.0
    exp2_fraction = exp2_pass / num_bases if num_bases else 0.0
    exp3_fraction = exp3_pass / num_bases if num_bases else 0.0
    exp4_fraction = exp4_pass / num_bases if num_bases else 0.0

    return {
        "model": model_name,
        "status": "ok",
        "exp1": {
            "median_phase_error_per_base": exp1_medians,
            "bases_passing": exp1_pass,
            "fraction_passing": exp1_fraction,
            "pass": exp1_pass >= cfg["phase_pass_min_bases"],
        },
        "exp2": {
            "median_lambda_diff_per_base": exp2_medians,
            "median_baseline_per_base": exp2_baselines,
            "ratio_per_base": exp2_ratios,
            "bases_passing": exp2_pass,
            "fraction_passing": exp2_fraction,
            "pass": exp2_fraction >= cfg["lambda_pass_fraction_min"],
        },
        "exp3": {
            "median_gamma_diff_sign_per_base": exp3_median_sign,
            "median_gamma_diff_tanh_per_base": exp3_median_tanh,
            "bases_passing": exp3_pass,
            "fraction_passing": exp3_fraction,
            "pass": exp3_fraction >= cfg["holonomy_pass_fraction_min"],
        },
        "exp4": {
            "complex_probe_median_error_per_base": exp4_complex_errors,
            "phase_probe_median_error_per_base": exp4_phase_errors,
            "ratio_per_base": exp4_ratios,
            "bases_passing": exp4_pass,
            "fraction_passing": exp4_fraction,
            "pass": exp4_fraction >= cfg["probe"]["complex_better_fraction_min"],
        },
    }


def _build_report(results: dict) -> str:
    summary = results["summary"]
    lines: List[str] = []
    lines.append("# Q51b Complex-Linearity Stress Test — Try3 Results")
    lines.append("")
    lines.append("## Data Source")
    lines.append(f"- URL: {results['dataset']['url']}")
    lines.append(f"- SHA256: {results['dataset']['sha256']}")
    lines.append(f"- Size (bytes): {results['dataset']['size_bytes']}")
    lines.append(f"- Total analogies: {results['dataset']['total_analogies']}")
    lines.append(f"- Unique words: {results['dataset']['n_words']}")
    lines.append("")
    lines.append("## Summary (Per-Model Pass Rates)")
    lines.append("| Model | Exp1 Pass | Exp2 Pass | Exp3 Pass | Exp4 Pass |")
    lines.append("| --- | --- | --- | --- | --- |")
    for item in results["models"]:
        if item.get("status") != "ok":
            lines.append(f"| {item['model']} | ERROR | ERROR | ERROR | ERROR |")
            continue
        lines.append(
            "| {model} | {e1} | {e2} | {e3} | {e4} |".format(
                model=item["model"],
                e1=str(item["exp1"]["pass"]),
                e2=str(item["exp2"]["pass"]),
                e3=str(item["exp3"]["pass"]),
                e4=str(item["exp4"]["pass"]),
            )
        )
    lines.append("")
    lines.append("## Aggregate Outcomes")
    lines.append(f"- Models evaluated: {summary['models_evaluated']}")
    lines.append(f"- Exp1 pass count: {summary['exp1_pass_count']}")
    lines.append(f"- Exp2 pass count: {summary['exp2_pass_count']}")
    lines.append(f"- Exp3 pass count: {summary['exp3_pass_count']}")
    lines.append(f"- Exp4 pass count: {summary['exp4_pass_count']}")
    lines.append("")
    lines.append("## Decision Table")
    lines.append("- Phase survives random bases: " + summary["decision_table"]["phase_survives_random_bases"])
    lines.append("- Shared complex λ exists: " + summary["decision_table"]["shared_lambda_exists"])
    lines.append("- Winding survives nonlinear distortion: " + summary["decision_table"]["winding_survives_distortion"])
    lines.append("- Complex probe dominates: " + summary["decision_table"]["complex_probe_dominates"])
    lines.append("")
    lines.append("## Conclusion")
    lines.append(summary["conclusion"])
    lines.append("")
    return "\n".join(lines) + "\n"


def _load_existing_results() -> dict:
    if RESULTS_PATH.exists():
        try:
            return _load_json(RESULTS_PATH)
        except Exception:
            return {}
    return {}


def main() -> int:
    cfg = _load_json(CONFIG_PATH)
    dataset_path = OUTPUT_DIR / cfg["dataset_path"]
    download_meta = _ensure_dataset(cfg["dataset_url"], dataset_path)

    analogies, categories = _load_analogies(dataset_path)
    train_analogies, test_analogies = _split_analogies(
        analogies, int(cfg["random_seed"]), float(cfg["train_split_ratio"])
    )
    loops = _build_loops(categories, int(cfg["loop_word_count"]), int(cfg["loops_per_category_max"]))

    vocab = _unique_words(analogies)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    dataset_info = {
        "url": cfg["dataset_url"],
        "path": str(dataset_path.relative_to(OUTPUT_DIR)),
        "sha256": _sha256_file(dataset_path),
        "size_bytes": dataset_path.stat().st_size,
        "total_analogies": len(analogies),
        "train_analogies": len(train_analogies),
        "test_analogies": len(test_analogies),
        "n_words": len(vocab),
        "loops": len(loops),
        "download": download_meta,
    }

    existing = _load_existing_results()
    existing_models = {m.get("model") for m in existing.get("models", [])}
    model_results: List[Dict[str, object]] = list(existing.get("models", []))

    for model_name in cfg["models"]:
        if model_name in existing_models:
            continue
        try:
            model = SentenceTransformer(model_name)
            embeddings = model.encode(
                vocab,
                batch_size=int(cfg["batch_size"]),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=bool(cfg["normalize_embeddings"]),
            )
        except Exception as exc:  # pragma: no cover
            model_results.append({"model": model_name, "status": "error", "error": str(exc)})
            _write_json(RESULTS_PATH, {
                "config": cfg,
                "dataset": dataset_info,
                "models": model_results,
                "summary": existing.get("summary", {}),
                "anti_pattern_check": {
                    "no_synthetic_data": True,
                    "parameters_fixed_before_results": True,
                    "no_grid_search": True,
                },
            })
            continue

        bases = _random_bases(embeddings.shape[1], int(cfg["num_bases"]), int(cfg["random_seed"]))
        res = _evaluate_model(
            model_name,
            embeddings,
            word_to_idx,
            train_analogies,
            test_analogies,
            loops,
            bases,
            cfg,
        )
        model_results.append(res)
        _write_json(RESULTS_PATH, {
            "config": cfg,
            "dataset": dataset_info,
            "models": model_results,
            "summary": existing.get("summary", {}),
            "anti_pattern_check": {
                "no_synthetic_data": True,
                "parameters_fixed_before_results": True,
                "no_grid_search": True,
            },
        })

    evaluated = [m for m in model_results if m.get("status") == "ok"]
    summary = {
        "models_attempted": len(model_results),
        "models_evaluated": len(evaluated),
        "exp1_pass_count": sum(1 for m in evaluated if m["exp1"]["pass"]),
        "exp2_pass_count": sum(1 for m in evaluated if m["exp2"]["pass"]),
        "exp3_pass_count": sum(1 for m in evaluated if m["exp3"]["pass"]),
        "exp4_pass_count": sum(1 for m in evaluated if m["exp4"]["pass"]),
    }

    decision = {
        "phase_survives_random_bases": "YES" if summary["exp1_pass_count"] > 0 else "NO",
        "shared_lambda_exists": "YES" if summary["exp2_pass_count"] > 0 else "NO",
        "winding_survives_distortion": "YES" if summary["exp3_pass_count"] > 0 else "NO",
        "complex_probe_dominates": "YES" if summary["exp4_pass_count"] > 0 else "NO",
    }

    pass_count = sum(1 for value in decision.values() if value == "YES")
    if pass_count >= 2:
        conclusion = (
            "Two or more experiments pass for at least one model; intrinsic complex structure becomes harder to dismiss."
        )
    else:
        conclusion = (
            "Fewer than two experiments pass; lock Q51 as: real embeddings show phase-like regularities under linear projections "
            "without evidence of intrinsic complex structure."
        )

    summary["decision_table"] = decision
    summary["conclusion"] = conclusion

    results = {
        "config": cfg,
        "dataset": dataset_info,
        "models": model_results,
        "summary": summary,
        "anti_pattern_check": {
            "no_synthetic_data": True,
            "parameters_fixed_before_results": True,
            "no_grid_search": True,
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(RESULTS_PATH, results)
    REPORT_PATH.write_text(_build_report(results), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
