from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


EPS = 1e-10


@dataclass
class FormulaMetrics:
    e: float
    grad_s: float
    sigma: float
    df: float
    r_simple: float
    r_full: float
    effective_rank: float
    isotropy_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "E": self.e,
            "grad_S": self.grad_s,
            "sigma": self.sigma,
            "Df": self.df,
            "R_simple": self.r_simple,
            "R_full": self.r_full,
            "effective_rank": self.effective_rank,
            "isotropy_score": self.isotropy_score,
            "inv_grad_S": 1.0 / max(self.grad_s, EPS),
        }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, EPS, None)
    return x / norms


def _pairwise_cosines(x: np.ndarray) -> np.ndarray:
    normed = _normalize_rows(x)
    sims = normed @ normed.T
    iu = np.triu_indices(normed.shape[0], k=1)
    return sims[iu]


def _positive_eigenvalues(x: np.ndarray) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if np.ndim(cov) == 0:
        return np.array([], dtype=float)
    eig = np.linalg.eigvalsh(cov)
    return np.sort(eig[eig > EPS])[::-1]


def compute_formula_metrics(embeddings: np.ndarray) -> FormulaMetrics:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (n, d)")
    if embeddings.shape[0] < 2:
        raise ValueError("need at least 2 embeddings")

    pairwise = _pairwise_cosines(embeddings)
    e = float(np.mean(pairwise))
    grad_s = float(np.std(pairwise))

    eig = _positive_eigenvalues(embeddings)
    d = embeddings.shape[1]

    if eig.size == 0:
        sigma = float("nan")
        df = float("nan")
        effective_rank = float("nan")
        isotropy = float("nan")
    else:
        pr = (eig.sum() ** 2) / np.sum(eig ** 2)
        sigma = float(np.clip(pr / max(d, 1), EPS, 1.0))

        p = eig / eig.sum()
        effective_rank = float(np.exp(-np.sum(p * np.log(np.clip(p, EPS, None)))))
        isotropy = float(eig.min() / eig.max())

        if eig.size < 3:
            df = float("nan")
        else:
            k = np.arange(1, eig.size + 1, dtype=float)
            log_k = np.log(k)
            log_eig = np.log(eig)
            slope, _ = np.polyfit(log_k, log_eig, 1)
            alpha = -float(slope)
            df = float(2.0 / alpha) if alpha > EPS else float("nan")

    r_simple = float(e / max(grad_s, EPS))
    r_full = float(r_simple * (sigma ** df)) if np.isfinite(sigma) and np.isfinite(df) else float("nan")

    return FormulaMetrics(
        e=e,
        grad_s=grad_s,
        sigma=sigma,
        df=df,
        r_simple=r_simple,
        r_full=r_full,
        effective_rank=effective_rank,
        isotropy_score=isotropy,
    )
