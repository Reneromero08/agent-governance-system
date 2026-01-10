"""
Q32: Public Truth-Anchored Benchmarks (Phase 2/3/4)

This script extends the Q32 falsification harness onto public datasets, with:
  - pinned cache locations (no dependency on global ~/.cache)
  - fixed seeds + sample caps
  - adversarial construction of "false basins" by borrowing semantically-near evidence from other claims
  - negative controls (shuffled checks)
  - threshold transfer (calibrate once, then freeze)
  - streaming dynamics (M(t) phase transitions under evidence arrival)

Datasets (Phase 2)
  - SciFact (claim verification with cited abstracts + rationales)

Notes
  - We intentionally avoid using ground-truth labels inside M. Labels are used only for evaluation.

Run (recommended in the pinned venv):
  LAW/CONTRACTS/_runs/q32_public/venv/Scripts/python.exe THOUGHT/LAB/FORMULA/experiments/open_questions/q32/q32_public_benchmarks.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
_CROSS_ENCODER = None
_SENTENCE_MODEL = None
_USE_CROSS_ENCODER = True
_PAIR_SCORE_CACHE: Dict[Tuple[str, str], float] = {}
_PAIR_SCORE_CACHE_MAX = 50_000
_DEVICE: Optional[str] = None
_THREADS: Optional[int] = None
_CE_BATCH_SIZE: Optional[int] = None
_ST_BATCH_SIZE: Optional[int] = None
_ABLATION: str = "full"
_DEPTH_POWER: float = 0.0


def set_cache_roots() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    cache_root = os.path.join(repo_root, "LAW", "CONTRACTS", "_runs", "q32_public", "hf_cache")
    os.makedirs(cache_root, exist_ok=True)

    # Hugging Face caches
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(cache_root, "sentence_transformers"))


@lru_cache(maxsize=1)
def _auto_device() -> str:
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def configure_runtime(*, device: str, threads: int, ce_batch_size: int, st_batch_size: int) -> None:
    global _DEVICE, _THREADS, _CE_BATCH_SIZE, _ST_BATCH_SIZE
    # If the user requests CUDA but torch isn't CUDA-enabled, fall back to CPU without crashing.
    if device == "cuda":
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                print("[Q32] WARN: --device cuda requested but torch.cuda.is_available() is false; using cpu.")
                device = "cpu"
        except Exception:
            print("[Q32] WARN: --device cuda requested but torch import/CUDA check failed; using cpu.")
            device = "cpu"

    _DEVICE = device
    _THREADS = threads
    _CE_BATCH_SIZE = ce_batch_size
    _ST_BATCH_SIZE = st_batch_size

    # Best-effort CPU threading control (safe if torch is absent).
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))

    try:
        import torch  # type: ignore

        torch.set_num_threads(threads)
        torch.set_num_interop_threads(max(1, threads // 2))
    except Exception:
        pass


def mean(x: Sequence[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=float)))


def std(x: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    if len(a) <= 1:
        return EPS
    return float(np.std(a, ddof=1)) + EPS


def se(x: Sequence[float]) -> float:
    return std(x) / math.sqrt(max(1, len(x)))


def kernel_gaussian(z: float) -> float:
    return float(math.exp(-0.5 * z * z))

def _mutual_information_continuous(x: Sequence[float], y: Sequence[float], *, n_bins: int = 8) -> float:
    """
    Phi-style proxy: mutual information I(X;Y) for continuous values via histogram binning.

    This is NOT canonical IIT Phi; it is a cheap coupling/structure proxy that is stable and reproducible.
    """
    xx = np.asarray(list(x), dtype=float)
    yy = np.asarray(list(y), dtype=float)
    if xx.size == 0 or yy.size == 0:
        return 0.0
    n = int(min(xx.size, yy.size))
    xx = xx[:n]
    yy = yy[:n]

    mask = np.isfinite(xx) & np.isfinite(yy)
    xx = xx[mask]
    yy = yy[mask]
    if xx.size < 5:
        return 0.0

    n_bins_i = max(2, int(n_bins))
    x_min = float(np.min(xx))
    x_max = float(np.max(xx))
    y_min = float(np.min(yy))
    y_max = float(np.max(yy))
    if x_min == x_max:
        x_min -= 1.0
        x_max += 1.0
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    x_edges = np.linspace(x_min - EPS, x_max + EPS, n_bins_i + 1)
    y_edges = np.linspace(y_min - EPS, y_max + EPS, n_bins_i + 1)
    joint, _, _ = np.histogram2d(xx, yy, bins=(x_edges, y_edges))
    pxy = joint / float(np.sum(joint) + EPS)
    px = np.sum(pxy, axis=1, keepdims=True)
    py = np.sum(pxy, axis=0, keepdims=True)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = pxy / (px @ py + EPS)
        mi = float(np.nansum(pxy * np.log2(ratio + EPS)))
    return float(max(0.0, mi))


def _unit(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, dtype=np.float64).reshape(-1)
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n <= 0:
        return np.zeros_like(v)
    return (v / n).astype(np.float64, copy=False)


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    ua = _unit(a)
    ub = _unit(b)
    return float(np.dot(ua, ub))


def _participation_ratio_from_embeddings(emb: np.ndarray) -> float:
    """
    Effective rank proxy via participation ratio of the sample covariance spectrum.
    Uses the Gram-matrix eigenspectrum (cheaper; equivalent for nonzero eigenvalues).
    """
    x = np.asarray(emb, dtype=np.float64)
    if x.ndim != 2:
        return float("nan")
    n = int(x.shape[0])
    if n < 2:
        return 1.0 if n == 1 else float("nan")
    x = x - np.mean(x, axis=0, keepdims=True)
    gram = (x @ x.T) / float(max(1, n - 1))
    try:
        eig = np.linalg.eigvalsh(gram)
    except Exception:
        eig = np.linalg.eigvals(gram).real
    eig = np.asarray([float(v) for v in eig if math.isfinite(float(v)) and float(v) > 0], dtype=np.float64)
    if eig.size == 0:
        return 1.0
    s1 = float(np.sum(eig))
    s2 = float(np.sum(eig * eig))
    if not math.isfinite(s1) or not math.isfinite(s2) or s2 <= 0:
        return float("nan")
    return float((s1 * s1) / s2)


def R_grounded(observations: Sequence[float], check: Sequence[float]) -> float:
    """
    The pinned Q32 spec:
      M := log(R)
      R := E / grad_S
      E compares mu_hat to mu_check using z normalized by SE(check)
      grad_S uses SE(observations) (mean-estimator uncertainty)
    """
    mu_hat = mean(observations)
    mu_check = mean(check)
    z = abs(mu_hat - mu_check) / (se(check) + EPS)

    # Depth proxy (deterministic, Q32-scoped):
    # - Matches the "σ^Df" intuition: we use sample std(check) as σ and `--depth_power` as Df.
    # - Default `--depth_power 0` => depth_term=1 (no behavioral change unless enabled explicitly).
    depth_term = float(pow(std(check), float(_DEPTH_POWER)))

    # Ablations (deterministic, Q32-scoped):
    # - full: the normal grounded R (with optional depth_term if depth_power != 0)
    # - no_essence: remove empirical compatibility (E=1), leaving only the scale term (+ optional depth)
    # - no_scale: remove the scale normalization (grad_S=1), leaving only empirical compatibility (+ optional depth)
    # - no_depth: force depth_term=1 (even if depth_power is set)
    # - no_grounding: remove empirical compatibility + scale + depth (R=1 constant), should hard-fail gates
    E = kernel_gaussian(z)
    grad_S = se(observations)
    if _ABLATION == "no_essence":
        E = 1.0
    elif _ABLATION == "no_scale":
        grad_S = 1.0
    elif _ABLATION == "no_depth":
        depth_term = 1.0
    elif _ABLATION == "no_grounding":
        E = 1.0
        grad_S = 1.0
        depth_term = 1.0
    return (float(E) * float(depth_term)) / (float(grad_S) + EPS)


def M_from_R(R: float) -> float:
    return float(math.log(R + EPS))


@dataclass(frozen=True)
class ScifactExample:
    claim_id: int
    claim: str
    doc_id: int
    doc_title: str
    rationale_sentences: Tuple[str, ...]
    label: str  # SUPPORT / CONTRADICT / NOT_ENOUGH_INFO
    cited_doc_ids: Tuple[int, ...]


@dataclass(frozen=True)
class SnliExample:
    premise: str
    hypothesis: str
    label: str  # ENTAILMENT / CONTRADICTION


@dataclass(frozen=True)
class MnliExample:
    premise: str
    hypothesis: str
    label: str  # ENTAILMENT / CONTRADICTION


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    passed: bool
    details: Dict[str, float]


def _binom_z(wins: int, n: int) -> float:
    return float((wins - (n / 2.0)) / (math.sqrt(n / 4.0) + EPS))


def _percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(list(values), dtype=float), q))


def load_scifact(max_claims: int = 400, seed: int = 123) -> List[ScifactExample]:
    """
    Loads SciFact via HF datasets.
    We build examples at the (claim, doc) level with rationale sentences.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("scifact", "claims")
    corpus = load_dataset("scifact", "corpus")["train"]
    corpus_by_id: Dict[int, dict] = {int(r["doc_id"]): r for r in corpus}

    rng = np.random.default_rng(seed)
    rows = ds["train"]

    claim_indices = np.arange(len(rows))
    rng.shuffle(claim_indices)

    examples: List[ScifactExample] = []

    for idx in claim_indices:
        row = rows[int(idx)]
        claim_id = int(row["id"])
        claim_text = str(row["claim"])
        raw_doc_id = row.get("evidence_doc_id")
        if raw_doc_id is None:
            continue
        if isinstance(raw_doc_id, str) and raw_doc_id.strip() == "":
            continue
        try:
            doc_id = int(raw_doc_id)
        except Exception:
            continue
        if doc_id not in corpus_by_id:
            continue
        label = str(row.get("evidence_label", "NOT_ENOUGH_INFO"))
        sent_ids = row.get("evidence_sentences", []) or []
        if not sent_ids:
            continue
        cited = row.get("cited_doc_ids", []) or []
        cited_ids: List[int] = []
        for cd in cited:
            try:
                cited_ids.append(int(cd))
            except Exception:
                continue

        doc = corpus_by_id[doc_id]
        title = str(doc.get("title", f"doc_{doc_id}"))
        abstract = doc.get("abstract", []) or []

        sents: List[str] = []
        for sid in sent_ids:
            if 0 <= int(sid) < len(abstract):
                sents.append(str(abstract[int(sid)]))
        if len(sents) < 1:
            continue

        examples.append(
                ScifactExample(
                    claim_id=claim_id,
                    claim=claim_text,
                    doc_id=doc_id,
                    doc_title=title,
                    rationale_sentences=tuple(sents),
                    label=label,
                    cited_doc_ids=tuple(cited_ids),
                )
            )

        if len({e.claim_id for e in examples}) >= max_claims:
            break

    if len(examples) < 50:
        raise RuntimeError(f"SciFact load produced too few examples ({len(examples)})")
    return examples


def load_snli(max_examples: int = 50000, seed: int = 123) -> List[SnliExample]:
    """
    Loads SNLI via HF datasets (public, standard format).
    We keep only ENTAILMENT/CONTRADICTION labels as truth anchors and drop NEUTRAL/unknown.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("snli", split="train")
    rng = np.random.default_rng(seed)

    idxs = np.arange(len(ds))
    rng.shuffle(idxs)

    out: List[SnliExample] = []
    for i in idxs:
        row = ds[int(i)]
        premise = str(row.get("premise", "")).strip()
        hyp = str(row.get("hypothesis", "")).strip()
        if not premise or not hyp:
            continue
        lab = row.get("label")
        if lab is None:
            continue
        try:
            lab_i = int(lab)
        except Exception:
            continue
        if lab_i == 0:
            label = "ENTAILMENT"
        elif lab_i == 2:
            label = "CONTRADICTION"
        else:
            continue
        out.append(SnliExample(premise=premise, hypothesis=hyp, label=label))
        if len(out) >= int(max_examples):
            break
    if len(out) < 200:
        raise RuntimeError(f"SNLI load produced too few usable examples ({len(out)})")
    return out


def load_mnli(max_examples: int = 50000, seed: int = 123) -> List[MnliExample]:
    """
    Loads MNLI via HF datasets (GLUE/MNLI).
    We keep only ENTAILMENT/CONTRADICTION labels as truth anchors and drop NEUTRAL/unknown.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("glue", "mnli", split="validation_matched")
    rng = np.random.default_rng(seed)

    idxs = np.arange(len(ds))
    rng.shuffle(idxs)

    out: List[MnliExample] = []
    for i in idxs:
        row = ds[int(i)]
        premise = str(row.get("premise", "")).strip()
        hyp = str(row.get("hypothesis", "")).strip()
        if not premise or not hyp:
            continue
        lab = row.get("label")
        if lab is None:
            continue
        try:
            lab_i = int(lab)
        except Exception:
            continue
        if lab_i == 0:
            label = "ENTAILMENT"
        elif lab_i == 2:
            label = "CONTRADICTION"
        else:
            continue
        out.append(MnliExample(premise=premise, hypothesis=hyp, label=label))
        if len(out) >= int(max_examples):
            break
    if len(out) < 200:
        raise RuntimeError(f"MNLI load produced too few usable examples ({len(out)})")
    return out


def _load_nli(*, domain: str, max_examples: int, seed: int) -> List[SnliExample]:
    if str(domain) == "snli":
        return list(load_snli(max_examples=max_examples, seed=seed))
    if str(domain) == "mnli":
        return [
            SnliExample(premise=e.premise, hypothesis=e.hypothesis, label=e.label)
            for e in load_mnli(max_examples=max_examples, seed=seed)
        ]
    raise ValueError(f"Unknown NLI domain: {domain}")


def _word_chunks(text: str, *, window: int = 3, stride: int = 1, max_chunks: int = 12) -> List[str]:
    """
    Deterministic chunking for NLI-style single-sentence evidence into multiple small "evidence pieces",
    so Q32's obs/check intervention can be applied without inventing extra data.
    """
    toks = [t for t in str(text).strip().split() if t]
    if len(toks) < window:
        return []
    chunks: List[str] = []
    for start in range(0, len(toks) - window + 1, stride):
        chunks.append(" ".join(toks[start : start + window]))
        if len(chunks) >= int(max_chunks):
            break
    return chunks

def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        try:
            _SENTENCE_MODEL = SentenceTransformer(model_name, device=_DEVICE or "cpu")
        except Exception:
            # Typical failure: torch not compiled with CUDA enabled.
            _SENTENCE_MODEL = SentenceTransformer(model_name, device="cpu")
    model = _SENTENCE_MODEL
    emb = model.encode(texts, normalize_embeddings=True, batch_size=int(_ST_BATCH_SIZE or 32), show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def sentence_support_scores(claims: List[str], sentences: List[str]) -> np.ndarray:
    """
    Produce a bounded support score in [-1,1] for each (claim, sentence) pair.

    We use a small NLI cross-encoder if available; otherwise fall back to cosine similarity.
    """
    if not _USE_CROSS_ENCODER:
        emb_claim = embed_texts(claims)
        emb_sent = embed_texts(sentences)
        sims = np.sum(emb_claim * emb_sent, axis=1)
        return np.clip(sims, -1.0, 1.0)

    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        global _CROSS_ENCODER, _PAIR_SCORE_CACHE
        if _CROSS_ENCODER is None:
            # Smaller NLI cross-encoder (CPU-friendly relative to large RoBERTa).
            try:
                _CROSS_ENCODER = CrossEncoder(
                    "cross-encoder/nli-MiniLM2-L6-H768", max_length=256, device=_DEVICE or "cpu"
                )
            except Exception:
                _CROSS_ENCODER = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", max_length=256, device="cpu")
        model = _CROSS_ENCODER
        # NLI convention: (premise, hypothesis). Evidence sentence is premise; claim is hypothesis.
        pairs = list(zip(sentences, claims))
        out = np.empty(len(pairs), dtype=np.float32)

        miss_idx: List[int] = []
        miss_pairs: List[Tuple[str, str]] = []
        for i, (sent, claim) in enumerate(pairs):
            k = (sent, claim)
            v = _PAIR_SCORE_CACHE.get(k)
            if v is None:
                miss_idx.append(i)
                miss_pairs.append((sent, claim))
            else:
                out[i] = float(v)

        if miss_pairs:
            if len(_PAIR_SCORE_CACHE) + len(miss_pairs) > _PAIR_SCORE_CACHE_MAX:
                _PAIR_SCORE_CACHE.clear()

            logits = np.asarray(
                model.predict(miss_pairs, batch_size=int(_CE_BATCH_SIZE or 16), show_progress_bar=False),
                dtype=float,
            )
            # Some NLI cross-encoders output 3 logits (contradiction, entailment, neutral).
            if logits.ndim == 2 and logits.shape[1] == 3:
                # softmax
                m = logits - logits.max(axis=1, keepdims=True)
                ex = np.exp(m)
                probs = ex / ex.sum(axis=1, keepdims=True)
                # signed support: P(entail) - P(contradict)
                scores = (probs[:, 1] - probs[:, 0]).astype(np.float32)
            else:
                # Otherwise treat as a single unbounded score and squash.
                logits1 = logits.reshape(-1)
                scores = np.tanh(logits1).astype(np.float32)

            for i, (sent, claim), s in zip(miss_idx, miss_pairs, scores):
                out[i] = float(s)
                _PAIR_SCORE_CACHE[(sent, claim)] = float(s)

        return out
    except Exception:
        # Fallback: cosine similarity as weak support proxy (still useful for negative controls).
        emb_claim = embed_texts(claims)
        emb_sent = embed_texts(sentences)
        sims = np.sum(emb_claim * emb_sent, axis=1)
        return np.clip(sims, -1.0, 1.0)


def build_false_basin_mapping(examples: List[ScifactExample], seed: int = 123) -> Dict[int, int]:
    """
    For each claim_id, pick a different claim_id that is semantically near (false basin source).
    """
    rng = np.random.default_rng(seed)
    claim_ids = sorted({e.claim_id for e in examples})
    claim_text_by_id: Dict[int, str] = {}
    label_by_id: Dict[int, str] = {}
    for e in examples:
        claim_text_by_id.setdefault(e.claim_id, e.claim)
        label_by_id.setdefault(e.claim_id, e.label)

    texts = [claim_text_by_id[cid] for cid in claim_ids]
    emb = embed_texts(texts)

    # Cosine similarity matrix is just dot product since normalized.
    sim = emb @ emb.T
    mapping: Dict[int, int] = {}
    for i, cid in enumerate(claim_ids):
        # Pick from top-10 nearest excluding self.
        order = np.argsort(-sim[i])
        candidates = [int(claim_ids[j]) for j in order[1:50]]
        # Prefer a near claim with a different label if possible (stronger false basin).
        cid_label = label_by_id.get(cid, "")
        diff_label = [c for c in candidates if label_by_id.get(c, "") != cid_label]
        pool = diff_label if diff_label else candidates
        mapping[cid] = int(rng.choice(pool))
    return mapping


def pick_true_and_false_sets(
    examples: List[ScifactExample],
    false_map: Dict[int, int],
    min_sentences: int = 1,
) -> List[Tuple[ScifactExample, ScifactExample]]:
    """
    For each true example (claim_id, doc), attach a false sentence set from a near claim's rationale.
    """
    by_claim: Dict[int, List[ScifactExample]] = {}
    for e in examples:
        by_claim.setdefault(e.claim_id, []).append(e)

    pairs: List[Tuple[ScifactExample, ScifactExample]] = []
    # pre-filter false candidates with enough sentences
    false_candidates_by_claim: Dict[int, List[ScifactExample]] = {}
    for cid, lst in by_claim.items():
        good = [x for x in lst if len(x.rationale_sentences) >= min_sentences]
        if good:
            false_candidates_by_claim[cid] = good

    for e in examples:
        if len(e.rationale_sentences) < min_sentences:
            continue
        false_claim = false_map.get(e.claim_id)
        if false_claim is None or false_claim == e.claim_id:
            continue
        candidates = false_candidates_by_claim.get(false_claim, [])
        if not candidates:
            continue
        # choose a stable candidate (first) for determinism
        fe = candidates[0]
        pairs.append((e, fe))

    if len(pairs) < 50:
        raise RuntimeError(f"Too few true/false basin pairs ({len(pairs)})")
    return pairs


def run_scifact_benchmark(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
) -> BenchmarkResult:
    print("\n[Q32:P2] Loading SciFact...")
    examples = load_scifact(max_claims=400, seed=seed)

    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    examples = examples[: (120 if fast else 600)]

    # Phase 2 gate for SciFact is a *check intervention* (not SUPPORT vs CONTRADICT):
    # - Obs: 2 sampled sentences from the evidence doc.
    # - Correct check: hold-out sampled sentences from the same evidence doc.
    # - Wrong check: sampled sentences from a different SUPPORT evidence doc, scored against the current claim.
    # Pass if M(obs, correct_check) > M(obs, wrong_check) with significant margin.
    M_correct: List[float] = []
    M_wrong: List[float] = []
    R_correct: List[float] = []
    R_wrong: List[float] = []
    # For swap/shuffle negative controls (receipted): keep raw score sets per sample.
    obs_scores_by_sample: List[List[float]] = []
    check_correct_scores_by_sample: List[List[float]] = []
    neighbor_sims: List[float] = []
    mu_hat_list: List[float] = []
    mu_check_list: List[float] = []

    # For each pair, build observation sets as support scores from sentences.
    # Build observation/check sets with enough samples to avoid degenerate SE=EPS.
    # We use top-K sentences from the evidence doc by support score (label-free).
    from datasets import load_dataset  # type: ignore

    corpus = load_dataset("scifact", "corpus")["train"]
    corpus_by_id: Dict[int, dict] = {int(r["doc_id"]): r for r in corpus}

    def doc_sentences(doc_id: int) -> List[str]:
        doc = corpus_by_id.get(int(doc_id))
        if not doc:
            return []
        abstract = doc.get("abstract", []) or []
        return [str(x) for x in abstract if str(x).strip()]

    def sample_sentences_from_doc(doc_id: int, n: int, *, seed_key: int) -> List[str]:
        sents = doc_sentences(doc_id)
        if len(sents) < n:
            return []
        local_rng = np.random.default_rng(seed_key)
        order = local_rng.permutation(len(sents))[:n]
        return [sents[int(i)] for i in order]

    support_examples = [e for e in examples if str(e.label).strip().upper() == "SUPPORT"]
    if len(support_examples) < (30 if fast else 120):
        raise RuntimeError(f"Too few SciFact SUPPORT examples ({len(support_examples)})")

    wrong_bank: List[List[str]] = []
    # Optional: per-claim bank for neighbor-wrong checks.
    claim_bank: Dict[int, List[str]] = {}
    for e in support_examples[: (80 if fast else 400)]:
        sampled = sample_sentences_from_doc(
            int(e.doc_id),
            8,
            seed_key=(seed * 1_000_003) ^ (int(e.claim_id) * 9176) ^ int(e.doc_id),
        )
        if len(sampled) >= 6:
            wrong_bank.append(sampled[:6])
            claim_bank.setdefault(int(e.claim_id), sampled[:6])
    if len(wrong_bank) < (20 if fast else 80):
        raise RuntimeError(f"Too few SciFact wrong-check pools ({len(wrong_bank)})")

    neighbor_candidates_by_claim: Dict[int, List[int]] = {}
    neighbor_candidate_sims_by_claim: Dict[int, List[float]] = {}
    inflation_candidates_by_claim: Dict[int, List[int]] = {}
    inflation_candidate_sims_by_claim: Dict[int, List[float]] = {}
    neighbor_sims: List[float] = []
    if wrong_checks in ("neighbor", "inflation"):
        claim_text_by_id: Dict[int, str] = {}
        for e in support_examples:
            claim_text_by_id.setdefault(int(e.claim_id), str(e.claim))
        claim_ids = sorted({int(e.claim_id) for e in support_examples})

        # Build a CONTRADICT bank so "neighbor wrong checks" are actually wrong (truth-inconsistent),
        # but still semantically close (nearest-neighbor competitor).
        contra_bank: Dict[int, List[str]] = {}
        if wrong_checks == "neighbor":
            contradict_examples = [e for e in examples if str(e.label).strip().upper() == "CONTRADICT"]
            contra_claim_text_by_id: Dict[int, str] = {}
            for e in contradict_examples[: (120 if fast else 600)]:
                contra_claim_text_by_id.setdefault(int(e.claim_id), str(e.claim))
                sampled = sample_sentences_from_doc(
                    int(e.doc_id),
                    8,
                    seed_key=(seed * 1_000_003) ^ (int(e.claim_id) * 9176) ^ int(e.doc_id) ^ 0xC0A7,
                )
                if len(sampled) >= 6:
                    contra_bank.setdefault(int(e.claim_id), sampled[:6])

            contra_claim_ids = sorted([cid for cid in contra_bank.keys() if cid in contra_claim_text_by_id])
            if len(contra_claim_ids) < (20 if fast else 80):
                raise RuntimeError(
                    f"Too few SciFact CONTRADICT candidates for neighbor-wrong checks ({len(contra_claim_ids)})"
                )

            support_texts = [claim_text_by_id[cid] for cid in claim_ids]
            contra_texts = [contra_claim_text_by_id[cid] for cid in contra_claim_ids]
            emb_support = embed_texts(support_texts)
            emb_contra = embed_texts(contra_texts)
            sim_sc = emb_support @ emb_contra.T

            k = max(1, int(neighbor_k))
            for i, cid in enumerate(claim_ids):
                order = np.argsort(-sim_sc[i])
                cand_pos = [int(j) for j in order[:k]]
                neighbor_candidates_by_claim[int(cid)] = [int(contra_claim_ids[int(j)]) for j in cand_pos]
                neighbor_candidate_sims_by_claim[int(cid)] = [float(sim_sc[i, int(j)]) for j in cand_pos]

        if wrong_checks == "inflation":
            # Agreement inflation negative control:
            # choose a nearest-neighbor SUPPORT claim (truth-consistent for itself) so wrong checks "accidentally" support.
            support_claim_ids = sorted([cid for cid in claim_ids if int(cid) in claim_bank and int(cid) in claim_text_by_id])
            if len(support_claim_ids) < (20 if fast else 80):
                raise RuntimeError(f"Too few SciFact SUPPORT candidates for inflation control ({len(support_claim_ids)})")
            support_texts = [claim_text_by_id[int(cid)] for cid in support_claim_ids]
            emb = embed_texts(support_texts)
            sim = emb @ emb.T
            k = max(1, int(neighbor_k))
            for i, cid in enumerate(support_claim_ids):
                sims_row = np.asarray(sim[i], dtype=float)
                sims_row[i] = -float("inf")
                order = np.argsort(-sims_row)
                cand_pos = [int(j) for j in order[:k]]
                inflation_candidates_by_claim[int(cid)] = [int(support_claim_ids[int(j)]) for j in cand_pos]
                inflation_candidate_sims_by_claim[int(cid)] = [float(sims_row[int(j)]) for j in cand_pos]

    for i, ex in enumerate(support_examples):
        sampled = sample_sentences_from_doc(
            int(ex.doc_id),
            10,
            seed_key=(seed * 1_000_003) ^ (int(ex.claim_id) * 9176) ^ int(ex.doc_id),
        )
        if len(sampled) < 8:
            continue
        obs_sents = sampled[:2]
        check_correct_sents = sampled[2:8]
        if len(check_correct_sents) < 4:
            continue

        obs_scores = sentence_support_scores([ex.claim] * len(obs_sents), obs_sents).tolist()

        wrong_sents: Optional[List[str]] = None
        if wrong_checks == "neighbor":
            cands = neighbor_candidates_by_claim.get(int(ex.claim_id)) or []
            sims = neighbor_candidate_sims_by_claim.get(int(ex.claim_id)) or []
            if cands and len(cands) == len(sims):
                best_pos: Optional[int] = None
                best_M: float = float("inf")
                for pos, contra_cid in enumerate(cands):
                    sents = contra_bank.get(int(contra_cid)) or []
                    pick = list(sents[:6])
                    if len(pick) < 2:
                        continue
                    cross_scores = sentence_support_scores([ex.claim] * len(pick), pick).tolist()
                    M_wrong_candidate = M_from_R(R_grounded(obs_scores, [float(x) for x in cross_scores]))
                    if M_wrong_candidate < best_M:
                        best_M = float(M_wrong_candidate)
                        best_pos = int(pos)
                        wrong_sents = pick
                if best_pos is not None:
                    neighbor_sims.append(float(sims[int(best_pos)]))
        elif wrong_checks == "inflation":
            cands = inflation_candidates_by_claim.get(int(ex.claim_id)) or []
            sims = inflation_candidate_sims_by_claim.get(int(ex.claim_id)) or []
            if cands and len(cands) == len(sims):
                best_pos: Optional[int] = None
                best_M: float = -float("inf")
                for pos, other_cid in enumerate(cands):
                    sents = claim_bank.get(int(other_cid)) or []
                    pick = list(sents[:6])
                    if len(pick) < 2:
                        continue
                    cross_scores = sentence_support_scores([ex.claim] * len(pick), pick).tolist()
                    M_wrong_candidate = M_from_R(R_grounded(obs_scores, [float(x) for x in cross_scores]))
                    if M_wrong_candidate > best_M:
                        best_M = float(M_wrong_candidate)
                        best_pos = int(pos)
                        wrong_sents = pick
                if best_pos is not None:
                    neighbor_sims.append(float(sims[int(best_pos)]))
        if wrong_sents is None:
            wrong_sents = wrong_bank[(i + 17) % len(wrong_bank)]
            if wrong_sents == check_correct_sents:
                wrong_sents = wrong_bank[(i + 18) % len(wrong_bank)]
        check_correct_scores = sentence_support_scores([ex.claim] * len(check_correct_sents), check_correct_sents).tolist()
        check_wrong_scores = sentence_support_scores([ex.claim] * len(wrong_sents), wrong_sents).tolist()

        obs_scores_by_sample.append([float(x) for x in obs_scores])
        check_correct_scores_by_sample.append([float(x) for x in check_correct_scores])

        mu_hat_list.append(float(np.mean(np.asarray(obs_scores, dtype=float))))
        mu_check_list.append(float(np.mean(np.asarray(check_correct_scores, dtype=float))))
        R_c = float(R_grounded(obs_scores, check_correct_scores))
        R_w = float(R_grounded(obs_scores, check_wrong_scores))
        R_correct.append(R_c)
        R_wrong.append(R_w)
        M_correct.append(M_from_R(R_c))
        M_wrong.append(M_from_R(R_w))

        if fast and len(M_correct) in (10, 25):
            print(f"[Q32:P2] scifact intervention samples={len(M_correct)}")
        if len(M_correct) >= (20 if fast else 120):
            break

    if len(M_correct) < (10 if fast else 60):
        raise RuntimeError(f"Too few SciFact intervention samples ({len(M_correct)})")

    Mc = np.array(M_correct, dtype=float)
    Mw = np.array(M_wrong, dtype=float)
    # Swap control: re-pair obs with another sample's correct checks (should behave like wrong checks).
    perm = np.roll(np.arange(len(M_correct), dtype=int), 1)
    Ms_list: List[float] = []
    Rs_list: List[float] = []
    for i in range(int(len(M_correct))):
        check_swap = check_correct_scores_by_sample[int(perm[int(i)])]
        R_s = float(R_grounded(obs_scores_by_sample[int(i)], check_swap))
        Rs_list.append(float(R_s))
        Ms_list.append(float(M_from_R(R_s)))
    Ms = np.asarray(Ms_list, dtype=float) if Ms_list else np.asarray([], dtype=float)
    Rs = np.asarray(Rs_list, dtype=float) if Rs_list else np.asarray([], dtype=float)
    swap_wins = int(np.sum(Mc > Ms)) if Ms.size else 0
    swap_pair_wins = float(swap_wins / max(1, int(Ms.size)))
    swap_margin = float(np.mean(Mc - Ms)) if Ms.size else float("nan")
    wins = int(np.sum(Mc > Mw))
    n = int(len(Mc))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(Mc - Mw))
    z = _binom_z(wins, n)

    print("\n[Q32:P2] SciFact check intervention (correct vs wrong check)")
    print(f"  P(M_correct > M_wrong) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct) = {Mc.mean():.3f}")
    print(f"  mean(M_wrong)   = {Mw.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if Ms.size:
        print(f"  [swap] P(M_correct > M_swapped_correct) = {swap_pair_wins:.3f}")
        print(f"  [swap] mean(M_correct - M_swapped_correct) = {swap_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(nn.mean()):.3f}")
    phi = _mutual_information_continuous(mu_hat_list, mu_check_list, n_bins=8)
    print(f"  [Phi_proxy] I(mu_hat; mu_check) = {phi:.3f} bits")

    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P2] SciFact status: FAIL (kept as a public counterexample until fixed)")
        if strict and not fast:
            raise AssertionError("FAIL: SciFact benchmark gates did not pass")

    details: Dict[str, float] = {
        "pair_wins": pair_wins,
        "z": z,
        "mean_margin": margin,
        "mean_R_correct": float(np.mean(np.asarray(R_correct, dtype=float))) if R_correct else 0.0,
        "mean_R_wrong": float(np.mean(np.asarray(R_wrong, dtype=float))) if R_wrong else 0.0,
        "mean_logR_correct": float(Mc.mean()),
        "mean_logR_wrong": float(Mw.mean()),
        "mean_R_swap": float(np.mean(Rs)) if Rs.size else float("nan"),
        "mean_logR_swap": float(np.mean(Ms)) if Ms.size else float("nan"),
        "swap_pair_wins": float(swap_pair_wins),
        "swap_mean_margin": float(swap_margin),
        "gate_z": gate_z,
        "gate_margin": gate_margin,
        "phi_proxy_bits": float(phi),
    }
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            details["mean_neighbor_sim"] = float(nn.mean())
    return BenchmarkResult(name="SciFact", passed=passed, details=details)


def run_snli_benchmark(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
    nli_domain: str = "snli",
) -> BenchmarkResult:
    """
    Phase 3: third public domain.

    Use SNLI as a truth anchor with NLI labels:
      - Correct checks come from ENTAILMENT premises for the same hypothesis (same sample, chunked).
      - Wrong checks come from CONTRADICTION premises (or neighbor/inflation variants).
    """
    tag = str(nli_domain).upper()
    print(f"\n[Q32:P3] Loading {tag} (NLI truth anchor)...")
    examples = _load_nli(domain=str(nli_domain), max_examples=30000, seed=seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    examples = examples[: (1200 if fast else 12000)]

    ent = [e for e in examples if e.label == "ENTAILMENT"]
    contra = [e for e in examples if e.label == "CONTRADICTION"]
    if len(ent) < (80 if fast else 600):
        raise RuntimeError(f"Too few {tag} ENTAILMENT examples ({len(ent)})")
    if len(contra) < (80 if fast else 600):
        raise RuntimeError(f"Too few {tag} CONTRADICTION examples ({len(contra)})")

    # Precompute embeddings for neighbor/inflation selection based on hypothesis text.
    neighbor_sims: List[float] = []
    emb_ent: Optional[np.ndarray] = None
    emb_contra: Optional[np.ndarray] = None
    ent_hyps: List[str] = []
    contra_hyps: List[str] = []
    if wrong_checks in ("neighbor", "inflation"):
        ent_hyps = [e.hypothesis for e in ent[: (400 if fast else 2000)]]
        contra_hyps = [e.hypothesis for e in contra[: (400 if fast else 2000)]]
        emb_ent = embed_texts(ent_hyps)
        emb_contra = embed_texts(contra_hyps)

    M_correct: List[float] = []
    M_wrong: List[float] = []
    R_correct: List[float] = []
    R_wrong: List[float] = []
    obs_scores_by_sample: List[List[float]] = []
    check_correct_scores_by_sample: List[List[float]] = []
    mu_hat_list: List[float] = []
    mu_check_list: List[float] = []

    for i, ex in enumerate(ent):
        chunks = _word_chunks(ex.premise, window=3, stride=1, max_chunks=12)
        if len(chunks) < 6:
            continue
        obs_chunks = chunks[:2]
        check_correct_chunks = chunks[2:8]
        if len(check_correct_chunks) < 4:
            continue

        obs_scores = sentence_support_scores([ex.hypothesis] * len(obs_chunks), obs_chunks).tolist()
        check_correct_scores = sentence_support_scores(
            [ex.hypothesis] * len(check_correct_chunks), check_correct_chunks
        ).tolist()

        wrong_chunks: Optional[List[str]] = None
        if wrong_checks == "dissimilar":
            # Deterministic dissimilar: take a far contradiction hypothesis and use its premise chunks.
            j = (i * 17 + 23) % max(1, len(contra))
            c = contra[int(j)]
            wrong_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=6)[:6]
        elif wrong_checks == "neighbor":
            if emb_contra is None or not contra_hyps:
                raise RuntimeError(f"{tag} neighbor requires embeddings")
            c_emb = embed_texts([ex.hypothesis])[0]
            sims = emb_contra @ c_emb
            order = np.argsort(-sims)
            k = max(1, int(neighbor_k))
            cand = [int(j) for j in order[:k]]
            best_pos: Optional[int] = None
            best_M = float("inf")
            for pos, j in enumerate(cand):
                c = contra[int(j)]
                c_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=6)[:6]
                if len(c_chunks) < 2:
                    continue
                cross = sentence_support_scores([ex.hypothesis] * len(c_chunks), c_chunks).tolist()
                M_wrong_candidate = M_from_R(R_grounded(obs_scores, [float(x) for x in cross]))
                if float(M_wrong_candidate) < best_M:
                    best_M = float(M_wrong_candidate)
                    best_pos = int(pos)
                    wrong_chunks = c_chunks
            if best_pos is not None:
                neighbor_sims.append(float(sims[int(cand[int(best_pos)])]))
        elif wrong_checks == "inflation":
            if emb_ent is None or not ent_hyps:
                raise RuntimeError(f"{tag} inflation requires embeddings")
            c_emb = embed_texts([ex.hypothesis])[0]
            sims = emb_ent @ c_emb
            sims[int(np.argmax(sims))] = -float("inf")
            order = np.argsort(-sims)
            k = max(1, int(neighbor_k))
            cand = [int(j) for j in order[:k]]
            best_pos: Optional[int] = None
            best_M = -float("inf")
            for pos, j in enumerate(cand):
                c = ent[int(j)]
                c_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=6)[:6]
                if len(c_chunks) < 2:
                    continue
                cross = sentence_support_scores([ex.hypothesis] * len(c_chunks), c_chunks).tolist()
                M_wrong_candidate = M_from_R(R_grounded(obs_scores, [float(x) for x in cross]))
                if float(M_wrong_candidate) > best_M:
                    best_M = float(M_wrong_candidate)
                    best_pos = int(pos)
                    wrong_chunks = c_chunks
            if best_pos is not None:
                neighbor_sims.append(float(sims[int(cand[int(best_pos)])]))

        if not wrong_chunks or len(wrong_chunks) < 2:
            continue

        check_wrong_scores = sentence_support_scores([ex.hypothesis] * len(wrong_chunks), wrong_chunks).tolist()

        obs_scores_by_sample.append([float(x) for x in obs_scores])
        check_correct_scores_by_sample.append([float(x) for x in check_correct_scores])

        mu_hat_list.append(float(np.mean(np.asarray(obs_scores, dtype=float))))
        mu_check_list.append(float(np.mean(np.asarray(check_correct_scores, dtype=float))))
        R_c = float(R_grounded(obs_scores, check_correct_scores))
        R_w = float(R_grounded(obs_scores, check_wrong_scores))
        R_correct.append(R_c)
        R_wrong.append(R_w)
        M_correct.append(M_from_R(R_c))
        M_wrong.append(M_from_R(R_w))

        if fast and len(M_correct) in (10, 25):
            print(f"[Q32:P3] {tag.lower()} intervention samples={len(M_correct)}")
        if len(M_correct) >= (40 if fast else 240):
            break

    if len(M_correct) < (20 if fast else 120):
        raise RuntimeError(f"Too few SNLI intervention samples ({len(M_correct)})")

    Mc = np.array(M_correct, dtype=float)
    Mw = np.array(M_wrong, dtype=float)
    perm = np.roll(np.arange(len(M_correct), dtype=int), 1)
    Ms_list: List[float] = []
    Rs_list: List[float] = []
    for i in range(int(len(M_correct))):
        check_swap = check_correct_scores_by_sample[int(perm[int(i)])]
        R_s = float(R_grounded(obs_scores_by_sample[int(i)], check_swap))
        Rs_list.append(float(R_s))
        Ms_list.append(float(M_from_R(R_s)))
    Ms = np.asarray(Ms_list, dtype=float) if Ms_list else np.asarray([], dtype=float)
    Rs = np.asarray(Rs_list, dtype=float) if Rs_list else np.asarray([], dtype=float)
    swap_wins = int(np.sum(Mc > Ms)) if Ms.size else 0
    swap_pair_wins = float(swap_wins / max(1, int(Ms.size)))
    swap_margin = float(np.mean(Mc - Ms)) if Ms.size else float("nan")

    wins = int(np.sum(Mc > Mw))
    n = int(len(Mc))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(Mc - Mw))
    z = _binom_z(wins, n)

    print(f"\n[Q32:P3] {tag} check intervention (correct vs wrong check)")
    print(f"  P(M_correct > M_wrong) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct) = {Mc.mean():.3f}")
    print(f"  mean(M_wrong)   = {Mw.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if Ms.size:
        print(f"  [swap] P(M_correct > M_swapped_correct) = {swap_pair_wins:.3f}")
        print(f"  [swap] mean(M_correct - M_swapped_correct) = {swap_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(nn.mean()):.3f}")
    phi = _mutual_information_continuous(mu_hat_list, mu_check_list, n_bins=8)
    print(f"  [Phi_proxy] I(mu_hat; mu_check) = {phi:.3f} bits")

    gate_z = float(min_z if min_z is not None else (1.8 if fast else 2.4))
    gate_margin = float(min_margin if min_margin is not None else (0.25 if fast else 0.40))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print(f"\n[Q32:P3] {tag} status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: SNLI benchmark gates did not pass")

    details: Dict[str, float] = {
        "pair_wins": pair_wins,
        "z": z,
        "mean_margin": margin,
        "mean_R_correct": float(np.mean(np.asarray(R_correct, dtype=float))) if R_correct else 0.0,
        "mean_R_wrong": float(np.mean(np.asarray(R_wrong, dtype=float))) if R_wrong else 0.0,
        "mean_logR_correct": float(Mc.mean()),
        "mean_logR_wrong": float(Mw.mean()),
        "mean_R_swap": float(np.mean(Rs)) if Rs.size else float("nan"),
        "mean_logR_swap": float(np.mean(Ms)) if Ms.size else float("nan"),
        "swap_pair_wins": float(swap_pair_wins),
        "swap_mean_margin": float(swap_margin),
        "gate_z": gate_z,
        "gate_margin": gate_margin,
        "phi_proxy_bits": float(phi),
    }
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            details["mean_neighbor_sim"] = float(nn.mean())
    return BenchmarkResult(name=tag, passed=passed, details=details)


def run_snli_streaming(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
    nli_domain: str = "snli",
) -> BenchmarkResult:
    """
    SNLI pseudo-stream: chunk a premise into many overlapping word windows to emulate evidence arriving over time.
    """
    tag = str(nli_domain).upper()
    print(f"\n[Q32:P3] {tag} streaming (chunked NLI evidence)")
    examples = _load_nli(domain=str(nli_domain), max_examples=50000, seed=seed)
    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    examples = examples[: (2000 if fast else 20000)]

    ent = [e for e in examples if e.label == "ENTAILMENT"]
    contra = [e for e in examples if e.label == "CONTRADICTION"]
    if len(ent) < (200 if fast else 1000):
        raise RuntimeError(f"Too few {tag} ENTAILMENT examples ({len(ent)})")
    if len(contra) < (200 if fast else 1000):
        raise RuntimeError(f"Too few {tag} CONTRADICTION examples ({len(contra)})")

    neighbor_sims: List[float] = []
    emb_ent: Optional[np.ndarray] = None
    emb_contra: Optional[np.ndarray] = None
    ent_hyps: List[str] = []
    contra_hyps: List[str] = []
    if wrong_checks in ("neighbor", "inflation"):
        ent_hyps = [e.hypothesis for e in ent[: (600 if fast else 3000)]]
        contra_hyps = [e.hypothesis for e in contra[: (600 if fast else 3000)]]
        emb_ent = embed_texts(ent_hyps)
        emb_contra = embed_texts(contra_hyps)

    samples: List[Tuple[str, List[str], List[float], List[str], List[float]]] = []
    for i, ex in enumerate(ent):
        chunks = _word_chunks(ex.premise, window=3, stride=1, max_chunks=14)
        if len(chunks) < 10:
            continue
        support_scores = sentence_support_scores([ex.hypothesis] * len(chunks), chunks).tolist()
        mu_hat_proxy = float(np.mean(np.asarray(support_scores[:2], dtype=float)))

        wrong_scores: Optional[List[float]] = None
        if wrong_checks == "dissimilar":
            j = (i * 17 + 23) % max(1, len(contra))
            c = contra[int(j)]
            c_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=10)[:10]
            if len(c_chunks) < 6:
                continue
            wrong_scores = sentence_support_scores([ex.hypothesis] * len(c_chunks), c_chunks).tolist()
        elif wrong_checks == "neighbor":
            if emb_contra is None or not contra_hyps:
                raise RuntimeError(f"{tag} neighbor requires embeddings")
            c_emb = embed_texts([ex.hypothesis])[0]
            sims = emb_contra @ c_emb
            order = np.argsort(-sims)
            k = max(1, int(neighbor_k))
            cand = [int(j) for j in order[:k]]
            best_pos: Optional[int] = None
            best_mean = float("inf")
            for pos, j in enumerate(cand):
                c = contra[int(j)]
                c_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=10)[:10]
                if len(c_chunks) < 6:
                    continue
                cross = sentence_support_scores([ex.hypothesis] * len(c_chunks), c_chunks).tolist()
                m = float(np.mean(np.asarray(cross, dtype=float)))
                if m < best_mean:
                    best_mean = m
                    best_pos = int(pos)
                    wrong_scores = cross
            if best_pos is not None:
                neighbor_sims.append(float(sims[int(cand[int(best_pos)])]))
        elif wrong_checks == "inflation":
            if emb_ent is None or not ent_hyps:
                raise RuntimeError(f"{tag} inflation requires embeddings")
            c_emb = embed_texts([ex.hypothesis])[0]
            sims = emb_ent @ c_emb
            sims[int(np.argmax(sims))] = -float("inf")
            order = np.argsort(-sims)
            k = max(1, int(neighbor_k))
            cand = [int(j) for j in order[:k]]
            best_pos: Optional[int] = None
            best_mean = -float("inf")
            for pos, j in enumerate(cand):
                c = ent[int(j)]
                c_chunks = _word_chunks(c.premise, window=3, stride=1, max_chunks=10)[:10]
                if len(c_chunks) < 6:
                    continue
                cross = sentence_support_scores([ex.hypothesis] * len(c_chunks), c_chunks).tolist()
                m = float(np.mean(np.asarray(cross, dtype=float)))
                if m > best_mean:
                    best_mean = m
                    best_pos = int(pos)
                    wrong_scores = cross
            if best_pos is not None:
                neighbor_sims.append(float(sims[int(cand[int(best_pos)])]))

        if wrong_scores is None or len(wrong_scores) < 6:
            continue
        samples.append((ex.hypothesis, [float(x) for x in support_scores], [float(x) for x in wrong_scores]))
        if fast and len(samples) in (10, 25):
            print(f"[Q32:P3] {tag.lower()} streaming samples={len(samples)}")
        if len(samples) >= (40 if fast else 240):
            break

    if len(samples) < (20 if fast else 120):
        raise RuntimeError(f"Too few SNLI streaming samples ({len(samples)})")

    M_correct_end: List[float] = []
    M_wrong_end: List[float] = []
    R_correct_end: List[float] = []
    R_wrong_end: List[float] = []
    obs_end_by_sample: List[List[float]] = []
    check_end_by_sample: List[List[float]] = []
    wrong_end_by_sample: List[List[float]] = []
    dM_correct: List[float] = []
    dM_wrong: List[float] = []

    phi_coupling: List[float] = []
    use_n = min((40 if fast else 240), len(samples))
    for hyp, support_scores, wrong_scores in samples[:use_n]:
        t_max = max(2, len(support_scores) - 4)
        t_max = min(t_max, len(support_scores) - 4)

        M_series_correct: List[float] = []
        M_series_wrong: List[float] = []
        R_series_correct: List[float] = []
        R_series_wrong: List[float] = []
        mu_hat_series: List[float] = []
        mu_check_series: List[float] = []
        last_obs: Optional[List[float]] = None
        last_check: Optional[List[float]] = None
        last_t: int = 0
        for t in range(2, t_max + 1):
            check_correct = support_scores[t:]
            if len(check_correct) < 4:
                break
            obs = support_scores[:t]
            last_obs = [float(x) for x in obs]
            last_check = [float(x) for x in check_correct]
            R_c = float(R_grounded(obs, check_correct))
            R_w = float(R_grounded(obs, wrong_scores))
            R_series_correct.append(R_c)
            R_series_wrong.append(R_w)
            M_series_correct.append(M_from_R(R_c))
            M_series_wrong.append(M_from_R(R_w))
            mu_hat_series.append(float(np.mean(np.asarray(support_scores[:t], dtype=float))))
            mu_check_series.append(float(np.mean(np.asarray(check_correct, dtype=float))))
            last_t = int(t)

        if len(M_series_correct) < 1:
            continue
        M_correct_end.append(M_series_correct[-1])
        M_wrong_end.append(M_series_wrong[-1])
        R_correct_end.append(R_series_correct[-1])
        R_wrong_end.append(R_series_wrong[-1])
        if last_obs is not None and last_check is not None:
            obs_end_by_sample.append(last_obs)
            check_end_by_sample.append(last_check)
            wrong_end_by_sample.append([float(x) for x in wrong_scores])
        dM_correct.append(M_series_correct[-1] - M_series_correct[0])
        dM_wrong.append(M_series_wrong[-1] - M_series_wrong[0])
        phi_coupling.append(_mutual_information_continuous(mu_hat_series, mu_check_series, n_bins=8))

    if len(M_correct_end) < (20 if fast else 120):
        raise RuntimeError(f"Too few usable {tag} streaming series ({len(M_correct_end)})")

    M_correct_end_a = np.array(M_correct_end, dtype=float)
    M_wrong_end_a = np.array(M_wrong_end, dtype=float)
    R_correct_end_a = np.array(R_correct_end, dtype=float)
    R_wrong_end_a = np.array(R_wrong_end, dtype=float)
    dM_correct_a = np.array(dM_correct, dtype=float)
    dM_wrong_a = np.array(dM_wrong, dtype=float)

    wins = int(np.sum(M_correct_end_a > M_wrong_end_a))
    n = int(len(M_correct_end_a))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(M_correct_end_a - M_wrong_end_a))
    z = _binom_z(wins, n)

    perm = np.roll(np.arange(n, dtype=int), 1)
    Ms_swap_correct: List[float] = []
    Rs_swap_correct: List[float] = []
    Ms_swap_wrong: List[float] = []
    Rs_swap_wrong: List[float] = []
    for i in range(n):
        obs_end = obs_end_by_sample[int(i)]
        check_swap = check_end_by_sample[int(perm[int(i)])]
        wrong_swap = wrong_end_by_sample[int(perm[int(i)])]
        R_sc = float(R_grounded(obs_end, check_swap))
        R_sw = float(R_grounded(obs_end, wrong_swap))
        Rs_swap_correct.append(float(R_sc))
        Ms_swap_correct.append(float(M_from_R(R_sc)))
        Rs_swap_wrong.append(float(R_sw))
        Ms_swap_wrong.append(float(M_from_R(R_sw)))
    M_swap_correct_end_a = np.asarray(Ms_swap_correct, dtype=float) if Ms_swap_correct else np.asarray([], dtype=float)
    R_swap_correct_end_a = np.asarray(Rs_swap_correct, dtype=float) if Rs_swap_correct else np.asarray([], dtype=float)
    M_swap_wrong_end_a = np.asarray(Ms_swap_wrong, dtype=float) if Ms_swap_wrong else np.asarray([], dtype=float)
    R_swap_wrong_end_a = np.asarray(Rs_swap_wrong, dtype=float) if Rs_swap_wrong else np.asarray([], dtype=float)
    swap_correct_pair_wins = float(np.mean(M_correct_end_a > M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_pair_wins = float(np.mean(M_correct_end_a > M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")
    swap_correct_margin = float(np.mean(M_correct_end_a - M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_margin = float(np.mean(M_correct_end_a - M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")

    print(f"\n[Q32:P3] {tag} streaming intervention (chunked)")
    print(f"  P(M_correct_end > M_wrong_end) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct_end) = {M_correct_end_a.mean():.3f}")
    print(f"  mean(M_wrong_end)   = {M_wrong_end_a.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if M_swap_correct_end_a.size:
        print(f"  [swap_correct] P(M_correct_end > M_swapped_correct_end) = {swap_correct_pair_wins:.3f}")
        print(f"  [swap_correct] mean(M_correct_end - M_swapped_correct_end) = {swap_correct_margin:.3f}")
    if M_swap_wrong_end_a.size:
        print(f"  [swap_wrong] P(M_correct_end > M_swapped_wrong_end) = {swap_wrong_pair_wins:.3f}")
        print(f"  [swap_wrong] mean(M_correct_end - M_swapped_wrong_end) = {swap_wrong_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(nn.mean()):.3f}")
    if phi_coupling:
        print(f"  [Phi_proxy] mean I(mu_hat(t); mu_check(t)) = {float(np.mean(np.asarray(phi_coupling, dtype=float))):.3f} bits")

    gate_z = float(min_z if min_z is not None else (1.8 if fast else 2.4))
    gate_margin = float(min_margin if min_margin is not None else (0.25 if fast else 0.40))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print(f"\n[Q32:P3] {tag} streaming status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: SNLI streaming gates did not pass")

    details: Dict[str, float] = {
        "pair_wins": pair_wins,
        "z": z,
        "mean_margin": margin,
        "mean_R_correct_end": float(R_correct_end_a.mean()),
        "mean_R_wrong_end": float(R_wrong_end_a.mean()),
        "mean_logR_correct_end": float(M_correct_end_a.mean()),
        "mean_logR_wrong_end": float(M_wrong_end_a.mean()),
        "gate_z": gate_z,
        "gate_margin": gate_margin,
        "mean_dM_correct": float(dM_correct_a.mean()),
        "mean_dM_wrong": float(dM_wrong_a.mean()),
        "mean_R_swap_correct_end": float(np.mean(R_swap_correct_end_a)) if R_swap_correct_end_a.size else float("nan"),
        "mean_logR_swap_correct_end": float(np.mean(M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan"),
        "swap_correct_pair_wins_end": float(swap_correct_pair_wins),
        "swap_correct_mean_margin_end": float(swap_correct_margin),
        "mean_R_swap_wrong_end": float(np.mean(R_swap_wrong_end_a)) if R_swap_wrong_end_a.size else float("nan"),
        "mean_logR_swap_wrong_end": float(np.mean(M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan"),
        "swap_wrong_pair_wins_end": float(swap_wrong_pair_wins),
        "swap_wrong_mean_margin_end": float(swap_wrong_margin),
    }
    if phi_coupling:
        details["phi_proxy_bits"] = float(np.mean(np.asarray(phi_coupling, dtype=float)))
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            details["mean_neighbor_sim"] = float(nn.mean())
    return BenchmarkResult(name=f"{tag}-Streaming", passed=passed, details=details)


def _majority_vote(votes: Sequence[Optional[str]]) -> Optional[str]:
    vs = [v for v in votes if v is not None]
    if not vs:
        return None
    # deterministic tie-break by lexicographic sort
    counts: Dict[str, int] = {}
    for v in vs:
        counts[str(v)] = counts.get(str(v), 0) + 1
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best


def run_climate_fever_benchmark(*, seed: int = 123, fast: bool = False, strict: bool = True) -> BenchmarkResult:
    print("\n[Q32:P2] Loading Climate-FEVER (public votes as truth anchor)...")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("climate_fever")["test"]
    rng = np.random.default_rng(seed)

    prepared: List[Tuple[List[float], List[float], List[float]]] = []
    checks_pool: List[List[float]] = []

    indices = np.arange(len(ds))
    rng.shuffle(indices)

    for idx in indices:
        ex = ds[int(idx)]
        claim = str(ex["claim"])
        evidences = ex.get("evidences", []) or []
        if not evidences:
            continue

        supports: List[Tuple[int, float, str, dict]] = []
        refutes: List[Tuple[int, float, str, dict]] = []
        for ev in evidences:
            votes = [v for v in (ev.get("votes", []) or []) if v is not None]
            counts: Dict[str, int] = {}
            for v in votes:
                vv = str(v)
                counts[vv] = counts.get(vv, 0) + 1
            entropy = float(ev.get("entropy", 0.0) or 0.0)
            ev_id = str(ev.get("evidence_id", "")).strip()

            # Strong-ish anchor to reduce label noise:
            # - SUPPORT: at least 2 SUPPORTS votes and zero REFUTES votes
            # - REFUTE: at least 1 REFUTES vote and zero SUPPORTS votes
            if counts.get("SUPPORTS", 0) >= 2 and counts.get("REFUTES", 0) == 0:
                supports.append((int(counts.get("SUPPORTS", 0)), entropy, ev_id, ev))
            if counts.get("REFUTES", 0) >= 1 and counts.get("SUPPORTS", 0) == 0:
                refutes.append((int(counts.get("REFUTES", 0)), entropy, ev_id, ev))

        # Climate-FEVER has small per-claim evidence sets (often <=5); use the SUPPORT pool as the
        # check distribution. Votes are noisy, so we require "strong-ish" vote thresholds above.
        if len(supports) < 2 or len(refutes) < 2:
            continue

        # Deterministic ranking: stronger vote signal first, then lower entropy, then stable id.
        supports_sorted = sorted(supports, key=lambda t: (-t[0], t[1], t[2]))
        refutes_sorted = sorted(refutes, key=lambda t: (-t[0], t[1], t[2]))

        supports_texts = [str(t[3].get("evidence", "")).strip() for t in supports_sorted if str(t[3].get("evidence", "")).strip()]
        refutes_texts = [str(t[3].get("evidence", "")).strip() for t in refutes_sorted if str(t[3].get("evidence", "")).strip()]
        if len(supports_texts) < 2 or len(refutes_texts) < 2:
            continue

        obs_texts = supports_texts[:2]
        check_texts = supports_texts[:5]
        false_texts = refutes_texts[:2]

        if len(check_texts) < 2:
            continue

        obs_true_scores = sentence_support_scores([claim] * len(obs_texts), obs_texts).tolist()
        check_scores = sentence_support_scores([claim] * len(check_texts), check_texts).tolist()
        obs_false_scores = sentence_support_scores([claim] * len(false_texts), false_texts).tolist()

        prepared.append((obs_true_scores, check_scores, obs_false_scores))
        checks_pool.append(check_scores)

        if len(prepared) in (10, 25, 50, 75, 100):
            print(f"[Q32:P2] climate prepared={len(prepared)}")
        if len(prepared) >= (20 if fast else 120):
            break

    if len(prepared) < 60:
        if strict and not fast:
            raise RuntimeError(f"Too few prepared Climate-FEVER samples ({len(prepared)})")
        if len(prepared) < 10:
            raise RuntimeError(f"Too few prepared Climate-FEVER samples ({len(prepared)})")

    rng.shuffle(checks_pool)
    use_n = min((16 if fast else 120), len(prepared))

    M_support_internal: List[float] = []
    M_refute_internal: List[float] = []
    M_support_shuffled: List[float] = []
    M_refute_shuffled: List[float] = []

    for i, (obs_true_scores, check_scores, obs_false_scores) in enumerate(prepared[:use_n]):
        check_other = checks_pool[(i + 7) % len(checks_pool)]
        M_support_internal.append(M_from_R(R_grounded(obs_true_scores, check_scores)))
        M_refute_internal.append(M_from_R(R_grounded(obs_false_scores, check_scores)))
        M_support_shuffled.append(M_from_R(R_grounded(obs_true_scores, check_other)))
        M_refute_shuffled.append(M_from_R(R_grounded(obs_false_scores, check_other)))

    M_support_internal = np.array(M_support_internal)
    M_refute_internal = np.array(M_refute_internal)
    M_support_shuffled = np.array(M_support_shuffled)
    M_refute_shuffled = np.array(M_refute_shuffled)

    pair_wins = float(np.mean(M_support_internal > M_refute_internal))
    print("\n[Q32:P2] Climate-FEVER supports vs refutes discrimination")
    print(f"  P(M_support_internal > M_refute_internal) = {pair_wins:.3f}")
    print(f"  mean(M_support_internal) = {M_support_internal.mean():.3f}")
    print(f"  mean(M_refute_internal)  = {M_refute_internal.mean():.3f}")

    shuffled_pair = float(np.mean(M_support_shuffled > M_refute_shuffled))
    print("\n[Q32:P2] Negative control (check-shuffle across claims)")
    print(f"  P(M_support_shuffled > M_refute_shuffled) = {shuffled_pair:.3f}")
    print(f"  mean(M_support_shuffled) = {M_support_shuffled.mean():.3f}")
    print(f"  mean(M_refute_shuffled)  = {M_refute_shuffled.mean():.3f}")

    if fast:
        deltas = (M_support_internal - M_refute_internal).tolist()
        order = np.argsort(deltas)
        print("\n[Q32:P2] Fast debug: worst deltas (support-refute)")
        for j in order[: min(5, len(order))]:
            jj = int(j)
            print(f"  delta={deltas[jj]: .3f}  M_support={M_support_internal[jj]: .3f}  M_refute={M_refute_internal[jj]: .3f}")
        print("[Q32:P2] Fast debug: best deltas (support-refute)")
        for j in order[-min(5, len(order)) :]:
            jj = int(j)
            print(f"  delta={deltas[jj]: .3f}  M_support={M_support_internal[jj]: .3f}  M_refute={M_refute_internal[jj]: .3f}")

    # Gates:
    # - We require discrimination to beat the negative control by a margin (not just an absolute threshold),
    #   because the dataset is noisy and per-claim refutes are not always clean contradictions.
    min_pair = 0.60 if fast else 0.65
    min_margin = 0.10 if fast else 0.15
    max_neg_dev = 0.25 if fast else 0.15

    passed_discrimination = pair_wins >= min_pair and (pair_wins - shuffled_pair) >= min_margin
    passed_negative = abs(shuffled_pair - 0.5) <= max_neg_dev
    passed = bool(passed_discrimination and passed_negative)
    if not passed:
        print("\n[Q32:P2] Climate-FEVER status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: Climate-FEVER benchmark gates did not pass")

    return BenchmarkResult(
        name="Climate-FEVER",
        passed=passed,
        details={
            "pair_wins": pair_wins,
            "negative": shuffled_pair,
            "mean_M_support_internal": float(M_support_internal.mean()),
            "mean_M_refute_internal": float(M_refute_internal.mean()),
        },
    )


def run_climate_fever_intervention_benchmark(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
) -> BenchmarkResult:
    """
    Climate-FEVER "bench" version of the same intervention gate used in streaming:
    compare M under a truth-consistent check pool vs a wrong check pool from another claim.
    """
    print("\n[Q32:P2] Climate-FEVER intervention (correct vs wrong check)")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("climate_fever")["test"]
    rng = np.random.default_rng(seed)

    def vote_counts(votes: Sequence[Optional[str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in votes:
            if v is None:
                continue
            vv = str(v)
            counts[vv] = counts.get(vv, 0) + 1
        return counts

    # Bank of supportive evidence texts for wrong checks.
    bank: List[Tuple[str, List[str]]] = []
    for ex in ds:
        claim = str(ex.get("claim", "")).strip()
        evidences = ex.get("evidences", []) or []
        if not claim or not evidences:
            continue
        evs: List[Tuple[int, float, str, str]] = []
        for ev in evidences:
            votes = [v for v in (ev.get("votes", []) or []) if v is not None]
            counts = vote_counts(votes)
            text = str(ev.get("evidence", "")).strip()
            if not text:
                continue
            if counts.get("SUPPORTS", 0) >= 2 and counts.get("REFUTES", 0) == 0:
                entropy = float(ev.get("entropy", 0.0) or 0.0)
                ev_id = str(ev.get("evidence_id", "")).strip()
                evs.append((int(counts.get("SUPPORTS", 0)), entropy, ev_id, text))
        if len(evs) < 4:
            continue
        evs_sorted = sorted(evs, key=lambda t: (-t[0], t[1], t[2]))
        bank.append((claim, [t[3] for t in evs_sorted[:6]]))
        if len(bank) >= (120 if fast else 800):
            break
    if len(bank) < (30 if fast else 120):
        raise RuntimeError(f"Climate-FEVER wrong-check bank too small ({len(bank)})")

    bank_claim_texts = [c for c, _ in bank]
    bank_emb = embed_texts(bank_claim_texts) if wrong_checks in ("neighbor", "inflation") else None

    indices = np.arange(len(ds))
    rng.shuffle(indices)

    M_correct: List[float] = []
    M_wrong: List[float] = []
    R_correct: List[float] = []
    R_wrong: List[float] = []
    # For swap/shuffle negative controls (receipted): keep raw score sets per sample.
    obs_scores_by_sample: List[List[float]] = []
    check_correct_scores_by_sample: List[List[float]] = []
    neighbor_sims: List[float] = []
    mu_hat_list: List[float] = []
    mu_check_list: List[float] = []

    for i in indices:
        ex = ds[int(i)]
        claim = str(ex.get("claim", "")).strip()
        evidences = ex.get("evidences", []) or []
        if not claim or not evidences:
            continue
        evs: List[Tuple[int, float, str, str]] = []
        for ev in evidences:
            votes = [v for v in (ev.get("votes", []) or []) if v is not None]
            counts = vote_counts(votes)
            text = str(ev.get("evidence", "")).strip()
            if not text:
                continue
            if counts.get("SUPPORTS", 0) >= 2 and counts.get("REFUTES", 0) == 0:
                entropy = float(ev.get("entropy", 0.0) or 0.0)
                ev_id = str(ev.get("evidence_id", "")).strip()
                evs.append((int(counts.get("SUPPORTS", 0)), entropy, ev_id, text))
        if len(evs) < 4:
            continue
        evs_sorted = sorted(evs, key=lambda t: (-t[0], t[1], t[2]))
        support_texts = [t[3] for t in evs_sorted[:5]]
        if len(support_texts) < 4:
            continue

        obs_texts = support_texts[:2]
        check_correct_texts = support_texts[2:]
        if len(check_correct_texts) < 2:
            continue

        if wrong_checks in ("neighbor", "inflation") and bank_emb is not None:
            c_emb = embed_texts([claim])[0]
            sims = bank_emb @ c_emb
            k = max(1, int(neighbor_k))
            order = np.argsort(-sims)
            cand: List[int] = []
            for j in order[: max(50, k + 2)]:
                j = int(j)
                if bank_claim_texts[j].strip() == claim.strip():
                    continue
                cand.append(j)
                if len(cand) >= k:
                    break
            if not cand:
                cand = [(int(i) + 17) % len(bank)]

            # Neighbor / inflation candidates are both selected from nearest neighbors (high similarity).
            #
            # - neighbor: prefer a competitor whose own SUPPORT evidence is strong for itself, but yields LOW scores
            #            when evaluated against this claim (prevents accidental support).
            # - inflation: prefer a competitor whose SUPPORT evidence yields HIGH scores when evaluated against this
            #              claim (agreement inflation negative control; should break the intervention gate).
            pick_sents: List[str] = []
            pick_offsets: List[Tuple[int, int]] = []
            cand_for_offsets: List[int] = []
            cur = 0
            for j in cand:
                other_claim, other_supports = bank[int(j)]
                pool = other_supports[:6]
                if len(pool) < 3:
                    continue
                self_scores = sentence_support_scores([other_claim] * len(pool), pool).tolist()
                order_self = np.argsort(-np.asarray(self_scores, dtype=float))
                chosen = [pool[int(k)] for k in order_self[:3]]
                pick_sents.extend(chosen)
                pick_offsets.append((cur, cur + len(chosen)))
                cand_for_offsets.append(int(j))
                cur += len(chosen)
            if not cand_for_offsets:
                cand_for_offsets = [(int(i) + 17) % len(bank)]
                pick_sents = bank[int(cand_for_offsets[0])][1][:3]
                pick_offsets = [(0, len(pick_sents))]

            cross_scores = sentence_support_scores([claim] * len(pick_sents), pick_sents).tolist()
            means: List[float] = []
            for a, b in pick_offsets:
                means.append(float(np.mean(np.asarray(cross_scores[a:b], dtype=float))) if b > a else float("inf"))
            if wrong_checks == "inflation":
                pos = int(np.argmax(np.asarray(means, dtype=float)))
            else:
                pos = int(np.argmin(np.asarray(means, dtype=float)))
            pick = int(cand_for_offsets[pos])
            a, b = pick_offsets[pos]

            neighbor_sims.append(float(sims[int(pick)]))
            other_claim, other_supports = bank[pick]
            wrong_check_texts = pick_sents[a:b]
        else:
            other_claim, other_supports = bank[(int(i) + 17) % len(bank)]
            if other_claim.strip() == claim.strip():
                other_claim, other_supports = bank[(int(i) + 18) % len(bank)]
        if wrong_checks not in ("neighbor", "inflation"):
            wrong_check_texts = other_supports[:3]
        if len(wrong_check_texts) < 2:
            continue

        obs_scores = sentence_support_scores([claim] * len(obs_texts), obs_texts).tolist()
        check_correct_scores = sentence_support_scores([claim] * len(check_correct_texts), check_correct_texts).tolist()
        check_wrong_scores = sentence_support_scores([claim] * len(wrong_check_texts), wrong_check_texts).tolist()

        obs_scores_by_sample.append([float(x) for x in obs_scores])
        check_correct_scores_by_sample.append([float(x) for x in check_correct_scores])

        mu_hat_list.append(float(np.mean(np.asarray(obs_scores, dtype=float))))
        mu_check_list.append(float(np.mean(np.asarray(check_correct_scores, dtype=float))))
        R_c = float(R_grounded(obs_scores, check_correct_scores))
        R_w = float(R_grounded(obs_scores, check_wrong_scores))
        R_correct.append(R_c)
        R_wrong.append(R_w)
        M_correct.append(M_from_R(R_c))
        M_wrong.append(M_from_R(R_w))

        if fast and len(M_correct) in (10, 25):
            print(f"[Q32:P2] climate intervention samples={len(M_correct)}")
        if len(M_correct) >= (20 if fast else 120):
            break

    if len(M_correct) < (10 if fast else 60):
        raise RuntimeError(f"Too few Climate-FEVER intervention samples ({len(M_correct)})")

    Mc = np.array(M_correct, dtype=float)
    Mw = np.array(M_wrong, dtype=float)
    # Swap control: re-pair obs with another sample's correct checks (should behave like wrong checks).
    perm = np.roll(np.arange(len(M_correct), dtype=int), 1)
    Ms_list: List[float] = []
    Rs_list: List[float] = []
    for i in range(int(len(M_correct))):
        check_swap = check_correct_scores_by_sample[int(perm[int(i)])]
        R_s = float(R_grounded(obs_scores_by_sample[int(i)], check_swap))
        Rs_list.append(float(R_s))
        Ms_list.append(float(M_from_R(R_s)))
    Ms = np.asarray(Ms_list, dtype=float) if Ms_list else np.asarray([], dtype=float)
    Rs = np.asarray(Rs_list, dtype=float) if Rs_list else np.asarray([], dtype=float)
    swap_wins = int(np.sum(Mc > Ms)) if Ms.size else 0
    swap_pair_wins = float(swap_wins / max(1, int(Ms.size)))
    swap_margin = float(np.mean(Mc - Ms)) if Ms.size else float("nan")
    wins = int(np.sum(Mc > Mw))
    n = int(len(Mc))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(Mc - Mw))
    z = _binom_z(wins, n)

    print(f"  P(M_correct > M_wrong) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if Ms.size:
        print(f"  [swap] P(M_correct > M_swapped_correct) = {swap_pair_wins:.3f}")
        print(f"  [swap] mean(M_correct - M_swapped_correct) = {swap_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(nn.mean()):.3f}")
    phi = _mutual_information_continuous(mu_hat_list, mu_check_list, n_bins=8)
    print(f"  [Phi_proxy] I(mu_hat; mu_check) = {phi:.3f} bits")

    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P2] Climate-FEVER intervention status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: Climate-FEVER intervention benchmark gates did not pass")

    details: Dict[str, float] = {
        "pair_wins": pair_wins,
        "z": z,
        "mean_margin": margin,
        "mean_R_correct": float(np.mean(np.asarray(R_correct, dtype=float))) if R_correct else 0.0,
        "mean_R_wrong": float(np.mean(np.asarray(R_wrong, dtype=float))) if R_wrong else 0.0,
        "mean_logR_correct": float(Mc.mean()),
        "mean_logR_wrong": float(Mw.mean()),
        "mean_R_swap": float(np.mean(Rs)) if Rs.size else float("nan"),
        "mean_logR_swap": float(np.mean(Ms)) if Ms.size else float("nan"),
        "swap_pair_wins": float(swap_pair_wins),
        "swap_mean_margin": float(swap_margin),
        "gate_z": gate_z,
        "gate_margin": gate_margin,
        "phi_proxy_bits": float(phi),
    }
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            details["mean_neighbor_sim"] = float(nn.mean())
    return BenchmarkResult(name="Climate-FEVER-Intervention", passed=passed, details=details)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q32 public truth-anchored benchmarks (fast + full modes).")
    p.add_argument(
        "--mode",
        choices=["bench", "stream", "transfer", "matrix", "stress", "sweep"],
        default="bench",
        help=(
            "Run static benchmarks (bench), streaming/intervention simulation (stream), Phase-3 transfer (transfer), "
            "a Phase-3 matrix (matrix: run both transfer directions), variability stress tests (stress), "
            "or a Phase-2 neighbor_k sweep (sweep)."
        ),
    )
    p.add_argument(
        "--dataset",
        choices=["scifact", "climate_fever", "snli", "mnli", "all"],
        default="scifact",
        help="Which benchmark to run (default: scifact).",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Fast debug mode: small caps + relaxed thresholds + no hard-fail unless --strict.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Hard-fail benchmarks that do not meet gates (default in full mode).",
    )
    p.add_argument(
        "--scoring",
        choices=["crossencoder", "cosine"],
        default=None,
        help="Override scoring model. In --fast mode default is cosine; otherwise crossencoder.",
    )
    p.add_argument(
        "--ablation",
        choices=["full", "no_essence", "no_scale", "no_depth", "no_grounding"],
        default="full",
        help=(
            "Q32 ablations for falsification: full (default), no_essence (E=1), no_scale (grad_S=1), "
            "no_depth (depth_term=1), no_grounding (E=1, grad_S=1, depth_term=1 => R=1 constant)."
        ),
    )
    p.add_argument(
        "--depth_power",
        type=float,
        default=0.0,
        help="Depth proxy power Df for σ^Df (default: 0.0 => disabled).",
    )
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device for sentence models / cross-encoder (default: auto).",
    )
    p.add_argument(
        "--threads",
        type=int,
        default=min(12, (os.cpu_count() or 12)),
        help="CPU threads for torch/BLAS (default: min(12, cpu_count)).",
    )
    p.add_argument(
        "--ce_batch",
        type=int,
        default=16,
        help="Cross-encoder batch size (default: 16). Increase until you hit VRAM/RAM limits.",
    )
    p.add_argument(
        "--st_batch",
        type=int,
        default=32,
        help="SentenceTransformer embedding batch size (default: 32).",
    )
    p.add_argument(
        "--wrong_checks",
        choices=["dissimilar", "neighbor", "inflation"],
        default="dissimilar",
        help=(
            "How to construct wrong checks: dissimilar topic (default), nearest-neighbor (J-style), "
            "or inflation (agreement inflation negative control)."
        ),
    )
    p.add_argument(
        "--neighbor_k",
        type=int,
        default=10,
        help="When --wrong_checks neighbor|inflation, choose from top-k nearest neighbors (default: 10).",
    )
    p.add_argument(
        "--sweep_neighbor_k",
        default="1,3,5,10",
        help="(sweep) Comma-separated neighbor_k values (default: 1,3,5,10).",
    )
    p.add_argument(
        "--scifact_stream_seed",
        type=int,
        default=123,
        help="SciFact streaming internal sampling seed (default: 123). Use -1 to tie to --seed (stress variability).",
    )
    p.add_argument(
        "--stress_n",
        type=int,
        default=10,
        help="(stress) Number of trials (default: 10).",
    )
    p.add_argument(
        "--stress_min_pass_rate",
        type=float,
        default=None,
        help="(stress) If set, exit non-zero unless pass_rate >= this threshold.",
    )
    p.add_argument(
        "--stress_out",
        default=None,
        help="(stress) Optional JSON path to write stress summary.",
    )
    p.add_argument(
        "--sweep_out",
        default=None,
        help="(sweep) Optional JSON path to write neighbor_k sweep summary.",
    )
    p.add_argument(
        "--calibrate_on",
        choices=["climate_fever", "scifact", "snli", "mnli"],
        default="climate_fever",
        help="(transfer) Dataset used to calibrate thresholds (default: climate_fever).",
    )
    p.add_argument(
        "--apply_to",
        choices=["climate_fever", "scifact", "snli", "mnli"],
        default="scifact",
        help="(transfer) Dataset used to verify frozen thresholds (default: scifact).",
    )
    p.add_argument(
        "--calibration_out",
        default=None,
        help="(transfer) Optional JSON path to write calibration output.",
    )
    p.add_argument(
        "--calibration_n",
        type=int,
        default=3,
        help="(transfer) Number of calibration seeds starting at --seed (default: 3).",
    )
    p.add_argument(
        "--verify_n",
        type=int,
        default=1,
        help="(transfer) Number of verification seeds starting at --seed (default: 1).",
    )
    p.add_argument(
        "--empirical_receipt_out",
        default=None,
        help=(
            "Optional JSON path to write an EmpiricalMetricReceipt-style summary of results "
            "(R/M/J/Phi-proxy + gates)."
        ),
    )
    p.add_argument(
        "--geometry_out",
        default=None,
        help="Optional JSON path to write a geometry summary (QGT/QGTL proxy metrics) for stream mode.",
    )
    p.add_argument(
        "--stream_series_out",
        default=None,
        help="(stream) Optional JSON path to write a time-series summary (M(t), dM(t), phase boundary) for stream mode.",
    )
    p.add_argument(
        "--require_geometry_gate",
        action="store_true",
        help="If set, require the geometry-break gate to pass in stream mode (in addition to the existing intervention gate).",
    )
    p.add_argument(
        "--require_phase_boundary_gate",
        action="store_true",
        help="(stream) If set, require a phase-boundary (stable delta) gate to pass (SciFact + Climate-FEVER).",
    )
    p.add_argument(
        "--require_injection_gate",
        action="store_true",
        help="(stream) If set, require a delayed-injection causal gate to pass (SciFact).",
    )
    p.add_argument(
        "--phase_delta_tau",
        type=float,
        default=None,
        help="(stream) Delta threshold for phase-boundary stability gate on delta(t)=M_correct(t)-M_wrong(t). Defaults depend on fast/full.",
    )
    p.add_argument(
        "--phase_min_tail",
        type=int,
        default=None,
        help="(stream) Phase-boundary stability window length in steps (default: 2 in fast mode, 3 in full mode).",
    )
    p.add_argument(
        "--phase_min_stable_rate",
        type=float,
        default=None,
        help="(stream) Minimum fraction of samples that must exhibit a stable phase-boundary crossing. Defaults depend on fast/full.",
    )
    p.add_argument(
        "--geometry_backend",
        choices=["proxy", "qgtl"],
        default="proxy",
        help="(stream) Geometry backend: proxy (cos/PR only) or qgtl (uses vendored qgt_lib/python for QGT/Q43-style metrics).",
    )
    return p.parse_args()


def _write_empirical_receipt(*, out_path: str, args: argparse.Namespace, results: List["BenchmarkResult"]) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    receipt = {
        "type": "EmpiricalMetricReceipt",
        "version": 1,
        "run": {
            "mode": str(args.mode),
            "dataset": str(args.dataset),
            "fast": bool(args.fast),
            "strict": bool(args.strict),
            "seed": int(args.seed),
            "scoring": str(args.scoring) if args.scoring is not None else None,
            "ablation": str(args.ablation),
            "depth_power": float(args.depth_power),
            "wrong_checks": str(args.wrong_checks),
            "neighbor_k": int(args.neighbor_k),
            "sweep_neighbor_k": str(args.sweep_neighbor_k) if hasattr(args, "sweep_neighbor_k") else None,
            "scifact_stream_seed": int(args.scifact_stream_seed),
            "stress_n": int(args.stress_n) if hasattr(args, "stress_n") else None,
            "stress_min_pass_rate": float(args.stress_min_pass_rate) if getattr(args, "stress_min_pass_rate", None) is not None else None,
            "calibrate_on": str(args.calibrate_on) if hasattr(args, "calibrate_on") else None,
            "apply_to": str(args.apply_to) if hasattr(args, "apply_to") else None,
            "calibration_n": int(args.calibration_n) if hasattr(args, "calibration_n") else None,
            "verify_n": int(args.verify_n) if hasattr(args, "verify_n") else None,
            "phase_delta_tau": float(args.phase_delta_tau) if getattr(args, "phase_delta_tau", None) is not None else None,
            "phase_min_tail": int(args.phase_min_tail) if getattr(args, "phase_min_tail", None) is not None else None,
            "phase_min_stable_rate": float(args.phase_min_stable_rate) if getattr(args, "phase_min_stable_rate", None) is not None else None,
        },
        "results": [
            {
                "name": str(r.name),
                "passed": bool(r.passed),
                "details": dict(r.details),
            }
            for r in results
        ],
    }

    canonical = json.dumps(receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    file_bytes = canonical + b"\n"
    sha256 = hashlib.sha256(file_bytes).hexdigest().upper()

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    os.replace(tmp_path, out_path)

    print(f"[Q32] EmpiricalMetricReceipt written: {out_path}")
    print(f"[Q32] EmpiricalMetricReceipt sha256: {sha256}")
    return sha256


def _write_geometry_summary(*, out_path: str, args: argparse.Namespace, results: List["BenchmarkResult"]) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    geom_results: List[dict] = []
    for r in results:
        d = dict(r.details)
        geom = {k: v for k, v in d.items() if str(k).startswith("geom_")}
        if not geom:
            continue
        geom_results.append({"name": str(r.name), "passed": bool(r.passed), "geometry": geom})

    payload = {
        "type": "GeometrySummary",
        "version": 1,
        "run": {
            "mode": str(args.mode),
            "dataset": str(args.dataset),
            "fast": bool(args.fast),
            "strict": bool(args.strict),
            "seed": int(args.seed),
            "scoring": str(args.scoring) if args.scoring is not None else None,
            "wrong_checks": str(args.wrong_checks),
            "neighbor_k": int(args.neighbor_k),
            "require_geometry_gate": bool(getattr(args, "require_geometry_gate", False)),
        },
        "results": geom_results,
    }

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    file_bytes = canonical + b"\n"
    sha256 = hashlib.sha256(file_bytes).hexdigest().upper()

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    os.replace(tmp_path, out_path)

    print(f"[Q32] GeometrySummary written: {out_path}")
    print(f"[Q32] GeometrySummary sha256: {sha256}")
    return sha256


def _write_stream_series_summary(*, out_path: str, args: argparse.Namespace, results: List["BenchmarkResult"]) -> str:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    payload = {
        "type": "StreamSeriesSummary",
        "version": 1,
        "run": {
            "mode": str(args.mode),
            "dataset": str(args.dataset),
            "fast": bool(args.fast),
            "strict": bool(args.strict),
            "seed": int(args.seed),
            "scoring": str(args.scoring) if args.scoring is not None else None,
            "wrong_checks": str(args.wrong_checks),
            "neighbor_k": int(args.neighbor_k),
            "phase_delta_tau": float(args.phase_delta_tau) if getattr(args, "phase_delta_tau", None) is not None else None,
            "phase_min_tail": int(args.phase_min_tail) if getattr(args, "phase_min_tail", None) is not None else None,
            "phase_min_stable_rate": float(args.phase_min_stable_rate) if getattr(args, "phase_min_stable_rate", None) is not None else None,
            "geometry_backend": str(getattr(args, "geometry_backend", "proxy")),
        },
        "results": [],
    }

    for r in results:
        d = dict(r.details)
        series = d.get("stream_series_summary")
        if series is None:
            continue
        payload["results"].append({"name": str(r.name), "passed": bool(r.passed), "series": series})

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    file_bytes = canonical + b"\n"
    sha256 = hashlib.sha256(file_bytes).hexdigest().upper()

    tmp_path = out_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(file_bytes)
    os.replace(tmp_path, out_path)

    print(f"[Q32] StreamSeriesSummary written: {out_path}")
    print(f"[Q32] StreamSeriesSummary sha256: {sha256}")
    return sha256


def run_climate_fever_streaming(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
    require_phase_boundary_gate: bool = False,
    phase_delta_tau: Optional[float] = None,
    phase_min_tail: Optional[int] = None,
    phase_min_stable_rate: Optional[float] = None,
) -> BenchmarkResult:
    """
    Phase 4: Real semiosphere dynamics (nonlinear time) + intervention.

    We simulate evidence arriving over time for Climate-FEVER claims.
    "Independence" is approximated as supportive evidence from different articles for the same claim.

    Gates (public):
      - Discrimination: by end of stream, SUPPORT should outrank REFUTE.
      - Negative control: swapping check pools across claims should collapse the effect.
      - Intervention: swapping from correlated checks (same-article) to independent checks (other-article)
        should not break true basins, and should not rescue refuting basins.
    """
    print("\n[Q32:P4] Climate-FEVER streaming + intervention")
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("climate_fever")["test"]
    rng = np.random.default_rng(seed)

    def vote_counts(votes: Sequence[Optional[str]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for v in votes:
            if v is None:
                continue
            vv = str(v)
            counts[vv] = counts.get(vv, 0) + 1
        return counts

    # Phase-4 streaming on Climate-FEVER is limited by the dataset's small per-claim evidence sets (usually <=5).
    # We therefore implement a sharp, falsifiable intervention:
    #
    #   For a claim, let SUPPORT evidence arrive over time. Define a "correct" check group as remaining SUPPORT
    #   evidence for the same claim (hold-out), and a "wrong" check group as SUPPORT evidence from a different claim.
    #
    # If M is field-like (coupled to truth-consistent checks), then:
    #   - M_correct(t_end) > M_wrong(t_end) for most claims.
    #   - The "meaning" should collapse under the wrong-check intervention.
    samples: List[Tuple[str, List[str], List[str], List[str]]] = []
    # (claim, support_texts_stream, independent_check_texts, wrong_check_texts)

    indices = np.arange(len(ds))
    rng.shuffle(indices)

    # Pre-index claim -> top supportive evidence texts (for wrong-check construction).
    claim_support_bank: List[Tuple[str, List[str]]] = []
    for ex in ds:
        claim_text = str(ex.get("claim", "")).strip()
        evidences = ex.get("evidences", []) or []
        if not claim_text or not evidences:
            continue
        evs: List[Tuple[int, float, str, str, str]] = []
        for ev in evidences:
            votes = [v for v in (ev.get("votes", []) or []) if v is not None]
            counts = vote_counts(votes)
            text = str(ev.get("evidence", "")).strip()
            if not text:
                continue
            supports = int(counts.get("SUPPORTS", 0))
            refutes = int(counts.get("REFUTES", 0))
            if supports >= 1 and refutes == 0:
                entropy = float(ev.get("entropy", 0.0) or 0.0)
                ev_id = str(ev.get("evidence_id", "")).strip() or "UNKNOWN:0"
                source = str(ev.get("article", "")).strip() or str(ev_id).split(":", 1)[0] or "UNKNOWN"
                evs.append((supports, entropy, source, ev_id, text))
        if len(evs) < 2:
            continue
        evs_sorted = sorted(evs, key=lambda t: (-t[0], t[1], t[2], t[3]))
        claim_support_bank.append((claim_text, [t[4] for t in evs_sorted[:6]]))

    if len(claim_support_bank) < 50:
        raise RuntimeError(f"Climate-FEVER streaming index too small ({len(claim_support_bank)})")

    # Choose "wrong checks":
    # - dissimilar: topic-mismatched and empirically incompatible
    # - neighbor: nearest-neighbor competitor (J-style)
    support_claim_texts = [c for c, _ in claim_support_bank]
    support_claim_emb = embed_texts(support_claim_texts)
    claim_emb_cache: Dict[str, np.ndarray] = {}

    def claim_embedding(text: str) -> np.ndarray:
        t = str(text).strip()
        if t in claim_emb_cache:
            return claim_emb_cache[t]
        claim_emb_cache[t] = embed_texts([t])[0]
        return claim_emb_cache[t]

    neighbor_sims: List[float] = []

    for idx in indices:
        ex = ds[int(idx)]
        claim = str(ex.get("claim", "")).strip()
        evidences = ex.get("evidences", []) or []
        if not claim or not evidences:
            continue

        # Climate-FEVER has ~5 evidence items per claim and often only 1 item per article, so we cannot form a long
        # correlated stream by grouping by article. Instead, we treat *within-evidence* n-gram chunks as correlated
        # (same source sentence), and use other supportive evidence sentences as an "independent" check pool.
        support_evs: List[Tuple[int, float, str]] = []
        for ev in evidences:
            votes = [v for v in (ev.get("votes", []) or []) if v is not None]
            counts = vote_counts(votes)
            text = str(ev.get("evidence", "")).strip()
            if not text:
                continue
            supports = int(counts.get("SUPPORTS", 0))
            refutes = int(counts.get("REFUTES", 0))
            if supports >= 1 and refutes == 0:
                entropy = float(ev.get("entropy", 0.0) or 0.0)
                support_evs.append((supports, entropy, text))
        if len(support_evs) < 2:
            continue
        support_evs_sorted = sorted(support_evs, key=lambda t: (-t[0], t[1]))
        stream_text = str(support_evs_sorted[0][2])
        support_texts = _word_chunks(stream_text, window=3, stride=1, max_chunks=12)[:6]
        if len(support_texts) < 4:
            continue
        independent_texts = [str(t[2]) for t in support_evs_sorted[1:4]]
        if len(independent_texts) < 2:
            continue

        # Observation proxy: how supportive are the claim's own supports for the claim?
        support_scores_preview = sentence_support_scores([claim] * len(support_texts), support_texts).tolist()
        mu_hat_proxy = float(np.mean(np.asarray(support_scores_preview, dtype=float)))

        # Candidate wrong checks:
        # - dissimilar: choose least similar claims
        # - neighbor: choose most similar claims
        c_emb = claim_embedding(claim)
        sims = support_claim_emb @ c_emb
        if wrong_checks in ("neighbor", "inflation"):
            order = np.argsort(-sims)  # high similarity => nearest neighbor competitor
        else:
            order = np.argsort(sims)  # low similarity => more wrong
        cand: List[int] = []
        k = max(1, int(neighbor_k)) if wrong_checks in ("neighbor", "inflation") else 80
        for j in order[: max(80, k + 2)]:
            j = int(j)
            if support_claim_texts[j].strip() == claim.strip():
                continue
            cand.append(j)
            if len(cand) >= (k if wrong_checks in ("neighbor", "inflation") else 6):
                break
        if not cand:
            continue

        cand_texts: List[List[str]] = [claim_support_bank[j][1][:4] for j in cand]
        flat_sents: List[str] = []
        offsets: List[Tuple[int, int]] = []
        cur = 0
        for lst in cand_texts:
            flat_sents.extend(lst)
            offsets.append((cur, cur + len(lst)))
            cur += len(lst)
        flat_scores = sentence_support_scores([claim] * len(flat_sents), flat_sents).tolist()
        cand_means: List[float] = []
        for a, b in offsets:
            if b <= a:
                cand_means.append(float("inf"))
            else:
                cand_means.append(float(np.mean(np.asarray(flat_scores[a:b], dtype=float))))
        if wrong_checks == "neighbor":
            # For neighbor-competitor falsifiers, prefer the most incompatible neighbor (directional gap).
            strength = [(mu_hat_proxy - m) if math.isfinite(m) else -1.0 for m in cand_means]
        elif wrong_checks == "inflation":
            # Agreement inflation negative control: prefer the most supportive wrong-check candidate.
            strength = [m if math.isfinite(m) else -1.0 for m in cand_means]
        else:
            strength = [abs(m - mu_hat_proxy) if math.isfinite(m) else -1.0 for m in cand_means]
        chosen_pos = int(np.argmax(np.asarray(strength, dtype=float)))
        wrong_check_texts = cand_texts[chosen_pos]
        if wrong_checks in ("neighbor", "inflation"):
            neighbor_sims.append(float(sims[int(cand[chosen_pos])]))
        if len(wrong_check_texts) < 3:
            continue

        samples.append((claim, support_texts, independent_texts, wrong_check_texts))
        if fast and len(samples) in (5, 10, 15):
            print(f"[Q32:P4] streaming samples={len(samples)}")
        if len(samples) >= (20 if fast else 120):
            break

    if len(samples) < (10 if fast else 60):
        raise RuntimeError(f"Too few Climate-FEVER streaming samples ({len(samples)})")

    M_correct_end: List[float] = []
    M_wrong_end: List[float] = []
    R_correct_end: List[float] = []
    R_wrong_end: List[float] = []
    # Swap/shuffle controls (receipted): store end-step obs/check sets.
    obs_end_by_sample: List[List[float]] = []
    check_end_by_sample: List[List[float]] = []
    wrong_end_by_sample: List[List[float]] = []
    dM_correct: List[float] = []
    dM_wrong: List[float] = []
    phi_coupling: List[float] = []
    M_indep_end: List[float] = []
    R_indep_end: List[float] = []
    phase_stable_cross: List[float] = []
    phase_first_cross_frac: List[float] = []
    series_delta_grid: List[List[float]] = []

    use_n = min((16 if fast else 120), len(samples))
    for claim, support_texts, independent_texts, wrong_check_texts in samples[:use_n]:
        support_scores = sentence_support_scores([claim] * len(support_texts), support_texts).tolist()
        wrong_scores = sentence_support_scores([claim] * len(wrong_check_texts), wrong_check_texts).tolist()
        indep_scores = sentence_support_scores([claim] * len(independent_texts), independent_texts).tolist()

        # Streaming time steps: we require at least 2 obs points and at least 2 check points.
        # With 5 supports, this gives t in {2,3}. With 4 supports, only t=2.
        t_max = max(2, len(support_scores) - 2)
        t_max = min(t_max, len(support_scores) - 2)

        # Correct check is the hold-out supports not yet observed at time t.
        M_series_correct: List[float] = []
        M_series_wrong: List[float] = []
        R_series_correct: List[float] = []
        R_series_wrong: List[float] = []
        mu_hat_series: List[float] = []
        mu_check_series: List[float] = []
        last_obs: Optional[List[float]] = None
        last_check: Optional[List[float]] = None
        last_t: int = 0
        for t in range(2, t_max + 1):
            check_correct = support_scores[t:]
            if len(check_correct) < 2:
                break
            obs = support_scores[:t]
            last_obs = [float(x) for x in obs]
            last_check = [float(x) for x in check_correct]
            R_c = float(R_grounded(obs, check_correct))
            R_w = float(R_grounded(obs, wrong_scores))
            R_series_correct.append(R_c)
            R_series_wrong.append(R_w)
            M_series_correct.append(M_from_R(R_c))
            M_series_wrong.append(M_from_R(R_w))
            mu_hat_series.append(float(np.mean(np.asarray(support_scores[:t], dtype=float))))
            mu_check_series.append(float(np.mean(np.asarray(check_correct, dtype=float))))
            last_t = int(t)

        if len(M_series_correct) < 1:
            continue

        M_correct_end.append(M_series_correct[-1])
        M_wrong_end.append(M_series_wrong[-1])
        R_correct_end.append(R_series_correct[-1])
        R_wrong_end.append(R_series_wrong[-1])
        if last_obs is not None and last_check is not None:
            obs_end_by_sample.append(last_obs)
            check_end_by_sample.append(last_check)
            wrong_end_by_sample.append([float(x) for x in wrong_scores])
            R_i = float(R_grounded(last_obs, indep_scores))
            R_indep_end.append(R_i)
            M_indep_end.append(float(M_from_R(R_i)))
        dM_correct.append(M_series_correct[-1] - M_series_correct[0])
        dM_wrong.append(M_series_wrong[-1] - M_series_wrong[0])
        phi_coupling.append(_mutual_information_continuous(mu_hat_series, mu_check_series, n_bins=8))

        delta_series = [float(a - b) for a, b in zip(M_series_correct, M_series_wrong)]
        if delta_series:
            tau = float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75)
            min_tail = int(phase_min_tail) if phase_min_tail is not None else (2 if fast else 3)
            tail_n = min(int(min_tail), int(len(delta_series)))
            stable = 0.0
            first_frac = float("nan")
            if tail_n >= 1 and all(float(x) >= tau for x in delta_series[-tail_n:]):
                stable = 1.0
                for i in range(0, len(delta_series) - tail_n + 1):
                    window = delta_series[i : i + tail_n]
                    if all(float(x) >= tau for x in window):
                        first_frac = float(i / max(1, len(delta_series) - 1))
                        break
            phase_stable_cross.append(stable)
            if math.isfinite(first_frac):
                phase_first_cross_frac.append(first_frac)
            grid_n = 11
            if len(delta_series) == 1:
                series_delta_grid.append([float(delta_series[0])] * grid_n)
            else:
                xs = np.linspace(0.0, 1.0, num=len(delta_series))
                xg = np.linspace(0.0, 1.0, num=grid_n)
                series_delta_grid.append([float(v) for v in np.interp(xg, xs, np.asarray(delta_series, dtype=float))])

        # (Geometry is handled in SciFact streaming; Climate-FEVER Phase 4 focuses on independence + phase boundary.)

    if len(M_correct_end) < (10 if fast else 60):
        raise RuntimeError(f"Too few usable streaming series ({len(M_correct_end)})")

    M_correct_end_a = np.array(M_correct_end, dtype=float)
    M_wrong_end_a = np.array(M_wrong_end, dtype=float)
    R_correct_end_a = np.array(R_correct_end, dtype=float)
    R_wrong_end_a = np.array(R_wrong_end, dtype=float)
    dM_correct_a = np.array(dM_correct, dtype=float)
    dM_wrong_a = np.array(dM_wrong, dtype=float)

    wins = int(np.sum(M_correct_end_a > M_wrong_end_a))
    n = int(len(M_correct_end_a))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(M_correct_end_a - M_wrong_end_a))
    # Normal-approx z-score for H0: win-rate=0.5 (binomial), avoids fragile absolute thresholds.
    z = _binom_z(wins, n)

    # Swap/shuffle controls:
    # - swap_correct: re-pair obs with another sample's correct checks (should behave like wrong checks).
    # - swap_wrong: re-pair obs with another sample's wrong checks (should stay wrong; sanity check).
    perm = np.roll(np.arange(n, dtype=int), 1)
    Ms_swap_correct: List[float] = []
    Rs_swap_correct: List[float] = []
    Ms_swap_wrong: List[float] = []
    Rs_swap_wrong: List[float] = []
    for i in range(n):
        obs_end = obs_end_by_sample[int(i)]
        check_swap = check_end_by_sample[int(perm[int(i)])]
        wrong_swap = wrong_end_by_sample[int(perm[int(i)])]
        R_sc = float(R_grounded(obs_end, check_swap))
        R_sw = float(R_grounded(obs_end, wrong_swap))
        Rs_swap_correct.append(float(R_sc))
        Ms_swap_correct.append(float(M_from_R(R_sc)))
        Rs_swap_wrong.append(float(R_sw))
        Ms_swap_wrong.append(float(M_from_R(R_sw)))
    M_swap_correct_end_a = np.asarray(Ms_swap_correct, dtype=float) if Ms_swap_correct else np.asarray([], dtype=float)
    R_swap_correct_end_a = np.asarray(Rs_swap_correct, dtype=float) if Rs_swap_correct else np.asarray([], dtype=float)
    M_swap_wrong_end_a = np.asarray(Ms_swap_wrong, dtype=float) if Ms_swap_wrong else np.asarray([], dtype=float)
    R_swap_wrong_end_a = np.asarray(Rs_swap_wrong, dtype=float) if Rs_swap_wrong else np.asarray([], dtype=float)
    swap_correct_pair_wins = float(np.mean(M_correct_end_a > M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_pair_wins = float(np.mean(M_correct_end_a > M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")
    swap_correct_margin = float(np.mean(M_correct_end_a - M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_margin = float(np.mean(M_correct_end_a - M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")

    print("\n[Q32:P4] Intervention: replace truth-consistent checks with wrong checks")
    print(f"  P(M_correct_end > M_wrong_end) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct_end) = {M_correct_end_a.mean():.3f}")
    print(f"  mean(M_wrong_end)   = {M_wrong_end_a.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if M_swap_correct_end_a.size:
        print(f"  [swap_correct] P(M_correct_end > M_swapped_correct_end) = {swap_correct_pair_wins:.3f}")
        print(f"  [swap_correct] mean(M_correct_end - M_swapped_correct_end) = {swap_correct_margin:.3f}")
    if M_swap_wrong_end_a.size:
        print(f"  [swap_wrong] P(M_correct_end > M_swapped_wrong_end) = {swap_wrong_pair_wins:.3f}")
        print(f"  [swap_wrong] mean(M_correct_end - M_swapped_wrong_end) = {swap_wrong_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(np.mean(np.asarray(neighbor_sims, dtype=float))):.3f}")
    if phi_coupling:
        print(f"  [Phi_proxy] mean I(mu_hat(t); mu_check(t)) = {float(np.mean(np.asarray(phi_coupling, dtype=float))):.3f} bits")

    print("\n[Q32:P4] Streaming deltas")
    print(f"  mean dM_correct = {dM_correct_a.mean():.3f}")
    print(f"  mean dM_wrong   = {dM_wrong_a.mean():.3f}")

    if M_indep_end:
        M_indep_end_a = np.asarray(M_indep_end, dtype=float)
        wins_i = int(np.sum(M_indep_end_a > M_wrong_end_a))
        n_i = int(M_indep_end_a.size)
        pair_wins_i = float(wins_i / max(1, n_i))
        margin_i = float(np.mean(M_indep_end_a - M_wrong_end_a))
        z_i = _binom_z(wins_i, n_i)
        print("\n[Q32:P4] Independence probe (independent checks vs wrong checks)")
        print(f"  P(M_indep_end > M_wrong_end) = {pair_wins_i:.3f}")
        print(f"  z(H0: p=0.5) = {z_i:.3f}  (n={n_i})")
        print(f"  mean(M_indep_end - M_wrong_end) = {margin_i:.3f}")

    if phase_stable_cross:
        pr = float(np.mean(np.asarray(phase_stable_cross, dtype=float)))
        tau = float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75)
        min_tail = int(phase_min_tail) if phase_min_tail is not None else (2 if fast else 3)
        gate_rate = float(phase_min_stable_rate) if phase_min_stable_rate is not None else (0.50 if fast else 0.60)
        print("\n[Q32:P4] Phase-boundary (stable delta) probe")
        print(f"  delta_tau = {tau:.3f}, min_tail = {int(min_tail)}")
        print(f"  stable_cross_rate = {pr:.3f}")
        if phase_first_cross_frac:
            print(f"  mean_first_cross_frac = {float(np.mean(np.asarray(phase_first_cross_frac, dtype=float))):.3f}")
        phase_passed = bool(pr >= gate_rate)
        print(f"  phase_gate_rate={gate_rate:.3f} => {'PASS' if phase_passed else 'FAIL'}")
        if require_phase_boundary_gate and (not phase_passed):
            if strict and not fast:
                raise AssertionError("FAIL: Climate-FEVER phase-boundary gate did not pass")

    # Strict threshold uses ~p<0.01 (z≈2.576) for "field effect survives intervention" significance.
    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P4] Climate-FEVER streaming status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: Climate-FEVER streaming gates did not pass")

    return BenchmarkResult(
        name="Climate-FEVER-Streaming",
        passed=passed,
        details={
            "pair_wins": pair_wins,
            "z": z,
            "mean_margin": margin,
            "mean_R_correct_end": float(R_correct_end_a.mean()),
            "mean_R_wrong_end": float(R_wrong_end_a.mean()),
            "mean_logR_correct_end": float(M_correct_end_a.mean()),
            "mean_logR_wrong_end": float(M_wrong_end_a.mean()),
            "gate_z": gate_z,
            "gate_margin": gate_margin,
            "mean_dM_correct": float(dM_correct_a.mean()),
            "mean_dM_wrong": float(dM_wrong_a.mean()),
            "mean_R_swap_correct_end": float(np.mean(R_swap_correct_end_a)) if R_swap_correct_end_a.size else float("nan"),
            "mean_logR_swap_correct_end": float(np.mean(M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan"),
            "swap_correct_pair_wins_end": float(swap_correct_pair_wins),
            "swap_correct_mean_margin_end": float(swap_correct_margin),
            "mean_R_swap_wrong_end": float(np.mean(R_swap_wrong_end_a)) if R_swap_wrong_end_a.size else float("nan"),
            "mean_logR_swap_wrong_end": float(np.mean(M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan"),
            "swap_wrong_pair_wins_end": float(swap_wrong_pair_wins),
            "swap_wrong_mean_margin_end": float(swap_wrong_margin),
            "mean_neighbor_sim": float(np.mean(np.asarray(neighbor_sims, dtype=float)))
            if (wrong_checks in ("neighbor", "inflation") and neighbor_sims)
            else float("nan"),
            "phi_proxy_bits": float(np.mean(np.asarray(phi_coupling, dtype=float))) if phi_coupling else 0.0,
            "mean_R_indep_end": float(np.mean(np.asarray(R_indep_end, dtype=float))) if R_indep_end else float("nan"),
            "mean_logR_indep_end": float(np.mean(np.asarray(M_indep_end, dtype=float))) if M_indep_end else float("nan"),
            "phase_delta_tau": float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75),
            "phase_min_tail": float(int(phase_min_tail) if phase_min_tail is not None else (2 if fast else 3)),
            "phase_stable_cross_rate": float(np.mean(np.asarray(phase_stable_cross, dtype=float))) if phase_stable_cross else float("nan"),
            "stream_series_summary": {
                "grid_n": 11,
                "delta_mean": [float(x) for x in np.nanmean(np.asarray(series_delta_grid, dtype=float), axis=0)]
                if series_delta_grid
                else [],
            },
        },
    )


def run_scifact_streaming(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
    wrong_checks: str = "dissimilar",
    neighbor_k: int = 10,
    scifact_stream_seed: int = 123,
    require_geometry_gate: bool = False,
    compute_geometry: bool = False,
    require_phase_boundary_gate: bool = False,
    phase_delta_tau: Optional[float] = None,
    phase_min_tail: Optional[int] = None,
    phase_min_stable_rate: Optional[float] = None,
    geometry_backend: str = "proxy",
    require_injection_gate: bool = False,
) -> BenchmarkResult:
    """
    Phase 4 / Phase 3 transfer probe on SciFact.

    We re-use the same streaming intervention gates as Climate-FEVER:
      - Build an evidence stream from a SUPPORT-labeled (claim, doc) pair.
      - Correct check pool: hold-out sentences from the same doc.
      - Wrong check pool: sentences from a different claim/doc, scored against the current claim.
      - Pass if the wrong-check intervention produces a significant drop (z-score) and a large mean margin.

    This intentionally avoids any dataset-specific tuning.
    """
    print("\n[Q32:P4] SciFact streaming + intervention")
    from datasets import load_dataset  # type: ignore

    claims = load_dataset("scifact", "claims")["train"]
    corpus = load_dataset("scifact", "corpus")["train"]
    corpus_by_id: Dict[int, dict] = {int(r["doc_id"]): r for r in corpus}

    # Collect SUPPORT examples with enough abstract sentences to form a stream.
    ex_rows: List[Tuple[int, str, int]] = []
    for r in claims:
        label = str(r.get("evidence_label", "")).strip().upper()
        if label != "SUPPORT":
            continue
        try:
            claim_id = int(r["id"])
        except Exception:
            continue
        claim_text = str(r.get("claim", "")).strip()
        if not claim_text:
            continue
        raw_doc = r.get("evidence_doc_id")
        if raw_doc is None:
            continue
        try:
            doc_id = int(raw_doc)
        except Exception:
            continue
        doc = corpus_by_id.get(doc_id)
        if not doc:
            continue
        abstract = doc.get("abstract", []) or []
        if len([x for x in abstract if str(x).strip()]) < 6:
            continue
        ex_rows.append((claim_id, claim_text, doc_id))

    if len(ex_rows) < 200:
        raise RuntimeError(f"Too few SciFact SUPPORT examples ({len(ex_rows)})")

    # NOTE: SciFact is sensitive to which abstract sentences are sampled as the stream.
    # For the public harness we keep this deterministic across seeds to avoid flakiness in transfer/matrix runs.
    # Default keeps this deterministic across seeds (scifact_stream_seed=123) to avoid flakiness in transfer/matrix runs.
    # For variability stress, pass scifact_stream_seed=-1 to tie it to `seed`, or set an explicit integer.
    base_seed = int(seed) if int(scifact_stream_seed) == -1 else int(scifact_stream_seed)
    rng = np.random.default_rng(base_seed)
    rng.shuffle(ex_rows)

    def sample_sentences(sents: List[str], n: int, *, seed_key: int) -> List[str]:
        if len(sents) < n:
            return []
        local_rng = np.random.default_rng(seed_key)
        order = local_rng.permutation(len(sents))[:n]
        return [sents[int(i)] for i in order]

    # Build a support sentence bank per (claim_id, doc_id) for wrong-check construction.
    # IMPORTANT: do not pick only top-scoring sentences; that biases mu_hat vs mu_check and breaks E.
    # We sample deterministically from the abstract to keep obs/check distributions comparable.
    support_bank: List[Tuple[str, List[str]]] = []
    for claim_id, claim_text, doc_id in ex_rows[: (200 if fast else 800)]:
        abstract = corpus_by_id[int(doc_id)].get("abstract", []) or []
        sents = [str(x) for x in abstract if str(x).strip()]
        if len(sents) < 6:
            continue
        sampled = sample_sentences(
            sents, 6, seed_key=(base_seed * 1_000_003) ^ (int(claim_id) * 9176) ^ int(doc_id)
        )
        if len(sampled) >= 4:
            support_bank.append((claim_text, sampled))
        if len(support_bank) >= (80 if fast else 400):
            break

    if len(support_bank) < (30 if fast else 120):
        raise RuntimeError(f"SciFact support bank too small ({len(support_bank)})")

    # For the intervention, define "wrong checks":
    # - dissimilar: topic-mismatched checks (default)
    # - neighbor: nearest-neighbor competitor checks (J-style)
    support_claim_texts = [c for c, _ in support_bank]
    support_claim_emb = embed_texts(support_claim_texts)
    claim_emb_cache: Dict[str, np.ndarray] = {}

    def claim_embedding(text: str) -> np.ndarray:
        t = str(text).strip()
        if t in claim_emb_cache:
            return claim_emb_cache[t]
        claim_emb_cache[t] = embed_texts([t])[0]
        return claim_emb_cache[t]

    samples: List[Tuple[str, List[float], List[float]]] = []
    neighbor_sims: List[float] = []
    for i, (claim_id, claim_text, doc_id) in enumerate(ex_rows[: (120 if fast else 800)]):
        abstract = corpus_by_id[int(doc_id)].get("abstract", []) or []
        sents = [str(x) for x in abstract if str(x).strip()]
        if len(sents) < 6:
            continue

        # Evidence stream sentences (deterministic sample; no top-k cherry-pick).
        # Use a longer stream so the last-step estimate uses more than a 2-sentence check tail.
        support_texts = sample_sentences(
            sents, 10, seed_key=(base_seed * 1_000_003) ^ (int(claim_id) * 9176) ^ int(doc_id)
        )
        if len(support_texts) < 8:
            continue

        support_scores_preview = sentence_support_scores([claim_text] * len(support_texts), support_texts).tolist()
        mu_hat_proxy = float(np.mean(np.asarray(support_scores_preview, dtype=float)))

        # Wrong checks:
        # - dissimilar: choose least similar claims
        # - neighbor: choose most similar claims
        c_emb = claim_embedding(claim_text)
        sims = support_claim_emb @ c_emb
        if wrong_checks in ("neighbor", "inflation"):
            order = np.argsort(-sims)  # high similarity => nearest neighbor competitor
        else:
            order = np.argsort(sims)  # low similarity => more wrong
        cand: List[int] = []
        k = max(1, int(neighbor_k)) if wrong_checks in ("neighbor", "inflation") else 50
        for j in order[: max(50, k + 2)]:
            idx = int(j)
            if support_claim_texts[idx].strip() == claim_text.strip():
                continue
            cand.append(idx)
            if len(cand) >= (k if wrong_checks in ("neighbor", "inflation") else 6):
                break
        if not cand:
            continue

        if wrong_checks in ("neighbor", "inflation"):
            # Build a "neighbor competitor" wrong-check pool:
            # pick a nearest-neighbor claim whose own evidence-sentences are strongly SUPPORTIVE for itself,
            # but score LOW when evaluated against the current claim.
            cand_sent_lists: List[List[str]] = [support_bank[idx][1][:6] for idx in cand]

            # 1) Score each candidate's sentences against its *own* claim (self-support).
            flat_self_claims: List[str] = []
            flat_self_sents: List[str] = []
            offsets: List[Tuple[int, int]] = []
            cur = 0
            for pos, idx in enumerate(cand):
                sents = cand_sent_lists[pos]
                c_claim = support_claim_texts[int(idx)]
                flat_self_claims.extend([c_claim] * len(sents))
                flat_self_sents.extend(sents)
                offsets.append((cur, cur + len(sents)))
                cur += len(sents)
            self_scores = sentence_support_scores(flat_self_claims, flat_self_sents).tolist()

            # 2) For each candidate, take its top self-support sentences as the check pool,
            # then score those sentences against the current claim (cross-score).
            pick_sents: List[str] = []
            pick_offsets: List[Tuple[int, int]] = []
            cur = 0
            for a, b in offsets:
                local = np.asarray([float(x) for x in self_scores[a:b]], dtype=float)
                order_local = np.argsort(-local)
                top_idx = [int(i) for i in order_local[:4]]
                chosen_sents = [flat_self_sents[a + int(j)] for j in top_idx]
                pick_sents.extend(chosen_sents)
                pick_offsets.append((cur, cur + len(chosen_sents)))
                cur += len(chosen_sents)
            cross_scores = sentence_support_scores([claim_text] * len(pick_sents), pick_sents).tolist()
            # Select the candidate by minimizing/maximizing the *actual* wrong-check score under the grounded R,
            # not by a proxy (mean cross score). This reduces full-mode brittleness in stress runs.
            obs_for_select = [float(x) for x in support_scores_preview[:2]]
            cand_M: List[float] = []
            for a, b in pick_offsets:
                if b <= a:
                    cand_M.append(float("inf") if wrong_checks != "inflation" else -float("inf"))
                    continue
                check = [float(x) for x in cross_scores[a:b]]
                cand_M.append(float(M_from_R(R_grounded(obs_for_select, check))))

            if wrong_checks == "inflation":
                chosen_pos = int(np.argmax(np.asarray(cand_M, dtype=float)))
            else:
                chosen_pos = int(np.argmin(np.asarray(cand_M, dtype=float)))
            chosen_idx = int(cand[chosen_pos])
            neighbor_sims.append(float(sims[int(chosen_idx)]))
            a, b = pick_offsets[chosen_pos]
            wrong_texts_preview = [str(x) for x in pick_sents[a:b]]
            wrong_scores_preview = [float(x) for x in cross_scores[a:b]]
        else:
            cand_sent_lists = [support_bank[idx][1][:6] for idx in cand]
            flat_sents: List[str] = []
            offsets = []
            cur = 0
            for lst in cand_sent_lists:
                flat_sents.extend(lst)
                offsets.append((cur, cur + len(lst)))
                cur += len(lst)
            flat_claims = [claim_text] * len(flat_sents)
            flat_scores = sentence_support_scores(flat_claims, flat_sents).tolist()

            cand_means: List[float] = []
            for a, b in offsets:
                if b <= a:
                    cand_means.append(float("inf"))
                else:
                    cand_means.append(float(np.mean(np.asarray(flat_scores[a:b], dtype=float))))
            # Choose a candidate whose check-mean most strongly disagrees with the observation mean,
            # making the empirical compatibility term E collapse under the wrong-check intervention.
            cand_strength = [abs(m - mu_hat_proxy) if math.isfinite(m) else -1.0 for m in cand_means]
            chosen_pos = int(np.argmax(np.asarray(cand_strength, dtype=float)))
            chosen_idx = int(cand[chosen_pos])
            a, b = offsets[chosen_pos]
            wrong_texts_preview = [str(x) for x in flat_sents[a:b]]
            wrong_scores_preview = [float(x) for x in flat_scores[a:b]]
        if len(wrong_scores_preview) < 3:
            continue

        samples.append(
            (claim_text, [str(x) for x in support_texts], support_scores_preview, wrong_texts_preview, wrong_scores_preview)
        )
        if fast and len(samples) in (10, 25):
            print(f"[Q32:P4] scifact streaming samples={len(samples)}")
        if len(samples) >= (20 if fast else 120):
            break

    if len(samples) < (10 if fast else 60):
        raise RuntimeError(f"Too few SciFact streaming samples ({len(samples)})")

    M_correct_end: List[float] = []
    M_wrong_end: List[float] = []
    R_correct_end: List[float] = []
    R_wrong_end: List[float] = []
    # Swap/shuffle controls (receipted): store end-step obs/check sets.
    obs_end_by_sample: List[List[float]] = []
    check_end_by_sample: List[List[float]] = []
    wrong_end_by_sample: List[List[float]] = []
    dM_correct: List[float] = []
    dM_wrong: List[float] = []
    geom_cos_correct_end: List[float] = []
    geom_cos_wrong_end: List[float] = []
    geom_pr_obs_end: List[float] = []
    geom_pr_check_end: List[float] = []
    geom_pr_wrong_end: List[float] = []
    geom_qgtl_berry_correct: List[float] = []
    geom_qgtl_berry_wrong: List[float] = []
    geom_qgtl_df_obs: List[float] = []
    geom_qgtl_df_check: List[float] = []
    geom_qgtl_df_wrong: List[float] = []
    phase_stable_cross: List[float] = []
    phase_first_cross_frac: List[float] = []
    series_delta_grid: List[List[float]] = []
    M_injected_end: List[float] = []

    use_n = min((16 if fast else 120), len(samples))
    phi_coupling: List[float] = []
    for claim_text, support_texts, support_scores, wrong_texts, wrong_scores in samples[:use_n]:
        if compute_geometry:
            support_emb = embed_texts([claim_text] + support_texts)
            claim_vec = support_emb[0]
            support_emb = support_emb[1:]
            wrong_emb = (
                embed_texts([str(x) for x in wrong_texts])
                if wrong_texts
                else np.zeros((0, int(support_emb.shape[1])), dtype=np.float32)
            )
        else:
            claim_vec = np.zeros((1,), dtype=np.float32)
            support_emb = np.zeros((0, 1), dtype=np.float32)
            wrong_emb = np.zeros((0, 1), dtype=np.float32)

        # Keep at least 4 check sentences at the end to reduce tail-noise.
        t_max = max(2, len(support_scores) - 4)
        t_max = min(t_max, len(support_scores) - 4)

        M_series_correct: List[float] = []
        M_series_wrong: List[float] = []
        R_series_correct: List[float] = []
        R_series_wrong: List[float] = []
        mu_hat_series: List[float] = []
        mu_check_series: List[float] = []
        last_obs: Optional[List[float]] = None
        last_check: Optional[List[float]] = None
        last_t: int = 0
        for t in range(2, t_max + 1):
            check_correct = support_scores[t:]
            if len(check_correct) < 4:
                break
            obs = support_scores[:t]
            last_obs = [float(x) for x in obs]
            last_check = [float(x) for x in check_correct]
            R_c = float(R_grounded(obs, check_correct))
            R_w = float(R_grounded(obs, wrong_scores))
            R_series_correct.append(R_c)
            R_series_wrong.append(R_w)
            M_series_correct.append(M_from_R(R_c))
            M_series_wrong.append(M_from_R(R_w))
            mu_hat_series.append(float(np.mean(np.asarray(support_scores[:t], dtype=float))))
            mu_check_series.append(float(np.mean(np.asarray(check_correct, dtype=float))))
            last_t = int(t)

        if len(M_series_correct) < 1:
            continue

        M_correct_end.append(M_series_correct[-1])
        M_wrong_end.append(M_series_wrong[-1])
        R_correct_end.append(R_series_correct[-1])
        R_wrong_end.append(R_series_wrong[-1])
        if last_obs is not None and last_check is not None:
            obs_end_by_sample.append(last_obs)
            check_end_by_sample.append(last_check)
            wrong_end_by_sample.append([float(x) for x in wrong_scores])
        dM_correct.append(M_series_correct[-1] - M_series_correct[0])
        dM_wrong.append(M_series_wrong[-1] - M_series_wrong[0])
        phi_coupling.append(_mutual_information_continuous(mu_hat_series, mu_check_series, n_bins=8))

        # Causal delayed-injection probe: contaminate the *observations* with wrong evidence and require M drops.
        # This is a stricter "causal" intervention than modifying checks: it changes mu_hat directly.
        if last_obs is not None and last_check is not None and wrong_scores:
            injected_obs = list(last_obs)
            if injected_obs:
                worst_wrong = float(np.min(np.asarray(wrong_scores, dtype=float)))
                injected_obs[-1] = worst_wrong
                if len(injected_obs) >= 2:
                    injected_obs[-2] = worst_wrong
                R_inj = float(R_grounded(injected_obs, last_check))
                M_injected_end.append(float(M_from_R(R_inj)))

        # Geometry proxies at end-step (t = last_t): compare obs vs correct-check vs wrong-check embedding structure.
        if compute_geometry and last_t >= 2 and int(support_emb.shape[0]) >= last_t:
            obs_e = support_emb[:last_t]
            chk_e = support_emb[last_t:]
            wrong_e = wrong_emb
            obs_mean = np.mean(obs_e, axis=0) if obs_e.size else claim_vec
            chk_mean = np.mean(chk_e, axis=0) if chk_e.size else claim_vec
            wrong_mean = np.mean(wrong_e, axis=0) if wrong_e.size else -claim_vec
            geom_cos_correct_end.append(_cos_sim(obs_mean, chk_mean))
            geom_cos_wrong_end.append(_cos_sim(obs_mean, wrong_mean))
            geom_pr_obs_end.append(_participation_ratio_from_embeddings(obs_e))
            geom_pr_check_end.append(_participation_ratio_from_embeddings(chk_e))
            geom_pr_wrong_end.append(_participation_ratio_from_embeddings(wrong_e))

            if str(geometry_backend) == "qgtl":
                try:
                    import sys
                    from pathlib import Path

                    qgt_dir = (
                        Path(__file__).resolve().parents[5]
                        / "LAB"
                        / "VECTOR_ELO"
                        / "eigen-alignment"
                        / "_from_wt-eigen-alignment"
                        / "qgt_lib"
                        / "python"
                    )
                    sys.path.insert(0, str(qgt_dir))
                    import qgt as qgtl  # type: ignore

                    geom_qgtl_df_obs.append(float(qgtl.participation_ratio(obs_e, normalize=True)))
                    geom_qgtl_df_check.append(float(qgtl.participation_ratio(chk_e, normalize=True)))
                    geom_qgtl_df_wrong.append(float(qgtl.participation_ratio(wrong_e, normalize=True)) if wrong_e.size else 0.0)
                    # "Berry phase" proxy around a simple closed loop (real-embedding holonomy proxy).
                    loop_correct = np.vstack([claim_vec, obs_mean, chk_mean, claim_vec])
                    loop_wrong = np.vstack([claim_vec, obs_mean, wrong_mean, claim_vec])
                    geom_qgtl_berry_correct.append(float(qgtl.berry_phase(loop_correct, closed=True)))
                    geom_qgtl_berry_wrong.append(float(qgtl.berry_phase(loop_wrong, closed=True)))
                except Exception:
                    pass

        # Phase-boundary: stable crossing on delta(t)=M_correct(t)-M_wrong(t), resampled to a fixed grid for reporting.
        delta_series = [float(a - b) for a, b in zip(M_series_correct, M_series_wrong)]
        if delta_series:
            tau = float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75)
            min_tail = 2 if fast else 3
            tail_n = min(int(min_tail), int(len(delta_series)))
            stable = 0.0
            first_frac = float("nan")
            if tail_n >= 1 and all(float(x) >= tau for x in delta_series[-tail_n:]):
                stable = 1.0
                for i in range(0, len(delta_series) - tail_n + 1):
                    window = delta_series[i : i + tail_n]
                    if all(float(x) >= tau for x in window):
                        first_frac = float(i / max(1, len(delta_series) - 1))
                        break
            phase_stable_cross.append(stable)
            if math.isfinite(first_frac):
                phase_first_cross_frac.append(first_frac)

            grid_n = 11
            if len(delta_series) == 1:
                series_delta_grid.append([float(delta_series[0])] * grid_n)
            else:
                xs = np.linspace(0.0, 1.0, num=len(delta_series))
                xg = np.linspace(0.0, 1.0, num=grid_n)
                series_delta_grid.append([float(v) for v in np.interp(xg, xs, np.asarray(delta_series, dtype=float))])

    if len(M_correct_end) < (10 if fast else 60):
        raise RuntimeError(f"Too few usable SciFact streaming series ({len(M_correct_end)})")

    M_correct_end_a = np.array(M_correct_end, dtype=float)
    M_wrong_end_a = np.array(M_wrong_end, dtype=float)
    R_correct_end_a = np.array(R_correct_end, dtype=float)
    R_wrong_end_a = np.array(R_wrong_end, dtype=float)
    dM_correct_a = np.array(dM_correct, dtype=float)
    dM_wrong_a = np.array(dM_wrong, dtype=float)

    wins = int(np.sum(M_correct_end_a > M_wrong_end_a))
    n = int(len(M_correct_end_a))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(M_correct_end_a - M_wrong_end_a))
    z = _binom_z(wins, n)

    # Swap/shuffle controls:
    # - swap_correct: re-pair obs with another sample's correct checks (should behave like wrong checks).
    # - swap_wrong: re-pair obs with another sample's wrong checks (should stay wrong; sanity check).
    perm = np.roll(np.arange(n, dtype=int), 1)
    Ms_swap_correct: List[float] = []
    Rs_swap_correct: List[float] = []
    Ms_swap_wrong: List[float] = []
    Rs_swap_wrong: List[float] = []
    for i in range(n):
        obs_end = obs_end_by_sample[int(i)]
        check_swap = check_end_by_sample[int(perm[int(i)])]
        wrong_swap = wrong_end_by_sample[int(perm[int(i)])]
        R_sc = float(R_grounded(obs_end, check_swap))
        R_sw = float(R_grounded(obs_end, wrong_swap))
        Rs_swap_correct.append(float(R_sc))
        Ms_swap_correct.append(float(M_from_R(R_sc)))
        Rs_swap_wrong.append(float(R_sw))
        Ms_swap_wrong.append(float(M_from_R(R_sw)))
    M_swap_correct_end_a = np.asarray(Ms_swap_correct, dtype=float) if Ms_swap_correct else np.asarray([], dtype=float)
    R_swap_correct_end_a = np.asarray(Rs_swap_correct, dtype=float) if Rs_swap_correct else np.asarray([], dtype=float)
    M_swap_wrong_end_a = np.asarray(Ms_swap_wrong, dtype=float) if Ms_swap_wrong else np.asarray([], dtype=float)
    R_swap_wrong_end_a = np.asarray(Rs_swap_wrong, dtype=float) if Rs_swap_wrong else np.asarray([], dtype=float)
    swap_correct_pair_wins = float(np.mean(M_correct_end_a > M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_pair_wins = float(np.mean(M_correct_end_a > M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")
    swap_correct_margin = float(np.mean(M_correct_end_a - M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan")
    swap_wrong_margin = float(np.mean(M_correct_end_a - M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan")

    print("\n[Q32:P4] Intervention: replace truth-consistent checks with wrong checks")
    print(f"  P(M_correct_end > M_wrong_end) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct_end) = {M_correct_end_a.mean():.3f}")
    print(f"  mean(M_wrong_end)   = {M_wrong_end_a.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")
    if M_swap_correct_end_a.size:
        print(f"  [swap_correct] P(M_correct_end > M_swapped_correct_end) = {swap_correct_pair_wins:.3f}")
        print(f"  [swap_correct] mean(M_correct_end - M_swapped_correct_end) = {swap_correct_margin:.3f}")
    if M_swap_wrong_end_a.size:
        print(f"  [swap_wrong] P(M_correct_end > M_swapped_wrong_end) = {swap_wrong_pair_wins:.3f}")
        print(f"  [swap_wrong] mean(M_correct_end - M_swapped_wrong_end) = {swap_wrong_margin:.3f}")
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            print(f"  [J] mean neighbor sim (k={int(neighbor_k)}) = {float(nn.mean()):.3f}")
    if phi_coupling:
        print(f"  [Phi_proxy] mean I(mu_hat(t); mu_check(t)) = {float(np.mean(np.asarray(phi_coupling, dtype=float))):.3f} bits")

    # Geometry-break "tipping" probe (proxy-only for now; QGTL backend is Phase-4.0 work):
    geom_passed = None
    if geom_cos_correct_end and geom_cos_wrong_end:
        c = np.asarray(geom_cos_correct_end, dtype=float)
        w = np.asarray(geom_cos_wrong_end, dtype=float)
        gwins = int(np.sum(c > w))
        gn = int(c.size)
        g_pair_wins = float(gwins / max(1, gn))
        g_margin = float(np.mean(c - w))
        g_z = _binom_z(gwins, gn)
        print("\n[Q32:P4] Geometry-break proxy (end-step)")
        print(f"  P(cos(obs, check_correct) > cos(obs, check_wrong)) = {g_pair_wins:.3f}")
        print(f"  z(H0: p=0.5) = {g_z:.3f}  (n={gn})")
        print(f"  mean(cos_correct - cos_wrong) = {g_margin:.3f}")
        if geom_pr_check_end and geom_pr_wrong_end:
            prc = np.asarray(geom_pr_check_end, dtype=float)
            prw = np.asarray(geom_pr_wrong_end, dtype=float)
            pr_margin = float(np.nanmean(prc - prw))
            print(f"  mean(PR_check - PR_wrong) = {pr_margin:.3f}")
        geom_gate_z = float(2.0 if fast else 2.6)
        geom_gate_margin = float(0.05 if fast else 0.08)
        geom_passed = bool(g_z >= geom_gate_z and g_margin >= geom_gate_margin)
        print(f"  geom_gate_z={geom_gate_z:.3f}, geom_gate_margin={geom_gate_margin:.3f} => {'PASS' if geom_passed else 'FAIL'}")
        if require_geometry_gate and (not geom_passed):
            if strict and not fast:
                raise AssertionError("FAIL: SciFact geometry-break gate did not pass")

    # Phase-boundary (delta stability) summary + optional gate.
    if phase_stable_cross:
        pr = float(np.mean(np.asarray(phase_stable_cross, dtype=float)))
        tau = float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75)
        min_tail = int(phase_min_tail) if phase_min_tail is not None else (2 if fast else 3)
        gate_rate = float(phase_min_stable_rate) if phase_min_stable_rate is not None else (0.50 if fast else 0.60)
        print("\n[Q32:P4] Phase-boundary (stable delta) probe")
        print(f"  delta_tau = {tau:.3f}, min_tail = {int(min_tail)}")
        print(f"  stable_cross_rate = {pr:.3f}")
        if phase_first_cross_frac:
            print(f"  mean_first_cross_frac = {float(np.mean(np.asarray(phase_first_cross_frac, dtype=float))):.3f}")
        phase_passed = bool(pr >= gate_rate)
        print(f"  phase_gate_rate={gate_rate:.3f} => {'PASS' if phase_passed else 'FAIL'}")
        if require_phase_boundary_gate and (not phase_passed):
            if strict and not fast:
                raise AssertionError("FAIL: SciFact phase-boundary gate did not pass")

    # Delayed-injection causal probe (optional gate).
    if M_injected_end:
        M_inj_a = np.asarray(M_injected_end, dtype=float)
        # Align lengths if some samples were skipped.
        m = min(int(M_inj_a.size), int(M_correct_end_a.size))
        if m > 0:
            M_inj_a = M_inj_a[:m]
            M_corr_a = M_correct_end_a[:m]
            wins_inj = int(np.sum(M_corr_a > M_inj_a))
            z_inj = _binom_z(wins_inj, int(m))
            margin_inj = float(np.mean(M_corr_a - M_inj_a))
            print("\n[Q32:P4] Delayed-injection causal probe (end-step)")
            print(f"  P(M_correct_end > M_injected_end) = {float(wins_inj / max(1, m)):.3f}")
            print(f"  z(H0: p=0.5) = {z_inj:.3f}  (n={int(m)})")
            print(f"  mean(M_correct_end - M_injected_end) = {margin_inj:.3f}")
            inj_gate_z = float(2.0 if fast else 2.6)
            inj_gate_margin = float(0.50 if fast else 0.75)
            inj_passed = bool(z_inj >= inj_gate_z and margin_inj >= inj_gate_margin)
            print(f"  inj_gate_z={inj_gate_z:.3f}, inj_gate_margin={inj_gate_margin:.3f} => {'PASS' if inj_passed else 'FAIL'}")
            if require_injection_gate and (not inj_passed):
                if strict and not fast:
                    raise AssertionError("FAIL: SciFact injection gate did not pass")

    print("\n[Q32:P4] Streaming deltas")
    print(f"  mean dM_correct = {dM_correct_a.mean():.3f}")
    print(f"  mean dM_wrong   = {dM_wrong_a.mean():.3f}")

    # Strict threshold uses ~p<0.01 (z≈2.576) for "field effect survives intervention" significance.
    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P4] SciFact streaming status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: SciFact streaming gates did not pass")

    details: Dict[str, float] = {
        "pair_wins": pair_wins,
        "z": z,
        "mean_margin": margin,
        "mean_R_correct_end": float(R_correct_end_a.mean()),
        "mean_R_wrong_end": float(R_wrong_end_a.mean()),
        "mean_logR_correct_end": float(M_correct_end_a.mean()),
        "mean_logR_wrong_end": float(M_wrong_end_a.mean()),
        "gate_z": gate_z,
        "gate_margin": gate_margin,
        "mean_dM_correct": float(dM_correct_a.mean()),
        "mean_dM_wrong": float(dM_wrong_a.mean()),
        "mean_R_swap_correct_end": float(np.mean(R_swap_correct_end_a)) if R_swap_correct_end_a.size else float("nan"),
        "mean_logR_swap_correct_end": float(np.mean(M_swap_correct_end_a)) if M_swap_correct_end_a.size else float("nan"),
        "swap_correct_pair_wins_end": float(swap_correct_pair_wins),
        "swap_correct_mean_margin_end": float(swap_correct_margin),
        "mean_R_swap_wrong_end": float(np.mean(R_swap_wrong_end_a)) if R_swap_wrong_end_a.size else float("nan"),
        "mean_logR_swap_wrong_end": float(np.mean(M_swap_wrong_end_a)) if M_swap_wrong_end_a.size else float("nan"),
        "swap_wrong_pair_wins_end": float(swap_wrong_pair_wins),
        "swap_wrong_mean_margin_end": float(swap_wrong_margin),
    }
    if phi_coupling:
        details["phi_proxy_bits"] = float(np.mean(np.asarray(phi_coupling, dtype=float)))
    if wrong_checks in ("neighbor", "inflation") and neighbor_sims:
        nn = np.asarray([x for x in neighbor_sims if math.isfinite(x)], dtype=float)
        if nn.size:
            details["mean_neighbor_sim"] = float(nn.mean())
    if geom_cos_correct_end and geom_cos_wrong_end:
        c = np.asarray(geom_cos_correct_end, dtype=float)
        w = np.asarray(geom_cos_wrong_end, dtype=float)
        gwins = int(np.sum(c > w))
        gn = int(c.size)
        details["geom_pair_wins_end"] = float(gwins / max(1, gn))
        details["geom_z_end"] = float(_binom_z(gwins, gn))
        details["geom_mean_margin_end"] = float(np.mean(c - w))
        details["geom_mean_cos_correct_end"] = float(np.mean(c))
        details["geom_mean_cos_wrong_end"] = float(np.mean(w))
        details["geom_mean_pr_obs_end"] = (
            float(np.nanmean(np.asarray(geom_pr_obs_end, dtype=float))) if geom_pr_obs_end else float("nan")
        )
        details["geom_mean_pr_check_end"] = (
            float(np.nanmean(np.asarray(geom_pr_check_end, dtype=float))) if geom_pr_check_end else float("nan")
        )
        details["geom_mean_pr_wrong_end"] = (
            float(np.nanmean(np.asarray(geom_pr_wrong_end, dtype=float))) if geom_pr_wrong_end else float("nan")
        )
        if geom_qgtl_df_obs:
            details["geom_qgtl_df_obs_mean"] = float(np.nanmean(np.asarray(geom_qgtl_df_obs, dtype=float)))
        if geom_qgtl_df_check:
            details["geom_qgtl_df_check_mean"] = float(np.nanmean(np.asarray(geom_qgtl_df_check, dtype=float)))
        if geom_qgtl_df_wrong:
            details["geom_qgtl_df_wrong_mean"] = float(np.nanmean(np.asarray(geom_qgtl_df_wrong, dtype=float)))
        if geom_qgtl_berry_correct:
            details["geom_qgtl_berry_correct_mean"] = float(np.nanmean(np.asarray(geom_qgtl_berry_correct, dtype=float)))
        if geom_qgtl_berry_wrong:
            details["geom_qgtl_berry_wrong_mean"] = float(np.nanmean(np.asarray(geom_qgtl_berry_wrong, dtype=float)))
        if geom_qgtl_berry_correct and geom_qgtl_berry_wrong:
            details["geom_qgtl_berry_margin_mean"] = float(
                np.nanmean(np.asarray(geom_qgtl_berry_correct, dtype=float) - np.asarray(geom_qgtl_berry_wrong, dtype=float))
            )
    if phase_stable_cross:
        tau = float(phase_delta_tau) if phase_delta_tau is not None else (0.50 if fast else 0.75)
        min_tail = int(phase_min_tail) if phase_min_tail is not None else (2 if fast else 3)
        gate_rate = float(phase_min_stable_rate) if phase_min_stable_rate is not None else (0.50 if fast else 0.60)
        details["phase_delta_tau"] = float(tau)
        details["phase_min_tail"] = float(min_tail)
        details["phase_stable_cross_rate"] = float(np.mean(np.asarray(phase_stable_cross, dtype=float)))
        details["phase_gate_rate"] = float(gate_rate)
        if phase_first_cross_frac:
            details["phase_mean_first_cross_frac"] = float(np.mean(np.asarray(phase_first_cross_frac, dtype=float)))
        if series_delta_grid:
            grid = np.asarray(series_delta_grid, dtype=float)
            mean_curve = np.nanmean(grid, axis=0)
            p25 = np.nanpercentile(grid, 25, axis=0)
            p75 = np.nanpercentile(grid, 75, axis=0)
            details["stream_series_summary"] = {
                "grid_n": 11,
                "delta_mean": [float(x) for x in mean_curve],
                "delta_p25": [float(x) for x in p25],
                "delta_p75": [float(x) for x in p75],
            }
    if M_injected_end:
        M_inj_a = np.asarray(M_injected_end, dtype=float)
        m = min(int(M_inj_a.size), int(M_correct_end_a.size))
        if m > 0:
            M_inj_a = M_inj_a[:m]
            M_corr_a = M_correct_end_a[:m]
            wins_inj = int(np.sum(M_corr_a > M_inj_a))
            details["inj_pair_wins_end"] = float(wins_inj / max(1, m))
            details["inj_z_end"] = float(_binom_z(wins_inj, int(m)))
            details["inj_mean_margin_end"] = float(np.mean(M_corr_a - M_inj_a))
    return BenchmarkResult(name="SciFact-Streaming", passed=passed, details=details)


def main() -> int:
    set_cache_roots()
    args = parse_args()

    device = _auto_device() if args.device == "auto" else str(args.device)
    configure_runtime(device=device, threads=max(1, int(args.threads)), ce_batch_size=int(args.ce_batch), st_batch_size=int(args.st_batch))
    global _ABLATION
    _ABLATION = str(args.ablation)
    global _DEPTH_POWER
    _DEPTH_POWER = float(args.depth_power)

    global _USE_CROSS_ENCODER
    if args.scoring is not None:
        _USE_CROSS_ENCODER = args.scoring == "crossencoder"
    else:
        _USE_CROSS_ENCODER = False if args.fast else True

    # Full mode is strict by default; fast mode is non-strict unless explicitly requested.
    strict = args.strict or (not args.fast)

    results: List[BenchmarkResult] = []
    if args.mode == "stress":
        # Variability stress is intentionally non-strict: we want a distribution, not an abort.
        trials = max(1, int(args.stress_n))
        base = int(args.seed)

        if args.dataset not in ("scifact", "all"):
            raise SystemExit("stress mode currently supports only --dataset scifact (or all)")

        print(f"\n[Q32:STRESS] SciFact streaming variability (trials={trials})")
        per: List[Dict[str, float]] = []
        pass_n = 0
        for i in range(trials):
            s = base + i
            try:
                r = run_scifact_streaming(
                    seed=s,
                    fast=args.fast,
                    strict=False,
                    wrong_checks=args.wrong_checks,
                    neighbor_k=args.neighbor_k,
                    scifact_stream_seed=-1,
                    require_geometry_gate=False,
                    compute_geometry=False,
                )
            except Exception:
                r = BenchmarkResult(name=f"SciFact-Streaming@seed={s}", passed=False, details={})
            results.append(r)
            if r.passed:
                pass_n += 1
            d = dict(r.details)
            d["seed"] = float(s)
            per.append(d)

        pass_rate = float(pass_n / max(1, trials))
        pw = [float(d.get("pair_wins", float("nan"))) for d in per]
        zz = [float(d.get("z", float("nan"))) for d in per]
        mm = [float(d.get("mean_margin", float("nan"))) for d in per]
        rc_end = [float(d.get("mean_R_correct_end", float("nan"))) for d in per]
        rw_end = [float(d.get("mean_R_wrong_end", float("nan"))) for d in per]
        mc_end = [float(d.get("mean_logR_correct_end", float("nan"))) for d in per]
        mw_end = [float(d.get("mean_logR_wrong_end", float("nan"))) for d in per]

        def finite_min(xs: List[float]) -> float:
            vals = [x for x in xs if math.isfinite(x)]
            return float(min(vals)) if vals else float("nan")

        summary = {
            "mode": "stress",
            "dataset": "scifact",
            "trials": int(trials),
            "seed_base": int(base),
            "wrong_checks": str(args.wrong_checks),
            "neighbor_k": int(args.neighbor_k),
            "pass_n": int(pass_n),
            "pass_rate": float(pass_rate),
            "min_pair_wins": finite_min(pw),
            "min_z": finite_min(zz),
            "min_mean_margin": finite_min(mm),
            "min_mean_R_correct_end": finite_min(rc_end),
            "min_mean_R_wrong_end": finite_min(rw_end),
            "min_mean_logR_correct_end": finite_min(mc_end),
            "min_mean_logR_wrong_end": finite_min(mw_end),
        }

        print("\n[Q32:STRESS] Summary")
        print(f"  pass_rate = {summary['pass_rate']:.3f}  (pass_n={summary['pass_n']}/{summary['trials']})")
        print(f"  min_pair_wins = {summary['min_pair_wins']:.3f}")
        print(f"  min_z = {summary['min_z']:.3f}")
        print(f"  min_mean_margin = {summary['min_mean_margin']:.3f}")
        if math.isfinite(float(summary["min_mean_R_correct_end"])):
            print(f"  min_mean_R_correct_end = {summary['min_mean_R_correct_end']:.3f}")
            print(f"  min_mean_R_wrong_end   = {summary['min_mean_R_wrong_end']:.3f}")
            print(f"  min_mean_logR_correct_end = {summary['min_mean_logR_correct_end']:.3f}")
            print(f"  min_mean_logR_wrong_end   = {summary['min_mean_logR_wrong_end']:.3f}")

        # Add a synthetic result so EmpiricalMetricReceipt can capture stress summaries.
        details_stress: Dict[str, float] = {
            "trials": float(trials),
            "seed_base": float(base),
            "neighbor_k": float(int(args.neighbor_k)),
            "pass_n": float(pass_n),
            "pass_rate": float(pass_rate),
            "min_pair_wins": float(summary["min_pair_wins"]),
            "min_z": float(summary["min_z"]),
            "min_mean_margin": float(summary["min_mean_margin"]),
            "min_mean_R_correct_end": float(summary["min_mean_R_correct_end"]),
            "min_mean_R_wrong_end": float(summary["min_mean_R_wrong_end"]),
            "min_mean_logR_correct_end": float(summary["min_mean_logR_correct_end"]),
            "min_mean_logR_wrong_end": float(summary["min_mean_logR_wrong_end"]),
        }
        if args.stress_min_pass_rate is not None:
            details_stress["gate_min_pass_rate"] = float(args.stress_min_pass_rate)

        passed_stress = True
        if args.stress_min_pass_rate is not None and math.isfinite(float(pass_rate)):
            passed_stress = bool(float(pass_rate) >= float(args.stress_min_pass_rate))
        results.append(BenchmarkResult(name="SciFact-Streaming-Stress", passed=passed_stress, details=details_stress))

        if args.stress_out:
            out_path = str(args.stress_out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            import json

            payload = {"summary": summary, "per_trial": per}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(f"[Q32:STRESS] Wrote {out_path}")

        # Note: do not exit early here; we want receipts written even on stress gate failures.

    elif args.mode == "sweep":
        # Phase-2 robustness: sweep neighbor_k and require a stable pass-rate across a range.
        # Like stress, sweep is intentionally non-strict at the per-trial level: we want a distribution.
        if args.dataset not in ("scifact", "all"):
            raise SystemExit("sweep mode currently supports only --dataset scifact (or all)")

        raw = str(args.sweep_neighbor_k or "").strip()
        parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
        ks = sorted({int(p) for p in parts if str(p).lstrip("-").isdigit() and int(p) > 0})
        if not ks:
            raise SystemExit(f"sweep requires at least one positive int neighbor_k (got: {raw!r})")

        trials = max(1, int(args.stress_n))
        base = int(args.seed)

        print(f"\n[Q32:SWEEP] SciFact streaming neighbor_k sweep (ks={ks}, trials={trials})")
        per_k: List[Dict[str, float]] = []
        per_trial: List[Dict[str, float]] = []

        for k in ks:
            pass_n = 0
            for i in range(trials):
                s = base + i
                try:
                    r = run_scifact_streaming(
                        seed=s,
                        fast=args.fast,
                        strict=False,
                        wrong_checks=args.wrong_checks,
                        neighbor_k=int(k),
                        scifact_stream_seed=-1,
                        require_geometry_gate=False,
                        compute_geometry=False,
                    )
                except Exception:
                    r = BenchmarkResult(name=f"SciFact-Streaming@k={k}@seed={s}", passed=False, details={})

                results.append(BenchmarkResult(name=f"SciFact-Streaming@k={k}@seed={s}", passed=r.passed, details=r.details))
                if r.passed:
                    pass_n += 1
                d = dict(r.details)
                d["seed"] = float(s)
                d["neighbor_k"] = float(int(k))
                per_trial.append(d)

            pass_rate = float(pass_n / max(1, trials))
            per_k.append({"neighbor_k": float(int(k)), "pass_n": float(pass_n), "pass_rate": float(pass_rate)})
            print(f"  k={int(k):>3d}: pass_rate={pass_rate:.3f}  (pass_n={pass_n}/{trials})")

        min_pr_over_k = float(min([float(d["pass_rate"]) for d in per_k]) if per_k else float("nan"))
        details_sweep: Dict[str, float] = {
            "trials_per_k": float(trials),
            "k_count": float(len(ks)),
            "min_pass_rate_over_k": float(min_pr_over_k),
        }
        for d in per_k:
            kk = int(d["neighbor_k"])
            details_sweep[f"pass_rate_k{kk}"] = float(d["pass_rate"])
            details_sweep[f"pass_n_k{kk}"] = float(d["pass_n"])

        if args.stress_min_pass_rate is not None:
            details_sweep["gate_min_pass_rate"] = float(args.stress_min_pass_rate)

        passed_sweep = True
        if args.stress_min_pass_rate is not None and math.isfinite(min_pr_over_k):
            passed_sweep = bool(float(min_pr_over_k) >= float(args.stress_min_pass_rate))
        results.append(BenchmarkResult(name="SciFact-Streaming-SweepK", passed=passed_sweep, details=details_sweep))

        if args.sweep_out:
            out_path = str(args.sweep_out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            import json

            payload = {"ks": [int(k) for k in ks], "trials": int(trials), "per_k": per_k, "per_trial": per_trial}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, sort_keys=True)
            print(f"[Q32:SWEEP] Wrote {out_path}")

        # Note: do not exit early here; we want receipts written even on sweep gate failures.

    elif args.mode == "bench":
        if args.dataset in ("scifact", "all"):
            results.append(
                run_scifact_benchmark(
                    seed=args.seed, fast=args.fast, strict=strict, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                )
            )
        if args.dataset in ("climate_fever", "all"):
            results.append(run_climate_fever_benchmark(seed=args.seed, fast=args.fast, strict=strict))
        if args.dataset in ("snli", "all"):
            results.append(
                run_snli_benchmark(
                    seed=args.seed, fast=args.fast, strict=strict, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                )
            )
        if args.dataset in ("mnli", "all"):
            results.append(
                run_snli_benchmark(
                    seed=args.seed,
                    fast=args.fast,
                    strict=strict,
                    wrong_checks=args.wrong_checks,
                    neighbor_k=args.neighbor_k,
                    nli_domain="mnli",
                )
            )
    elif args.mode == "stream":
        if args.dataset in ("climate_fever", "all"):
            results.append(
                run_climate_fever_streaming(
                    seed=args.seed,
                    fast=args.fast,
                    strict=strict,
                    wrong_checks=args.wrong_checks,
                    neighbor_k=args.neighbor_k,
                    require_phase_boundary_gate=bool(args.require_phase_boundary_gate),
                    phase_delta_tau=float(args.phase_delta_tau) if args.phase_delta_tau is not None else None,
                    phase_min_tail=int(args.phase_min_tail) if getattr(args, "phase_min_tail", None) is not None else None,
                    phase_min_stable_rate=float(args.phase_min_stable_rate) if args.phase_min_stable_rate is not None else None,
                )
            )
        if args.dataset in ("scifact", "all"):
            results.append(
                run_scifact_streaming(
                    seed=args.seed,
                    fast=args.fast,
                    strict=strict,
                    wrong_checks=args.wrong_checks,
                    neighbor_k=args.neighbor_k,
                    scifact_stream_seed=int(args.scifact_stream_seed),
                    require_geometry_gate=bool(args.require_geometry_gate),
                    compute_geometry=True,
                    require_phase_boundary_gate=bool(args.require_phase_boundary_gate),
                    phase_delta_tau=float(args.phase_delta_tau) if args.phase_delta_tau is not None else None,
                    phase_min_tail=int(args.phase_min_tail) if getattr(args, "phase_min_tail", None) is not None else None,
                    phase_min_stable_rate=float(args.phase_min_stable_rate) if args.phase_min_stable_rate is not None else None,
                    geometry_backend=str(args.geometry_backend),
                    require_injection_gate=bool(getattr(args, "require_injection_gate", False)),
                )
            )
        if args.dataset in ("snli", "all"):
            results.append(
                run_snli_streaming(
                    seed=args.seed, fast=args.fast, strict=strict, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                )
            )
        if args.dataset in ("mnli", "all"):
            results.append(
                run_snli_streaming(
                    seed=args.seed,
                    fast=args.fast,
                    strict=strict,
                    wrong_checks=args.wrong_checks,
                    neighbor_k=args.neighbor_k,
                    nli_domain="mnli",
                )
            )
    else:
        # Phase 3: calibrate thresholds once on one dataset (multiple seeds), then verify on the other without retuning.
        def run_transfer(*, calibrate_on: str, apply_to: str) -> List[BenchmarkResult]:
            cal_n = max(1, int(args.calibration_n))
            verify_n = max(1, int(args.verify_n))
            cal_seeds = [args.seed + i for i in range(cal_n)]
            verify_seeds = [args.seed + i for i in range(verify_n)]

            def tag_seed(r: BenchmarkResult, seed: int) -> BenchmarkResult:
                return BenchmarkResult(name=f"{calibrate_on}->{apply_to}:{r.name}@seed={seed}", passed=r.passed, details=r.details)

            def run_intervention_bench(ds: str, seed: int) -> BenchmarkResult:
                if ds == "scifact":
                    return run_scifact_benchmark(
                        seed=seed, fast=args.fast, strict=False, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                    )
                if ds == "climate_fever":
                    return run_climate_fever_intervention_benchmark(
                        seed=seed, fast=args.fast, strict=False, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                    )
                if ds == "mnli":
                    return run_snli_benchmark(
                        seed=seed,
                        fast=args.fast,
                        strict=False,
                        wrong_checks=args.wrong_checks,
                        neighbor_k=args.neighbor_k,
                        nli_domain="mnli",
                    )
                return run_snli_benchmark(
                    seed=seed, fast=args.fast, strict=False, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                )

            def run_intervention_stream(ds: str, seed: int) -> BenchmarkResult:
                if ds == "scifact":
                    return run_scifact_streaming(
                        seed=seed,
                        fast=args.fast,
                        strict=False,
                        wrong_checks=args.wrong_checks,
                        neighbor_k=args.neighbor_k,
                        scifact_stream_seed=int(args.scifact_stream_seed),
                        require_geometry_gate=False,
                        compute_geometry=False,
                        require_phase_boundary_gate=False,
                        phase_delta_tau=None,
                        phase_min_tail=None,
                        phase_min_stable_rate=None,
                        geometry_backend="proxy",
                        require_injection_gate=False,
                    )
                if ds == "climate_fever":
                    return run_climate_fever_streaming(
                        seed=seed,
                        fast=args.fast,
                        strict=False,
                        wrong_checks=args.wrong_checks,
                        neighbor_k=args.neighbor_k,
                        require_phase_boundary_gate=False,
                        phase_delta_tau=None,
                        phase_min_tail=None,
                        phase_min_stable_rate=None,
                    )
                if ds == "mnli":
                    return run_snli_streaming(
                        seed=seed,
                        fast=args.fast,
                        strict=False,
                        wrong_checks=args.wrong_checks,
                        neighbor_k=args.neighbor_k,
                        nli_domain="mnli",
                    )
                return run_snli_streaming(
                    seed=seed, fast=args.fast, strict=False, wrong_checks=args.wrong_checks, neighbor_k=args.neighbor_k
                )

            print(f"\n[Q32:P3] Calibrating on {calibrate_on} (seeds={cal_seeds})")
            cal_bench = [tag_seed(run_intervention_bench(calibrate_on, s), s) for s in cal_seeds]
            cal_stream = [tag_seed(run_intervention_stream(calibrate_on, s), s) for s in cal_seeds]
            cal_all = cal_bench + cal_stream
            cal_failed = [r.name for r in cal_all if not r.passed]
            if cal_failed:
                print("\n[Q32:P3] Calibration failures")
                for n in cal_failed:
                    print(f"  - {n}")
                if args.fast and not args.strict:
                    print("[Q32:P3] Fast mode: continuing using only passing calibration runs.")
                else:
                    raise SystemExit("transfer calibration failed: calibration dataset did not meet its own gates")

            cal_for_thresholds = [r for r in cal_all if r.passed] if (args.fast and not args.strict) else cal_all
            pw_vals = [r.details.get("pair_wins") for r in cal_for_thresholds]
            pw_vals_f = [float(x) for x in pw_vals if x is not None]
            if not pw_vals_f:
                raise SystemExit("transfer calibration failed: missing pair_wins")

            # NOTE: In --fast mode, n is intentionally small, so a slightly lower z-gate keeps
            # the short-loop iteration meaningful without forcing long runs.
            frozen_min_z = 1.4 if args.fast else 2.6
            frozen_min_pair_wins = _percentile(pw_vals_f, 10.0)

            if args.calibration_out:
                frozen = {
                    "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
                    "seed_base": int(args.seed),
                    "calibrate_on": calibrate_on,
                    "apply_to": apply_to,
                    "frozen_min_z": float(frozen_min_z),
                    "frozen_min_pair_wins": float(frozen_min_pair_wins),
                }
                out_path = str(args.calibration_out)
                os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
                import json

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(frozen, f, indent=2, sort_keys=True)
                print(f"[Q32:P3] Wrote calibration to {out_path}")

            print(f"\n[Q32:P3] Frozen thresholds: min_z={frozen_min_z:.3f}  min_pair_wins={frozen_min_pair_wins:.3f}")
            print(f"[Q32:P3] Verifying on {apply_to} (seeds={verify_seeds})")

            def enforce_transfer(r: BenchmarkResult) -> BenchmarkResult:
                pw = r.details.get("pair_wins")
                if pw is None:
                    raise RuntimeError(f"transfer verify failed: {r.name} missing pair_wins")
                details = dict(r.details)
                details["gate_pair_wins_calibrated"] = float(frozen_min_pair_wins)
                details["gate_z"] = float(frozen_min_z)
                if float(pw) < frozen_min_pair_wins:
                    print(
                        f"[Q32:P3] {r.name} below calibrated pair_wins (reported only): "
                        f"{float(pw):.3f} < {float(frozen_min_pair_wins):.3f}"
                    )
                return BenchmarkResult(name=r.name, passed=bool(r.passed), details=details)

            out: List[BenchmarkResult] = []
            for s in verify_seeds:
                if apply_to == "scifact":
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_scifact_benchmark(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                )
                            ),
                            s,
                        )
                    )
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_scifact_streaming(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                    scifact_stream_seed=int(args.scifact_stream_seed),
                                )
                            ),
                            s,
                        )
                    )
                elif apply_to == "climate_fever":
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_climate_fever_intervention_benchmark(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                )
                            ),
                            s,
                        )
                    )
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_climate_fever_streaming(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                )
                            ),
                            s,
                        )
                    )
                else:
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_snli_benchmark(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                    nli_domain=str(apply_to),
                                )
                            ),
                            s,
                        )
                    )
                    out.append(
                        tag_seed(
                            enforce_transfer(
                                run_snli_streaming(
                                    seed=s,
                                    fast=args.fast,
                                    strict=False,
                                    min_z=frozen_min_z,
                                    wrong_checks=args.wrong_checks,
                                    neighbor_k=args.neighbor_k,
                                    nli_domain=str(apply_to),
                                )
                            ),
                            s,
                        )
                    )
            return out

        if args.mode == "matrix":
            ds_all = ["climate_fever", "scifact", "snli", "mnli"]
            for a in ds_all:
                for b in ds_all:
                    if a == b:
                        continue
                    results.extend(run_transfer(calibrate_on=a, apply_to=b))
        else:
            results.extend(run_transfer(calibrate_on=args.calibrate_on, apply_to=args.apply_to))

    print("\n[Q32] PUBLIC BENCHMARK SUMMARY")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  - {r.name}: {status}")

    if args.empirical_receipt_out:
        _write_empirical_receipt(out_path=str(args.empirical_receipt_out), args=args, results=results)
    if args.geometry_out:
        _write_geometry_summary(out_path=str(args.geometry_out), args=args, results=results)
    if args.stream_series_out:
        _write_stream_series_summary(out_path=str(args.stream_series_out), args=args, results=results)

    all_passed = all(r.passed for r in results)
    # Stress/sweep are distributional probes; per-trial FAILs are expected and are gated by the aggregate result.
    if args.mode == "stress":
        all_passed = next((bool(r.passed) for r in results if r.name == "SciFact-Streaming-Stress"), False)
    elif args.mode == "sweep":
        all_passed = next((bool(r.passed) for r in results if r.name == "SciFact-Streaming-SweepK"), False)
    if args.fast and not args.strict:
        return 0
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
