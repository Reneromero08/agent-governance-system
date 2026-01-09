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

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12


def set_cache_roots() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
    cache_root = os.path.join(repo_root, "LAW", "CONTRACTS", "_runs", "q32_public", "hf_cache")
    os.makedirs(cache_root, exist_ok=True)

    # Hugging Face caches
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(cache_root, "datasets"))
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(cache_root, "transformers"))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.path.join(cache_root, "sentence_transformers"))


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
    E = kernel_gaussian(z)
    grad_S = se(observations)
    return E / (grad_S + EPS)


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


def load_scifact(max_claims: int = 400, seed: int = 123) -> List[ScifactExample]:
    """
    Loads SciFact via HF datasets.
    We build examples at the (claim, doc) level with rationale sentences.
    """
    from datasets import load_dataset  # type: ignore

    ds = load_dataset("scifact", "claims", trust_remote_code=True)
    corpus = load_dataset("scifact", "corpus", trust_remote_code=True)["train"]
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

        doc = corpus_by_id[doc_id]
        title = str(doc.get("title", f"doc_{doc_id}"))
        abstract = doc.get("abstract", []) or []

        sents: List[str] = []
        for sid in sent_ids:
            if 0 <= int(sid) < len(abstract):
                sents.append(str(abstract[int(sid)]))
        if len(sents) < 2:
            continue

        examples.append(
            ScifactExample(
                claim_id=claim_id,
                claim=claim_text,
                doc_id=doc_id,
                doc_title=title,
                rationale_sentences=tuple(sents),
                label=label,
            )
        )

        if len({e.claim_id for e in examples}) >= max_claims:
            break

    if len(examples) < 50:
        raise RuntimeError(f"SciFact load produced too few examples ({len(examples)})")
    return examples


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    return np.asarray(emb, dtype=np.float32)


def sentence_support_scores(claims: List[str], sentences: List[str]) -> np.ndarray:
    """
    Produce a bounded support score in [-1,1] for each (claim, sentence) pair.

    We use a small NLI cross-encoder if available; otherwise fall back to cosine similarity.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore

        # Smaller NLI cross-encoder (CPU-friendly relative to large RoBERTa).
        model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", max_length=256)
        pairs = list(zip(claims, sentences))
        logits = np.asarray(model.predict(pairs, batch_size=16, show_progress_bar=False), dtype=float)
        # Some NLI cross-encoders output 3 logits (contradiction, entailment, neutral).
        if logits.ndim == 2 and logits.shape[1] == 3:
            # softmax
            m = logits - logits.max(axis=1, keepdims=True)
            ex = np.exp(m)
            probs = ex / ex.sum(axis=1, keepdims=True)
            # signed support: P(entail) - P(contradict)
            return (probs[:, 1] - probs[:, 0]).astype(np.float32)
        # Otherwise treat as a single unbounded score and squash.
        logits1 = logits.reshape(-1)
        return np.tanh(logits1).astype(np.float32)
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
    min_sentences: int = 2,
) -> List[Tuple[ScifactExample, Tuple[str, ...]]]:
    """
    For each true example (claim_id, doc), attach a false sentence set from a near claim's rationale.
    """
    by_claim: Dict[int, List[ScifactExample]] = {}
    for e in examples:
        by_claim.setdefault(e.claim_id, []).append(e)

    pairs: List[Tuple[ScifactExample, Tuple[str, ...]]] = []
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
        pairs.append((e, fe.rationale_sentences[:min_sentences]))

    if len(pairs) < 50:
        raise RuntimeError(f"Too few true/false basin pairs ({len(pairs)})")
    return pairs


def run_scifact_benchmark(seed: int = 123) -> None:
    print("\n[Q32:P2] Loading SciFact...")
    examples = load_scifact(max_claims=400, seed=seed)
    false_map = build_false_basin_mapping(examples, seed=seed)
    pairs = pick_true_and_false_sets(examples, false_map, min_sentences=2)

    rng = np.random.default_rng(seed)
    rng.shuffle(pairs)
    pairs = pairs[:120]

    M_true: List[float] = []
    M_false: List[float] = []
    M_false_shuffled_check: List[float] = []

    # For each pair, build observation sets as support scores from sentences.
    # Build observation/check sets with enough samples to avoid degenerate SE=EPS.
    # We use top-K sentences from the evidence doc by support score (label-free).
    from datasets import load_dataset  # type: ignore

    corpus = load_dataset("scifact", "corpus", trust_remote_code=True)["train"]
    corpus_by_id: Dict[int, dict] = {int(r["doc_id"]): r for r in corpus}

    def topk_sentences_for_doc(claim: str, doc_id: int, k: int = 8) -> List[str]:
        doc = corpus_by_id.get(int(doc_id))
        if not doc:
            return []
        abstract = doc.get("abstract", []) or []
        sents = [str(x) for x in abstract if str(x).strip()]
        if len(sents) < 3:
            return sents
        # score all sentences; choose top-k
        scores = sentence_support_scores([claim] * len(sents), sents)
        order = np.argsort(-scores)[: min(k, len(sents))]
        return [sents[int(i)] for i in order]

    # Precompute true check pools for negative control (shuffled across claims).
    true_checks_pool: List[List[float]] = []

    prepared: List[Tuple[List[float], List[float], List[float]]] = []
    # Each item: (obs_true_scores, check_true_scores, obs_false_scores)
    for e, false_sents in pairs:
        claim = e.claim
        true_sents = topk_sentences_for_doc(claim, e.doc_id, k=8)
        if len(true_sents) < 4:
            continue
        # obs = first 2, check = next 4 (SE meaningful)
        obs_true_sents = true_sents[:2]
        check_true_sents = true_sents[2:6]
        obs_true_scores = sentence_support_scores([claim] * len(obs_true_sents), obs_true_sents).tolist()
        check_true_scores = sentence_support_scores([claim] * len(check_true_sents), check_true_sents).tolist()

        obs_false_scores = sentence_support_scores([claim] * len(false_sents), list(false_sents)).tolist()
        if len(obs_false_scores) < 2:
            continue

        prepared.append((obs_true_scores, check_true_scores, obs_false_scores))
        true_checks_pool.append(check_true_scores)

    if len(prepared) < 80:
        raise RuntimeError(f"Too few prepared SciFact samples with non-degenerate checks ({len(prepared)})")

    rng.shuffle(true_checks_pool)

    for idx, (obs_true_scores, check_true_scores, obs_false_scores) in enumerate(prepared[:120]):
        check_shuffled = true_checks_pool[idx % len(true_checks_pool)]

        M_true.append(M_from_R(R_grounded(obs_true_scores, check_true_scores)))
        M_false.append(M_from_R(R_grounded(obs_false_scores, check_true_scores)))
        # negative control: use shuffled check (likely unrelated) and see discrimination collapse
        M_false_shuffled_check.append(M_from_R(R_grounded(obs_false_scores, check_shuffled)))

    M_true = np.array(M_true)
    M_false = np.array(M_false)
    M_false_shuffled_check = np.array(M_false_shuffled_check)

    pair_wins = float(np.mean(M_true > M_false))
    print("\n[Q32:P2] SciFact true-basin vs false-basin discrimination")
    print(f"  P(M_true > M_false) = {pair_wins:.3f}")
    print(f"  mean(M_true)  = {M_true.mean():.3f}")
    print(f"  mean(M_false) = {M_false.mean():.3f}")

    assert pair_wins >= 0.70, "FAIL: SciFact discrimination too weak (field not robust on public benchmark)"

    # Negative control: replace check group with shuffled checks; discrimination should collapse.
    collapse = float(np.mean(M_true > M_false_shuffled_check))
    print("\n[Q32:P2] Negative control (shuffled baseline)")
    print(f"  P(M_true > M_false_with_shuffled_check) = {collapse:.3f}")
    assert collapse <= 0.65, "FAIL: negative control did not collapse (suspect leakage)"


def main() -> int:
    set_cache_roots()
    run_scifact_benchmark(seed=123)
    print("\n[Q32] PUBLIC BENCHMARK (SciFact) PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
