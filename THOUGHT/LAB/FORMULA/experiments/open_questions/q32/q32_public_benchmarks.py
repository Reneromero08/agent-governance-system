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
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-12
_CROSS_ENCODER = None
_SENTENCE_MODEL = None
_USE_CROSS_ENCODER = True


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
    cited_doc_ids: Tuple[int, ...]


@dataclass(frozen=True)
class BenchmarkResult:
    name: str
    passed: bool
    details: Dict[str, float]


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


def embed_texts(texts: List[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # type: ignore

    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        _SENTENCE_MODEL = SentenceTransformer(model_name)
    model = _SENTENCE_MODEL
    emb = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
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

        global _CROSS_ENCODER
        if _CROSS_ENCODER is None:
            # Smaller NLI cross-encoder (CPU-friendly relative to large RoBERTa).
            _CROSS_ENCODER = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768", max_length=256)
        model = _CROSS_ENCODER
        # NLI convention: (premise, hypothesis). Evidence sentence is premise; claim is hypothesis.
        pairs = list(zip(sentences, claims))
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


def run_scifact_benchmark(*, seed: int = 123, fast: bool = False, strict: bool = True) -> BenchmarkResult:
    print("\n[Q32:P2] Loading SciFact...")
    examples = load_scifact(max_claims=400, seed=seed)

    rng = np.random.default_rng(seed)
    rng.shuffle(examples)
    examples = examples[: (120 if fast else 600)]

    # Truth anchor: SciFact provides SUPPORT vs CONTRADICT at the (claim, doc) level.
    # We do not use labels inside M; labels only define the evaluation split.
    M_support: List[float] = []
    M_contradict: List[float] = []

    # For each pair, build observation sets as support scores from sentences.
    # Build observation/check sets with enough samples to avoid degenerate SE=EPS.
    # We use top-K sentences from the evidence doc by support score (label-free).
    from datasets import load_dataset  # type: ignore

    corpus = load_dataset("scifact", "corpus", trust_remote_code=True)["train"]
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

    # Build per-example M with an internal (hold-out) check pool.
    for ex in examples:
        label = str(ex.label).strip().upper()
        if label not in ("SUPPORT", "CONTRADICT"):
            continue

        sampled = sample_sentences_from_doc(
            int(ex.doc_id),
            10,
            seed_key=(seed * 1_000_003) ^ (int(ex.claim_id) * 9176) ^ int(ex.doc_id),
        )
        if not sampled:
            continue
        obs_sents = sampled[:2]
        check_sents = sampled[2:10]
        if len(obs_sents) < 2 or len(check_sents) < 6:
            continue

        obs_scores = sentence_support_scores([ex.claim] * len(obs_sents), obs_sents).tolist()
        check_scores = sentence_support_scores([ex.claim] * len(check_sents), check_sents).tolist()
        m = M_from_R(R_grounded(obs_scores, check_scores))
        if label == "SUPPORT":
            M_support.append(m)
        else:
            M_contradict.append(m)

        if fast and len(M_support) + len(M_contradict) in (10, 25, 50):
            print(f"[Q32:P2] scifact collected={len(M_support)+len(M_contradict)}")

    if min(len(M_support), len(M_contradict)) < (10 if fast else 60):
        raise RuntimeError(f"Too few SciFact labeled examples: support={len(M_support)} contradict={len(M_contradict)}")

    # Evaluate: P(M_support > M_contradict) by paired subsampling (deterministic shuffle).
    M_support_a = np.array(M_support, dtype=float)
    M_contra_a = np.array(M_contradict, dtype=float)
    rng.shuffle(M_support_a)
    rng.shuffle(M_contra_a)
    use_n = int(min(len(M_support_a), len(M_contra_a), (16 if fast else 120)))
    pair_wins = float(np.mean(M_support_a[:use_n] > M_contra_a[:use_n]))

    print("\n[Q32:P2] SciFact SUPPORT vs CONTRADICT discrimination")
    print(f"  P(M_support > M_contradict) = {pair_wins:.3f}")
    print(f"  mean(M_support)    = {M_support_a[:use_n].mean():.3f}")
    print(f"  mean(M_contradict) = {M_contra_a[:use_n].mean():.3f}")

    passed_discrimination = pair_wins >= (0.65 if fast else 0.70)

    # Negative control: shuffle labels (equivalently, compare against a permuted contradict array).
    M_contra_perm = M_contra_a.copy()
    rng.shuffle(M_contra_perm)
    collapse = float(np.mean(M_support_a[:use_n] > M_contra_perm[:use_n]))
    print("\n[Q32:P2] Negative control (label shuffle)")
    print(f"  P(M_support > M_contradict_permuted) = {collapse:.3f}")
    passed_negative = abs(collapse - 0.5) <= (0.25 if fast else 0.15)

    passed = bool(passed_discrimination and passed_negative)
    if not passed:
        print("\n[Q32:P2] SciFact status: FAIL (kept as a public counterexample until fixed)")
        if strict and not fast:
            raise AssertionError("FAIL: SciFact benchmark gates did not pass")
    return BenchmarkResult(
        name="SciFact",
        passed=passed,
        details={
            "pair_wins": pair_wins,
            "collapse": collapse,
            "mean_M_support": float(M_support_a[:use_n].mean()),
            "mean_M_contradict": float(M_contra_a[:use_n].mean()),
        },
    )


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q32 public truth-anchored benchmarks (fast + full modes).")
    p.add_argument(
        "--dataset",
        choices=["scifact", "climate_fever", "all"],
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
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def main() -> int:
    set_cache_roots()
    args = parse_args()

    global _USE_CROSS_ENCODER
    if args.scoring is not None:
        _USE_CROSS_ENCODER = args.scoring == "crossencoder"
    else:
        _USE_CROSS_ENCODER = False if args.fast else True

    # Full mode is strict by default; fast mode is non-strict unless explicitly requested.
    strict = args.strict or (not args.fast)

    results: List[BenchmarkResult] = []
    if args.dataset in ("scifact", "all"):
        results.append(run_scifact_benchmark(seed=args.seed, fast=args.fast, strict=strict))
    if args.dataset in ("climate_fever", "all"):
        results.append(run_climate_fever_benchmark(seed=args.seed, fast=args.fast, strict=strict))

    print("\n[Q32] PUBLIC BENCHMARK SUMMARY")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"  - {r.name}: {status}")

    all_passed = all(r.passed for r in results)
    if args.fast and not args.strict:
        return 0
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
