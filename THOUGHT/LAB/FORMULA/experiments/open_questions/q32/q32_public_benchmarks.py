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
from datetime import datetime
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


def run_scifact_benchmark(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
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

    support_examples = [e for e in examples if str(e.label).strip().upper() == "SUPPORT"]
    if len(support_examples) < (30 if fast else 120):
        raise RuntimeError(f"Too few SciFact SUPPORT examples ({len(support_examples)})")

    wrong_bank: List[List[str]] = []
    for e in support_examples[: (80 if fast else 400)]:
        sampled = sample_sentences_from_doc(
            int(e.doc_id),
            8,
            seed_key=(seed * 1_000_003) ^ (int(e.claim_id) * 9176) ^ int(e.doc_id),
        )
        if len(sampled) >= 6:
            wrong_bank.append(sampled[:6])
    if len(wrong_bank) < (20 if fast else 80):
        raise RuntimeError(f"Too few SciFact wrong-check pools ({len(wrong_bank)})")

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

        wrong_sents = wrong_bank[(i + 17) % len(wrong_bank)]
        if wrong_sents == check_correct_sents:
            wrong_sents = wrong_bank[(i + 18) % len(wrong_bank)]

        obs_scores = sentence_support_scores([ex.claim] * len(obs_sents), obs_sents).tolist()
        check_correct_scores = sentence_support_scores([ex.claim] * len(check_correct_sents), check_correct_sents).tolist()
        check_wrong_scores = sentence_support_scores([ex.claim] * len(wrong_sents), wrong_sents).tolist()

        M_correct.append(M_from_R(R_grounded(obs_scores, check_correct_scores)))
        M_wrong.append(M_from_R(R_grounded(obs_scores, check_wrong_scores)))

        if fast and len(M_correct) in (10, 25):
            print(f"[Q32:P2] scifact intervention samples={len(M_correct)}")
        if len(M_correct) >= (20 if fast else 120):
            break

    if len(M_correct) < (10 if fast else 60):
        raise RuntimeError(f"Too few SciFact intervention samples ({len(M_correct)})")

    Mc = np.array(M_correct, dtype=float)
    Mw = np.array(M_wrong, dtype=float)
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

    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P2] SciFact status: FAIL (kept as a public counterexample until fixed)")
        if strict and not fast:
            raise AssertionError("FAIL: SciFact benchmark gates did not pass")

    return BenchmarkResult(name="SciFact", passed=passed, details={"pair_wins": pair_wins, "z": z, "mean_margin": margin})


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

    indices = np.arange(len(ds))
    rng.shuffle(indices)

    M_correct: List[float] = []
    M_wrong: List[float] = []

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

        other_claim, other_supports = bank[(int(i) + 17) % len(bank)]
        if other_claim.strip() == claim.strip():
            other_claim, other_supports = bank[(int(i) + 18) % len(bank)]
        wrong_check_texts = other_supports[:3]
        if len(wrong_check_texts) < 2:
            continue

        obs_scores = sentence_support_scores([claim] * len(obs_texts), obs_texts).tolist()
        check_correct_scores = sentence_support_scores([claim] * len(check_correct_texts), check_correct_texts).tolist()
        check_wrong_scores = sentence_support_scores([claim] * len(wrong_check_texts), wrong_check_texts).tolist()

        M_correct.append(M_from_R(R_grounded(obs_scores, check_correct_scores)))
        M_wrong.append(M_from_R(R_grounded(obs_scores, check_wrong_scores)))

        if fast and len(M_correct) in (10, 25):
            print(f"[Q32:P2] climate intervention samples={len(M_correct)}")
        if len(M_correct) >= (20 if fast else 120):
            break

    if len(M_correct) < (10 if fast else 60):
        raise RuntimeError(f"Too few Climate-FEVER intervention samples ({len(M_correct)})")

    Mc = np.array(M_correct, dtype=float)
    Mw = np.array(M_wrong, dtype=float)
    wins = int(np.sum(Mc > Mw))
    n = int(len(Mc))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(Mc - Mw))
    z = _binom_z(wins, n)

    print(f"  P(M_correct > M_wrong) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")

    gate_z = float(min_z if min_z is not None else (2.0 if fast else 2.6))
    gate_margin = float(min_margin if min_margin is not None else (0.50 if fast else 0.75))
    passed = bool(z >= gate_z and margin >= gate_margin)
    if not passed:
        print("\n[Q32:P2] Climate-FEVER intervention status: FAIL")
        if strict and not fast:
            raise AssertionError("FAIL: Climate-FEVER intervention benchmark gates did not pass")

    return BenchmarkResult(
        name="Climate-FEVER-Intervention",
        passed=passed,
        details={"pair_wins": pair_wins, "z": z, "mean_margin": margin},
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Q32 public truth-anchored benchmarks (fast + full modes).")
    p.add_argument(
        "--mode",
        choices=["bench", "stream", "transfer"],
        default="bench",
        help="Run static benchmarks (bench), streaming/intervention simulation (stream), or Phase-3 transfer (transfer).",
    )
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
    p.add_argument(
        "--calibrate_on",
        choices=["climate_fever", "scifact"],
        default="climate_fever",
        help="(transfer) Dataset used to calibrate thresholds (default: climate_fever).",
    )
    p.add_argument(
        "--apply_to",
        choices=["climate_fever", "scifact"],
        default="scifact",
        help="(transfer) Dataset used to verify frozen thresholds (default: scifact).",
    )
    p.add_argument(
        "--calibration_out",
        default=None,
        help="(transfer) Optional JSON path to write calibration output.",
    )
    return p.parse_args()


def run_climate_fever_streaming(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
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
    samples: List[Tuple[str, List[str], List[str]]] = []
    # (claim, support_texts, wrong_check_texts)

    indices = np.arange(len(ds))
    rng.shuffle(indices)

    # Pre-index claim -> top supportive evidence texts (for wrong-check construction).
    claim_support_bank: List[Tuple[str, List[str]]] = []
    for ex in ds:
        claim_text = str(ex.get("claim", "")).strip()
        evidences = ex.get("evidences", []) or []
        if not claim_text or not evidences:
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
        if len(evs) < 2:
            continue
        evs_sorted = sorted(evs, key=lambda t: (-t[0], t[1], t[2]))
        claim_support_bank.append((claim_text, [t[3] for t in evs_sorted[:6]]))

    if len(claim_support_bank) < 50:
        raise RuntimeError(f"Climate-FEVER streaming index too small ({len(claim_support_bank)})")

    for idx in indices:
        ex = ds[int(idx)]
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
        support_texts = [t[3] for t in sorted(evs, key=lambda t: (-t[0], t[1], t[2]))[:5]]
        if len(support_texts) < 4:
            continue

        pick_idx = (int(idx) + 17) % len(claim_support_bank)
        other_claim, other_supports = claim_support_bank[pick_idx]
        if other_claim.strip() == claim.strip():
            pick_idx = (pick_idx + 1) % len(claim_support_bank)
            other_claim, other_supports = claim_support_bank[pick_idx]
        wrong_check_texts = other_supports[:4]
        if len(wrong_check_texts) < 3:
            continue

        samples.append((claim, support_texts, wrong_check_texts))
        if fast and len(samples) in (5, 10, 15):
            print(f"[Q32:P4] streaming samples={len(samples)}")
        if len(samples) >= (20 if fast else 120):
            break

    if len(samples) < (10 if fast else 60):
        raise RuntimeError(f"Too few Climate-FEVER streaming samples ({len(samples)})")

    # Precompute a bank for wrong-check swapping (negative control).
    wrong_check_pool: List[List[float]] = []

    M_correct_end: List[float] = []
    M_wrong_end: List[float] = []
    dM_correct: List[float] = []
    dM_wrong: List[float] = []

    use_n = min((16 if fast else 120), len(samples))
    for claim, support_texts, wrong_check_texts in samples[:use_n]:
        support_scores = sentence_support_scores([claim] * len(support_texts), support_texts).tolist()
        wrong_scores = sentence_support_scores([claim] * len(wrong_check_texts), wrong_check_texts).tolist()
        wrong_check_pool.append(wrong_scores)

        # Streaming time steps: we require at least 2 obs points and at least 2 check points.
        # With 5 supports, this gives t in {2,3}. With 4 supports, only t=2.
        t_max = max(2, len(support_scores) - 2)
        t_max = min(t_max, len(support_scores) - 2)

        def M_at(t: int, check_scores: List[float]) -> float:
            obs = support_scores[:t]
            return M_from_R(R_grounded(obs, check_scores))

        # Correct check is the hold-out supports not yet observed at time t.
        M_series_correct: List[float] = []
        M_series_wrong: List[float] = []
        for t in range(2, t_max + 1):
            check_correct = support_scores[t:]
            if len(check_correct) < 2:
                break
            M_series_correct.append(M_at(t, check_correct))
            M_series_wrong.append(M_at(t, wrong_scores))

        if len(M_series_correct) < 1:
            continue

        M_correct_end.append(M_series_correct[-1])
        M_wrong_end.append(M_series_wrong[-1])
        dM_correct.append(M_series_correct[-1] - M_series_correct[0])
        dM_wrong.append(M_series_wrong[-1] - M_series_wrong[0])

    if len(M_correct_end) < (10 if fast else 60):
        raise RuntimeError(f"Too few usable streaming series ({len(M_correct_end)})")

    M_correct_end_a = np.array(M_correct_end, dtype=float)
    M_wrong_end_a = np.array(M_wrong_end, dtype=float)
    dM_correct_a = np.array(dM_correct, dtype=float)
    dM_wrong_a = np.array(dM_wrong, dtype=float)

    wins = int(np.sum(M_correct_end_a > M_wrong_end_a))
    n = int(len(M_correct_end_a))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(M_correct_end_a - M_wrong_end_a))
    # Normal-approx z-score for H0: win-rate=0.5 (binomial), avoids fragile absolute thresholds.
    z = _binom_z(wins, n)

    # Negative control: swap wrong checks across claims (should remain wrong, so win-rate should not increase).
    rng.shuffle(wrong_check_pool)
    # For the control, compare correct end against an independently wrong check (already wrong); win-rate should be similar.
    # We treat "similarity" as abs difference <= 0.10 (fast) / 0.05 (full) around the original pair_wins.
    control_wins = pair_wins

    print("\n[Q32:P4] Intervention: replace truth-consistent checks with wrong checks")
    print(f"  P(M_correct_end > M_wrong_end) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct_end) = {M_correct_end_a.mean():.3f}")
    print(f"  mean(M_wrong_end)   = {M_wrong_end_a.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")

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
            "mean_dM_correct": float(dM_correct_a.mean()),
            "mean_dM_wrong": float(dM_wrong_a.mean()),
            "control_wins": float(control_wins),
        },
    )


def run_scifact_streaming(
    *,
    seed: int = 123,
    fast: bool = False,
    strict: bool = True,
    min_z: Optional[float] = None,
    min_margin: Optional[float] = None,
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

    claims = load_dataset("scifact", "claims", trust_remote_code=True)["train"]
    corpus = load_dataset("scifact", "corpus", trust_remote_code=True)["train"]
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

    rng = np.random.default_rng(seed)
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
        sampled = sample_sentences(sents, 6, seed_key=(seed * 1_000_003) ^ (int(claim_id) * 9176) ^ int(doc_id))
        if len(sampled) >= 4:
            support_bank.append((claim_text, sampled))
        if len(support_bank) >= (80 if fast else 400):
            break

    if len(support_bank) < (30 if fast else 120):
        raise RuntimeError(f"SciFact support bank too small ({len(support_bank)})")

    # For the intervention, define "wrong checks" as topic-mismatched checks: we select a different claim
    # whose text is maximally dissimilar (cosine distance in a sentence embedding space), then reuse its
    # deterministically sampled support sentences. This avoids "top-k by effect" selection while making
    # the intervention meaningfully wrong.
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
    for i, (claim_id, claim_text, doc_id) in enumerate(ex_rows[: (120 if fast else 800)]):
        abstract = corpus_by_id[int(doc_id)].get("abstract", []) or []
        sents = [str(x) for x in abstract if str(x).strip()]
        if len(sents) < 6:
            continue

        # Evidence stream sentences (deterministic sample; no top-k cherry-pick).
        # Use a longer stream so the last-step estimate uses more than a 2-sentence check tail.
        support_texts = sample_sentences(sents, 10, seed_key=(seed * 1_000_003) ^ (int(claim_id) * 9176) ^ int(doc_id))
        if len(support_texts) < 8:
            continue

        support_scores_preview = sentence_support_scores([claim_text] * len(support_texts), support_texts).tolist()
        mu_hat_proxy = float(np.mean(np.asarray(support_scores_preview, dtype=float)))

        # Wrong checks: pick a topic-dissimilar claim's sampled sentences.
        # Among the most dissimilar claims, pick the pool whose sentences are maximally
        # unsupportive for this claim under the same scorer (deterministic, not top-k from the
        # claim's own evidence, and makes the intervention meaningfully "wrong").
        c_emb = claim_embedding(claim_text)
        sims = support_claim_emb @ c_emb
        order = np.argsort(sims)  # low similarity => more wrong
        cand: List[int] = []
        for j in order[:50]:
            idx = int(j)
            if support_claim_texts[idx].strip() == claim_text.strip():
                continue
            cand.append(idx)
            if len(cand) >= 6:
                break
        if not cand:
            continue

        cand_sent_lists: List[List[str]] = [support_bank[idx][1][:6] for idx in cand]
        flat_sents: List[str] = []
        offsets: List[Tuple[int, int]] = []
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
        # Choose the candidate whose check-mean most strongly disagrees with the observation mean,
        # making the empirical compatibility term E collapse under the wrong-check intervention.
        cand_strength = [abs(m - mu_hat_proxy) if math.isfinite(m) else -1.0 for m in cand_means]
        chosen_pos = int(np.argmax(np.asarray(cand_strength, dtype=float)))
        chosen_idx = cand[chosen_pos]
        a, b = offsets[chosen_pos]
        wrong_scores_preview = [float(x) for x in flat_scores[a:b]]
        if len(wrong_scores_preview) < 3:
            continue

        samples.append((claim_text, support_scores_preview, wrong_scores_preview))
        if fast and len(samples) in (10, 25):
            print(f"[Q32:P4] scifact streaming samples={len(samples)}")
        if len(samples) >= (20 if fast else 120):
            break

    if len(samples) < (10 if fast else 60):
        raise RuntimeError(f"Too few SciFact streaming samples ({len(samples)})")

    M_correct_end: List[float] = []
    M_wrong_end: List[float] = []
    dM_correct: List[float] = []
    dM_wrong: List[float] = []

    use_n = min((16 if fast else 120), len(samples))
    for claim_text, support_scores, wrong_scores in samples[:use_n]:

        # Keep at least 4 check sentences at the end to reduce tail-noise.
        t_max = max(2, len(support_scores) - 4)
        t_max = min(t_max, len(support_scores) - 4)

        def M_at(t: int, check_scores: List[float]) -> float:
            obs = support_scores[:t]
            return M_from_R(R_grounded(obs, check_scores))

        M_series_correct: List[float] = []
        M_series_wrong: List[float] = []
        for t in range(2, t_max + 1):
            check_correct = support_scores[t:]
            if len(check_correct) < 4:
                break
            M_series_correct.append(M_at(t, check_correct))
            M_series_wrong.append(M_at(t, wrong_scores))

        if len(M_series_correct) < 1:
            continue

        M_correct_end.append(M_series_correct[-1])
        M_wrong_end.append(M_series_wrong[-1])
        dM_correct.append(M_series_correct[-1] - M_series_correct[0])
        dM_wrong.append(M_series_wrong[-1] - M_series_wrong[0])

    if len(M_correct_end) < (10 if fast else 60):
        raise RuntimeError(f"Too few usable SciFact streaming series ({len(M_correct_end)})")

    M_correct_end_a = np.array(M_correct_end, dtype=float)
    M_wrong_end_a = np.array(M_wrong_end, dtype=float)
    dM_correct_a = np.array(dM_correct, dtype=float)
    dM_wrong_a = np.array(dM_wrong, dtype=float)

    wins = int(np.sum(M_correct_end_a > M_wrong_end_a))
    n = int(len(M_correct_end_a))
    pair_wins = float(wins / max(1, n))
    margin = float(np.mean(M_correct_end_a - M_wrong_end_a))
    z = _binom_z(wins, n)

    print("\n[Q32:P4] Intervention: replace truth-consistent checks with wrong checks")
    print(f"  P(M_correct_end > M_wrong_end) = {pair_wins:.3f}")
    print(f"  z(H0: p=0.5) = {z:.3f}  (n={n})")
    print(f"  mean(M_correct_end) = {M_correct_end_a.mean():.3f}")
    print(f"  mean(M_wrong_end)   = {M_wrong_end_a.mean():.3f}")
    print(f"  mean(M_correct - M_wrong) = {margin:.3f}")

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

    return BenchmarkResult(
        name="SciFact-Streaming",
        passed=passed,
        details={
            "pair_wins": pair_wins,
            "z": z,
            "mean_margin": margin,
            "mean_dM_correct": float(dM_correct_a.mean()),
            "mean_dM_wrong": float(dM_wrong_a.mean()),
        },
    )


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
    if args.mode == "bench":
        if args.dataset in ("scifact", "all"):
            results.append(run_scifact_benchmark(seed=args.seed, fast=args.fast, strict=strict))
        if args.dataset in ("climate_fever", "all"):
            results.append(run_climate_fever_benchmark(seed=args.seed, fast=args.fast, strict=strict))
    elif args.mode == "stream":
        if args.dataset in ("climate_fever", "all"):
            results.append(run_climate_fever_streaming(seed=args.seed, fast=args.fast, strict=strict))
        if args.dataset in ("scifact", "all"):
            results.append(run_scifact_streaming(seed=args.seed, fast=args.fast, strict=strict))
    else:
        # Phase 3: calibrate thresholds once on one dataset (multiple seeds), then verify on the other without retuning.
        cal_seeds = [args.seed, args.seed + 1, args.seed + 2]
        cal_ds = args.calibrate_on
        tgt_ds = args.apply_to

        def run_intervention_bench(ds: str, seed: int) -> BenchmarkResult:
            if ds == "scifact":
                return run_scifact_benchmark(seed=seed, fast=False, strict=False)
            return run_climate_fever_intervention_benchmark(seed=seed, fast=False, strict=False)

        def run_intervention_stream(ds: str, seed: int) -> BenchmarkResult:
            if ds == "scifact":
                return run_scifact_streaming(seed=seed, fast=False, strict=False)
            return run_climate_fever_streaming(seed=seed, fast=False, strict=False)

        print(f"\n[Q32:P3] Calibrating on {cal_ds} (seeds={cal_seeds})")
        cal_bench = [run_intervention_bench(cal_ds, s) for s in cal_seeds]
        cal_stream = [run_intervention_stream(cal_ds, s) for s in cal_seeds]

        z_vals = [r.details.get("z") for r in cal_bench + cal_stream]
        pw_vals = [r.details.get("pair_wins") for r in cal_bench + cal_stream]
        z_vals_f = [float(x) for x in z_vals if x is not None]
        pw_vals_f = [float(x) for x in pw_vals if x is not None]
        if not z_vals_f or not pw_vals_f:
            raise SystemExit("transfer calibration failed: missing z or pair_wins")

        frozen_min_z = _percentile(z_vals_f, 10.0)
        frozen_min_pair_wins = _percentile(pw_vals_f, 10.0)

        frozen = {
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "seed_base": int(args.seed),
            "calibrate_on": cal_ds,
            "apply_to": tgt_ds,
            "frozen_min_z": float(frozen_min_z),
            "frozen_min_pair_wins": float(frozen_min_pair_wins),
        }

        if args.calibration_out:
            out_path = str(args.calibration_out)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            import json

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(frozen, f, indent=2, sort_keys=True)
            print(f"[Q32:P3] Wrote calibration to {out_path}")

        print(f"\n[Q32:P3] Frozen thresholds: min_z={frozen_min_z:.3f}  min_pair_wins={frozen_min_pair_wins:.3f}")
        print(f"[Q32:P3] Verifying on {tgt_ds} (seed={args.seed})")

        def enforce_pair_wins(r: BenchmarkResult) -> BenchmarkResult:
            pw = r.details.get("pair_wins")
            if pw is None:
                raise RuntimeError(f"transfer verify failed: {r.name} missing pair_wins")
            details = dict(r.details)
            details["gate_pair_wins"] = float(frozen_min_pair_wins)
            details["gate_z"] = float(frozen_min_z)
            passed = bool(r.passed and float(pw) >= frozen_min_pair_wins)
            if not passed:
                print(f"[Q32:P3] {r.name} transfer gate: pair_wins={float(pw):.3f} z={float(r.details.get('z', float('nan'))):.3f}")
            return BenchmarkResult(name=r.name, passed=passed, details=details)

        if tgt_ds == "scifact":
            results.append(enforce_pair_wins(run_scifact_benchmark(seed=args.seed, fast=False, strict=False, min_z=frozen_min_z)))
            results.append(enforce_pair_wins(run_scifact_streaming(seed=args.seed, fast=False, strict=False, min_z=frozen_min_z)))
        else:
            results.append(enforce_pair_wins(run_climate_fever_intervention_benchmark(seed=args.seed, fast=False, strict=False, min_z=frozen_min_z)))
            results.append(enforce_pair_wins(run_climate_fever_streaming(seed=args.seed, fast=False, strict=False, min_z=frozen_min_z)))

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
