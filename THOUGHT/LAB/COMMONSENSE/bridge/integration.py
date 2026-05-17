"""Bridge integration: connects fact_extractor -> resolver -> verdict.

Provides:
    check_output(text)       -> full Verdict
    commonsense_fragment(text) -> dict for truth attractor C_epistemic
    batch_check(texts)       -> list of Verdicts

Usage:
    from bridge.integration import check_output, commonsense_fragment

    verdict = check_output("Birds can fly unless they are penguins.")
    print(verdict.status, verdict.emits)

    fragment = commonsense_fragment("The Earth is flat. Gravity is a myth.")
    # fragment = {"score": 0.0, "confidence": 1.0, "hard_fails": [...]}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_BRIDGE_DIR = Path(__file__).resolve().parent
_COMMONSENSE_DIR = _BRIDGE_DIR.parent


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """Result of running COMMONSENSE resolution on extracted facts."""
    status: str                     # "pass", "soft_fail", "hard_fail"
    score: float                    # 0.0 to 1.0
    confidence: float               # 0.0 to 1.0
    input_text: str = ""
    extracted_facts: List[str] = field(default_factory=list)
    selected_ids: List[str] = field(default_factory=list)
    derived_facts: List[str] = field(default_factory=list)
    emits: List[Dict[str, Any]] = field(default_factory=list)
    hard_fails: List[Dict[str, Any]] = field(default_factory=list)
    soft_fails: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["input_text"] = self.input_text[:200]
        return d


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

_resolver = None
_codebook = None
_db = None


def _get_resolver():
    global _resolver
    if _resolver is None:
        import importlib.util
        import sys
        spec = importlib.util.spec_from_file_location(
            "resolver", str(_COMMONSENSE_DIR / "resolver.py"))
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        sys.modules["resolver"] = mod
        spec.loader.exec_module(mod)
        _resolver = mod
    return _resolver


def _get_codebook():
    global _codebook
    if _codebook is None:
        cb_path = _COMMONSENSE_DIR / "CODEBOOK.json"
        _codebook = json.loads(cb_path.read_text(encoding="utf-8"))
    return _codebook


def _get_db():
    global _db
    if _db is None:
        db_path = _COMMONSENSE_DIR / "db.example.json"
        _db = json.loads(db_path.read_text(encoding="utf-8"))
    return _db


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def check_output(
    text: str,
    extract_method: str = "regex",
    llm_callable=None,
) -> Verdict:
    """Run the full pipeline: extract facts -> resolve -> classify verdict.

    Args:
        text: Raw model output text.
        extract_method: "regex" (default) or "prompt".
        llm_callable: Required if extract_method="prompt".

    Returns:
        Verdict with status, score, confidence, and detail.
    """
    from .fact_extractor import extract_facts, extract_facts_prompt

    if extract_method == "prompt":
        if llm_callable is None:
            raise ValueError("extract_method='prompt' requires llm_callable")
        facts = extract_facts_prompt(text, llm_callable)
    else:
        facts = extract_facts(text, method=extract_method)

    resolver_mod = _get_resolver()
    codebook = _get_codebook()
    db = _get_db()

    result = resolver_mod.resolve(db, facts, codebook=codebook)

    hard_fails = [
        e for e in result.get("emits", [])
        if isinstance(e, dict) and e.get("type") == "hard_fail"
    ]
    soft_fails = [
        e for e in result.get("emits", [])
        if isinstance(e, dict) and e.get("type") != "hard_fail"
    ]

    if hard_fails:
        status = "hard_fail"
        score = 0.0
        confidence = 1.0
    elif soft_fails:
        status = "soft_fail"
        score = 0.5
        confidence = 0.8
    else:
        status = "pass"
        score = 1.0
        confidence = 0.9

    return Verdict(
        status=status,
        score=score,
        confidence=confidence,
        input_text=text,
        extracted_facts=facts,
        selected_ids=result.get("selected_ids", []),
        derived_facts=result.get("derived_facts", []),
        emits=result.get("emits", []),
        hard_fails=hard_fails,
        soft_fails=soft_fails,
    )


def commonsense_fragment(text: str) -> Dict[str, Any]:
    """Truth attractor fragment: score a proposition via COMMONSENSE verification.

    Plugs directly into C_epistemic alongside factual verification,
    logical consistency, and self-consistency fragments.

    Returns a dict with score, confidence, and detail suitable for
    aggregation across multiple fragments.
    """
    verdict = check_output(text)
    return {
        "fragment": "commonsense",
        "score": verdict.score,
        "confidence": verdict.confidence,
        "status": verdict.status,
        "hard_fail_count": len(verdict.hard_fails),
        "soft_fail_count": len(verdict.soft_fails),
        "emits": verdict.emits,
    }


def batch_check(texts: List[str], **kwargs) -> List[Verdict]:
    """Run check_output on a list of texts."""
    return [check_output(t, **kwargs) for t in texts]
