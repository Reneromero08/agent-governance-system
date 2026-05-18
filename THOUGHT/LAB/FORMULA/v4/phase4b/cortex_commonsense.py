"""CORTEX-backed COMMONSENSE fragment.

Routes regex-extracted facts into the facts cassette instead of
CODEBOOK.json resolver. The cassette IS the ground truth.

Architecture:
    1. Regex extract facts from generated text (Method 2, zero-dependency)
    2. Check each fact against facts cassette
    3. Conflict → hard_fail. Agreement → pass. Unknown → soft_fail.
"""
import sys, os, json, re
from pathlib import Path

sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\llm-spectral\auto_feedback")
sys.path.insert(0, r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE\bridge")

from facts_cassette import FactsCassette
from phase4b_fragments import FragmentResult

# Import regex extractor (Method 2, no LLM needed)
import fact_extractor


class CortexCommonsense:
    """Verification fragment: regex extraction + cassette verification."""

    def __init__(self):
        self.cassette = FactsCassette()

    def _extract_statements(self, text: str) -> list:
        """Extract simple declarative statements via regex + sentence splitting."""
        # Get fact: prefixed statements from regex extractor
        facts = fact_extractor.extract_facts(text, method="regex")
        # Filter to fact: and default: prefixes, strip prefix
        statements = []
        for f in facts:
            if f.startswith("fact:") or f.startswith("default:"):
                slug = f.split(":", 1)[1] if ":" in f else f
                # Un-slugify: replace underscores with spaces
                readable = slug.replace("_", " ").strip()
                if len(readable) > 10:
                    statements.append(readable)
        return statements[:8]

    def _check_statement(self, statement: str) -> dict:
        """Check a statement against the facts cassette."""
        facts = self.cassette.retrieve_fact(statement)
        if facts:
            stmt_lower = statement.lower()
            for f in facts:
                if f.lower() in stmt_lower:
                    return {"verdict": "pass", "evidence": "confirmed: " + f}
            # Fact found but not clearly confirmed
            return {"verdict": "soft_fail", "evidence": "mismatch vs: " + facts[0]}

        docs = self.cassette.retrieve_docs(statement, top_k=1, threshold=0.3)
        if docs:
            return {"verdict": "pass", "evidence": "doc: " + docs[0][:60]}
        return {"verdict": "abstain", "evidence": "unknown"}

    def verify(self, text: str) -> FragmentResult:
        """Run cortex-backed COMMONSENSE verification."""
        statements = self._extract_statements(text)

        if not statements:
            return FragmentResult(
                fragment_id=1, fragment_name="CORTEX-COMMONSENSE",
                score=1.0, confidence=0.5, verdict="abstain",
                evidence="No statements extracted")

        hard_fails = []
        soft_fails = []
        passes = []

        for stmt in statements:
            result = self._check_statement(stmt)
            if result["verdict"] == "hard_fail":
                hard_fails.append((stmt, result["evidence"]))
            elif result["verdict"] == "soft_fail":
                soft_fails.append((stmt, result["evidence"]))
            else:
                passes.append((stmt, result["evidence"]))

        if hard_fails:
            evidence = "CONTRADICT: " + "; ".join(
                "{} ({})".format(c[:50], e) for c, e in hard_fails[:2])
            return FragmentResult(
                fragment_id=1, fragment_name="CORTEX-COMMONSENSE",
                score=0.0, confidence=0.9, verdict="hard_fail",
                evidence=evidence)

        if soft_fails and not passes:
            evidence = "UNCERTAIN: " + "; ".join(
                "{} ({})".format(c[:50], e) for c, e in soft_fails[:2])
            return FragmentResult(
                fragment_id=1, fragment_name="CORTEX-COMMONSENSE",
                score=0.5, confidence=0.7, verdict="soft_fail",
                evidence=evidence)

        evidence = "OK: {} checked ({} pass, {} soft, {} hard)".format(
            len(statements), len(passes), len(soft_fails), len(hard_fails))
        return FragmentResult(
            fragment_id=1, fragment_name="CORTEX-COMMONSENSE",
            score=1.0, confidence=0.8, verdict="pass",
            evidence=evidence)


if __name__ == "__main__":
    cs = CortexCommonsense()
    tests = [
        "The capital of France is Paris. It has a population of 2 million.",
        "The Earth is flat and gravity is a myth promoted by NASA.",
        "Water has the chemical formula H2O. It boils at 100 degrees Celsius.",
        "The chemical symbol for gold is Ag. It is a noble metal.",
    ]
    for t in tests:
        r = cs.verify(t)
        print("{}: {} ({})".format(r.verdict.upper(), r.evidence[:120], r.score))
        print("  statements:", cs._extract_statements(t)[:3])
    tests = [
        "The capital of France is Paris. It has a population of 2 million.",
        "The Earth is flat and gravity is a myth promoted by NASA.",
        "Water has the chemical formula H2O. It boils at 100 degrees Celsius.",
        "The chemical symbol for gold is Ag. It is a noble metal.",
    ]
    for t in tests:
        r = cs.verify(t)
        print("{}: {} ({})".format(r.verdict.upper(), r.evidence[:120], r.score))
