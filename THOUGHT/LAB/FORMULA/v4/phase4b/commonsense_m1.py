"""Method 1 COMMONSENSE — LFM 2.5 extracts structured facts, resolver checks them.

Wires the LLM-based fact extraction (Method 1) into the COMMONSENSE
verification fragment. Uses LFM 2.5 to extract facts from model output,
then resolves against CODEBOOK.json via the COMMONSENSE bridge.
"""
import sys, os, json
from pathlib import Path

PHASE4B = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\FORMULA\v4\phase4b"
GGUF = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\TINY_COMPRESS\extensions\03_flat_llm"
BRIDGE = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE\bridge"
COMMONSENSE = r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE"
for p in [PHASE4B, GGUF, BRIDGE, COMMONSENSE]:
    if p not in sys.path: sys.path.insert(0, p)

# Make relative imports work inside bridge/integration.py
import fact_extractor
sys.modules["bridge.fact_extractor"] = fact_extractor

# Must set __package__ before importing integration
import importlib.util
spec = importlib.util.spec_from_file_location("bridge.integration",
    r"D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\COMMONSENSE\bridge\integration.py")
integration = importlib.util.module_from_spec(spec)
sys.modules["bridge.integration"] = integration
integration.__package__ = "bridge"
spec.loader.exec_module(integration)
check_output = integration.check_output

from gguf_backend import GgufBackend


class Method1Commonsense:
    """COMMONSENSE fragment using LFM 2.5 for Method 1 fact extraction."""

    def __init__(self, llm=None):
        if llm is not None:
            self.llm = llm
            self._own_llm = False
        else:
            self.llm = GgufBackend()
            self._own_llm = True
        self._call_count = 0

    def _extract(self, text: str) -> str:
        """LLM callable for fact_extractor."""
        prompt = (
            "Extract all factual claims, defaults, exceptions, and invariants from the "
            "following text. Return ONLY a JSON list of strings. Prefix each with one of: "
            "fact:, default:, exception:, invariant:, ref:.\n\n"
            "Text:\n" + text[:500] + "\n\nJSON list:"
        )
        return self.llm.generate(prompt, max_tokens=200, temperature=0.0)

    def verify(self, text: str):
        """Run Method 1 COMMONSENSE verification."""
        from phase4b_fragments import FragmentResult

        try:
            verdict = check_output(text, extract_method="prompt", llm_callable=self._extract)
            self._call_count += 1

            if verdict.status == "hard_fail":
                score, conf = 0.0, verdict.confidence
                frag_verdict = "hard_fail"
                evidence = "M1 HARD_FAIL: {} violations".format(len(verdict.hard_fails))
            elif verdict.status == "soft_fail":
                score, conf = 0.5, verdict.confidence
                frag_verdict = "soft_fail"
                evidence = "M1 SOFT_FAIL: {} warnings".format(len(verdict.soft_fails))
            else:
                score, conf = 1.0, verdict.confidence
                frag_verdict = "pass"
                evidence = "M1 PASS: {} facts extracted, {} derived".format(
                    len(verdict.extracted_facts), len(verdict.derived_facts))

            return FragmentResult(
                fragment_id=1, fragment_name="COMMONSENSE-M1",
                score=score, confidence=conf, verdict=frag_verdict,
                evidence=evidence,
                details={"extracted": verdict.extracted_facts[:5], "derived": verdict.derived_facts[:3]},
            )
        except Exception as e:
            from phase4b_fragments import FragmentResult
            return FragmentResult(
                fragment_id=1, fragment_name="COMMONSENSE-M1",
                score=0.5, confidence=0.3, verdict="abstain",
                evidence="M1 error: {}".format(str(e)[:100]),
            )


if __name__ == "__main__":
    cs = Method1Commonsense()
    tests = [
        "The capital of France is Paris. This is a well-known fact.",
        "The Earth is flat and gravity is a myth promoted by NASA.",
        "Water has the chemical formula H2O. It boils at 100 degrees Celsius.",
        "Vaccines cause autism according to many studies.",
    ]
    for t in tests:
        r = cs.verify(t)
        print("{} -> {} ({})".format(r.verdict, r.evidence[:80], r.score))
