"""COMMONSENSE Bridge: fact-extraction and integration layer.

Connects unstructured model output to the COMMONSENSE resolver.
"""

from .fact_extractor import extract_facts, extract_facts_prompt, extract_facts_regex
from .integration import check_output, commonsense_fragment, batch_check, Verdict

__all__ = [
    "extract_facts",
    "extract_facts_prompt",
    "extract_facts_regex",
    "check_output",
    "commonsense_fragment",
    "batch_check",
    "Verdict",
]
