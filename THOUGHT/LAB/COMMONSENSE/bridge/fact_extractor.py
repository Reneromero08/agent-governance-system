"""Fact-extraction bridge: unstructured model output -> structured fact-sets.

Method 2 (regex + keyword pipeline) is the primary, zero-dependency path.
Method 1 (prompt-based) wraps an LLM callable for higher accuracy.
Method 3 (tiny classifier) is stubbed for future upgrade.

Input: raw text string
Output: list of fact strings compatible with resolver.resolve(fact_set)
"""

from __future__ import annotations

import re
from typing import List, Optional, Callable, Dict


# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

CANON_REF_RE = re.compile(r"@(?:CANON|C:)[/\w.-]+", re.IGNORECASE)
DEFAULT_KEYWORDS = re.compile(
    r"\b(normally|usually|typically|in general|tends? to|by default)\b",
    re.IGNORECASE,
)
EXCEPTION_KEYWORDS = re.compile(
    r"\b(unless|except|however|but not|except for|other than)\b",
    re.IGNORECASE,
)
INVARIANT_KEYWORDS = re.compile(
    r"\b(must\b(?!\s+have)|must\s+not|always|never|invariant|required|mandatory|shall\s+not)\b",
    re.IGNORECASE,
)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


# ---------------------------------------------------------------------------
# Method 2: Regex + Keyword Pipeline (zero-dependency)
# ---------------------------------------------------------------------------

def extract_facts_regex(text: str) -> List[str]:
    """Extract fact strings from raw text using regex and keyword heuristics.

    Prefixes:
        ref:       canon references (@CANON/..., @C:...)
        default:   sentences containing default language (normally, usually)
        exception: sentences containing exception language (unless, except)
        invariant: sentences containing hard-constraint language (must, never)
        fact:      all other declarative sentences
    """
    facts: List[str] = []
    text = text.strip()
    if not text:
        return facts

    # 1. Canon references -- scan whole text
    for m in CANON_REF_RE.finditer(text):
        facts.append(f"ref:{m.group(0).lstrip('@')}")

    # 2. Sentence-level classification
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sentences) == 0:
        sentences = [text]

    for idx, sent in enumerate(sentences):
        sent_clean = re.sub(r"\s+", " ", sent).strip()
        if len(sent_clean) < 3:
            continue

        has_default = bool(DEFAULT_KEYWORDS.search(sent_clean))
        has_exception = bool(EXCEPTION_KEYWORDS.search(sent_clean))
        has_invariant = bool(INVARIANT_KEYWORDS.search(sent_clean))

        # Prefer invariant over default over fact
        if has_invariant and has_exception:
            facts.append(f"invariant:{_slugify(sent_clean)}")
            facts.append(f"exception:invariant_{_slugify(sent_clean)}")
        elif has_invariant:
            facts.append(f"invariant:{_slugify(sent_clean)}")
        elif has_default and has_exception:
            base = _slugify(re.sub(EXCEPTION_KEYWORDS, "", sent_clean, count=1).strip())
            facts.append(f"default:{base}")
            facts.append(f"exception:{base}")
        elif has_default:
            facts.append(f"default:{_slugify(sent_clean)}")
        elif has_exception:
            facts.append(f"exception:{_slugify(sent_clean)}")
        else:
            facts.append(f"fact:{_slugify(sent_clean)}")

    # 3. Deduplicate preserving order
    seen = set()
    deduped: List[str] = []
    for f in facts:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ---------------------------------------------------------------------------
# Method 1: Prompt-Based Extraction (requires an LLM callable)
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """Extract all factual claims, canon references, defaults, exceptions, and invariants from the following text. Return ONLY a JSON list of strings with these prefixes:

- "fact:<short_slug>" for declarative factual claims
- "ref:<path>" for canon references (@CANON/..., @C:...)
- "default:<short_slug>" for default/normally/generally statements
- "exception:<short_slug>" for unless/except/but-not exceptions
- "invariant:<short_slug>" for must/never/always/invariant statements

Rules:
- Slugs must be lowercase, underscore_separated, max 10 words
- One entry per distinct claim
- No duplicates
- No markdown formatting, just the JSON list

Text:
{text}

JSON list:"""


def extract_facts_prompt(text: str, llm_callable: Callable[[str], str]) -> List[str]:
    """Extract facts using an LLM prompted with structured extraction instructions.

    Args:
        text: Raw model output text.
        llm_callable: Function that takes a prompt string and returns a response string.
    """
    import json

    prompt = EXTRACTION_PROMPT.format(text=text)
    response = llm_callable(prompt)

    # Try to parse JSON from the response
    try:
        # Find JSON array in response (may have surrounding text)
        start = response.find("[")
        end = response.rfind("]")
        if start != -1 and end != -1 and end > start:
            facts = json.loads(response[start:end + 1])
            if isinstance(facts, list):
                return [str(f) for f in facts if isinstance(f, str)]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: return empty
    return []


# ---------------------------------------------------------------------------
# Default extractor (Method 2)
# ---------------------------------------------------------------------------

def extract_facts(text: str, method: str = "regex") -> List[str]:
    """Primary entry point. Uses regex extraction by default."""
    if method == "regex":
        return extract_facts_regex(text)
    raise ValueError(f"Unknown extraction method: {method}. Use 'regex' or call extract_facts_prompt directly.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slugify(text: str, max_words: int = 10) -> str:
    """Convert a sentence to a compact lowercase slug."""
    # Remove non-alpha except spaces
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Collapse whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip().lower()
    # Take first max_words words
    words = cleaned.split()[:max_words]
    slug = "_".join(words)
    # Cap length
    return slug[:120]


# ---------------------------------------------------------------------------
# Method 3 stub (tiny classifier -- future upgrade)
# ---------------------------------------------------------------------------

def extract_facts_classifier(text: str, model_path: Optional[str] = None) -> List[str]:
    """Stub for future DistilBERT-based fact classifier.

    Currently falls back to regex extraction.
    """
    return extract_facts_regex(text)
