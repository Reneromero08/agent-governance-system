"""Symbolic handle expansion for facts.

Rules:
- If a fact starts with '@', treat it as a symbol.
  - If it matches CODEBOOK.symbols, expand to those predicates.
  - Else if CODEBOOK.ids_as_symbols is true and the symbol matches an entry ID (without @),
    expand to a canonical predicate: id:<ID> (optional hook for later).
  - Else pass through unchanged (but record as unresolved).
- If a fact equals an entry ID and CODEBOOK.ids_as_symbols is true, optionally expand to @<ID>.
  (By default we DO NOT auto-expand bare IDs to avoid ambiguity.)
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def expand_facts(facts: List[str], codebook: Dict[str, Any], db_ids: List[str] | None = None) -> Tuple[List[str], List[Dict[str, str]]]:
    symbols = (codebook or {}).get("symbols", {}) or {}
    ids_as_symbols = bool((codebook or {}).get("ids_as_symbols", False))

    unresolved: List[Dict[str, str]] = []
    out: List[str] = []

    idset = set(db_ids or [])

    for f in facts:
        if not isinstance(f, str):
            f = str(f)

        if f.startswith("@"):
            if f in symbols:
                out.extend([str(x) for x in symbols[f]])
            else:
                # optional: @ID as symbol
                if ids_as_symbols and f[1:] in idset:
                    out.append(f"id:{f[1:]}")
                else:
                    out.append(f)
                    unresolved.append({"symbol": f, "reason": "unknown_symbol"})
        else:
            out.append(f)

    # de-dupe while preserving order
    seen = set()
    deduped = []
    for x in out:
        if x not in seen:
            seen.add(x)
            deduped.append(x)

    return deduped, unresolved
