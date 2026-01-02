from __future__ import annotations

import difflib
from typing import Dict, List, Tuple

def similarity_ratio(a_lines: List[str], b_lines: List[str]) -> float:
    return difflib.SequenceMatcher(a=a_lines, b=b_lines).ratio()

def diff_summary(a_lines: List[str], b_lines: List[str], context_lines: int = 3, max_diff_lines: int = 500) -> str:
    udiff = list(difflib.unified_diff(
        a_lines, b_lines,
        fromfile="a", tofile="b",
        n=context_lines,
        lineterm=""
    ))
    if len(udiff) > max_diff_lines:
        udiff = udiff[:max_diff_lines] + ["... (diff truncated)"]
    return "\n".join(udiff)

def unique_lines(a_lines: List[str], b_lines: List[str], max_lines: int = 200) -> Tuple[List[str], List[str]]:
    sm = difflib.SequenceMatcher(a=a_lines, b=b_lines)
    unique_a: List[str] = []
    unique_b: List[str] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("delete", "replace"):
            unique_a.extend(a_lines[i1:i2])
        if tag in ("insert", "replace"):
            unique_b.extend(b_lines[j1:j2])
    if len(unique_a) > max_lines:
        unique_a = unique_a[:max_lines] + ["... (unique_to_a truncated)\n"]
    if len(unique_b) > max_lines:
        unique_b = unique_b[:max_lines] + ["... (unique_to_b truncated)\n"]
    return unique_a, unique_b
