from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .blocks import Block, split_blocks, block_fingerprint

@dataclass(frozen=True)
class MergePlan:
    base: str  # "a" or "b"
    strategy: str
    # operations: deterministic, hash-addressable via fingerprints
    append_unique_block_fingerprints: List[str]

def plan_append_unique_blocks(a_text: str, b_text: str, base: str = "a") -> MergePlan:
    a_blocks = split_blocks(a_text)
    b_blocks = split_blocks(b_text)

    a_fps = [block_fingerprint(bl) for bl in a_blocks]
    b_fps = [block_fingerprint(bl) for bl in b_blocks]

    a_set = set(a_fps)
    b_set = set(b_fps)

    if base == "a":
        extras = [fp for fp in b_fps if fp not in a_set and fp != "BLANK"]
    else:
        extras = [fp for fp in a_fps if fp not in b_set and fp != "BLANK"]

    # de-dupe while preserving order
    seen = set()
    ordered_extras: List[str] = []
    for fp in extras:
        if fp in seen:
            continue
        seen.add(fp)
        ordered_extras.append(fp)

    return MergePlan(base=base, strategy="append_unique_blocks", append_unique_block_fingerprints=ordered_extras)

def apply_append_unique_blocks(a_text: str, b_text: str, plan: MergePlan) -> str:
    a_blocks = split_blocks(a_text)
    b_blocks = split_blocks(b_text)
    a_fp_to_block: Dict[str, Block] = {}
    b_fp_to_block: Dict[str, Block] = {}

    for bl in a_blocks:
        fp = block_fingerprint(bl)
        # keep first occurrence for determinism
        a_fp_to_block.setdefault(fp, bl)
    for bl in b_blocks:
        fp = block_fingerprint(bl)
        b_fp_to_block.setdefault(fp, bl)

    if plan.base == "a":
        merged = list(a_blocks)
        src = b_fp_to_block
    else:
        merged = list(b_blocks)
        src = a_fp_to_block

    # ensure we end with a single blank line before appends
    if merged and merged[-1].text.strip() != "":
        merged.append(Block(kind="blank", text="\n", anchor=None))

    for fp in plan.append_unique_block_fingerprints:
        bl = src.get(fp)
        if bl is None:
            # missing fingerprint, skip deterministically
            continue
        merged.append(bl)
        # keep blocks separated
        if not (merged and merged[-1].text.endswith("\n")):
            merged.append(Block(kind="blank", text="\n", anchor=None))
        else:
            merged.append(Block(kind="blank", text="\n", anchor=None))

    # collapse excessive trailing blanks deterministically
    while len(merged) >= 2 and merged[-1].text.strip() == "" and merged[-2].text.strip() == "":
        merged.pop()

    return "".join(bl.text for bl in merged)
