from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple

_HEADING_RE = re.compile(r"^(#{1,6})\s+.+$")

@dataclass(frozen=True)
class Block:
    kind: str               # "heading" | "paragraph" | "blank"
    text: str               # raw text
    anchor: str | None      # heading line when kind=="heading"

def split_blocks(text: str) -> List[Block]:
    # Deterministic block splitter tuned for markdown:
    # - Heading lines become anchors.
    # - Paragraphs are runs separated by blank lines.
    lines = text.splitlines(True)
    blocks: List[Block] = []
    buf: List[str] = []
    buf_kind: str | None = None

    def flush():
        nonlocal buf, buf_kind
        if not buf:
            return
        t = "".join(buf)
        if buf_kind == "heading":
            anchor = t.splitlines()[0].strip()
            blocks.append(Block(kind="heading", text=t, anchor=anchor))
        else:
            blocks.append(Block(kind="paragraph", text=t, anchor=None))
        buf = []
        buf_kind = None

    for line in lines:
        if _HEADING_RE.match(line.rstrip("\n")):
            flush()
            blocks.append(Block(kind="heading", text=line, anchor=line.rstrip("\n")))
            continue

        if line.strip() == "":
            # blank line splits paragraphs
            flush()
            blocks.append(Block(kind="blank", text=line, anchor=None))
            continue

        if buf_kind is None:
            buf_kind = "paragraph"
        buf.append(line)

    flush()
    return blocks

def blocks_to_text(blocks: List[Block]) -> str:
    return "".join(b.text for b in blocks)

def block_fingerprint(block: Block) -> str:
    # Stable fingerprint ignores block kind for matching purposes except blank lines.
    # Keep blanks as blanks so we can preserve readability in appends.
    if block.kind == "blank":
        return "BLANK"
    # Normalize internal whitespace minimally to avoid accidental drift.
    t = block.text.replace("\r\n", "\n").replace("\r", "\n")
    return t
