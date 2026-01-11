#!/usr/bin/env python3
"""
Structure-Aware Markdown Chunker (Phase 1.5)

Splits markdown documents on header boundaries, preserving semantic hierarchy.
Each chunk knows its header depth, text, and parent relationship.

Architecture:
    # H1 Title           → chunk boundary (depth=1)
    ## H2 Section        → chunk boundary (depth=2)
    ### H3 Subsection    → chunk boundary (depth=3)
    #### H4              → chunk boundary (depth=4)
    ##### H5             → keep with parent unless >token_limit
    body text            → accumulate until next header or size limit

Navigation:
    - Query returns chunk with header_depth=3
    - Want broader context? Follow parent_chunk_id → depth=2
    - Want deeper? Query children where parent_chunk_id = this.chunk_id
    - "Go to next #" = find sibling at same depth
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# Configuration
MAX_TOKENS = 500  # Soft limit - will split body text if exceeded
MIN_CHUNK_TOKENS = 50  # Don't create tiny chunks


@dataclass
class MarkdownChunk:
    """A semantically-bounded chunk of markdown content."""

    content: str                          # Full chunk text (header + body)
    chunk_hash: str                       # SHA-256 of content
    token_count: int                      # Approximate token count

    # Hierarchy fields
    header_depth: Optional[int] = None    # 1-6 for headers, None for body-only
    header_text: Optional[str] = None     # "## Section Name" or None
    header_title: Optional[str] = None    # "Section Name" (no # prefix)

    # Position tracking
    start_line: int = 0                   # 1-indexed line number in source
    end_line: int = 0                     # 1-indexed line number in source
    chunk_index: int = 0                  # Sequential index in file

    # Parent relationship (set during tree building)
    parent_index: Optional[int] = None    # Index of parent chunk

    # Children tracking (computed)
    child_indices: List[int] = field(default_factory=list)


def count_tokens(text: str) -> int:
    """Approximate token count (words / 0.75 ≈ tokens)."""
    return int(len(text.split()) / 0.75)


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of text."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def parse_header(line: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse a markdown header line.

    Returns:
        (depth, title) or (None, None) if not a header
    """
    match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
    if match:
        depth = len(match.group(1))
        title = match.group(2).strip()
        return depth, title
    return None, None


def chunk_markdown(text: str, max_tokens: int = MAX_TOKENS) -> List[MarkdownChunk]:
    """Split markdown text into structure-aware chunks.

    Algorithm:
    1. Split on header boundaries (# ## ### etc.)
    2. Each header starts a new chunk
    3. Body text accumulates until next header or size limit
    4. Large body sections split at paragraph boundaries
    5. Parent-child relationships computed from header depth

    Args:
        text: Markdown text to chunk
        max_tokens: Soft limit for chunk size (default 500)

    Returns:
        List of MarkdownChunk with hierarchy information
    """
    lines = text.split('\n')
    chunks: List[MarkdownChunk] = []

    current_content: List[str] = []
    current_header_depth: Optional[int] = None
    current_header_text: Optional[str] = None
    current_header_title: Optional[str] = None
    current_start_line = 1

    def flush_chunk():
        """Create chunk from accumulated content."""
        nonlocal current_content, current_start_line
        nonlocal current_header_depth, current_header_text, current_header_title

        if not current_content:
            return

        content = '\n'.join(current_content)
        tokens = count_tokens(content)

        # If chunk is too large, split at paragraph boundaries
        if tokens > max_tokens and current_header_depth is None:
            # Body-only chunk that's too large - split it
            sub_chunks = split_large_body(
                current_content,
                current_start_line,
                max_tokens
            )
            for sub in sub_chunks:
                sub.chunk_index = len(chunks)
                chunks.append(sub)
        else:
            # Normal chunk or header-bounded chunk
            chunk = MarkdownChunk(
                content=content,
                chunk_hash=compute_hash(content),
                token_count=tokens,
                header_depth=current_header_depth,
                header_text=current_header_text,
                header_title=current_header_title,
                start_line=current_start_line,
                end_line=current_start_line + len(current_content) - 1,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)

        # Reset for next chunk
        current_content = []
        current_header_depth = None
        current_header_text = None
        current_header_title = None

    # Process lines
    for i, line in enumerate(lines, start=1):
        depth, title = parse_header(line)

        if depth is not None:
            # Found a header - flush previous chunk
            flush_chunk()

            # Start new chunk with this header
            current_start_line = i
            current_header_depth = depth
            current_header_text = line.strip()
            current_header_title = title
            current_content = [line]
        else:
            # Body line
            if not current_content:
                current_start_line = i
            current_content.append(line)

            # Check if body is getting too large (will be split in flush_chunk)
            # We don't force-flush here because flush_chunk handles large bodies
            # via split_large_body() which respects paragraph boundaries

    # Flush final chunk
    flush_chunk()

    # Build parent-child relationships
    build_hierarchy(chunks)

    return chunks


def split_large_body(lines: List[str], start_line: int, max_tokens: int) -> List[MarkdownChunk]:
    """Split a large body section at paragraph boundaries.

    Args:
        lines: Lines of text (no headers)
        start_line: Starting line number
        max_tokens: Target chunk size

    Returns:
        List of body-only chunks
    """
    chunks = []
    current_para: List[str] = []
    current_start = start_line

    for i, line in enumerate(lines):
        # Paragraph break = empty line
        if not line.strip() and current_para:
            content = '\n'.join(current_para)
            if count_tokens(content) >= MIN_CHUNK_TOKENS:
                chunks.append(MarkdownChunk(
                    content=content,
                    chunk_hash=compute_hash(content),
                    token_count=count_tokens(content),
                    header_depth=None,
                    header_text=None,
                    header_title=None,
                    start_line=current_start,
                    end_line=current_start + len(current_para) - 1,
                    chunk_index=0  # Will be set by caller
                ))
                current_para = []
                current_start = start_line + i + 1
            else:
                current_para.append(line)
        else:
            current_para.append(line)

            # Force split if way over limit
            content = '\n'.join(current_para)
            if count_tokens(content) > max_tokens * 1.5:
                chunks.append(MarkdownChunk(
                    content=content,
                    chunk_hash=compute_hash(content),
                    token_count=count_tokens(content),
                    header_depth=None,
                    header_text=None,
                    header_title=None,
                    start_line=current_start,
                    end_line=current_start + len(current_para) - 1,
                    chunk_index=0
                ))
                current_para = []
                current_start = start_line + i + 1

    # Flush remaining
    if current_para:
        content = '\n'.join(current_para)
        chunks.append(MarkdownChunk(
            content=content,
            chunk_hash=compute_hash(content),
            token_count=count_tokens(content),
            header_depth=None,
            header_text=None,
            header_title=None,
            start_line=current_start,
            end_line=current_start + len(current_para) - 1,
            chunk_index=0
        ))

    return chunks if chunks else [MarkdownChunk(
        content='\n'.join(lines),
        chunk_hash=compute_hash('\n'.join(lines)),
        token_count=count_tokens('\n'.join(lines)),
        start_line=start_line,
        end_line=start_line + len(lines) - 1,
        chunk_index=0
    )]


def build_hierarchy(chunks: List[MarkdownChunk]) -> None:
    """Build parent-child relationships based on header depth.

    Algorithm:
    - Track "open" headers at each depth level
    - When a new header appears, it becomes a child of the nearest open header
      with lower depth (higher in hierarchy)
    - Body-only chunks are children of the most recent header

    Modifies chunks in place.
    """
    # Stack of (depth, index) for open headers at each level
    header_stack: List[Tuple[int, int]] = []
    last_header_index: Optional[int] = None

    for i, chunk in enumerate(chunks):
        if chunk.header_depth is not None:
            # This is a header chunk
            depth = chunk.header_depth

            # Pop headers at same or deeper level (they're now closed)
            while header_stack and header_stack[-1][0] >= depth:
                header_stack.pop()

            # Parent is the top of stack (if any)
            if header_stack:
                parent_depth, parent_idx = header_stack[-1]
                chunk.parent_index = parent_idx
                chunks[parent_idx].child_indices.append(i)

            # Push this header onto stack
            header_stack.append((depth, i))
            last_header_index = i

        else:
            # Body-only chunk - child of most recent header
            if last_header_index is not None:
                chunk.parent_index = last_header_index
                chunks[last_header_index].child_indices.append(i)


def get_chunk_path(chunks: List[MarkdownChunk], index: int) -> List[str]:
    """Get the hierarchical path to a chunk (for breadcrumb navigation).

    Returns list like: ["# Document", "## Section", "### Subsection"]
    """
    path = []
    current = chunks[index]

    while current is not None:
        if current.header_text:
            path.insert(0, current.header_text)
        if current.parent_index is not None:
            current = chunks[current.parent_index]
        else:
            break

    return path


def get_siblings(chunks: List[MarkdownChunk], index: int) -> Tuple[Optional[int], Optional[int]]:
    """Get previous and next sibling indices at same header depth.

    Returns:
        (prev_sibling_index, next_sibling_index) - None if no sibling
    """
    chunk = chunks[index]
    if chunk.header_depth is None:
        return None, None

    depth = chunk.header_depth
    parent_idx = chunk.parent_index

    prev_sibling = None
    next_sibling = None

    for i, c in enumerate(chunks):
        if c.header_depth == depth and c.parent_index == parent_idx:
            if i < index:
                prev_sibling = i
            elif i > index and next_sibling is None:
                next_sibling = i
                break

    return prev_sibling, next_sibling


# ============================================================================
# Demo / Test
# ============================================================================

def demo():
    """Demo the markdown chunker."""
    sample = """# Document Title

This is the introduction paragraph with some context.

## Section One

Content for section one goes here. This explains the first topic.

### Subsection 1.1

Deep dive into subtopic 1.1 with detailed information.

### Subsection 1.2

Another subtopic with its own content.

## Section Two

Second major section of the document.

### Subsection 2.1

Details for section two's first subtopic.

#### Sub-subsection 2.1.1

Even deeper nesting with specific details.

## Section Three

Final section wrapping up the document.
"""

    print("=== Structure-Aware Markdown Chunker Demo ===\n")

    chunks = chunk_markdown(sample)

    print(f"Created {len(chunks)} chunks:\n")

    for i, chunk in enumerate(chunks):
        depth_str = f"depth={chunk.header_depth}" if chunk.header_depth else "body"
        parent_str = f"parent={chunk.parent_index}" if chunk.parent_index is not None else "root"
        children_str = f"children={chunk.child_indices}" if chunk.child_indices else ""

        print(f"[{i}] {depth_str}, {parent_str} {children_str}")
        if chunk.header_text:
            print(f"    Header: {chunk.header_text}")
        print(f"    Lines: {chunk.start_line}-{chunk.end_line}, ~{chunk.token_count} tokens")
        print(f"    Hash: {chunk.chunk_hash[:16]}...")
        print()

    # Demo navigation
    print("=== Navigation Demo ===\n")

    # Find a subsection
    for i, chunk in enumerate(chunks):
        if chunk.header_depth == 3:
            print(f"Chunk [{i}]: {chunk.header_text}")
            print(f"  Path: {' > '.join(get_chunk_path(chunks, i))}")
            prev_sib, next_sib = get_siblings(chunks, i)
            if prev_sib is not None:
                print(f"  Prev sibling: [{prev_sib}] {chunks[prev_sib].header_text}")
            if next_sib is not None:
                print(f"  Next sibling: [{next_sib}] {chunks[next_sib].header_text}")
            if chunk.parent_index is not None:
                print(f"  Parent: [{chunk.parent_index}] {chunks[chunk.parent_index].header_text}")
            print()


if __name__ == "__main__":
    demo()
