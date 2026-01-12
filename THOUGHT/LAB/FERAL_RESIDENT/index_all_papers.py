#!/usr/bin/env python3
"""
Bulk index all papers from paper_pipeline.py into the geometric memory.

Reads the PAPERS list from paper_pipeline.py and indexes each one.
"""

import sys
from pathlib import Path

FERAL_DIR = Path(__file__).parent
sys.path.insert(0, str(FERAL_DIR))

from paper_pipeline import PAPERS
from paper_indexer import PaperIndexer

PAPERS_DIR = FERAL_DIR / "research" / "papers"
MD_DIR = PAPERS_DIR / "markdown"


def main():
    print("=" * 60)
    print("BULK PAPER INDEXER - B.1 Paper Flooding")
    print("=" * 60)
    print(f"Total papers: {len(PAPERS)}")
    print()

    indexer = PaperIndexer()

    success = 0
    failed = []

    for i, paper in enumerate(PAPERS, 1):
        paper_id = paper["id"]
        name = paper["name"]
        category = paper["category"]

        print(f"[{i}/{len(PAPERS)}] {paper_id} - {name}")

        # Determine markdown path
        if paper["type"] == "markdown":
            md_path = FERAL_DIR / paper["source"]
        else:
            md_path = MD_DIR / f"{paper_id}.md"

        if not md_path.exists():
            print(f"  [SKIP] Markdown not found: {md_path}")
            failed.append({"id": paper_id, "reason": "markdown_missing"})
            continue

        try:
            # Register paper (if not already registered)
            try:
                indexer.register_paper(paper_id, name, name, category)
            except ValueError:
                pass  # Already registered

            # Set markdown path
            indexer.set_markdown_path(paper_id, str(md_path))

            # Index
            result = indexer.index_paper(paper_id)
            chunks = len(result.get("chunks", []))
            print(f"  [OK] Indexed {chunks} chunks")
            success += 1

        except Exception as e:
            print(f"  [ERROR] {e}")
            failed.append({"id": paper_id, "reason": str(e)})

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Successfully indexed: {success}/{len(PAPERS)}")
    print(f"Failed: {len(failed)}")

    if failed:
        print("\nFailed papers:")
        for f in failed:
            print(f"  - {f['id']}: {f['reason']}")

    # Print final stats
    print()
    print("Final stats:")
    stats = indexer.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
