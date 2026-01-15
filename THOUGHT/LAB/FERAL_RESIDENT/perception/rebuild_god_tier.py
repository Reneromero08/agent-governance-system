#!/usr/bin/env python3
"""
COMPLETE FORMAT & REBUILD - GOD TIER Papers

!!! NUCLEAR OPTION !!!
- Deletes feral_eternal.db COMPLETELY
- Clears ALL vectors, interactions, threads, receipts
- Starts from ZERO
- Only GOD TIER papers remain

Usage:
    python rebuild_god_tier.py
"""

import json
import sys
import re
from pathlib import Path
from datetime import datetime

# Setup paths
PERCEPTION_DIR = Path(__file__).parent
FERAL_DIR = PERCEPTION_DIR.parent
RESEARCH_DIR = PERCEPTION_DIR / "research"
PAPERS_DIR = RESEARCH_DIR / "papers"
GOD_TIER_DIR = PAPERS_DIR / "god_tier"
MANIFEST_PATH = PAPERS_DIR / "manifest.json"
DB_DIR = FERAL_DIR / "data" / "db"
DB_PATH = DB_DIR / "feral_eternal.db"

# Add paths for imports
REPO_ROOT = FERAL_DIR.parent.parent.parent  # agent-governance-system/
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"
sys.path.insert(0, str(CAPABILITY_PATH))  # For GeometricReasoner
sys.path.insert(0, str(FERAL_DIR))
sys.path.insert(0, str(PERCEPTION_DIR))


def extract_arxiv_id(filename: str) -> str:
    """Extract arxiv ID from filename like 2310.06816.md"""
    return filename.replace('.md', '')


def extract_short_name(arxiv_id: str, content: str) -> str:
    """Try to extract a short name from the paper content."""
    # Look for title in first few lines
    lines = content.split('\n')[:20]
    for line in lines:
        if line.startswith('# ') and len(line) > 3:
            title = line[2:].strip()
            # Extract first significant word(s) as short name
            words = re.sub(r'[^a-zA-Z0-9\s]', '', title).split()
            if words:
                return words[0][:20]
    return arxiv_id


def guess_category(content: str, filename: str) -> str:
    """Guess paper category from content keywords."""
    content_lower = content.lower()

    if 'hyperdimensional' in content_lower or 'vsa' in content_lower:
        return 'hdc_vsa'
    elif 'vec2text' in content_lower or 'inversion' in content_lower:
        return 'vec2text'
    elif 'latent' in content_lower and 'reasoning' in content_lower:
        return 'latent_reasoning'
    elif 'compression' in content_lower or 'compress' in content_lower:
        return 'compression'
    elif 'platonic' in content_lower or 'representation' in content_lower:
        return 'representation'
    elif 'sentence' in content_lower and 'embed' in content_lower:
        return 'sentence_embed'
    elif 'memory' in content_lower and 'transformer' in content_lower:
        return 'memory'
    elif 'retrieval' in content_lower and 'dense' in content_lower:
        return 'dense_retrieval'
    elif 'splade' in content_lower or 'sparse' in content_lower:
        return 'sparse_retrieval'
    elif 'clip' in content_lower or 'multimodal' in content_lower:
        return 'multimodal'
    elif 'embed' in content_lower:
        return 'text_embed'
    elif 'chain of thought' in content_lower or 'cot' in content_lower:
        return 'chain_of_thought'
    elif 'vector database' in content_lower or 'hnsw' in content_lower:
        return 'vector_db'
    elif 'rag' in content_lower or 'retrieval augmented' in content_lower:
        return 'rag'
    elif 'contrastive' in content_lower:
        return 'contrastive'
    elif 'attention' in content_lower:
        return 'attention'
    else:
        return 'research'


def step1_nuke_everything():
    """NUKE the database - delete EVERYTHING, start from ZERO."""
    print("\n" + "=" * 60)
    print("STEP 1: NUKE DATABASE - FORMAT FROM ZERO")
    print("=" * 60)

    # Delete database file
    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / (1024 * 1024)
        DB_PATH.unlink()
        print(f"  DELETED: {DB_PATH} ({size_mb:.1f} MB)")
        print(f"    - ALL vectors: GONE")
        print(f"    - ALL interactions: GONE")
        print(f"    - ALL threads: GONE")
        print(f"    - ALL receipts: GONE")
        print(f"    - ALL memories: GONE")
    else:
        print(f"  No existing database found")

    # Also clear any other db files that might exist
    for db_file in DB_DIR.glob("*.db"):
        if db_file != DB_PATH:
            db_file.unlink()
            print(f"  DELETED: {db_file.name}")

    # Ensure db directory exists
    DB_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  Database directory ready: {DB_DIR}")
    print(f"  Starting from ZERO")


def step2_reset_manifest():
    """Reset manifest.json for GOD TIER papers."""
    print("\n" + "=" * 60)
    print("STEP 2: Reset Manifest for GOD TIER")
    print("=" * 60)

    # Create fresh manifest
    manifest = {
        "version": "2.0.0",
        "created": datetime.utcnow().isoformat(),
        "source": "god_tier",
        "papers": {},
        "aliases": {},
        "categories": {},
        "stats": {"total": 0, "indexed": 0}
    }

    # Scan god_tier directory
    god_tier_papers = list(GOD_TIER_DIR.glob("*.md"))
    print(f"  Found {len(god_tier_papers)} GOD TIER papers")

    for paper_path in sorted(god_tier_papers):
        arxiv_id = extract_arxiv_id(paper_path.name)

        # Read content for metadata extraction
        try:
            content = paper_path.read_text(encoding='utf-8', errors='replace')
        except:
            content = ""

        short_name = extract_short_name(arxiv_id, content)
        category = guess_category(content, paper_path.name)

        # Register paper with god_tier path
        manifest["papers"][arxiv_id] = {
            "arxiv_id": arxiv_id,
            "short_name": short_name,
            "title": short_name,
            "category": category,
            "primary_symbol": f"@Paper-{arxiv_id}",
            "alias_symbol": f"@Paper-{short_name}",
            "pdf_path": None,
            "markdown_path": f"god_tier/{paper_path.name}",
            "status": "converted",
            "chunks": [],
            "Df_values": [],
            "created_at": datetime.utcnow().isoformat(),
            "indexed_at": None
        }

        manifest["aliases"][short_name] = arxiv_id
        manifest["stats"]["total"] += 1

        # Track category
        if category not in manifest["categories"]:
            manifest["categories"][category] = {"count": 0, "papers": []}
        manifest["categories"][category]["count"] += 1
        manifest["categories"][category]["papers"].append(arxiv_id)

    # Save manifest
    MANIFEST_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding='utf-8'
    )
    print(f"  Manifest saved: {len(manifest['papers'])} papers registered")
    print(f"  Categories: {list(manifest['categories'].keys())}")

    return manifest


def step3_index_papers():
    """Index all GOD TIER papers with heading structure."""
    print("\n" + "=" * 60)
    print("STEP 3: Index GOD TIER Papers")
    print("=" * 60)

    from paper_indexer import PaperIndexer

    indexer = PaperIndexer(str(PAPERS_DIR))

    # Get list of papers to index
    papers = indexer.list_papers(status='converted')
    print(f"  Papers to index: {len(papers)}")

    success = 0
    failed = []
    total_chunks = 0

    for i, paper in enumerate(papers, 1):
        arxiv_id = paper['arxiv_id']
        print(f"  [{i}/{len(papers)}] {arxiv_id}...", end=" ", flush=True)

        try:
            result = indexer.index_paper(arxiv_id)
            chunks = len(result.get('chunks', []))
            total_chunks += chunks
            print(f"OK ({chunks} chunks)")
            success += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append({"id": arxiv_id, "error": str(e)})

    print(f"\n  Indexed: {success}/{len(papers)} papers")
    print(f"  Total chunks: {total_chunks}")

    if failed:
        print(f"  Failed papers:")
        for f in failed[:5]:
            print(f"    - {f['id']}: {f['error'][:50]}")

    return success, total_chunks


def step4_load_into_database():
    """Load indexed papers into vector database."""
    print("\n" + "=" * 60)
    print("STEP 4: Load Papers into Database")
    print("=" * 60)

    # Import modules with proper path setup
    MEMORY_DIR = FERAL_DIR / "memory"
    sys.path.insert(0, str(MEMORY_DIR))

    # Import the underlying modules directly to avoid relative import issues
    from geometric_reasoner import GeometricReasoner
    from resident_db import ResidentDB

    print(f"  Creating fresh database: {DB_PATH}")

    # Initialize database
    db = ResidentDB(str(DB_PATH))
    reasoner = GeometricReasoner()

    # Load papers from manifest
    manifest = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    papers = [p for p in manifest['papers'].values() if p['status'] == 'indexed']

    print(f"  Papers to load: {len(papers)}")

    chunks_loaded = 0
    papers_loaded = 0

    for paper in papers:
        arxiv_id = paper['arxiv_id']
        markdown_path = PAPERS_DIR / paper['markdown_path']

        if not markdown_path.exists():
            continue

        # Chunk by headings
        content = markdown_path.read_text(encoding='utf-8', errors='replace')
        chunks = chunk_by_headings_simple(content)

        for chunk in chunks:
            # Embed chunk
            chunk_text = f"@Paper-{arxiv_id} {chunk['heading']}\n{chunk['content'][:2000]}"
            state = reasoner.initialize(chunk_text)

            # Store vector
            vector_id = db.store_vector(
                vector=state.vector,
                Df=state.Df,
                composition_op='paper_load',
                parent_ids=None
            )

            # Store receipt with paper metadata
            import hashlib
            content_hash = hashlib.sha256(chunk['content'].encode()).hexdigest()[:16]
            db.store_receipt(
                operation='paper_load',
                input_hashes=[content_hash],
                output_hash=state.receipt()['vector_hash'],
                metadata={
                    'paper_id': arxiv_id,
                    'heading': chunk['heading'],
                    'content': chunk['content'][:2000],
                    'alias': paper.get('alias_symbol', ''),
                    'category': paper.get('category', '')
                }
            )
            chunks_loaded += 1

        papers_loaded += 1
        if papers_loaded % 10 == 0:
            print(f"    Loaded {papers_loaded}/{len(papers)} papers...")

    db.close()

    print(f"  Papers loaded: {papers_loaded}")
    print(f"  Chunks loaded: {chunks_loaded}")

    return {'papers_loaded': papers_loaded, 'chunks_loaded': chunks_loaded}


def chunk_by_headings_simple(content: str) -> list:
    """Simple heading-based chunking."""
    chunks = []
    current = {"heading": "# Preamble", "level": 0, "content": ""}

    for line in content.split('\n'):
        if line.startswith('#') and not line.startswith('#!'):
            stripped = line.lstrip('#')
            level = len(line) - len(stripped)
            if stripped.startswith(' ') or stripped == '':
                if current["content"].strip():
                    chunks.append(current)
                current = {"heading": line.strip(), "level": level, "content": ""}
                continue
        current["content"] += line + "\n"

    if current["content"].strip():
        chunks.append(current)

    return chunks


def step5_initialize_thread():
    """Initialize the eternal thread for constellation."""
    print("\n" + "=" * 60)
    print("STEP 5: Initialize Eternal Thread")
    print("=" * 60)

    MEMORY_DIR = FERAL_DIR / "memory"
    if str(MEMORY_DIR) not in sys.path:
        sys.path.insert(0, str(MEMORY_DIR))

    from resident_db import ResidentDB

    db = ResidentDB(str(DB_PATH))

    # Create eternal thread if not exists
    thread = db.get_thread('eternal')
    if thread is None:
        thread = db.create_thread('eternal')
        print(f"  Created eternal thread")
    else:
        print(f"  Eternal thread exists")

    # Get stats
    stats = db.get_stats()
    print(f"  Database stats:")
    print(f"    Vectors: {stats['vector_count']}")
    print(f"    Receipts: {stats['receipt_count']}")
    print(f"    Threads: {stats['thread_count']}")

    db.close()
    return stats


def step6_verify():
    """Verify the rebuild was successful."""
    print("\n" + "=" * 60)
    print("STEP 6: Verification")
    print("=" * 60)

    import sqlite3

    with sqlite3.connect(str(DB_PATH)) as conn:
        # Count paper_load receipts
        cursor = conn.execute("SELECT COUNT(*) FROM receipts WHERE operation='paper_load'")
        paper_chunks = cursor.fetchone()[0]

        # Count total vectors
        cursor = conn.execute("SELECT COUNT(*) FROM vectors")
        vectors = cursor.fetchone()[0]

        # Sample headings
        cursor = conn.execute("""
            SELECT metadata FROM receipts
            WHERE operation='paper_load'
            LIMIT 5
        """)
        samples = cursor.fetchall()

    print(f"  Paper chunks in DB: {paper_chunks}")
    print(f"  Total vectors: {vectors}")

    print(f"\n  Sample paper headings:")
    for row in samples:
        try:
            meta = json.loads(row[0])
            print(f"    - {meta.get('paper_id', '?')}: {meta.get('heading', '?')[:50]}")
        except:
            pass

    # Final status
    db_size = DB_PATH.stat().st_size / (1024 * 1024)
    print(f"\n  Database size: {db_size:.1f} MB")
    print(f"  Status: {'SUCCESS' if paper_chunks > 0 else 'FAILED'}")

    return paper_chunks > 0


def main():
    print("\n" + "=" * 60)
    print("GOD TIER DATABASE REBUILD")
    print("=" * 60)
    print(f"Source: {GOD_TIER_DIR}")
    print(f"Target: {DB_PATH}")

    # Execute rebuild steps
    step1_nuke_everything()
    step2_reset_manifest()
    step3_index_papers()
    step4_load_into_database()
    step5_initialize_thread()
    success = step6_verify()

    print("\n" + "=" * 60)
    print("REBUILD COMPLETE" if success else "REBUILD FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
