# perception/ - Sensory interface: world -> manifold
# Handles all inbound boundary operations

from .paper_pipeline import (
    download_arxiv_pdf,
    pdf_to_markdown,
    process_paper,
    run_pipeline,
)
from .paper_indexer import PaperIndexer, PaperChunk, PaperRecord
from .index_all_papers import index_all_papers

__all__ = [
    # paper_pipeline
    "download_arxiv_pdf",
    "pdf_to_markdown",
    "process_paper",
    "run_pipeline",
    # paper_indexer
    "PaperIndexer",
    "PaperChunk",
    "PaperRecord",
    # index_all_papers
    "index_all_papers",
]
