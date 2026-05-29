"""
Catalytic Chat Package

Roadmap Phase: Phase 1 — Substrate + deterministic indexing
"""

from .section_extractor import SectionExtractor, Section, extract_sections
from .section_indexer import SectionIndexer, build_index
from .symbol_registry import Symbol, SymbolRegistry, add_symbol
from .embedding_engine import ChatEmbeddingEngine, get_embedding_engine

__all__ = [
    "SectionExtractor",
    "Section",
    "extract_sections",
    "SectionIndexer",
    "build_index",
    "Symbol",
    "SymbolRegistry",
    "add_symbol",
    "ChatEmbeddingEngine",
    "get_embedding_engine",
]

__version__ = "0.1.0"


