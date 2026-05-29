# emergence/ - Pattern & protocol formation (System 1)
# Unconscious pattern recognition and language development

from .emergence import (
    load_thread_history,
    count_symbol_refs,
    count_vector_hashes,
    measure_compression,
    detect_new_patterns,
    compute_E_histogram,
    track_Df_over_time,
    compute_mind_distance_from_start,
    count_own_vector_refs,
    extract_composition_graph,
    compute_communication_mode_distribution,
    compute_canonical_reuse_rate,
    store_emergence_receipt,
    detect_protocols,
    print_emergence_report,
)
from .symbol_evolution import (
    SymbolEvolutionTracker,
    PointerRatioTracker,
    ECompressionTracker,
    NotationRegistry,
    CommunicationModeTimeline,
    EvolutionReceiptStore,
)
from .symbolic_compiler import (
    SymbolicCompiler,
    CompressionLevel,
    RenderResult,
    RoundTripVerification,
    SymbolEntry,
    HybridSymbolRegistry,
)

__all__ = [
    # emergence functions
    "load_thread_history",
    "count_symbol_refs",
    "count_vector_hashes",
    "measure_compression",
    "detect_new_patterns",
    "compute_E_histogram",
    "track_Df_over_time",
    "compute_mind_distance_from_start",
    "count_own_vector_refs",
    "extract_composition_graph",
    "compute_communication_mode_distribution",
    "compute_canonical_reuse_rate",
    "store_emergence_receipt",
    "detect_protocols",
    "print_emergence_report",
    # symbol_evolution
    "SymbolEvolutionTracker",
    "PointerRatioTracker",
    "ECompressionTracker",
    "NotationRegistry",
    "CommunicationModeTimeline",
    "EvolutionReceiptStore",
    # symbolic_compiler
    "SymbolicCompiler",
    "CompressionLevel",
    "RenderResult",
    "RoundTripVerification",
    "SymbolEntry",
    "HybridSymbolRegistry",
]
