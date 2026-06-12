"""Central .holo paths — single source of truth."""
from pathlib import Path

REPO = next(p for p in Path(__file__).resolve().parents if (p / ".git").exists())
HOLO_MODELS = REPO / "THOUGHT/LAB/HOLO/_models"

# Catalytic input (from distiller)
CATALYTIC_27B = REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_27b_catalytic_k256.holo"
CATALYTIC_05B = REPO / "THOUGHT/LAB/EIGEN_BUDDY/cybernetic_truth/qwen_0_5b_k128.holo"

# Cavity sieved
CAVITATED_27B = HOLO_MODELS / "qwen_27b_cavitated.holo"

# Wormhole modules (cavity-compressed)
LLM_WORMHOLE = HOLO_MODELS / "qwen_27b_llm_cavity_wormhole.holo"
VISUAL_WORMHOLE = HOLO_MODELS / "qwen_27b_visual_cavity_wormhole.holo"
AUX_WORMHOLE = HOLO_MODELS / "qwen_27b_aux_cavity_wormhole.holo"

# Legacy (pre-cavity)
LLM_WORMHOLE_LEGACY = HOLO_MODELS / "qwen_27b_llm_wormhole.holo"
VISUAL_WORMHOLE_LEGACY = HOLO_MODELS / "qwen_27b_visual_wormhole.holo"

# Tuned
LLM_TUNED = HOLO_MODELS / "qwen_27b_llm_tuned.holo"

# Manifests
CATALYTIC_MANIFEST = Path(__file__).resolve().parent / "catalytic_manifest.json"

# 0.5B
WORMHOLE_05B = HOLO_MODELS / "qwen_0_5b_wormhole.holo"

# All modules dict for loaders
MODULE_PATHS = {
    "llm":    LLM_WORMHOLE,
    "visual": VISUAL_WORMHOLE,
    "aux":    AUX_WORMHOLE,
}
