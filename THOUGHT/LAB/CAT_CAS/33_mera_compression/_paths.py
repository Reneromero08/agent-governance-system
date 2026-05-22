"""Central .holo paths — single source of truth."""
from pathlib import Path

REPO = Path(r"D:\CCC 2.0\AI\agent-governance-system")
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
CATALYTIC_MANIFEST = REPO / "THOUGHT/LAB/CAT_CAS/33_mera_compression/catalytic_manifest.json"

# 0.5B
WORMHOLE_05B = HOLO_MODELS / "qwen_0_5b_wormhole.holo"

# All modules dict for loaders
MODULE_PATHS = {
    "llm":    LLM_WORMHOLE,
    "visual": VISUAL_WORMHOLE,
    "aux":    AUX_WORMHOLE,
}
