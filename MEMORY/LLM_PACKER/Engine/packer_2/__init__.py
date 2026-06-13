"""
Packer 2 -- Standalone CAT_CAS codebase cleaner.

Non-destructive copy of THOUGHT/LAB/CAT_CAS stripped to essential source files
(.py .md .rs .toml). Outputs a clean folder + zip to MEMORY/LLM_PACKER/_packs/.

Usage:
  python -m MEMORY.LLM_PACKER.packer_2
  python -m MEMORY.LLM_PACKER.packer_2 --source THOUGHT/LAB/CAT_CAS --no-zip
"""

from .cleaner import main

__all__ = ["main"]
