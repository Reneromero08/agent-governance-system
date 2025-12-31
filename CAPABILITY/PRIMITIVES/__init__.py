"""
CATALYTIC-DPT Primitives

Smallest possible catalytic computing kernel.
"""

from .cas_store import CatalyticStore, normalize_relpath
from .hash_toolbelt import hash_ast, hash_describe, hash_grep, hash_read_text
from .ledger import Ledger
from .merkle import build_manifest_root, verify_manifest_root
from .restore_proof import RestorationProofValidator, canonical_json_bytes
from .restore_runner import restore_bundle, restore_chain, RESTORE_CODES
from .skills import SkillRegistry, SkillNotFoundError, canonical_json, resolve_adapter, RegistryError, CapabilityHashMismatch
from .verify_bundle import verify_bundle
from .scratch import CatalyticScratch

# Export modules too for flexibility
from . import cas_store
from . import hash_toolbelt
from . import ledger
from . import merkle
from . import restore_proof
from . import restore_runner
from . import skills
from . import verify_bundle as verify_bundle_mod
from . import fs_guard
from . import scratch

__all__ = [
    "CatalyticStore",
    "normalize_relpath",
    "hash_ast",
    "hash_describe",
    "hash_grep",
    "hash_read_text",
    "Ledger",
    "build_manifest_root",
    "verify_manifest_root",
    "RestorationProofValidator",
    "canonical_json_bytes",
    "restore_bundle",
    "restore_chain",
    "RESTORE_CODES",
    "SkillRegistry",
    "SkillNotFoundError",
    "canonical_json",
    "resolve_adapter",
    "RegistryError",
    "CapabilityHashMismatch",
    "verify_bundle",
    "cas_store",
    "hash_toolbelt",
    "ledger",
    "merkle",
    "restore_proof",
    "restore_runner",
    "skills",
    "fs_guard",
    "scratch",
    "CatalyticScratch"
]
