"""
LLM Packer - Modular implementation (Phase 1).

Output structure (canonical):
- FULL/     : Single-file combined outputs
- SPLIT/    : Chunked section files
- LITE/     : Compressed high-signal outputs
- Internal Archive (inside pack): `<pack>/archive/pack.zip` + scope-prefixed `.txt` siblings
- External Archive (outside pack): `MEMORY/LLM_PACKER/_packs/_archive/<pack_name>.zip`

FORBIDDEN: COMBINED/, FULL_COMBINED/, SPLIT_LITE/
"""

from .core import (
    PackScope,
    SCOPE_AGS,
    SCOPE_LAB,
    SCOPES,
    hash_file,
    read_text,
    estimate_tokens,
    build_state_manifest,
    manifest_digest,
    verify_manifest,
    baseline_path_for_scope,
    load_baseline,
    write_json,
    make_pack,
    PROJECT_ROOT,
)

from .cli import main

__all__ = [
    "PackScope",
    "SCOPE_AGS",
    "SCOPE_LAB",
    "SCOPES",
    "hash_file",
    "read_text",
    "estimate_tokens",
    "build_state_manifest",
    "manifest_digest",
    "verify_manifest",
    "baseline_path_for_scope",
    "load_baseline",
    "write_json",
    "make_pack",
    "PROJECT_ROOT",
    "main",
]
