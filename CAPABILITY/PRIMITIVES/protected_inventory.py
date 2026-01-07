#!/usr/bin/env python3
"""
Protected Artifacts Inventory (CRYPTO_SAFE.0)

Defines and manages the canonical inventory of protected artifacts that must be
sealed before public distribution.

Phase: 2.4.2 Protected Artifact Inventory
Dependencies: Phase 2.4.1C (write surface enforcement complete)

This module provides:
- Protected artifact class definitions
- Path pattern matching
- Allowed location rules
- Distribution policy enforcement
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Set
import fnmatch
import hashlib


class DistributionPolicy(Enum):
    """Distribution rules for protected artifacts."""
    PLAINTEXT_NEVER = "plaintext_never"  # Must always be sealed
    PLAINTEXT_INTERNAL = "plaintext_internal"  # Allowed in working tree only
    PLAINTEXT_ALLOWED = "plaintext_allowed"  # Can be distributed as plaintext


class ArtifactClass(Enum):
    """Protected artifact classes."""
    VECTOR_DATABASE = "vector_database"
    CAS_BLOB = "cas_blob"
    PROOF_OUTPUT = "proof_output"
    COMPRESSION_ADVANTAGE = "compression_advantage"
    PACK_OUTPUT = "pack_output"
    SEMANTIC_INDEX = "semantic_index"


@dataclass
class ProtectedPattern:
    """A protected artifact pattern with enforcement rules."""

    artifact_class: ArtifactClass
    patterns: List[str]  # Glob patterns (sorted for determinism)
    allowed_locations: List[str]  # Allowed directory patterns (sorted)
    distribution_policy: DistributionPolicy
    description: str

    def __post_init__(self):
        """Ensure deterministic ordering."""
        self.patterns = sorted(self.patterns)
        self.allowed_locations = sorted(self.allowed_locations)

    def matches_path(self, path: Path) -> bool:
        """Check if a path matches this pattern.

        Args:
            path: Path to check (relative to repo root)

        Returns:
            True if path matches any pattern
        """
        path_str = str(path).replace('\\', '/')
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in self.patterns)

    def is_allowed_location(self, path: Path) -> bool:
        """Check if path is in an allowed location.

        Args:
            path: Path to check (relative to repo root)

        Returns:
            True if path is in an allowed location
        """
        if not self.allowed_locations:
            return False

        path_str = str(path).replace('\\', '/')
        return any(fnmatch.fnmatch(path_str, pattern) for pattern in self.allowed_locations)

    def to_dict(self) -> dict:
        """Serialize to canonical JSON format."""
        return {
            "artifact_class": self.artifact_class.value,
            "patterns": self.patterns,
            "allowed_locations": self.allowed_locations,
            "distribution_policy": self.distribution_policy.value,
            "description": self.description
        }


@dataclass
class ProtectedInventory:
    """Canonical inventory of protected artifacts.

    This is the single source of truth for what must be protected.
    If this inventory is incomplete, crypto-safe becomes theater.
    """

    patterns: List[ProtectedPattern] = field(default_factory=list)
    version: str = "1.0.0"

    def __post_init__(self):
        """Ensure deterministic ordering."""
        self.patterns = sorted(self.patterns, key=lambda p: (p.artifact_class.value, tuple(p.patterns)))

    def add_pattern(self, pattern: ProtectedPattern) -> None:
        """Add a protected pattern to inventory."""
        self.patterns.append(pattern)
        self.patterns = sorted(self.patterns, key=lambda p: (p.artifact_class.value, tuple(p.patterns)))

    def find_matching_patterns(self, path: Path) -> List[ProtectedPattern]:
        """Find all patterns that match a given path.

        Args:
            path: Path to check (relative to repo root)

        Returns:
            List of matching ProtectedPattern objects (sorted)
        """
        matches = [p for p in self.patterns if p.matches_path(path)]
        return sorted(matches, key=lambda p: (p.artifact_class.value, tuple(p.patterns)))

    def is_protected(self, path: Path) -> bool:
        """Check if a path is protected.

        Args:
            path: Path to check (relative to repo root)

        Returns:
            True if path matches any protected pattern
        """
        return len(self.find_matching_patterns(path)) > 0

    def requires_sealing(self, path: Path, context: str = "public") -> bool:
        """Check if a path requires sealing in the given context.

        Args:
            path: Path to check (relative to repo root)
            context: Distribution context ("public", "internal", "working")

        Returns:
            True if path must be sealed in this context
        """
        matches = self.find_matching_patterns(path)
        if not matches:
            return False

        # If any matching pattern requires sealing, we must seal
        for pattern in matches:
            if pattern.distribution_policy == DistributionPolicy.PLAINTEXT_NEVER:
                return True
            if pattern.distribution_policy == DistributionPolicy.PLAINTEXT_INTERNAL:
                if context == "public":
                    return True

        return False

    def to_dict(self) -> dict:
        """Serialize to canonical JSON format."""
        return {
            "version": self.version,
            "patterns": [p.to_dict() for p in self.patterns]
        }

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize to canonical JSON string."""
        return json.dumps(self.to_dict(), separators=(',', ':'), sort_keys=True, indent=indent)

    def hash(self) -> str:
        """Compute deterministic hash of inventory."""
        canonical = json.dumps(self.to_dict(), separators=(',', ':'), sort_keys=True)
        return hashlib.sha256(canonical.encode('utf-8')).hexdigest()

    @classmethod
    def from_dict(cls, data: dict) -> 'ProtectedInventory':
        """Deserialize from dictionary."""
        patterns = [
            ProtectedPattern(
                artifact_class=ArtifactClass(p["artifact_class"]),
                patterns=p["patterns"],
                allowed_locations=p["allowed_locations"],
                distribution_policy=DistributionPolicy(p["distribution_policy"]),
                description=p["description"]
            )
            for p in data["patterns"]
        ]
        return cls(patterns=patterns, version=data["version"])

    @classmethod
    def from_json(cls, json_str: str) -> 'ProtectedInventory':
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> 'ProtectedInventory':
        """Load inventory from file."""
        with open(path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())

    def save(self, path: Path) -> None:
        """Save inventory to file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


def get_default_inventory() -> ProtectedInventory:
    """Get the default protected artifacts inventory.

    This is the canonical definition of what must be protected.
    AIRTIGHT: If this inventory is incomplete, crypto-safe becomes theater.

    Returns:
        ProtectedInventory with default patterns
    """
    inventory = ProtectedInventory(version="1.2.0")

    # ==========================================================================
    # VECTOR DATABASES (embeddings) - PLAINTEXT_NEVER
    # These contain semantic embeddings that encode meaning.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.VECTOR_DATABASE,
        patterns=[
            "NAVIGATION/CORTEX/db/*.db",
            "NAVIGATION/CORTEX/_generated/*.db",
            "THOUGHT/LAB/CAT_CHAT/**/*.db",
            "THOUGHT/LAB/CAT_CHAT/CAT_CORTEX/db/*.db",
            "THOUGHT/LAB/CAT_CHAT/CAT_CORTEX/_generated/*.db",
            "**/cat_chat_index.db"
        ],
        allowed_locations=[
            "NAVIGATION/CORTEX/**",
            "THOUGHT/LAB/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_NEVER,
        description="Vector databases containing semantic embeddings (384-dim vectors)"
    ))

    # ==========================================================================
    # CAS BLOBS (content-addressed storage) - PLAINTEXT_INTERNAL
    # Immutable blobs indexed by hash. Contains deduplicated content.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.CAS_BLOB,
        patterns=[
            ".ags-cas/**",
            "CAPABILITY/PRIMITIVES/.ags-cas/**"
        ],
        allowed_locations=[
            ".ags-cas/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_INTERNAL,
        description="Content-addressed storage blobs (immutable, hash-indexed)"
    ))

    # ==========================================================================
    # COMPRESSION ADVANTAGE ARTIFACTS - PLAINTEXT_NEVER
    # Proofs and benchmarks that reveal compression strategies.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.COMPRESSION_ADVANTAGE,
        patterns=[
            "NAVIGATION/PROOFS/COMPRESSION/*.json",
            "LAW/CONTRACTS/_runs/_tmp/compression_proof/*.db",
            "LAW/CONTRACTS/_runs/_tmp/compression_proof/*.json"
        ],
        allowed_locations=[
            "NAVIGATION/PROOFS/COMPRESSION/**",
            "LAW/CONTRACTS/_runs/_tmp/compression_proof/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_NEVER,
        description="Compression proofs revealing semantic evaluation databases and benchmark data"
    ))

    # ==========================================================================
    # PACK OUTPUTS - PLAINTEXT_NEVER (upgraded from INTERNAL)
    # Pack manifests contain CAS hashes for EVERY file - reveals structure.
    # Pack metadata reveals compression ratios, file organization.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.PACK_OUTPUT,
        patterns=[
            "_PACK_RUN/**",
            "MEMORY/LLM_PACKER/_packs/**/*.json",  # ALL JSON (manifests, proofs, refs)
            "MEMORY/LLM_PACKER/_packs/**/*.db",
            "MEMORY/LLM_PACKER/_packs/**/meta/**",  # Pack metadata directories
            "MEMORY/LLM_PACKER/_packs/**/*.md",  # Generated pack content
            # Pipeline/test run manifests (contain file hashes)
            "LAW/CONTRACTS/_runs/**/*MANIFEST*.json",  # PRE_MANIFEST, POST_MANIFEST
            "LAW/CONTRACTS/_runs/**/PACK_MANIFEST.json",  # Nested pack manifests
            "LAW/CONTRACTS/_runs/**/*.db",  # Any database in test runs
            "LAW/CONTRACTS/_runs/**/meta/**"  # Metadata directories in test runs
        ],
        allowed_locations=[
            "MEMORY/LLM_PACKER/_packs/**",
            "_PACK_RUN/**",
            "LAW/CONTRACTS/_runs/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_NEVER,
        description="Pack outputs: manifests with CAS hashes, proofs, metadata (reveals structure)"
    ))

    # ==========================================================================
    # PROOF OUTPUTS - PLAINTEXT_INTERNAL
    # Proof receipts and verification outputs.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.PROOF_OUTPUT,
        patterns=[
            "NAVIGATION/PROOFS/GREEN_STATE.json",
            "NAVIGATION/PROOFS/PROOF_MANIFEST.json",
            "**/PROTECTED_MANIFEST.json",
            "**/SEALED_ARTIFACTS.json"
        ],
        allowed_locations=[
            "NAVIGATION/PROOFS/**",
            "LAW/CONTRACTS/_runs/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_INTERNAL,
        description="Proof outputs and manifests (internal verification only)"
    ))

    # ==========================================================================
    # SEMANTIC INDEXES - PLAINTEXT_NEVER
    # Generated indexes that reveal codebase structure and organization.
    # ==========================================================================
    inventory.add_pattern(ProtectedPattern(
        artifact_class=ArtifactClass.SEMANTIC_INDEX,
        patterns=[
            # Databases
            "**/semantic_eval.db",
            "**/vector_store.db",
            "**/*_cassette.db",
            # Generated indexes (reveal structure)
            "NAVIGATION/CORTEX/_generated/*.json",
            "NAVIGATION/CORTEX/meta/*.json",
            "**/SECTION_INDEX.json",
            "**/SUMMARY_INDEX.json",
            "**/FILE_INDEX.json",
            "**/CORTEX_META.json",
            # Cassette configuration
            "NAVIGATION/CORTEX/network/cassettes.json",
            # AIRTIGHT: Catch-all for any .db file containing embeddings
            # This prevents leaks if vector databases appear in unexpected locations
            "**/system1.db",
            "**/system2.db",
            "**/system3.db",
            "**/cortex.db",
            "**/codebase_full.db",
            "**/instructions.db",
            "**/swarm_instructions.db"
        ],
        allowed_locations=[
            "LAW/CONTRACTS/_runs/_tmp/**",
            "NAVIGATION/CORTEX/**",
            "THOUGHT/LAB/**"
        ],
        distribution_policy=DistributionPolicy.PLAINTEXT_NEVER,
        description="Semantic indexes and generated metadata revealing codebase structure"
    ))

    return inventory


if __name__ == "__main__":
    # Self-test and inventory generation
    print("Protected Artifacts Inventory - Self-Test")
    print("=" * 60)

    inventory = get_default_inventory()

    print(f"\nInventory version: {inventory.version}")
    print(f"Total patterns: {len(inventory.patterns)}")
    print(f"Inventory hash: {inventory.hash()}")

    print("\n" + "=" * 60)
    print("Protected Patterns:")
    print("=" * 60)

    for pattern in inventory.patterns:
        print(f"\nClass: {pattern.artifact_class.value}")
        print(f"Policy: {pattern.distribution_policy.value}")
        print(f"Description: {pattern.description}")
        print(f"Patterns: {', '.join(pattern.patterns)}")
        print(f"Allowed: {', '.join(pattern.allowed_locations) if pattern.allowed_locations else 'NONE'}")

    # Test path matching
    print("\n" + "=" * 60)
    print("Path Matching Tests:")
    print("=" * 60)

    test_paths = [
        "NAVIGATION/CORTEX/db/system1.db",
        "NAVIGATION/PROOFS/COMPRESSION/COMPRESSION_PROOF_DATA.json",
        ".ags-cas/objects/abc123",
        "_PACK_RUN/LATEST/output.txt",
        "README.md",
        "CAPABILITY/PRIMITIVES/cas_store.py"
    ]

    for test_path in test_paths:
        path = Path(test_path)
        is_protected = inventory.is_protected(path)
        requires_seal = inventory.requires_sealing(path, context="public")
        matches = inventory.find_matching_patterns(path)

        status = "[PROTECTED]" if is_protected else "[PUBLIC]"
        seal = " (MUST SEAL)" if requires_seal else ""

        print(f"\n{status}{seal}: {test_path}")
        if matches:
            for match in matches:
                print(f"  -> {match.artifact_class.value}: {match.distribution_policy.value}")

    # Save inventory
    print("\n" + "=" * 60)
    print("Saving inventory...")
    output_path = Path(__file__).parent / "PROTECTED_INVENTORY.json"
    inventory.save(output_path)
    print(f"Saved to: {output_path}")

    # Verify round-trip
    loaded = ProtectedInventory.load(output_path)
    assert loaded.hash() == inventory.hash(), "Round-trip hash mismatch!"
    print("[OK] Round-trip verification passed")

    print("\n" + "=" * 60)
    print("Self-test complete!")
