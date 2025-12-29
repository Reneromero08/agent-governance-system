from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class SkillError(Exception):
    """Base class for skill errors."""


class SkillNotFoundError(SkillError):
    """Skill ID not found in registry."""


class CapabilityHashMismatch(SkillError):
    """Capability hash does not match computed hash."""


class RegistryError(SkillError):
    """Registry is malformed or invalid."""


def canonical_json(obj: Any) -> bytes:
    """Canonicalize JSON object per SPECTRUM-04 v1.1.0."""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    ).encode('utf-8')


@dataclass(frozen=True)
class Skill:
    skill_id: str
    capability_hash: str
    version: str
    human_name: Optional[str] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class SkillRegistry:
    version: str
    skills: Dict[str, Skill]

    @classmethod
    def load(cls, path: Path) -> SkillRegistry:
        path = Path(path)
        if not path.exists():
            raise RegistryError(f"Registry file not found: {path}")

        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except json.JSONDecodeError as e:
            raise RegistryError(f"Invalid JSON in registry: {e}")

        if not isinstance(data, dict):
            raise RegistryError("Registry must be a JSON object")

        version = data.get("registry_version")
        if not version:
            raise RegistryError("Missing registry_version")

        skills_data = data.get("skills", {})
        if not isinstance(skills_data, dict):
            raise RegistryError("skills field must be an object")

        skills = {}
        for skill_id, skill_data in skills_data.items():
            if not isinstance(skill_data, dict):
                raise RegistryError(f"Skill entry {skill_id} must be an object")
            
            cap_hash = skill_data.get("capability_hash")
            if not cap_hash:
                raise RegistryError(f"Skill {skill_id} missing capability_hash")
                
            skills[skill_id] = Skill(
                skill_id=skill_id,
                capability_hash=cap_hash,
                version=skill_data.get("version", "0.0.0"),
                human_name=skill_data.get("human_name"),
                description=skill_data.get("description"),
            )

        return cls(version=version, skills=skills)

    def resolve(self, skill_id: str) -> Skill:
        if skill_id not in self.skills:
            raise SkillNotFoundError(f"Skill '{skill_id}' not found in registry")
        return self.skills[skill_id]


def resolve_adapter(
    skill_id: str, 
    registry_path: Path, 
    capabilities_path: Path
) -> Dict[str, Any]:
    """
    Resolve a skill_id to its full adapter specification.
    
    1. Load registry and find skill -> capability_hash
    2. Load capabilities and find capability_hash -> adapter
    3. Verify integrity
    """
    registry = SkillRegistry.load(registry_path)
    skill = registry.resolve(skill_id)
    
    capabilities_path = Path(capabilities_path)
    if not capabilities_path.exists():
        raise RegistryError(f"Capabilities file not found: {capabilities_path}")
        
    try:
        caps_data = json.loads(capabilities_path.read_text(encoding='utf-8'))
    except json.JSONDecodeError as e:
        raise RegistryError(f"Invalid JSON in capabilities: {e}")

    capabilities = caps_data.get("capabilities", {})
    
    if skill.capability_hash not in capabilities:
        raise SkillNotFoundError(
            f"Skill '{skill_id}' references unknown capability hash: {skill.capability_hash}"
        )
        
    cap_entry = capabilities[skill.capability_hash]
    
    # SPECTRUM-05/PHASE 6.5 Integrity Check:
    # Ensure the capability entry actually hashes to the capability_hash.
    # The entry in CAPABILITIES.json is expected to be { "adapter_spec_hash": "...", "adapter": ... }
    # Per roadmap: capability_hash = sha256(canonical_json({adapter_schema, adapter_payload, ...}))
    # But usually CAPABILITIES.json keys ARE the hashes.
    # We should verify that the CONTENT of the capability entry matches the hash if possible,
    # OR at least that the key matches the skill's claim.
    
    # For now, we trust the map lookup returns the object associated with that hash,
    # but we should ideally re-hash the content if we had the raw preimage.
    # In Phase 6.5, we rely on the `adapter_spec_hash` field inside the entry matching the key.
    
    stored_hash = cap_entry.get("adapter_spec_hash")
    if stored_hash != skill.capability_hash:
        raise CapabilityHashMismatch(
            f"Capability integrity error: Key {skill.capability_hash} maps to object with hash {stored_hash}"
        )
        
    return cap_entry.get("adapter")
