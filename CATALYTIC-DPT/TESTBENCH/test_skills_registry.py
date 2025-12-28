import sys
from pathlib import Path
import json
import pytest

# Add CATALYTIC-DPT to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

from PRIMITIVES.skills import (
    SkillRegistry, 
    canonical_json, 
    resolve_adapter,
    SkillNotFoundError,
    RegistryError,
    CapabilityHashMismatch
)

@pytest.fixture
def temp_registry_dir(tmp_path):
    d = tmp_path / "registry_test"
    d.mkdir()
    return d

def test_canonical_json():
    obj = {"b": 2, "a": 1}
    expected = b'{"a":1,"b":2}'
    assert canonical_json(obj) == expected

def test_load_valid_registry(temp_registry_dir):
    reg_path = temp_registry_dir / "registry.json"
    content = {
        "registry_version": "1.0.0",
        "skills": {
            "test-skill": {
                "capability_hash": "abc12345",
                "version": "1.0.0",
                "human_name": "Test Skill"
            }
        }
    }
    reg_path.write_text(json.dumps(content))
    
    registry = SkillRegistry.load(reg_path)
    assert registry.version == "1.0.0"
    assert "test-skill" in registry.skills
    skill = registry.resolve("test-skill")
    assert skill.capability_hash == "abc12345"
    assert skill.human_name == "Test Skill"

def test_load_missing_registry(temp_registry_dir):
    with pytest.raises(RegistryError, match="not found"):
        SkillRegistry.load(temp_registry_dir / "missing.json")

def test_load_malformed_registry(temp_registry_dir):
    reg_path = temp_registry_dir / "bad.json"
    reg_path.write_text("{not valid json")
    with pytest.raises(RegistryError, match="Invalid JSON"):
        SkillRegistry.load(reg_path)

def test_resolve_missing_skill(temp_registry_dir):
    reg_path = temp_registry_dir / "registry.json"
    reg_path.write_text(json.dumps({"registry_version": "1.0", "skills": {}}))
    registry = SkillRegistry.load(reg_path)
    with pytest.raises(SkillNotFoundError):
        registry.resolve("missing-skill")

def test_e2e_resolution(temp_registry_dir):
    # Setup Registry
    reg_path = temp_registry_dir / "registry.json"
    reg_data = {
        "registry_version": "1.0",
        "skills": {
            "valid-skill": {
                "capability_hash": "deadbeef",
                "version": "1.0"
            }
        }
    }
    reg_path.write_text(json.dumps(reg_data))
    
    # Setup Capabilities
    caps_path = temp_registry_dir / "capabilities.json"
    caps_data = {
        "registry_version": "1.0",
        "capabilities": {
            "deadbeef": {
                "adapter_spec_hash": "deadbeef",
                "adapter": {"cmd": "fake"}
            }
        }
    }
    caps_path.write_text(json.dumps(caps_data))
    
    # Resolve
    adapter = resolve_adapter("valid-skill", reg_path, caps_path)
    assert adapter == {"cmd": "fake"}

def test_e2e_integrity_fail(temp_registry_dir):
    # Setup Registry with hash A
    reg_path = temp_registry_dir / "registry.json"
    reg_path.write_text(json.dumps({
        "registry_version": "1.0", 
        "skills": {"hack-skill": {"capability_hash": "hashA"}}
    }))
    
    # Setup Capabilities where key is HashA but internal hash is HashB
    caps_path = temp_registry_dir / "capabilities.json"
    caps_path.write_text(json.dumps({
        "registry_version": "1.0",
        "capabilities": {
            "hashA": {
                "adapter_spec_hash": "hashB", 
                "adapter": {}
            }
        }
    }))
    
    with pytest.raises(CapabilityHashMismatch):
        resolve_adapter("hack-skill", reg_path, caps_path)
