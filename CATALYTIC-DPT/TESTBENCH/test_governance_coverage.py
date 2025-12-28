
import os
import sys
import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

def test_roadmap_deliverables_exist():
    """Verify that key files promised in ROADMAP_V2.3.md actually exist."""
    roadmap_path = REPO_ROOT / "CATALYTIC-DPT" / "ROADMAP_V2.3.md"
    assert roadmap_path.exists()
    
    # Key files promised in Phase 7, 8, 9
    required_files = [
        "CATALYTIC-DPT/SCHEMAS/swarm.schema.json",
        "CATALYTIC-DPT/PIPELINES/swarm_runtime.py",
        "CATALYTIC-DPT/SCHEMAS/VERSIONING_POLICY.md",
        "CATALYTIC-DPT/RELEASE_CHECKLIST.md",
        "CONTEXT/decisions/ADR-023-capability-revocation-semantics.md",
        "TOOLS/ags.py",
        "CATALYTIC-DPT/PIPELINES/pipeline_runtime.py"
    ]
    
    for rel_path in required_files:
        assert (REPO_ROOT / rel_path).exists(), f"Missing roadmap deliverable: {rel_path}"

def test_canon_mentions_key_concepts():
    """Verify that CANON/CONTRACT.md and AGENTS.md mention key governance concepts."""
    contract = (REPO_ROOT / "CANON" / "CONTRACT.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    
    # Concepts that must be in law
    required_concepts = [
        "Commit ceremony",
        "Traceable Identity",
        "Output roots",
        "Determinism"
    ]
    
    for concept in required_concepts:
        assert concept in contract or concept in agents, f"Concept '{concept}' not found in CANON/CONTRACT.md or AGENTS.md"

def test_router_receipt_artifacts_in_schema():
    """Verify that ags_plan.schema.json includes the 'router' field (Phase 8)."""
    schema_path = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "ags_plan.schema.json"
    import json
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    
    assert "router" in schema["properties"], "ags_plan.schema.json missing 'router' property (Phase 8)"

def test_policy_snapshot_logic_exists():
    """Verify that ags.py contains logic for POLICY.json snapshotting (Phase 6.9)."""
    ags_py = (REPO_ROOT / "TOOLS" / "ags.py").read_text(encoding="utf-8")
    
    assert "revoked_capabilities" in ags_py, "TOOLS/ags.py missing revocation snapshot logic"
    assert "POLICY.json" in ags_py, "TOOLS/ags.py does not write POLICY.json"

def test_swarm_schema_validity():
    """Verify swarm.schema.json is valid JSON and contains expected structure (Phase 7)."""
    schema_path = REPO_ROOT / "CATALYTIC-DPT" / "SCHEMAS" / "swarm.schema.json"
    import json
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    
    assert schema.get("title") == "CAT-DPT Swarm Specification (DAG of Pipelines)"
    assert "nodes" in schema["properties"]
    assert "edges" in schema["properties"]

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
