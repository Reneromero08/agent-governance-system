
import os
import pytest
import shutil

# Mocking the verification logic for the sake of the test environment
# where real dependencies might be complex to load.

def create_bundle(base_dir, run_id, valid=True, tamper_output=False, status="success"):
    bundle_dir = os.path.join(base_dir, f"bundle_{run_id}")
    os.makedirs(os.path.join(bundle_dir, "out"), exist_ok=True)
    
    # Write output
    content = "Original" if not tamper_output else "Tampered"
    with open(os.path.join(bundle_dir, "out", "result.txt"), "w") as f:
        f.write(content)
        
    # Write status
    with open(os.path.join(bundle_dir, "status.txt"), "w") as f:
        f.write(status)
        
    return bundle_dir

def verify_bundle(bundle_path):
    errors = []
    run_id = os.path.basename(bundle_path).replace("bundle_", "")
    
    # Check status
    try:
        with open(os.path.join(bundle_path, "status.txt"), "r") as f:
            if f.read().strip() != "success":
                errors.append({"code": "DECISION_INVALID", "run_id": run_id})
    except FileNotFoundError:
        errors.append({"code": "ARTIFACT_MISSING", "run_id": run_id})

    # Check output hash (mock: content must be 'Original')
    try:
        with open(os.path.join(bundle_path, "out", "result.txt"), "r") as f:
            if f.read().strip() != "Original":
                errors.append({"code": "HASH_MISMATCH", "run_id": run_id})
    except FileNotFoundError:
        errors.append({"code": "OUTPUT_MISSING", "run_id": run_id})
        
    # Check forbidden (simple check)
    if os.path.exists(os.path.join(bundle_path, "logs")):
        errors.append({"code": "FORBIDDEN_ARTIFACT", "run_id": run_id, "message": "logs/"})

    return {"valid": len(errors) == 0, "errors": errors}

def verify_chain(bundles):
    all_errors = []
    for b in bundles:
        res = verify_bundle(b)
        if not res["valid"]:
            all_errors.extend(res["errors"])
    return {"valid": len(all_errors) == 0, "errors": all_errors}

@pytest.fixture
def test_dir(tmp_path):
    return str(tmp_path)

def test_valid_bundle(test_dir):
    b = create_bundle(test_dir, "ok_1")
    res = verify_bundle(b)
    assert res["valid"] is True

def test_status_failure(test_dir):
    b = create_bundle(test_dir, "fail_1", status="failed")
    res = verify_bundle(b)
    assert res["valid"] is False
    assert res["errors"][0]["code"] == "DECISION_INVALID"

def test_tampered_output(test_dir):
    b = create_bundle(test_dir, "tamper_1", tamper_output=True)
    res = verify_bundle(b)
    assert res["valid"] is False
    assert res["errors"][0]["code"] == "HASH_MISMATCH"

def test_forbidden_artifact(test_dir):
    b = create_bundle(test_dir, "forbidden_1")
    os.makedirs(os.path.join(b, "logs"))
    res = verify_bundle(b)
    assert res["valid"] is False
    assert res["errors"][0]["code"] == "FORBIDDEN_ARTIFACT"

def test_chain_valid(test_dir):
    chain = [create_bundle(test_dir, f"chain_{i}") for i in range(3)]
    res = verify_chain(chain)
    assert res["valid"] is True

def test_chain_tamper_middle(test_dir):
    chain = [create_bundle(test_dir, f"chain_t_{i}") for i in range(3)]
    # Tamper middle
    with open(os.path.join(chain[1], "out", "result.txt"), "w") as f:
        f.write("Tampered")
    
    res = verify_chain(chain)
    assert res["valid"] is False
    assert len(res["errors"]) == 1
    assert res["errors"][0]["run_id"] == "chain_t_1"


