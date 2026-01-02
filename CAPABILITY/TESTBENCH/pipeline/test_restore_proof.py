import os
import pytest

@pytest.fixture
def repo_root():
    """Returns the root directory of the repository where this test is located"""
    # Get the directory where this test file is located
    test_file_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up to the CAPABILITY/TESTBENCH/pipeline directory
    return os.path.normpath(os.path.join(test_file_dir, "../../.."))

@pytest.fixture
def proof_schema_path(repo_root):
    """Returns the path to the proof schema file"""
    schema_path = os.path.join(repo_root, "schemas", "proof_v1.json")

    if not os.path.exists(schema_path):
        raise FileNotFoundError(
            f"Proof schema file not found at {schema_path}. "
            "Please ensure the file exists at this location."
        )

    return schema_path