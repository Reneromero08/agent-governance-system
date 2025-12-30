import sys
import pytest
from pathlib import Path

# Define REPO_ROOT as the parent directory 3 levels up from this file
REPO_ROOT = Path(__file__).resolve().parents[3]

# Add REPO_ROOT to Python path to ensure imports work
sys.path.insert(0, str(REPO_ROOT))

def test_spectrum02_resume():
    """Pytest entry point."""
    from CAPABILITY.PIPELINES.spectrum.runner_spectrum02_resume import RunnerSPECTRUM02Resume
    assert RunnerSPECTRUM02Resume().run_all()

if __name__ == "__main__":
    pytest.main([__file__])