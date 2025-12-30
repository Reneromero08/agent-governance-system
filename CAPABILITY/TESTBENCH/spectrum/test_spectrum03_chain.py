from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "MCP" / "server.py"

import importlib.util
spec = importlib.util.spec_from_file_location("mcp_server", SERVER_PATH)
mcp_server_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mcp_server_module)

MCPTerminalServer = mcp_server_module.MCPTerminalServer
VALIDATOR_SEMVER = mcp_server_module.VALIDATOR_SEMVER
get_validator_build_id = mcp_server_module.get_validator_build_id

sys.path.append(REPO_ROOT / "spectrum")
from CAPABILITY.PRIMITIVES.verify_bundle import BundleVerifier


def verify_spectrum03_chain(
    run_dirs: list[Path],
    strict_order: bool = False
) -> dict:
    """Verify a chain of SPECTRUM-02 bundles.

    This is a compatibility wrapper that uses the new BundleVerifier primitive.

    Checks:
    1. Verify each individual bundle passes.
    2. Verify the chain as a whole, ensuring all runs have valid SPECTRUM-02 bundles.

    :param run_dirs: List of directories containing SPECTRUM-02 bundles.
    :type run_dirs: list[Path]
    :param strict_order: Whether to verify that each bundle is processed after its predecessor (default False).
    :return: A dictionary indicating the validity and errors for the chain verification.

    """
    # Verify individual bundles
    results = [runner.run_all() for _ in range(len(run_dirs))]

    # Combine validation results into a single result dictionary
    combined_result = {
        'valid': all(results),
        'errors': [
            {'code': r.get('code'), 'run_id': r['run_id']} for r, success in zip(results, results) if not success
        ]
    }

    return combined_result


if __name__ == "__main__":
    # Assume run_dirs is a list of directories containing valid SPECTRUM-02 bundles.
    run_dirs = [run_dir for run_dir in Path("/path/to/runs").iterdir() if run_dir.is_dir()]
    result = verify_spectrum03_chain(run_dirs)
    print(result)
