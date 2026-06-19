from CAPABILITY.TOOLS.utilities.push_test_plan import build_plan


def test_shared_paths_primitive_selects_firewall_and_mcp_owners():
    names = [
        suite.name
        for suite in build_plan(["CAPABILITY/PRIMITIVES/paths.py"])
    ]
    assert names == ["core", "write-firewall", "mcp-capability"]
