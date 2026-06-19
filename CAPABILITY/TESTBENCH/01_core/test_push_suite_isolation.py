from CAPABILITY.TOOLS.utilities.push_test_plan import build_plan, pytest_command


def test_mcp_capability_suite_disables_xdist_even_when_available():
    suite = next(
        suite
        for suite in build_plan(["CAPABILITY/MCP/server.py"])
        if suite.name == "mcp-capability"
    )
    command = pytest_command(
        suite,
        workers=4,
        python_executable="python",
        xdist_available=True,
    )
    assert suite.xdist is False
    assert "-n" not in command


def test_core_suite_keeps_xdist_parallelism():
    core = build_plan(["README.md"])[0]
    command = pytest_command(
        core,
        workers=4,
        python_executable="python",
        xdist_available=True,
    )
    assert core.xdist is True
    assert command[-4:] == ["-n", "4", "--dist=loadfile"]
