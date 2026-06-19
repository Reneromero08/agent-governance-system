from pathlib import Path

import pytest

import CAPABILITY.TOOLS.utilities.push_test_plan as push_test_plan
from CAPABILITY.TOOLS.utilities.push_test_plan import (
    CORE_IGNORES,
    RISK_GROUPS,
    PlanError,
    TestSuite as Suite,
    build_plan,
    changed_paths,
    normalize_paths,
    plan_payload,
    pytest_command,
    repo_python,
    requires_embeddings,
)


def suite_names(paths):
    return [suite.name for suite in build_plan(paths)]


def test_normalize_paths_is_cross_platform_and_deterministic():
    assert normalize_paths(["b.py", r"NAVIGATION\CORTEX\semantic\x.py", "b.py"]) == [
        "NAVIGATION/CORTEX/semantic/x.py",
        "b.py",
    ]


def test_unrelated_change_runs_core_only():
    assert suite_names(["README.md"]) == ["core"]


def test_every_core_ignore_has_exactly_one_conditional_owner():
    owned = [path for group in RISK_GROUPS for path in group.tests]
    assert len(owned) == len(set(owned))
    assert CORE_IGNORES == tuple(owned)
    assert all(group.tests for group in RISK_GROUPS)


def test_global_test_infrastructure_change_selects_every_risk_group():
    assert suite_names(["pytest.ini"]) == ["core", *(group.name for group in RISK_GROUPS)]


@pytest.mark.parametrize(
    ("path", "expected_group"),
    [
        ("CAPABILITY/PRIMITIVES/write_firewall.py", "write-firewall"),
        ("CAPABILITY/TOOLS/codebook_lookup.py", "symbol-resolution"),
        ("CAPABILITY/TOOLS/ags.py", "mcp-capability"),
        ("CAPABILITY/PRIMITIVES/skill_index.py", "skill-discovery"),
        ("NAVIGATION/CORTEX/network/geometric_cassette.py", "cassette-network"),
        ("CAPABILITY/PRIMITIVES/model_registry.py", "embeddings"),
    ],
)
def test_each_ignored_suite_has_an_owning_path_trigger(path, expected_group):
    assert expected_group in suite_names([path])


def test_skill_discovery_tracks_skill_metadata_not_every_skill_implementation():
    assert "skill-discovery" in suite_names(
        ["CAPABILITY/SKILLS/utilities/arxiv-to-md/SKILL.md"]
    )
    assert "skill-discovery" not in suite_names(
        ["CAPABILITY/SKILLS/utilities/arxiv-to-md/run.py"]
    )


def test_semantic_change_runs_all_semantic_dependents():
    names = suite_names(["NAVIGATION/CORTEX/semantic/embeddings.py"])
    assert names == ["core", "skill-discovery", "cassette-network", "embeddings"]
    assert requires_embeddings(["NAVIGATION/CORTEX/semantic/embeddings.py"])


def test_changing_a_conditional_test_selects_its_suite():
    for group in RISK_GROUPS:
        assert group.name in suite_names([group.tests[0]])


def test_exhaustive_is_single_unfiltered_suite():
    plan = build_plan(["README.md"], exhaustive=True)
    assert len(plan) == 1
    assert plan[0].name == "exhaustive"
    assert plan[0].extra_args == ()


def test_plan_hash_is_stable_and_changes_with_scope(monkeypatch):
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/repo/.venv/python")
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)
    first = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=0)
    second = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=0)
    embedding = plan_payload(
        ["CAPABILITY/PRIMITIVES/model_registry.py"],
        base_ref="main",
        exhaustive=False,
        workers=0,
    )
    assert first["plan_hash"] == second["plan_hash"]
    assert first["plan_hash"] != embedding["plan_hash"]


def test_plan_hash_does_not_depend_on_machine_interpreter_path(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/machine-a/.venv/python")
    first = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=4)
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/machine-b/.venv/python")
    second = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=4)
    assert first["python"] != second["python"]
    assert first["plan_hash"] == second["plan_hash"]


def test_plan_payload_reports_why_a_group_was_selected(monkeypatch):
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/repo/.venv/python")
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)
    path = "CAPABILITY/PRIMITIVES/write_firewall.py"
    payload = plan_payload([path], base_ref="main", exhaustive=False, workers=0)
    assert payload["risk_groups"] == [
        {"name": "write-firewall", "matched_paths": [path]}
    ]


def test_changed_paths_unions_merge_base_and_direct_diffs(monkeypatch):
    monkeypatch.setattr(push_test_plan, "resolve_base_ref", lambda explicit=None: "origin/main")
    monkeypatch.setattr(
        push_test_plan,
        "_git_diff_lines",
        lambda spec: {
            "origin/main...HEAD": ["local_only.py"],
            "origin/main..HEAD": ["local_only.py", "remote_removed.py"],
        }[spec],
    )
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    paths, base = changed_paths()
    assert base == "origin/main"
    assert paths == ["local_only.py", "remote_removed.py"]


def test_invalid_explicit_base_fails_closed(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: False)
    with pytest.raises(PlanError, match="explicit base ref"):
        push_test_plan.resolve_base_ref("not-a-ref")


def test_zero_ci_base_means_initial_history_not_current_origin(monkeypatch):
    monkeypatch.setenv("AGS_PUSH_BASE", "0" * 40)
    monkeypatch.setattr(
        push_test_plan,
        "_git_ref_exists",
        lambda candidate: candidate == "origin/main",
    )
    assert push_test_plan.resolve_base_ref() is None


def test_repo_python_prefers_repository_virtualenv(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(push_test_plan, "PROJECT_ROOT", tmp_path)
    if push_test_plan.os.name == "nt":
        interpreter = tmp_path / ".venv" / "Scripts" / "python.exe"
    else:
        interpreter = tmp_path / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True)
    interpreter.touch()

    assert repo_python() == str(interpreter)


def test_pytest_command_uses_selected_interpreter(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)
    command = pytest_command(
        Suite("core", ("CAPABILITY/TESTBENCH",)),
        workers=4,
        python_executable="/repo/.venv/python",
    )
    assert command[0] == "/repo/.venv/python"
    assert "-n" not in command


def test_pytest_command_does_not_probe_xdist_when_workers_are_disabled(monkeypatch):
    monkeypatch.setattr(
        push_test_plan,
        "_interpreter_has_xdist",
        lambda _: (_ for _ in ()).throw(AssertionError("xdist probe should not run")),
    )
    command = pytest_command(
        Suite("focused", ("test_one.py",)),
        workers=0,
        python_executable="python",
    )
    assert "-n" not in command
