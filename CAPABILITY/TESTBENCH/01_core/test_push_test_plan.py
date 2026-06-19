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
    resolve_base_ref,
    selected_risk_groups,
    _git_commit_exists,
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
    monkeypatch.setattr(push_test_plan, "resolve_base_ref", lambda explicit=None, force_all=False: "origin/main")
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: True)
    monkeypatch.setattr(
        push_test_plan,
        "_git_diff_lines",
        lambda spec: {
            "origin/main...HEAD": ["local_only.py"],
            "origin/main..HEAD": ["local_only.py", "remote_removed.py"],
        }[spec],
    )
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    paths, base, fallback = changed_paths()
    assert base == "origin/main"
    assert fallback is None
    assert paths == ["local_only.py", "remote_removed.py"]


def test_invalid_explicit_base_fails_closed(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: False)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: False)
    with pytest.raises(PlanError, match="explicit base ref"):
        push_test_plan.resolve_base_ref("not-a-ref")


def test_zero_ci_base_means_initial_history_not_current_origin(monkeypatch):
    monkeypatch.setenv("AGS_PUSH_BASE", "0" * 40)
    monkeypatch.setattr(
        push_test_plan,
        "_git_ref_exists",
        lambda candidate: candidate == "origin/main",
    )
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: True)
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


# --- Forced-push / missing-base resilience tests ---

ORPHAN_SHA = "3d44f83fdf4e0d31108d90678a24db5a3dddf669"
AVAILABLE_REF = "origin/main"


def test_raw_sha_not_treated_as_existing_without_commit_object(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: False)
    monkeypatch.setenv("AGS_PUSH_BASE", ORPHAN_SHA)
    monkeypatch.delenv("AGS_PUSH_FORCED", raising=False)
    with pytest.raises(PlanError, match="does not name an available commit"):
        resolve_base_ref()


def test_available_base_retains_normal_diff_selection(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    monkeypatch.setattr(
        push_test_plan,
        "_git_diff_lines",
        lambda spec: ["a.py"] if "HEAD" in spec else [],
    )
    monkeypatch.delenv("AGS_PUSH_BASE", raising=False)
    monkeypatch.setenv("AGS_PUSH_FORCED", "false")
    paths, base, fallback = changed_paths()
    assert base in ("@{upstream}", "origin/main")
    assert "a.py" in paths
    assert fallback is None


def test_normal_missing_base_strictly_fails(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: False)
    monkeypatch.setenv("AGS_PUSH_BASE", ORPHAN_SHA)
    monkeypatch.setenv("AGS_PUSH_FORCED", "false")
    with pytest.raises(PlanError, match="does not name an available commit"):
        resolve_base_ref()


def test_forced_push_missing_base_selects_all_risk_groups(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: False)
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/repo/.venv/python")
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)
    monkeypatch.setenv("AGS_PUSH_BASE", ORPHAN_SHA)
    monkeypatch.setenv("AGS_PUSH_FORCED", "true")

    paths, base, fallback = changed_paths()
    expected_group_names = tuple(group.name for group in RISK_GROUPS)

    assert base is None
    assert fallback is not None
    assert fallback["forced_push_fallback"] is True
    assert fallback["base_ref"] == ORPHAN_SHA
    assert fallback["base_status"] == "missing-forced-push"

    payload = plan_payload(paths, base_ref=base, exhaustive=False, workers=0, forced_fallback=fallback)
    selected = [item["name"] for item in payload["risk_groups"]]
    assert "core" in [s["name"] for s in payload["suites"]]
    assert all(name in selected for name in expected_group_names)
    assert payload["forced_push_fallback"] is True
    assert len(payload["changed_paths"]) == 0


def test_forced_fallback_plan_hash_is_deterministic(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    monkeypatch.setattr(push_test_plan, "repo_python", lambda: "/repo/.venv/python")
    monkeypatch.setattr(push_test_plan, "_interpreter_has_xdist", lambda _: False)

    fallback = {"base_ref": ORPHAN_SHA, "base_status": "missing-forced-push", "forced_push_fallback": True}
    first = plan_payload([], base_ref=None, exhaustive=False, workers=0, forced_fallback=fallback)
    second = plan_payload([], base_ref=None, exhaustive=False, workers=0, forced_fallback=fallback)
    assert first["plan_hash"] == second["plan_hash"]


def test_forced_push_with_available_base_uses_normal_selection(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    monkeypatch.setattr(
        push_test_plan,
        "_git_diff_lines",
        lambda spec: ["a.py"] if "HEAD" in spec else [],
    )
    monkeypatch.setenv("AGS_PUSH_BASE", AVAILABLE_REF)
    monkeypatch.setenv("AGS_PUSH_FORCED", "true")

    paths, base, fallback = changed_paths()
    assert base == AVAILABLE_REF
    assert fallback is None
    assert "a.py" in paths


def test_pr_never_uses_forced_push_fallback(monkeypatch):
    monkeypatch.setattr(push_test_plan, "_git_ref_exists", lambda _: True)
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: False)
    monkeypatch.setattr(push_test_plan, "_git_lines", lambda args, required=False: [])
    monkeypatch.setenv("AGS_PUSH_BASE", ORPHAN_SHA)
    monkeypatch.setenv("AGS_PUSH_FORCED", "false")

    with pytest.raises(PlanError, match="does not name an available commit"):
        resolve_base_ref()


def test_zero_sha_retains_initial_history_behavior(monkeypatch):
    monkeypatch.setenv("AGS_PUSH_BASE", "0" * 40)
    monkeypatch.setattr(
        push_test_plan,
        "_git_ref_exists",
        lambda candidate: candidate == "origin/main",
    )
    monkeypatch.setattr(push_test_plan, "_git_commit_exists", lambda _: True)
    assert resolve_base_ref() is None


def test_no_duplicate_risk_group_ownership():
    owned = [path for group in RISK_GROUPS for path in group.tests]
    assert len(owned) == len(set(owned))


def test_force_all_risk_groups_selects_every_group(monkeypatch):
    selected = selected_risk_groups([], force_all=True)
    assert len(selected) == len(RISK_GROUPS)
    assert all(group.name in [g.name for g, _ in selected] for group in RISK_GROUPS)


def test_force_all_flag_builds_plan_with_all_conditional_suites(monkeypatch):
    plan = build_plan([], force_all=True)
    names = [s.name for s in plan]
    assert "core" in names
    for group in RISK_GROUPS:
        assert group.name in names
