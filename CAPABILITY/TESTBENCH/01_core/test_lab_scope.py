from pathlib import Path

import pytest

from CAPABILITY.TOOLS.utilities.ags_lab import opencode_inline_policy
from CAPABILITY.TOOLS.utilities import lab_scope
from CAPABILITY.TOOLS.utilities.lab_scope import (
    LAB_ROOT,
    ZERO_SHA,
    PathClass,
    PushRef,
    ScopeViolation,
    canonical_runtime_path,
    classify_path,
    classify_paths,
    introduced_commits,
    normalize_git_path,
    parse_name_status_z,
    parse_push_refs,
)


def test_three_path_classes_are_immutable_and_marker_does_not_authorize_scope():
    assert classify_path("THOUGHT/LAB/experiment/result.py") == PathClass.LAB_PAYLOAD
    assert classify_path("THOUGHT/LAB/.agent-root") == PathClass.LAB_BOUNDARY_CONTROL
    assert classify_path("THOUGHT/LAB/.codex/rules.toml") == PathClass.LAB_BOUNDARY_CONTROL
    assert classify_path("CAPABILITY/experiment/.agent-root") == PathClass.MAIN
    assert classify_path("AGENTS.md") == PathClass.MAIN


@pytest.mark.parametrize(
    "path",
    ("", "/absolute", "../escape", "THOUGHT/LAB/../CAPABILITY/x", "a\\b"),
)
def test_git_paths_must_be_canonical_repository_relative(path: str):
    with pytest.raises(ScopeViolation):
        normalize_git_path(path)


def test_payload_cannot_mix_with_main_or_boundary_control():
    with pytest.raises(ScopeViolation, match="LAB SCOPE VIOLATION"):
        classify_paths(("THOUGHT/LAB/x.py", "CHANGELOG.md"))
    with pytest.raises(ScopeViolation, match="LAB_BOUNDARY_CONTROL"):
        classify_paths(("THOUGHT/LAB/x.py", "THOUGHT/LAB/AGENTS.md"))


def test_boundary_maintenance_may_include_main_enforcement_but_not_payload():
    result = classify_paths(("THOUGHT/LAB/.agent-root", ".githooks/pre-commit"))
    assert result.route == PathClass.LAB_BOUNDARY_CONTROL


def test_name_status_parser_keeps_both_rename_endpoints_and_odd_filenames():
    payload = (
        b"R100\x00THOUGHT/LAB/old name.py\x00CAPABILITY/new\nname.py\x00"
        b"M\x00THOUGHT/LAB/ok.py\x00"
    )
    assert parse_name_status_z(payload) == (
        "THOUGHT/LAB/old name.py",
        "CAPABILITY/new\nname.py",
        "THOUGHT/LAB/ok.py",
    )


def test_push_ref_parser_accepts_git_format_and_rejects_malformed_sha():
    text = f"refs/heads/lab {'a' * 40} refs/heads/lab {ZERO_SHA}\n"
    assert parse_push_refs(text) == (
        PushRef("refs/heads/lab", "a" * 40, "refs/heads/lab", ZERO_SHA),
    )
    with pytest.raises(ScopeViolation, match="malformed SHA"):
        parse_push_refs("refs/heads/lab nope refs/heads/lab " + ZERO_SHA)


def test_new_branch_excludes_remote_baselines(monkeypatch):
    calls: list[list[str]] = []

    def fake_git(args, **_kwargs):
        calls.append(list(args))
        return "c" * 40

    monkeypatch.setattr(lab_scope, "_git", fake_git)
    refs = (PushRef("refs/heads/lab", "a" * 40, "refs/heads/lab", ZERO_SHA),)
    assert introduced_commits(refs, "origin") == ("c" * 40,)
    assert calls == [["rev-list", "--reverse", "a" * 40, "--not", "--remotes=origin"]]


def test_existing_branch_uses_remote_to_local_range(monkeypatch):
    calls: list[list[str]] = []

    def fake_git(args, **_kwargs):
        calls.append(list(args))
        return ""

    monkeypatch.setattr(lab_scope, "_git", fake_git)
    monkeypatch.setattr(lab_scope, "_git_is_ancestor", lambda _remote, _local: True)
    refs = (PushRef("refs/heads/lab", "a" * 40, "refs/heads/lab", "b" * 40),)
    assert introduced_commits(refs, "origin") == ()
    assert calls == [["rev-list", "--reverse", f"{'b' * 40}..{'a' * 40}"]]


def test_rebased_existing_branch_excludes_remote_baselines(monkeypatch):
    calls: list[list[str]] = []

    def fake_git(args, **_kwargs):
        calls.append(list(args))
        return "c" * 40

    monkeypatch.setattr(lab_scope, "_git", fake_git)
    monkeypatch.setattr(lab_scope, "_git_is_ancestor", lambda _remote, _local: False)
    refs = (PushRef("refs/heads/lab", "a" * 40, "refs/heads/lab", "b" * 40),)
    assert introduced_commits(refs, "origin") == ("c" * 40,)
    assert calls == [["rev-list", "--reverse", "a" * 40, "--not", "--remotes=origin"]]


def test_merge_commit_is_rejected_before_path_classification(monkeypatch):
    monkeypatch.setattr(
        lab_scope,
        "_git",
        lambda args, **_kwargs: f"{'a' * 40} {'b' * 40} {'c' * 40}"
        if args[:2] == ["rev-list", "--parents"]
        else b"",
    )
    with pytest.raises(ScopeViolation, match="merge commit"):
        lab_scope.commit_paths("a" * 40)


def test_runtime_paths_reject_symlink_or_parent_escape(tmp_path: Path):
    lab = tmp_path / "lab"
    lab.mkdir()
    inside = lab / "inside"
    assert canonical_runtime_path(inside, beneath=lab) == inside
    with pytest.raises(ScopeViolation, match="escapes"):
        canonical_runtime_path(lab / ".." / "main", beneath=lab)


def test_opencode_denies_direct_outside_edits_and_boundary_controls():
    policy = opencode_inline_policy()
    permission = policy["permission"]
    assert permission["edit"]["*"] == "deny"
    assert permission["edit"][f"{LAB_ROOT}/**"] == "allow"
    for relative in (".agent-root", "AGENTS.md", "LAB_CONTRACT.md", "opencode.json"):
        assert permission["edit"][f"{LAB_ROOT}/{relative}"] == "deny"
    assert permission["edit"][f"{LAB_ROOT}/.codex/**"] == "deny"
    assert permission["read"][str(lab_scope.REPO_ROOT / "AGENTS.md")] == "deny"
    assert permission["bash"]["git *--no-verify*"] == "deny"


@pytest.mark.parametrize(
    "command",
    (
        "printf data > result.txt",
        "sed -i s/a/b/ result.txt",
        "cp input output",
        "mv input output",
        "rm result.txt",
        "install input output",
        "perl -pi -e s/a/b/ result.txt",
        "python -c open('result.txt','w').write('x')",
        "node -e fs.writeFileSync('result.txt','x')",
        "tar -xf payload.tar",
        "ln -s target link",
    ),
)
def test_opencode_does_not_choke_normal_lab_shell_mutations(command: str):
    assert command
    permission = opencode_inline_policy()["permission"]["bash"]
    assert permission["*"] == "allow"
    assert set(permission) == {"*", "git *--no-verify*", "git -c core.hooksPath=*"}
