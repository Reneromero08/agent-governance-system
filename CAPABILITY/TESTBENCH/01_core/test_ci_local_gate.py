import CAPABILITY.TOOLS.utilities.ci_local_gate as ci_local_gate


def test_status_entry_paths_handles_renames_and_windows_separators():
    assert ci_local_gate._status_entry_paths("R  old.py -> new.py") == ("old.py", "new.py")
    assert ci_local_gate._status_entry_paths(r"?? folder\file.py") == ("folder/file.py",)


def test_non_exempt_status_lines_include_untracked_and_ignore_thought():
    status = "\n".join(
        [
            " M CAPABILITY/TOOLS/utilities/push_test_plan.py",
            "?? scratch.txt",
            "?? THOUGHT/LAB/probe.txt",
        ]
    )
    assert ci_local_gate._non_exempt_status_lines(status) == [
        " M CAPABILITY/TOOLS/utilities/push_test_plan.py",
        "?? scratch.txt",
    ]


def test_rename_is_not_exempt_when_only_destination_is_in_thought():
    status = "R  CAPABILITY/old.py -> THOUGHT/LAB/old.py"
    assert ci_local_gate._non_exempt_status_lines(status) == [status]


def test_rename_is_exempt_when_both_paths_are_in_thought():
    status = "R  THOUGHT/LAB/old.py -> THOUGHT/LAB/new.py"
    assert ci_local_gate._non_exempt_status_lines(status) == []


def test_ensure_clean_tree_fails_before_expensive_checks(monkeypatch, capsys):
    monkeypatch.setattr(
        ci_local_gate,
        "_git_stdout",
        lambda args, required=False: "?? untracked.py",
    )
    assert not ci_local_gate._ensure_clean_tree("before checks")
    assert "before checks" in capsys.readouterr().err


def test_ensure_clean_tree_allows_only_thought_changes(monkeypatch):
    monkeypatch.setattr(
        ci_local_gate,
        "_git_stdout",
        lambda args, required=False: " M THOUGHT/LAB/experiment.py",
    )
    assert ci_local_gate._ensure_clean_tree("before checks")


def test_resolve_base_sha_dereferences_commit(monkeypatch):
    calls = []

    def fake_git(args, required=False):
        calls.append((args, required))
        return "b" * 40

    monkeypatch.setattr(ci_local_gate, "_git_stdout", fake_git)
    assert ci_local_gate._resolve_base_sha("origin/main") == "b" * 40
    assert calls == [
        (["git", "rev-parse", "--verify", "origin/main^{commit}"], True)
    ]
    assert ci_local_gate._resolve_base_sha(None) is None


def test_head_movement_is_rejected(monkeypatch, capsys):
    monkeypatch.setattr(
        ci_local_gate,
        "_git_stdout",
        lambda args, required=False: "c" * 40,
    )
    assert not ci_local_gate._ensure_head_unchanged("a" * 40)
    error = capsys.readouterr().err
    assert "HEAD changed during verification" in error
    assert "a" * 40 in error
    assert "c" * 40 in error


def test_unchanged_head_is_accepted(monkeypatch):
    head = "a" * 40
    monkeypatch.setattr(
        ci_local_gate,
        "_git_stdout",
        lambda args, required=False: head,
    )
    assert ci_local_gate._ensure_head_unchanged(head)
