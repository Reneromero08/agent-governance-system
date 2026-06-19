import CAPABILITY.TOOLS.utilities.ci_local_gate as ci_local_gate


def test_status_entry_path_handles_renames_and_windows_separators():
    assert ci_local_gate._status_entry_path("R  old.py -> new.py") == "new.py"
    assert ci_local_gate._status_entry_path(r"?? folder\file.py") == "folder/file.py"


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
