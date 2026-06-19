from pathlib import Path

import pytest

from CAPABILITY.PRIMITIVES.paths import (
    is_portable_absolute,
    normalize_relpath,
    portable_parts,
    resolve_under_root,
)
from CAPABILITY.PRIMITIVES.write_firewall import FirewallViolation, WriteFirewall


def test_portable_path_syntax_is_host_independent(tmp_path: Path):
    assert is_portable_absolute(r"C:\tmp\output.txt")
    assert is_portable_absolute("/tmp/output.txt")
    assert is_portable_absolute(r"\\server\share\output.txt")
    assert not is_portable_absolute(r"LAW\CONTRACTS\_runs\output.txt")

    assert portable_parts(r"LAW\CONTRACTS\..\CANON") == (
        "LAW",
        "CONTRACTS",
        "..",
        "CANON",
    )
    assert normalize_relpath(r"LAW\CONTRACTS\_runs\output.txt") == (
        "LAW/CONTRACTS/_runs/output.txt"
    )
    with pytest.raises(ValueError, match="Absolute path"):
        normalize_relpath(r"C:\tmp\output.txt")
    with pytest.raises(ValueError, match="traversal"):
        normalize_relpath(r"LAW\CONTRACTS\..\CANON")

    resolved = resolve_under_root(r"LAW\CONTRACTS\_runs\output.txt", root=tmp_path)
    assert resolved == tmp_path / "LAW" / "CONTRACTS" / "_runs" / "output.txt"


def make_firewall(project_root: Path) -> WriteFirewall:
    (project_root / "LAW" / "CONTRACTS" / "_runs" / "_tmp").mkdir(parents=True)
    return WriteFirewall(
        tmp_roots=["LAW/CONTRACTS/_runs/_tmp"],
        durable_roots=["LAW/CONTRACTS/_runs"],
        project_root=project_root,
    )


def test_windows_relative_write_targets_real_repository_components(tmp_path: Path):
    firewall = make_firewall(tmp_path)
    firewall.safe_write(r"LAW\CONTRACTS\_runs\_tmp\portable.txt", "ok", kind="tmp")
    assert (tmp_path / "LAW" / "CONTRACTS" / "_runs" / "_tmp" / "portable.txt").read_text() == "ok"


def test_classification_uses_project_relative_domain_not_absolute_tmp_ancestor(tmp_path: Path):
    project_root = tmp_path / "outer_tmp_name" / "repo"
    firewall = make_firewall(project_root)
    durable = project_root / "LAW" / "CONTRACTS" / "_runs" / "receipt.json"
    assert firewall.classify_path(durable) == "durable"


def test_durable_unlink_requires_commit_gate(tmp_path: Path):
    firewall = make_firewall(tmp_path)
    target = tmp_path / "LAW" / "CONTRACTS" / "_runs" / "durable.txt"
    target.write_text("durable")

    with pytest.raises(FirewallViolation) as exc_info:
        firewall.safe_unlink(target)
    assert exc_info.value.error_code == "FIREWALL_DURABLE_WRITE_BEFORE_COMMIT"
    assert target.exists()

    firewall.open_commit_gate()
    firewall.safe_unlink(target)
    assert not target.exists()
