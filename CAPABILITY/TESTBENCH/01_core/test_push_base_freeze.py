import CAPABILITY.TOOLS.utilities.ci_local_gate as ci_local_gate


def test_freeze_base_returns_label_sha_and_immutable_diff_ref(monkeypatch):
    monkeypatch.setattr(
        ci_local_gate,
        "resolve_base_ref",
        lambda explicit=None: "origin/main",
    )
    monkeypatch.setattr(
        ci_local_gate,
        "_resolve_base_sha",
        lambda base_ref: "b" * 40,
    )
    assert ci_local_gate._freeze_base(None) == (
        "origin/main",
        "b" * 40,
        "b" * 40,
    )


def test_freeze_base_preserves_initial_history(monkeypatch):
    monkeypatch.setattr(
        ci_local_gate,
        "resolve_base_ref",
        lambda explicit=None: None,
    )
    monkeypatch.setattr(
        ci_local_gate,
        "_resolve_base_sha",
        lambda base_ref: None,
    )
    assert ci_local_gate._freeze_base("0" * 40) == (
        None,
        None,
        "0" * 40,
    )
