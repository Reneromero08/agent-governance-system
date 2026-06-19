from pathlib import Path

from CAPABILITY.TOOLS.utilities.pre_push_guard import (
    ZERO_SHA,
    PushRef,
    introduces_commits,
    load_receipt_head,
    parse_push_refs,
    validate_push,
)


def ref(local_ref: str, local_sha: str, remote_ref: str = "refs/heads/main") -> PushRef:
    return PushRef(local_ref, local_sha, remote_ref, ZERO_SHA)


def test_parse_push_refs_accepts_git_pre_push_format():
    parsed = parse_push_refs(
        "refs/heads/main aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "refs/heads/main bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
    )
    assert parsed == (
        PushRef(
            "refs/heads/main",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "refs/heads/main",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        ),
    )


def test_empty_and_deletion_only_pushes_introduce_no_commits():
    deletion = ref("(delete)", ZERO_SHA, "refs/heads/obsolete")
    assert not introduces_commits(())
    assert not introduces_commits((deletion,))
    assert validate_push((deletion,), None).allowed


def test_receipt_must_match_the_commit_actually_being_pushed():
    checked_out_head = "a" * 40
    pushed_head = "b" * 40
    decision = validate_push(
        (ref("refs/heads/feature", pushed_head, "refs/heads/feature"),),
        checked_out_head,
    )
    assert not decision.allowed
    assert pushed_head in decision.reason


def test_matching_branch_tip_is_approved():
    pushed_head = "c" * 40
    decision = validate_push(
        (ref("refs/heads/main", pushed_head),),
        pushed_head,
    )
    assert decision.allowed


def test_multiple_distinct_tips_require_separate_pushes():
    first = ref("refs/heads/one", "1" * 40, "refs/heads/one")
    second = ref("refs/heads/two", "2" * 40, "refs/heads/two")
    decision = validate_push((first, second), "1" * 40)
    assert not decision.allowed
    assert "multiple distinct commit tips" in decision.reason


def test_annotated_tag_uses_resolved_commit():
    tag_object = "d" * 40
    tagged_commit = "e" * 40
    tag = ref("refs/tags/v1.2.3", tag_object, "refs/tags/v1.2.3")
    decision = validate_push(
        (tag,),
        tagged_commit,
        resolver=lambda _: tagged_commit,
    )
    assert decision.allowed


def test_load_receipt_head_rejects_legacy_and_accepts_json(tmp_path: Path):
    token = tmp_path / "ALLOW_PUSH.token"
    token.write_text("CI_OK\n", encoding="utf-8")
    assert load_receipt_head(token) is None

    token.write_text(
        '{"type":"CI_OK","head":"ffffffffffffffffffffffffffffffffffffffff"}\n',
        encoding="utf-8",
    )
    assert load_receipt_head(token) == "f" * 40
