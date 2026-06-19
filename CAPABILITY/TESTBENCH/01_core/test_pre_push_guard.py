import json
from pathlib import Path

import pytest

from CAPABILITY.TOOLS.utilities.pre_push_guard import (
    ZERO_SHA,
    PushRef,
    introduces_commits,
    load_receipt,
    load_receipt_head,
    parse_push_refs,
    validate_push,
)


def ref(
    local_ref: str,
    local_sha: str,
    remote_ref: str = "refs/heads/main",
    remote_sha: str = ZERO_SHA,
) -> PushRef:
    return PushRef(local_ref, local_sha, remote_ref, remote_sha)


def valid_receipt(
    head: str = "f" * 40,
    *,
    base_sha: str | None = "b" * 40,
    mode: str = "full",
) -> dict:
    if mode == "exhaustive":
        suites = ["exhaustive"]
        risk_groups = []
    else:
        suites = ["core", "embeddings"]
        risk_groups = ["embeddings"]
    return {
        "type": "CI_OK",
        "head": head,
        "base_ref": "origin/main" if base_sha else None,
        "base_sha": base_sha,
        "mode": mode,
        "plan_hash": "a" * 64,
        "risk_groups": risk_groups,
        "suites": suites,
        "timestamp": "2026-06-19T00:00:00+00:00",
    }


def test_parse_push_refs_accepts_git_pre_push_format():
    parsed = parse_push_refs(
        "refs/heads/main aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa "
        "refs/heads/main bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\n"
    )
    assert parsed == (
        PushRef(
            "refs/heads/main",
            "a" * 40,
            "refs/heads/main",
            "b" * 40,
        ),
    )


def test_parse_push_refs_rejects_malformed_sha():
    with pytest.raises(ValueError, match="malformed SHA"):
        parse_push_refs("refs/heads/main nope refs/heads/main " + "b" * 40)


def test_empty_and_deletion_only_pushes_introduce_no_commits():
    deletion = ref("(delete)", ZERO_SHA, "refs/heads/obsolete")
    assert not introduces_commits(())
    assert not introduces_commits((deletion,))
    assert validate_push((deletion,), None).allowed


def test_receipt_must_match_the_commit_actually_being_pushed():
    pushed_head = "c" * 40
    decision = validate_push(
        (ref("refs/heads/feature", pushed_head, "refs/heads/feature"),),
        valid_receipt(head="a" * 40, base_sha=None),
    )
    assert not decision.allowed
    assert pushed_head in decision.reason


def test_matching_existing_branch_tip_and_remote_base_are_approved():
    pushed_head = "c" * 40
    remote_base = "b" * 40
    decision = validate_push(
        (ref("refs/heads/main", pushed_head, remote_sha=remote_base),),
        valid_receipt(head=pushed_head, base_sha=remote_base),
    )
    assert decision.allowed


def test_remote_advance_invalidates_receipt_even_when_head_is_unchanged():
    pushed_head = "c" * 40
    decision = validate_push(
        (ref("refs/heads/main", pushed_head, remote_sha="d" * 40),),
        valid_receipt(head=pushed_head, base_sha="b" * 40),
    )
    assert not decision.allowed
    assert "remote is at" in decision.reason


def test_new_ref_requires_tested_base_to_be_ancestor():
    pushed_head = "c" * 40
    receipt = valid_receipt(head=pushed_head, base_sha="b" * 40)
    allowed = validate_push(
        (ref("refs/heads/new", pushed_head, "refs/heads/new"),),
        receipt,
        ancestor_checker=lambda ancestor, descendant: True,
    )
    blocked = validate_push(
        (ref("refs/heads/new", pushed_head, "refs/heads/new"),),
        receipt,
        ancestor_checker=lambda ancestor, descendant: False,
    )
    assert allowed.allowed
    assert not blocked.allowed
    assert "not an ancestor" in blocked.reason


def test_multiple_distinct_tips_or_remote_bases_require_separate_pushes():
    first = ref("refs/heads/one", "1" * 40, "refs/heads/one", "a" * 40)
    second_tip = ref("refs/heads/two", "2" * 40, "refs/heads/two", "a" * 40)
    decision = validate_push((first, second_tip), valid_receipt(head="1" * 40, base_sha="a" * 40))
    assert not decision.allowed
    assert "multiple distinct commit tips" in decision.reason

    second_base = ref("refs/heads/two", "1" * 40, "refs/heads/two", "b" * 40)
    decision = validate_push((first, second_base), valid_receipt(head="1" * 40, base_sha="a" * 40))
    assert not decision.allowed
    assert "multiple remote bases" in decision.reason


def test_annotated_tag_uses_resolved_commit():
    tag_object = "d" * 40
    tagged_commit = "e" * 40
    tag = ref("refs/tags/v1.2.3", tag_object, "refs/tags/v1.2.3")
    decision = validate_push(
        (tag,),
        valid_receipt(head=tagged_commit, base_sha=None),
        resolver=lambda _: tagged_commit,
    )
    assert decision.allowed


def test_load_receipt_head_rejects_legacy_and_accepts_complete_json(tmp_path: Path):
    token = tmp_path / "ALLOW_PUSH.token"
    token.write_text("CI_OK\n", encoding="utf-8")
    assert load_receipt_head(token) is None

    token.write_text(json.dumps(valid_receipt()) + "\n", encoding="utf-8")
    assert load_receipt_head(token) == "f" * 40
    assert load_receipt(token) == valid_receipt()


def test_load_receipt_rejects_incomplete_or_inconsistent_schema(tmp_path: Path):
    token = tmp_path / "ALLOW_PUSH.token"
    cases = [
        {"type": "CI_OK", "head": "f" * 40},
        {**valid_receipt(), "head": "not-a-sha"},
        {**valid_receipt(), "base_sha": "short"},
        {**valid_receipt(), "plan_hash": "short"},
        {**valid_receipt(), "mode": "fast"},
        {**valid_receipt(), "suites": []},
        {**valid_receipt(), "suites": ["core", "core"]},
        {**valid_receipt(), "risk_groups": ["embeddings", "embeddings"]},
        {**valid_receipt(), "risk_groups": ["cassette-network"]},
        {**valid_receipt(), "timestamp": "not-a-time"},
        {**valid_receipt(mode="exhaustive"), "risk_groups": ["embeddings"]},
        {**valid_receipt(mode="exhaustive"), "suites": ["core", "exhaustive"]},
    ]
    for payload in cases:
        token.write_text(json.dumps(payload), encoding="utf-8")
        assert load_receipt(token) is None
