import pytest

from context_pack import pack_context


def test_manual_context_is_ordered_budgeted_and_receipted(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("alpha " * 100, encoding="utf-8")
    second.write_text("beta " * 100, encoding="utf-8")
    packed, manifest = pack_context(
        workspace=str(tmp_path),
        read_roots=[str(tmp_path)],
        context_files=["first.txt", "second.txt"],
        context_texts=["manual tail"],
        token_budget=80,
        tokenizer="cl100k_base",
    )
    assert packed.startswith("[CONTEXT SOURCE: file:")
    assert manifest["included_tokens"] == 80
    assert manifest["sources"][0]["included_tokens"] > 0
    assert any(source["truncated"] for source in manifest["sources"])
    assert manifest["sources"][0]["sha256"]


def test_context_file_must_be_inside_declared_read_scope(tmp_path):
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    with pytest.raises(ValueError, match="escapes read scope"):
        pack_context(
            workspace=str(tmp_path),
            read_roots=[str(allowed)],
            context_files=[str(outside)],
            context_texts=[],
            token_budget=10,
            tokenizer="cl100k_base",
        )


def test_context_requires_explicit_budget(tmp_path):
    with pytest.raises(ValueError, match="token budget"):
        pack_context(
            workspace=str(tmp_path),
            read_roots=[str(tmp_path)],
            context_files=[],
            context_texts=["manual"],
            token_budget=0,
            tokenizer="cl100k_base",
        )
