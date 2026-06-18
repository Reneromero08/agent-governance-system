from CAPABILITY.TOOLS.utilities.push_test_plan import (
    EMBEDDING_TESTS,
    build_plan,
    normalize_paths,
    plan_payload,
    requires_embeddings,
)


def test_normalize_paths_is_cross_platform_and_deterministic():
    assert normalize_paths(["b.py", r"NAVIGATION\CORTEX\semantic\x.py", "b.py"]) == [
        "NAVIGATION/CORTEX/semantic/x.py",
        "b.py",
    ]


def test_unrelated_change_runs_core_only():
    plan = build_plan(["README.md"])
    assert [suite.name for suite in plan] == ["core"]
    assert all(test not in plan[0].paths for test in EMBEDDING_TESTS)


def test_embedding_source_change_adds_real_embedding_suite():
    paths = ["NAVIGATION/CORTEX/semantic/embeddings.py"]
    assert requires_embeddings(paths)
    plan = build_plan(paths)
    assert [suite.name for suite in plan] == ["core", "embeddings"]
    assert plan[1].paths == EMBEDDING_TESTS


def test_model_registry_change_adds_embedding_suite():
    assert requires_embeddings(["CAPABILITY/PRIMITIVES/model_registry.py"])


def test_exhaustive_is_single_unfiltered_suite():
    plan = build_plan(["README.md"], exhaustive=True)
    assert len(plan) == 1
    assert plan[0].name == "exhaustive"
    assert plan[0].extra_args == ()


def test_plan_hash_is_stable_and_changes_with_scope():
    first = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=0)
    second = plan_payload(["README.md"], base_ref="main", exhaustive=False, workers=0)
    embedding = plan_payload(
        ["NAVIGATION/CORTEX/semantic/embeddings.py"],
        base_ref="main",
        exhaustive=False,
        workers=0,
    )
    assert first["plan_hash"] == second["plan_hash"]
    assert first["plan_hash"] != embedding["plan_hash"]
