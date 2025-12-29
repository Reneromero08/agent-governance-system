# test_ags_plan_router_happy_path
def test_ags_plan_router_happy_path(tmp_path: Path) -> None:
    pipeline_id = "ags-router-happy"
    tmp_root = "ags_router_happy"
    plan_out = tmp_path / "plan.json"

    plan_obj = {
        "plan_version": "1.0",
        "pipeline_id": "ignored-by-override",
        "steps": [
            {
                "step_id": "s1",
                "command": [sys.executable, "-c", "pass"],
                "jobspec": _valid_jobspec(tmp_root=tmp_root),
            }
        ],
    }
    router_code = json.dumps(plan_obj)
    r1 = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            pipeline_id,
        ]
    )
    assert r1.returncode == 0, r1.stdout + r1.stderr
    b1 = plan_out.read_bytes()

    r2 = _run_ags(
        [
            "plan",
            "--router",
            sys.executable,
            "--router-arg=-c",
            f"--router-arg={router_code}",
            "--out",
            str(plan_out),
            "--pipeline-id",
            pipeline_id,
        ]
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
    b2 = plan_out.read_bytes()
    assert b1 == b2

    # Route + run should succeed using the validated plan output.
    pipeline_dir = REPO_ROOT / "LAW" / "_runs" / f"{pipeline_id}"
    try:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"LAW/CORTEX/_generated/_tmp/{tmp_root}_noop.txt")

        rr = _run_ags(["route", "--plan", str(plan_out), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
        assert rr.returncode == 0, rr.stdout + rr.stderr
        run = _run_ags(["run", "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs", "--strict", "--skip-preflight"])
        assert run.returncode == 0, run.stdout + run.stderr
    finally:
        _rm(pipeline_dir)
        _rm(REPO_ROOT / "LAW" / "_runs" / f"pipeline-{pipeline_id}-s1-a1")
        _rm(REPO_ROOT / f"LAW/CORTEX/_generated/_tmp/{tmp_root}_noop.txt")


# test_ags_route_rejects_missing_step_command
def test_ags_route_rejects_missing_step_command(tmp_path: Path) -> None:
    pipeline_id = "ags-router-missing-command"
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps({"steps": [{"step_id": "s1", "jobspec": _valid_jobspec(tmp_root="ags_router_missing_cmd")}]}),
        encoding="utf-8",
    )
    r = _run_ags(["route", "--plan", str(plan_path), "--pipeline-id", pipeline_id, "--runs-root", "LAW/CONTRACTS/_runs"])
    assert r.returncode != 0
    assert "MISSING_STEP_COMMAND" in r.stderr
