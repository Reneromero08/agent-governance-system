from __future__ import annotations

import shutil
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_memoization_miss_then_hit_then_invalidate(tmp_path: Path) -> None:
    # Import runtime from TOOLS (repo-root module).
    import sys

    sys.path.insert(0, str(REPO_ROOT / "TOOLS"))
    sys.path.insert(0, str(REPO_ROOT / "CATALYTIC-DPT"))

    from catalytic_runtime import CatalyticRuntime, PROJECT_ROOT  # type: ignore
    from PRIMITIVES.memo_cache import compute_job_cache_key  # type: ignore

    assert PROJECT_ROOT == REPO_ROOT

    run_id = "memoization-test-run"
    run_dir = REPO_ROOT / "CONTRACTS" / "_runs" / run_id
    cache_root = REPO_ROOT / "CONTRACTS" / "_runs" / "_cache" / "jobs"

    catalytic_domain = "CONTRACTS/_runs/_tmp/memoization-domain"
    durable_output = "CORTEX/_generated/memoization_output.txt"
    side_effect = "CONTRACTS/_runs/_tmp/memoization_side_effect.txt"

    # Clean any prior state.
    _rm(run_dir)
    _rm(REPO_ROOT / durable_output)
    _rm(REPO_ROOT / side_effect)

    try:
        runtime1 = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[catalytic_domain],
            durable_outputs=[durable_output],
            intent="memoization test",
            strict=True,
            memoize=True,
            validator_semver="0.1.0",
            validator_build_id="memo-test",
        )

        cmd_a = [
            "python3",
            "-c",
            (
                "from pathlib import Path;"
                f"Path('{durable_output}').write_text('A', encoding='utf-8');"
                f"Path('{side_effect}').write_text('EXEC_A', encoding='utf-8')"
            ),
        ]
        assert runtime1.run(cmd_a) == 0

        proof_bytes_1 = (run_dir / "PROOF.json").read_bytes()
        roots_bytes_1 = (run_dir / "DOMAIN_ROOTS.json").read_bytes()

        input_domain_roots = runtime1._compute_input_domain_roots()
        cache_key = compute_job_cache_key(
            jobspec=runtime1.build_jobspec(),
            input_domain_roots=input_domain_roots,
            validator_semver=runtime1.validator_semver,
            validator_build_id=runtime1.validator_build_id,
            strict=runtime1.strict,
        )

        cache_dir = cache_root / cache_key
        assert cache_dir.exists()
        assert (cache_dir / "PROOF.json").exists()
        assert (cache_dir / "DOMAIN_ROOTS.json").exists()
        assert (cache_dir / "OUTPUTS" / durable_output).exists()

        # Force visibility of hit by removing output and providing a command that would change it.
        _rm(REPO_ROOT / durable_output)
        _rm(REPO_ROOT / side_effect)

        runtime2 = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[catalytic_domain],
            durable_outputs=[durable_output],
            intent="memoization test",
            strict=True,
            memoize=True,
            validator_semver="0.1.0",
            validator_build_id="memo-test",
        )
        cmd_b = [
            "python3",
            "-c",
            (
                "from pathlib import Path;"
                f"Path('{durable_output}').write_text('B', encoding='utf-8');"
                f"Path('{side_effect}').write_text('EXEC_B', encoding='utf-8')"
            ),
        ]
        assert runtime2.run(cmd_b) == 0

        # On cache hit, command must not execute: output restored to cached A, side-effect file absent.
        assert (REPO_ROOT / durable_output).read_text(encoding="utf-8") == "A"
        assert not (REPO_ROOT / side_effect).exists()

        # Proof artifacts must be byte-identical on hit.
        assert (run_dir / "PROOF.json").read_bytes() == proof_bytes_1
        assert (run_dir / "DOMAIN_ROOTS.json").read_bytes() == roots_bytes_1

        # Ledger hit must be observable deterministically.
        ledger_text = (run_dir / "LEDGER.jsonl").read_text(encoding="utf-8")
        assert f"memoization:hit key={cache_key}" in ledger_text

        # Invalidate cache by changing strictness mode.
        _rm(REPO_ROOT / durable_output)
        _rm(REPO_ROOT / side_effect)

        runtime3 = CatalyticRuntime(
            run_id=run_id,
            catalytic_domains=[catalytic_domain],
            durable_outputs=[durable_output],
            intent="memoization test",
            strict=False,
            memoize=True,
            validator_semver="0.1.0",
            validator_build_id="memo-test",
        )
        cmd_c = [
            "python3",
            "-c",
            (
                "from pathlib import Path;"
                f"Path('{durable_output}').write_text('C', encoding='utf-8');"
                f"Path('{side_effect}').write_text('EXEC_C', encoding='utf-8')"
            ),
        ]
        assert runtime3.run(cmd_c) == 0
        assert (REPO_ROOT / durable_output).read_text(encoding="utf-8") == "C"
        assert (REPO_ROOT / side_effect).read_text(encoding="utf-8") == "EXEC_C"
    finally:
        # Cleanup to keep repo workspace clean.
        _rm(run_dir)
        _rm(REPO_ROOT / durable_output)
        _rm(REPO_ROOT / side_effect)
