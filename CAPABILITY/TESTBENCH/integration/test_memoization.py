import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def test_memoization_miss_then_hit_then_invalidate(tmp_path: Path) -> None:
    # Import runtime from CAPABILITY.TOOLS (repo-root module).
    sys.path.insert(0, str(REPO_ROOT / "CAPABILITY" / "TOOLS" / "catalytic"))
    # sys.path cleanup

    import catalytic_runtime as runtime  # type: ignore
    from CAPABILITY.PRIMITIVES.memo_cache import compute_job_cache_key  # type: ignore

    assert REPO_ROOT == Path(__file__).resolve().parents[3]

    run_id = "memoization-test-run"
    run_dir = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / run_id
    cache_root = REPO_ROOT / "LAW" / "CONTRACTS" / "_runs" / "_cache" / "jobs"

    catalytic_domain = "LAW/CONTRACTS/_runs/_tmp/memoization-domain"
    durable_output = "NAVIGATION/CORTEX/_generated/memoization_output.txt"
    side_effect = "LAW/CONTRACTS/_runs/_tmp/memoization_side_effect.txt"

    # Clean any prior state.
    _rm(run_dir)
    _rm(cache_root)
    _rm(REPO_ROOT / durable_output)
    _rm(REPO_ROOT / side_effect)

    try:
        runtime.run(
            cmd_a=[
                sys.executable,
                "-c",
                (
                    "from pathlib import Path;"
                    f"Path('{durable_output}').write_text('A', encoding='utf-8');"
                    f"Path('{side_effect}').write_text('EXEC_A', encoding='utf-8')"
                ),
            ]
        )
        assert runtime.run(cmd_a) == 0

        proof_bytes_1 = (run_dir / "PROOF.json").read_bytes()
        roots_bytes_1 = (run_dir / "DOMAIN_ROOTS.json").read_bytes()

        input_domain_roots = runtime._compute_input_domain_roots()
        cache_key = compute_job_cache_key(
            jobspec=runtime.build_jobspec(),
            input_domain_roots=input_domain_roots,
            validator_semver=runtime.validator_semver,
            validator_build_id=runtime.validator_build_id,
            strict=runtime.strict,
        )

        cache_dir = cache_root / cache_key
        assert cache_dir.exists()
        assert (cache_dir / "PROOF.json").exists()
        assert (cache_dir / "DOMAIN_ROOTS.json").exists()
        assert (cache_dir / "OUTPUTS" / durable_output).exists()

        # Force visibility of hit by removing output and providing a command that would change it.
        _rm(REPO_ROOT / durable_output)
        _rm(REPO_ROOT / side_effect)

        runtime.run(
            cmd_b=[
                sys.executable,
                "-c",
                (
                    "from pathlib import Path;"
                    f"Path('{durable_output}').write_text('B', encoding='utf-8');"
                    f"Path('{side_effect}').write_text('EXEC_B', encoding='utf-8')"
                ),
        ]
        )
        assert runtime.run(cmd_b) == 0

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

        runtime.run(
            cmd_c=[
                sys.executable,
                "-c",
                (
                    "from pathlib import Path;"
                    f"Path('{durable_output}').write_text('C', encoding='utf-8');"
                    f"Path('{side_effect}').write_text('EXEC_C', encoding='utf-8')"
                ),
        ]
        )
        assert runtime.run(cmd_c) == 0
        assert (REPO_ROOT / durable_output).read_text(encoding="utf-8") == "C"
        assert (REPO_ROOT / side_effect).read_text(encoding="utf-8") == "EXEC_C"
    finally:
        # Cleanup to keep repo workspace clean.
        _rm(run_dir)
        _rm(cache_root)
        _rm(REPO_ROOT / durable_output)
        _rm(REPO_ROOT / side_effect)
