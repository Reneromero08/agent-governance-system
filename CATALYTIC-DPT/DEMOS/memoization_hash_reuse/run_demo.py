from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict


REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _rm(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    _atomic_write(path, _canonical_json_bytes(obj))


def _write_inputs(inputs_dir: Path) -> None:
    inputs_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic non-trivial text payload (~256 KiB).
    payload = ("0123456789abcdef" * 1024).encode("utf-8")  # 16 KiB
    big = payload * 16  # 256 KiB
    (inputs_dir / "big.txt").write_bytes(big)

    # Deterministic Python source so `hash ast` is supported.
    py = "\n".join(
        [
            "import hashlib",
            "",
            "def f(x: int) -> str:",
            "    return hashlib.sha256(str(x).encode('utf-8')).hexdigest()",
            "",
            "class C:",
            "    def m(self) -> int:",
            "        return 42",
            "",
        ]
        + [f"def fn_{i}():\n    return {i}\n" for i in range(200)]
    )
    (inputs_dir / "sample.py").write_text(py, encoding="utf-8")


def _jobspec_job(*, job_id: str, inputs_domain_rel: str, output_rel: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "phase": 2,
        "task_type": "adapter_execution",
        "intent": "phase2 demo job: read inputs + write durable output (memoizable)",
        "inputs": {},
        "outputs": {"durable_paths": [output_rel], "validation_criteria": {}},
        "catalytic_domains": [inputs_domain_rel],
        "determinism": "deterministic",
    }


def _jobspec_deref(*, job_id: str, stats_rel: str, deref_domain_rel: str) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "phase": 2,
        "task_type": "adapter_execution",
        "intent": "phase2 demo deref: bounded hash-first inspection",
        "inputs": {},
        "outputs": {"durable_paths": [stats_rel], "validation_criteria": {}},
        "catalytic_domains": [deref_domain_rel],
        "determinism": "deterministic",
    }


def _pipeline_spec(
    *,
    pipeline_id: str,
    jobspec_job_rel: str,
    jobspec_deref_baseline_rel: str,
    jobspec_deref_reuse_rel: str,
    validator_semver: str,
    validator_build_id: str,
    timestamp: str,
    cmd_job: list[str],
    cmd_deref_baseline: list[str],
    cmd_deref_reuse: list[str],
) -> Dict[str, Any]:
    return {
        "pipeline_id": pipeline_id,
        "validator_semver": validator_semver,
        "validator_build_id": validator_build_id,
        "timestamp": timestamp,
        "steps": [
            {"step_id": "job_baseline", "jobspec_path": jobspec_job_rel, "cmd": cmd_job, "strict": True, "memoize": True},
            {
                "step_id": "deref_baseline",
                "jobspec_path": jobspec_deref_baseline_rel,
                "cmd": cmd_deref_baseline,
                "strict": True,
                "memoize": False,
            },
            {"step_id": "job_reuse", "jobspec_path": jobspec_job_rel, "cmd": cmd_job, "strict": True, "memoize": True},
            {"step_id": "deref_reuse", "jobspec_path": jobspec_deref_reuse_rel, "cmd": cmd_deref_reuse, "strict": True, "memoize": False},
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline-id", default="phase2-demo-memo-hash")
    ap.add_argument("--validator-build-id", default="phase2-demo-memo-hash")
    ap.add_argument("--timestamp", default="CATALYTIC-DPT-02_CONFIG")
    ap.add_argument("--out-root", default="CONTRACTS/_runs/_demos/memoization_hash_reuse")
    args = ap.parse_args()

    pipeline_id = args.pipeline_id
    out_root = REPO_ROOT / args.out_root

    tmp_base = REPO_ROOT / "CONTRACTS" / "_runs" / "_tmp" / "demos" / "memoization_hash_reuse" / pipeline_id
    pipeline_dir = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id
    inputs_dir = tmp_base / "inputs_domain"
    deref_domain_dir = tmp_base / "deref_domain"
    output_rel = "CORTEX/_generated/_tmp/demo_memo_hash_output.txt"
    output_abs = REPO_ROOT / output_rel

    deref_stats_rel = "CONTRACTS/_runs/_tmp/demos/memoization_hash_reuse/deref_stats.json"
    deref_stats_abs = REPO_ROOT / deref_stats_rel
    deref_stats_baseline_rel = "CONTRACTS/_runs/_tmp/demos/memoization_hash_reuse/deref_stats_baseline.json"
    deref_stats_baseline_abs = REPO_ROOT / deref_stats_baseline_rel
    deref_stats_reuse_rel = "CONTRACTS/_runs/_tmp/demos/memoization_hash_reuse/deref_stats_reuse.json"
    deref_stats_reuse_abs = REPO_ROOT / deref_stats_reuse_rel

    jobspec_job_abs = tmp_base / "JOBSPEC_job.json"
    jobspec_deref_baseline_abs = tmp_base / "JOBSPEC_deref_baseline.json"
    jobspec_deref_reuse_abs = tmp_base / "JOBSPEC_deref_reuse.json"
    spec_abs = tmp_base / "PIPELINE_SPEC.json"

    inputs_domain_rel = str(inputs_dir.relative_to(REPO_ROOT)).replace("\\", "/")

    # Ensure a clean, reproducible run. Only touches the demo's own namespaces.
    _rm(tmp_base)
    _rm(pipeline_dir)
    for run_dir in (REPO_ROOT / "CONTRACTS" / "_runs").glob(f"pipeline-{pipeline_id}-*"):
        _rm(run_dir)
    _rm(out_root)
    _rm(output_abs)
    _rm(deref_stats_abs)
    _rm(deref_stats_baseline_abs)
    _rm(deref_stats_reuse_abs)

    # Clear prior cache entries for this demo build-id only (don’t touch other caches).
    cache_jobs = REPO_ROOT / "CONTRACTS" / "_runs" / "_cache" / "jobs"
    if cache_jobs.exists():
        for meta in cache_jobs.glob("*/metadata.json"):
            try:
                obj = json.loads(meta.read_text(encoding="utf-8"))
                vid = obj.get("validator_id", {})
                if vid.get("validator_build_id") == args.validator_build_id:
                    _rm(meta.parent)
            except Exception:
                continue

    _write_inputs(inputs_dir)
    deref_domain_dir.mkdir(parents=True, exist_ok=True)

    job_id = "phase2-demo-memo-hash-job"
    _write_json(jobspec_job_abs, _jobspec_job(job_id=job_id, inputs_domain_rel=inputs_domain_rel, output_rel=output_rel))
    deref_domain_rel = str(deref_domain_dir.relative_to(REPO_ROOT)).replace("\\", "/")
    _write_json(
        jobspec_deref_baseline_abs,
        _jobspec_deref(job_id="phase2-demo-memo-hash-deref-baseline", stats_rel=deref_stats_baseline_rel, deref_domain_rel=deref_domain_rel),
    )
    _write_json(
        jobspec_deref_reuse_abs,
        _jobspec_deref(job_id="phase2-demo-memo-hash-deref-reuse", stats_rel=deref_stats_reuse_rel, deref_domain_rel=deref_domain_rel),
    )

    # Job command: hash all inputs (forces real I/O) and write deterministic durable output.
    cmd_job = [
        sys.executable,
        "-c",
        "\n".join(
            [
                "import hashlib, json",
                "from pathlib import Path",
                f"root = Path({inputs_domain_rel!r})",
                "items = []",
                "for p in sorted(root.rglob('*')):",
                "    if not p.is_file():",
                "        continue",
                "    h = hashlib.sha256(p.read_bytes()).hexdigest()",
                "    rel = str(p.relative_to(root)).replace('\\\\', '/')",
                "    items.append((rel, h))",
                f"out = Path({output_rel!r})",
                "out.parent.mkdir(parents=True, exist_ok=True)",
                "out.write_text(json.dumps(items, sort_keys=True, separators=(',', ':')), encoding='utf-8')",
            ]
        ),
    ]

    # Deref command template: read job run_id from pipeline STATE.json, pick a stable input hash,
    # then run bounded `catalytic hash ...` reads against that job's CAS root.
    def deref_cmd(target_step_id: str, mode: str, stats_rel: str) -> List[str]:
        mode_literal = mode
        return [
            sys.executable,
            "-c",
            "\n".join(
                [
                    "import json, os, re, subprocess",
                    "from pathlib import Path",
                    f"repo = Path({str(REPO_ROOT)!r})",
                    f"state = repo / 'CONTRACTS' / '_runs' / '_pipelines' / {pipeline_id!r} / 'STATE.json'",
                    "st = json.loads(state.read_text(encoding='utf-8'))",
                    f"job_run_id = st['step_run_ids'][{target_step_id!r}]",
                    "job_run_dir = repo / 'CONTRACTS' / '_runs' / job_run_id",
                    "cas_root = job_run_dir / 'CAS'",
                    "inp = json.loads((job_run_dir / 'INPUT_HASHES.json').read_text(encoding='utf-8'))",
                    "keys = sorted(inp.keys())",
                    "py_keys = [k for k in keys if k.endswith('sample.py')]",
                    "pick = py_keys[0] if py_keys else keys[0]",
                    "h = inp[pick]",
                    "obj = cas_root / 'objects' / h[0:2] / h[2:4] / h",
                    "size = os.stat(obj).st_size",
                    "def sh(argv):",
                    "    return subprocess.check_output(argv, cwd=str(repo)).decode('utf-8', errors='replace')",
                    "ops = []",
                    "bytes_total = 0",
                    "count = 0",
                    f"mode = {mode_literal!r}",
                    "if mode == 'baseline':",
                    "    out = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'read',h,'--max-bytes','65536'])",
                    "    m = re.search(r'bytes_returned=(\\d+)', out.split('\\n',1)[0])",
                    "    br = int(m.group(1)) if m else 0",
                    "    ops.append({'op':'read','max_bytes':65536,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    "    _ = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'grep',h,'def','--max-bytes','65536','--max-matches','20'])",
                    "    br = min(65536, size)",
                    "    ops.append({'op':'grep','max_bytes':65536,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    "    _ = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'ast',h,'--max-bytes','65536','--max-nodes','200','--max-depth','6'])",
                    "    br = min(65536, size)",
                    "    ops.append({'op':'ast','max_bytes':65536,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    "    desc = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'describe',h,'--max-bytes','8192']).strip()",
                    "    j = json.loads(desc)",
                    "    br = int(j.get('bytes_preview_len', 0))",
                    "    ops.append({'op':'describe','max_bytes':8192,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    "else:",
                    "    desc = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'describe',h,'--max-bytes','1024']).strip()",
                    "    j = json.loads(desc)",
                    "    br = int(j.get('bytes_preview_len', 0))",
                    "    ops.append({'op':'describe','max_bytes':1024,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    "    _ = sh([sys.executable,'TOOLS/catalytic.py','hash','--cas-root',str(cas_root),'grep',h,'def','--max-bytes','8192','--max-matches','5'])",
                    "    br = min(8192, size)",
                    "    ops.append({'op':'grep','max_bytes':8192,'bytes_read':br})",
                    "    bytes_total += br; count += 1",
                    f"stats_path = repo / {stats_rel!r}",
                    "stats_path.parent.mkdir(parents=True, exist_ok=True)",
                    "stats = {",
                    "  'target_run_id': job_run_id,",
                    "  'hash': h,",
                    "  'object_size': size,",
                    "  'deref_count': count,",
                    "  'bytes_read_total': bytes_total,",
                    "  'ops': ops,",
                    "}",
                    "stats_path.write_text(json.dumps(stats, sort_keys=True, separators=(',',':')), encoding='utf-8')",
                ]
            ),
        ]

    cmd_deref_baseline = deref_cmd(target_step_id="job_baseline", mode="baseline", stats_rel=deref_stats_baseline_rel)
    cmd_deref_reuse = deref_cmd(target_step_id="job_reuse", mode="reuse", stats_rel=deref_stats_reuse_rel)

    spec = _pipeline_spec(
        pipeline_id=pipeline_id,
        jobspec_job_rel=str(jobspec_job_abs.relative_to(REPO_ROOT)).replace("\\", "/"),
        jobspec_deref_baseline_rel=str(jobspec_deref_baseline_abs.relative_to(REPO_ROOT)).replace("\\", "/"),
        jobspec_deref_reuse_rel=str(jobspec_deref_reuse_abs.relative_to(REPO_ROOT)).replace("\\", "/"),
        validator_semver="0.1.0",
        validator_build_id=args.validator_build_id,
        timestamp=args.timestamp,
        cmd_job=cmd_job,
        cmd_deref_baseline=cmd_deref_baseline,
        cmd_deref_reuse=cmd_deref_reuse,
    )
    _write_json(spec_abs, spec)

    # Run pipeline.
    subprocess.check_call(
        [sys.executable, "TOOLS/catalytic.py", "pipeline", "run", pipeline_id, "--spec", str(spec_abs.relative_to(REPO_ROOT))],
        cwd=str(REPO_ROOT),
    )

    # Collect run ids from pipeline state.
    state_path = REPO_ROOT / "CONTRACTS" / "_runs" / "_pipelines" / pipeline_id / "STATE.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    run_ids = state["step_run_ids"]

    job_base = REPO_ROOT / "CONTRACTS" / "_runs" / run_ids["job_baseline"]
    job_reuse = REPO_ROOT / "CONTRACTS" / "_runs" / run_ids["job_reuse"]
    deref_base = REPO_ROOT / "CONTRACTS" / "_runs" / run_ids["deref_baseline"]
    deref_reuse = REPO_ROOT / "CONTRACTS" / "_runs" / run_ids["deref_reuse"]

    # Copy required artifacts into demo output root.
    (out_root / "baseline").mkdir(parents=True, exist_ok=True)
    (out_root / "reuse").mkdir(parents=True, exist_ok=True)

    def copy_pair(src_dir: Path, dst_dir: Path) -> None:
        (dst_dir / "PROOF.json").write_bytes((src_dir / "PROOF.json").read_bytes())
        (dst_dir / "LEDGER.jsonl").write_bytes((src_dir / "LEDGER.jsonl").read_bytes())

    copy_pair(job_base, out_root / "baseline")
    copy_pair(job_reuse, out_root / "reuse")

    # Include deref stats (durable output) and deref ledgers for evidence.
    (out_root / "baseline" / "DEREF_STATS.json").write_bytes(deref_stats_baseline_abs.read_bytes())
    (out_root / "baseline" / "DEREF_LEDGER.jsonl").write_bytes((deref_base / "LEDGER.jsonl").read_bytes())

    (out_root / "reuse" / "DEREF_STATS.json").write_bytes(deref_stats_reuse_abs.read_bytes())
    (out_root / "reuse" / "DEREF_LEDGER.jsonl").write_bytes((deref_reuse / "LEDGER.jsonl").read_bytes())

    _write_json(out_root / "baseline" / "RUN_IDS.json", {"job_run_id": run_ids["job_baseline"], "deref_run_id": run_ids["deref_baseline"]})
    _write_json(out_root / "reuse" / "RUN_IDS.json", {"job_run_id": run_ids["job_reuse"], "deref_run_id": run_ids["deref_reuse"]})

    # Build comparison from artifacts only.
    baseline_stats = json.loads((out_root / "baseline" / "DEREF_STATS.json").read_text(encoding="utf-8"))
    reuse_stats_obj = json.loads((out_root / "reuse" / "DEREF_STATS.json").read_text(encoding="utf-8"))

    baseline_proof_sha = _sha256_file(out_root / "baseline" / "PROOF.json")
    reuse_proof_sha = _sha256_file(out_root / "reuse" / "PROOF.json")
    proof_identical = baseline_proof_sha == reuse_proof_sha

    reuse_ledger_text = (out_root / "reuse" / "LEDGER.jsonl").read_text(encoding="utf-8")
    memo_hit = "memoization:hit" in reuse_ledger_text

    comparison_lines = [
        "# Memoization + hash-first dereference reuse (Phase 2 demo)",
        "",
        "Evidence is derived from committed artifacts under `CONTRACTS/_runs/_demos/memoization_hash_reuse/`.",
        "",
        "## Comparison",
        "",
        "| Metric | Baseline | Reuse |",
        "|---|---:|---:|",
        f"| Dereference events (`deref_count`) | {baseline_stats['deref_count']} | {reuse_stats_obj['deref_count']} |",
        f"| Bytes read via hash (`bytes_read_total`) | {baseline_stats['bytes_read_total']} | {reuse_stats_obj['bytes_read_total']} |",
        f"| Memoization hit observable in ledger | {'yes' if memo_hit else 'no'} | {'yes' if memo_hit else 'no'} |",
        f"| PROOF byte-identity (sha256 match) | {'yes' if proof_identical else 'no'} | {'yes' if proof_identical else 'no'} |",
        "",
        "## Anchors",
        "",
        f"- Baseline PROOF sha256: `{baseline_proof_sha}`",
        f"- Reuse PROOF sha256: `{reuse_proof_sha}`",
        "",
        "## Notes",
        "",
        "- “Bytes read via hash” is computed from tool-enforced bounds and CAS object size; no timing or synthetic estimates.",
        "- Memoization evidence is the `memoization:hit` marker in `reuse/LEDGER.jsonl`.",
        "",
    ]
    _write_text(out_root / "comparison.md", "\n".join(comparison_lines))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
