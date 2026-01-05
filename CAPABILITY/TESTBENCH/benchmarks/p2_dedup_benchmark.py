#!/usr/bin/env python3
"""
P.2.5: Deduplication Benchmark for CAS-backed Packer

Measures deduplication savings when using CAS-addressed manifests vs.
full file bodies in packs.

Outputs:
- JSON fixture: raw metrics for reproducibility
- Markdown report: human-readable summary

Usage:
    python CAPABILITY/TESTBENCH/benchmarks/p2_dedup_benchmark.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from MEMORY.LLM_PACKER.Engine.packer import core as packer_core
from CAPABILITY.CAS import cas as cas_mod


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    scope_key: str
    full_pack_bytes: int
    lite_manifest_bytes: int
    cas_unique_blobs: int
    cas_total_bytes: int
    dedup_savings_pct: float
    pack_generation_time_ms: float
    cas_dedup_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "scope_key": self.scope_key,
            "full_pack_bytes": self.full_pack_bytes,
            "lite_manifest_bytes": self.lite_manifest_bytes,
            "cas_unique_blobs": self.cas_unique_blobs,
            "cas_total_bytes": self.cas_total_bytes,
            "dedup_savings_pct": round(self.dedup_savings_pct, 2),
            "pack_generation_time_ms": round(self.pack_generation_time_ms, 2),
            "cas_dedup_count": self.cas_dedup_count,
        }


def _count_cas_blobs(cas_root: Path) -> tuple[int, int]:
    """
    Count CAS blobs and total bytes.
    
    Returns: (blob_count, total_bytes)
    """
    if not cas_root.exists():
        return 0, 0
    
    blob_count = 0
    total_bytes = 0
    
    for prefix1_dir in cas_root.iterdir():
        if not prefix1_dir.is_dir():
            continue
        for prefix2_dir in prefix1_dir.iterdir():
            if not prefix2_dir.is_dir():
                continue
            for blob_file in prefix2_dir.iterdir():
                if blob_file.is_file() and not blob_file.name.endswith('.tmp'):
                    blob_count += 1
                    total_bytes += blob_file.stat().st_size
    
    return blob_count, total_bytes


def _measure_directory_size(dir_path: Path) -> int:
    """Recursively measure total bytes in directory."""
    total = 0
    for item in dir_path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def run_benchmark(
    scope: packer_core.PackScope,
    tmp_dir: Path,
    *,
    max_total_bytes: int = 50 * 1024 * 1024,
) -> BenchmarkResult:
    """
    Run deduplication benchmark for a given scope.
    
    Measures:
    - Full pack size (with file bodies)
    - LITE manifest size (CAS refs only)
    - CAS storage efficiency
    - Time to generate pack
    """
    cas_root = tmp_dir / "cas"
    runs_dir = tmp_dir / "runs"
    cas_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporarily override CAS root
    import CAPABILITY.CAS.cas as cas_module
    old_cas_root = cas_module._CAS_ROOT
    cas_module._CAS_ROOT = cas_root
    
    try:
        # Measure pack generation time
        start_time = time.perf_counter()
        
        pack_dir = packer_core.make_pack(
            scope_key=scope.key,
            mode="full",
            profile="full",
            split_lite=True,
            out_dir=tmp_dir / "pack",
            combined=False,
            stamp="benchmark",
            zip_enabled=False,
            max_total_bytes=max_total_bytes,
            max_entry_bytes=10 * 1024 * 1024,
            max_entries=50_000,
            allow_duplicate_hashes=True,
            p2_runs_dir=runs_dir,
            p2_cas_root=cas_root,
        )
        
        end_time = time.perf_counter()
        pack_generation_time_ms = (end_time - start_time) * 1000
        
        # Measure full pack size (repo/ directory contains actual file bodies)
        full_pack_bytes = _measure_directory_size(pack_dir / "repo")
        
        # Measure LITE manifest size
        lite_dir = pack_dir / "LITE"
        lite_manifest_bytes = _measure_directory_size(lite_dir)
        
        # Count CAS blobs
        cas_unique_blobs, cas_total_bytes = _count_cas_blobs(cas_root)
        
        # Read manifest to count deduplicated entries
        manifest_path = lite_dir / "PACK_MANIFEST.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        
        # Count unique refs (deduplicated by CAS)
        unique_refs = set()
        for entry in manifest["entries"]:
            unique_refs.add(entry["ref"])
        
        cas_dedup_count = len(manifest["entries"]) - len(unique_refs)
        dedup_savings_pct = ((full_pack_bytes - lite_manifest_bytes) / full_pack_bytes * 100) if full_pack_bytes > 0 else 0.0
        
        return BenchmarkResult(
            scope_key=scope.key,
            full_pack_bytes=full_pack_bytes,
            lite_manifest_bytes=lite_manifest_bytes,
            cas_unique_blobs=cas_unique_blobs,
            cas_total_bytes=cas_total_bytes,
            dedup_savings_pct=dedup_savings_pct,
            pack_generation_time_ms=pack_generation_time_ms,
            cas_dedup_count=cas_dedup_count,
        )
    finally:
        cas_module._CAS_ROOT = old_cas_root


def generate_report(results: List[BenchmarkResult], output_dir: Path) -> None:
    """Generate JSON fixture and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON fixture (reproducible, machine-readable)
    fixture_data = {
        "version": "P2.5.0",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "results": [r.to_dict() for r in results],
    }
    
    fixture_path = output_dir / "dedup_benchmark_fixture.json"
    fixture_path.write_text(json.dumps(fixture_data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    
    # Markdown report (human-readable)
    report_lines = [
        "# P.2.5: Deduplication Benchmark Report",
        "",
        f"**Generated**: {fixture_data['timestamp']}",
        "",
        "## Summary",
        "",
        "| Scope | Full Pack (MB) | LITE Manifest (KB) | Savings (%) | CAS Blobs | CAS Total (MB) | Dedup Count | Gen Time (ms) |",
        "|-------|----------------|-------------------|-------------|-----------|----------------|-------------|---------------|",
    ]
    
    for result in results:
        report_lines.append(
            f"| {result.scope_key} "
            f"| {result.full_pack_bytes / (1024 * 1024):.2f} "
            f"| {result.lite_manifest_bytes / 1024:.2f} "
            f"| {result.dedup_savings_pct:.2f} "
            f"| {result.cas_unique_blobs} "
            f"| {result.cas_total_bytes / (1024 * 1024):.2f} "
            f"| {result.cas_dedup_count} "
            f"| {result.pack_generation_time_ms:.2f} |"
        )
    
    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "- **Full Pack**: Total size of `repo/` directory with all file bodies included",
        "- **LITE Manifest**: Total size of `LITE/` directory with only CAS references",
        "- **Savings**: Percentage reduction in size when using LITE manifests vs full packs",
        "- **CAS Blobs**: Number of unique content-addressed blobs stored in CAS",
        "- **CAS Total**: Total bytes stored in CAS (deduplicated storage)",
        "- **Dedup Count**: Number of files deduplicated by CAS (same content, different paths)",
        "- **Gen Time**: Time to generate pack in milliseconds",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python CAPABILITY/TESTBENCH/benchmarks/p2_dedup_benchmark.py",
        "```",
        "",
        f"**Fixture**: `{fixture_path.relative_to(REPO_ROOT)}`",
        "",
    ])
    
    report_path = output_dir / "DEDUP_BENCHMARK_REPORT.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"✓ Benchmark complete")
    print(f"  Fixture: {fixture_path}")
    print(f"  Report:  {report_path}")


def main() -> None:
    """Main benchmark entry point."""
    print("Running P.2.5 Deduplication Benchmark...")
    
    # Create temporary directory for benchmark (must be under _packs/)
    tmp_dir = REPO_ROOT / "MEMORY" / "LLM_PACKER" / "_packs" / "_system" / "benchmark_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    results: List[BenchmarkResult] = []
    
    # Benchmark AGS scope
    print("  Benchmarking AGS scope...")
    ags_tmp = tmp_dir / "ags"
    ags_tmp.mkdir(exist_ok=True)
    ags_result = run_benchmark(packer_core.SCOPE_AGS, ags_tmp)
    results.append(ags_result)
    print(f"    Full pack: {ags_result.full_pack_bytes / (1024 * 1024):.2f} MB")
    print(f"    LITE manifest: {ags_result.lite_manifest_bytes / 1024:.2f} KB")
    print(f"    Savings: {ags_result.dedup_savings_pct:.2f}%")
    
    # Generate reports
    output_dir = REPO_ROOT / "MEMORY" / "LLM_PACKER" / "_packs" / "_system" / "benchmarks"
    generate_report(results, output_dir)


if __name__ == "__main__":
    main()
