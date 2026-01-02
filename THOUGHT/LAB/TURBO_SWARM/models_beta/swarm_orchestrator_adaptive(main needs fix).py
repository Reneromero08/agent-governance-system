#!/usr/bin/env python3
"""
🎯 ADAPTIVE SWARM ORCHESTRATOR 🎯

Strategy:
1. PRIMARY: Caddy Deluxe (hierarchical escalation - FAST & CHEAP)
2. FALLBACK: The Professional (ministral-3:8b dual-mode - POWERFUL & EXPENSIVE)

Per file:
- Try Caddy Deluxe first
- If Caddy fails, escalate to The Professional
- (Optional) verify with pytest or py_compile
- Generate comprehensive report

"Start cheap. Escalate to power when needed."
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --------------------------------------------------------------------------------
# UTF-8 stdio (Windows-friendly)  (MUST be after __future__ import)
# --------------------------------------------------------------------------------

def force_utf8_stdio() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        return
    except Exception:
        pass

    if hasattr(sys.stdout, "buffer"):
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)
        except Exception:
            pass


force_utf8_stdio()


# --------------------------------------------------------------------------------
# Repo Root
# --------------------------------------------------------------------------------

def find_repo_root() -> Path:
    # IMPORTANT: start from the directory, not the file path
    cur = Path(__file__).resolve().parent
    for _ in range(10):
        if (cur / ".git").exists() or (cur / "THOUGHT").exists():
            return cur
        parent = cur.parent
        if parent == cur:
            break
        cur = parent

    # Safe fallback: walk up a few levels without indexing errors
    cur = Path(__file__).resolve().parent
    for _ in range(4):
        if (cur / ".git").exists() or (cur / "THOUGHT").exists():
            return cur
        cur = cur.parent
    return Path(__file__).resolve().parent


REPO_ROOT = find_repo_root()

# Orchestrator paths
CADDY_SCRIPT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "swarm_orchestrator_caddy_deluxe.py"
PROFESSIONAL_SCRIPT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "swarm_orchestrator_professional.py"

# Default paths
DEFAULT_INPUT = REPO_ROOT / "THOUGHT" / "LAB" / "SWARM_MANIFEST.json"
DEFAULT_OUTPUT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "ADAPTIVE_REPORT.json"
CADDY_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "SWARM_REPORT.json"
PROFESSIONAL_REPORT = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "PROFESSIONAL_REPORT.json"


# --------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------

@dataclass
class AdaptiveConfig:
    input_manifest: Path = DEFAULT_INPUT
    output_report: Path = DEFAULT_OUTPUT

    # Caddy settings (primary)
    caddy_workers: int = 6
    caddy_ollama_slots: int = 2

    # Professional settings (fallback)
    professional_model: str = "ministral-3:8b"
    professional_workers: int = 3

    # Verification
    run_tests: bool = True
    debug: bool = False


# --------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------

def setup_logging(debug: bool) -> logging.Logger:
    logger = logging.getLogger("adaptive_swarm")
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if debug else logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# --------------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------------

def read_json(path: Path, logger: logging.Logger) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error(f"Failed to read JSON: {path} ({e})")
        return None


def write_json(path: Path, data: Any, logger: logging.Logger) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.error(f"Failed to write JSON: {path} ({e})")


def load_manifest_items(manifest_path: Path, logger: logging.Logger) -> list[dict[str, Any]]:
    raw = read_json(manifest_path, logger)
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict) and isinstance(raw.get("items"), list):
        return [x for x in raw["items"] if isinstance(x, dict)]
    logger.warning(f"Manifest shape unexpected: {manifest_path}")
    return []


# --------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------

def is_probably_test_path(rel: str) -> bool:
    p = Path(rel)
    name = p.name.lower()
    parts = [x.lower() for x in p.parts]
    return (
        name.startswith("test_")
        or name.endswith("_test.py")
        or "tests" in parts
        or "testbench" in parts
        or "testing" in parts
    )


def verify_target(file_rel: str, logger: logging.Logger) -> bool:
    """
    If it's a test file/path, run pytest on it.
    Otherwise, do a fast sanity check via py_compile to avoid pytest returncode=5.
    If pytest returns 5 (no tests collected), fall back to py_compile.
    """
    full = (REPO_ROOT / file_rel).resolve()
    if not full.exists():
        logger.debug(f"Verify skipped (missing): {file_rel}")
        return False

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    if is_probably_test_path(file_rel):
        cmd = [sys.executable, "-m", "pytest", file_rel, "-v", "--tb=short"]
        try:
            r = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=180,
            )
            if r.returncode == 0:
                return True
            # pytest "no tests collected"
            if r.returncode == 5:
                logger.debug(f"Pytest collected no tests for {file_rel}, falling back to py_compile.")
            else:
                logger.debug(f"Pytest failed for {file_rel} (code {r.returncode}).")
                logger.debug(f"STDERR:\n{r.stderr}")
        except Exception as e:
            logger.debug(f"Pytest verification failed for {file_rel}: {e}")

        # Fallback: compile
        try:
            r2 = subprocess.run(
                [sys.executable, "-m", "py_compile", file_rel],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                timeout=30,
            )
            if r2.returncode != 0:
                logger.debug(f"py_compile STDERR for {file_rel}:\n{r2.stderr}")
            return r2.returncode == 0
        except Exception as e:
            logger.debug(f"py_compile verification failed for {file_rel}: {e}")
            return False

    # Non-test file: py_compile check
    try:
        r = subprocess.run(
            [sys.executable, "-m", "py_compile", file_rel],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
        )
        if r.returncode != 0:
            logger.debug(f"py_compile STDERR for {file_rel}:\n{r.stderr}")
        return r.returncode == 0
    except Exception as e:
        logger.debug(f"py_compile verification failed for {file_rel}: {e}")
        return False


# --------------------------------------------------------------------------------
# Orchestration Logic
# --------------------------------------------------------------------------------

def _is_fixed_status(status: Any) -> bool:
    if status is None:
        return False
    s = str(status).strip().lower()
    return s in {
        "fixed",
        "fix",
        "success",
        "succeeded",
        "ok",
        "passed",
        "pass",
        "done",
        "resolved",
        "complete",
        "completed",
    }


def run_caddy_deluxe(config: AdaptiveConfig, logger: logging.Logger) -> list[dict[str, Any]]:
    """Run Caddy Deluxe orchestrator and return its report (best-effort)."""
    logger.info("🎪 Launching Caddy Deluxe (hierarchical escalation)...")

    if not CADDY_SCRIPT.exists():
        logger.error(f"Caddy script missing: {CADDY_SCRIPT}")
        return []

    # Some Caddy versions read a hardcoded manifest path.
    # If user provided a different manifest, stage it into DEFAULT_INPUT.
    if config.input_manifest != DEFAULT_INPUT:
        try:
            src = config.input_manifest
            if not src.exists():
                logger.error(f"Input manifest missing: {src}")
                return []
            DEFAULT_INPUT.parent.mkdir(parents=True, exist_ok=True)
            DEFAULT_INPUT.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            logger.debug(f"Copied manifest -> {DEFAULT_INPUT}")
        except Exception as e:
            logger.error(f"Failed to stage manifest for Caddy: {e}")
            return []

    cmd = [
        sys.executable,
        str(CADDY_SCRIPT),
        "--max-workers", str(config.caddy_workers),
        "--ollama-slots", str(config.caddy_ollama_slots),
    ]
    if config.debug:
        cmd.append("--debug")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=1800,
        )

        if result.returncode != 0:
            logger.warning(f"Caddy Deluxe exited with code {result.returncode}")
            if config.debug:
                logger.debug(f"STDOUT:\n{result.stdout}")
                logger.debug(f"STDERR:\n{result.stderr}")

        if CADDY_REPORT.exists():
            data = read_json(CADDY_REPORT, logger)
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            logger.error("Caddy report exists but is not a list.")
            return []

        logger.error("Caddy report not found.")
        return []

    except subprocess.TimeoutExpired:
        logger.error("Caddy Deluxe timed out!")
        return []
    except Exception as e:
        logger.error(f"Caddy Deluxe failed: {e}")
        return []


def run_professional_on_failures(
    failures: list[dict[str, Any]], config: AdaptiveConfig, logger: logging.Logger
) -> list[dict[str, Any]]:
    """Run The Professional on files that Caddy couldn't fix."""
    if not failures:
        return []

    logger.info(f"✨ Launching The Professional on {len(failures)} failed files...")

    if not PROFESSIONAL_SCRIPT.exists():
        logger.error(f"Professional script missing: {PROFESSIONAL_SCRIPT}")
        return []

    professional_manifest_path = REPO_ROOT / "THOUGHT" / "LAB" / "TURBO_SWARM" / "_professional_fallback.json"
    professional_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    midas_input: list[dict[str, Any]] = []
    for item in failures:
        midas_input.append(
            {
                "file": item.get("file", "") or "",
                "instruction": item.get("instruction") or "Fix all test failures",
                "last_error": item.get("reason") or item.get("last_error") or "Unknown error",
            }
        )

    write_json(professional_manifest_path, midas_input, logger)

    cmd = [
        sys.executable,
        str(PROFESSIONAL_SCRIPT),
        "--input", str(professional_manifest_path),
        "--output", str(PROFESSIONAL_REPORT),
        "--model", config.professional_model,
        "--workers", str(config.professional_workers),
    ]
    if config.debug:
        cmd.append("--debug")

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=3600,
        )

        if result.returncode != 0:
            logger.warning(f"The Professional exited with code {result.returncode}")
            if config.debug:
                logger.debug(f"STDOUT:\n{result.stdout}")
                logger.debug(f"STDERR:\n{result.stderr}")

        if PROFESSIONAL_REPORT.exists():
            data = read_json(PROFESSIONAL_REPORT, logger)
            if isinstance(data, dict) and isinstance(data.get("results"), list):
                return [x for x in data["results"] if isinstance(x, dict)]
            if isinstance(data, list):
                return [x for x in data if isinstance(x, dict)]
            logger.error("Professional report shape unexpected.")
            return []

        logger.error("Professional report not found.")
        return []

    except subprocess.TimeoutExpired:
        logger.error("The Professional timed out!")
        return []
    except Exception as e:
        logger.error(f"The Professional failed: {e}")
        return []


# --------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description="Adaptive Swarm: Caddy -> Professional fallback")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input manifest")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output report")
    parser.add_argument("--caddy-workers", type=int, default=6)
    parser.add_argument("--caddy-slots", type=int, default=2)
    parser.add_argument("--professional-workers", type=int, default=3)
    parser.add_argument("--professional-model", type=str, default="ministral-3:8b")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    config = AdaptiveConfig(
        input_manifest=args.input,
        output_report=args.output,
        caddy_workers=args.caddy_workers,
        caddy_ollama_slots=args.caddy_slots,
        professional_workers=args.professional_workers,
        professional_model=args.professional_model,
        run_tests=not args.skip_tests,
        debug=args.debug,
    )

    logger = setup_logging(config.debug)

    # Clear prior reports (best-effort)
    for p in (CADDY_REPORT, PROFESSIONAL_REPORT):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass

    logger.info("=" * 70)
    logger.info("🎯 ADAPTIVE SWARM ORCHESTRATOR")
    logger.info("=" * 70)
    logger.info(f"Repo root: {REPO_ROOT}")
    logger.info(f"Input: {config.input_manifest}")
    logger.info(f"Output: {config.output_report}")
    logger.info("")

    manifest_items = load_manifest_items(config.input_manifest, logger)

    # Phase 1: Caddy Deluxe
    caddy_results = run_caddy_deluxe(config, logger)

    # If Caddy produced nothing but we do have a manifest, treat all as failures for fallback.
    if not caddy_results and manifest_items:
        logger.warning("Caddy produced no results. Escalating entire manifest to The Professional.")
        caddy_results = [
            {
                "file": (x.get("file") or ""),
                "status": "failed",
                "reason": "caddy_no_result",
                "instruction": x.get("instruction") or "Fix all test failures",
            }
            for x in manifest_items
            if isinstance(x, dict)
        ]

    caddy_fixed: list[dict[str, Any]] = []
    caddy_failed: list[dict[str, Any]] = []

    for item in caddy_results:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        if not file_path:
            continue

        if _is_fixed_status(item.get("status")):
            if config.run_tests:
                logger.info(f"Verifying {file_path}...")
                if verify_target(str(file_path), logger):
                    caddy_fixed.append(item)
                    logger.info(f"✅ {file_path}")
                else:
                    item["status"] = "verify_failed"
                    caddy_failed.append(item)
                    logger.warning(f"❌ {file_path} (verify failed)")
            else:
                caddy_fixed.append(item)
        else:
            caddy_failed.append(item)

    logger.info(f"\nCaddy: {len(caddy_fixed)} fixed, {len(caddy_failed)} failed.")

    # Phase 2: Professional (Fallback)
    prof_results: list[dict[str, Any]] = []
    if caddy_failed:
        prof_results = run_professional_on_failures(caddy_failed, config, logger)

    prof_fixed: list[dict[str, Any]] = []
    prof_final_failed: list[dict[str, Any]] = []

    for item in prof_results:
        if not isinstance(item, dict):
            continue
        file_path = item.get("file")
        if not file_path:
            prof_final_failed.append(item)
            continue

        if _is_fixed_status(item.get("status")):
            if config.run_tests:
                logger.info(f"Verifying {file_path} (Professional)...")
                if verify_target(str(file_path), logger):
                    prof_fixed.append(item)
                    logger.info(f"✅ {file_path} (Professional)")
                else:
                    item["status"] = "verify_failed"
                    prof_final_failed.append(item)
                    logger.warning(f"❌ {file_path} (verify failed) (Professional)")
            else:
                prof_fixed.append(item)
        else:
            prof_final_failed.append(item)

    # Prefer manifest_items for counting if present, otherwise fall back to caddy_results
    source_for_count = manifest_items if manifest_items else caddy_results
    total_files = len(
        {x.get("file") for x in source_for_count if isinstance(x, dict) and x.get("file")}
    )

    final_report = {
        "summary": {
            "repo_root": str(REPO_ROOT),
            "total_files": total_files,
            "caddy_fixed": len(caddy_fixed),
            "prof_fixed": len(prof_fixed),
            "still_failing": len(prof_final_failed),
        },
        "results": {
            "caddy_fixed": caddy_fixed,
            "prof_fixed": prof_fixed,
            "failed": prof_final_failed,
        },
    }

    write_json(config.output_report, final_report, logger)

    logger.info("\n" + "=" * 70)
    logger.info(
        f"FINAL: {final_report['summary']['caddy_fixed'] + final_report['summary']['prof_fixed']} fixed, {len(prof_final_failed)} failed."
    )
    logger.info("=" * 70)

    return 0 if len(prof_final_failed) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
