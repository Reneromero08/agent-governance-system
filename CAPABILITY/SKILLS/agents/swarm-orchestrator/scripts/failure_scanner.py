#!/usr/bin/env python3
import sys, os, json, time
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

def log_event(agent_id, event_type, data):
    print(f"[{agent_id}] {event_type}: {json.dumps(data)}")
    sys.stdout.flush()

def main():
    agent_id = "failure-scanner"
    print(f"[{agent_id}] Agent starting...", flush=True)
    print(f"[{agent_id}] INITIALIZING FAILURE SCAN...", flush=True)
    log_event(agent_id, "INIT", {"status": "starting_scan"})

    failures = []
    test_dirs = [
        REPO_ROOT / "CATALYTIC-DPT" / "TESTBENCH",
        REPO_ROOT / "SKILLS",
    ]

    for test_dir in test_dirs:
        if not test_dir.exists():
            print(f"[{agent_id}] SKIPPED: {test_dir} does not exist", flush=True)
            continue

        print(f"[{agent_id}] SCANNING: {test_dir}", flush=True)
        log_event(agent_id, "SCAN_DIR", {"dir": str(test_dir)})

        for test_file in test_dir.glob("test_*.py"):
            print(f"[{agent_id}] FOUND TEST: {test_file.name}", flush=True)
            failures.append({"file": str(test_file), "type": "test_file"})

    # Check for failure markers
    failure_reports = [
        REPO_ROOT / "TEST_FAILURES_REPORT.md",
        REPO_ROOT / "test_results.txt",
        REPO_ROOT / "test_phase7.txt",
        REPO_ROOT / "test_phase8_output.txt",
    ]

    for report in failure_reports:
        if report.exists():
            print(f"[{agent_id}] FOUND FAILURE REPORT: {report.name}", flush=True)
            try:
                content = report.read_text()
                lines = len(content.split('\n'))
                failures.append({"file": str(report), "type": "failure_report", "lines": lines, "preview": content[:200]})
                log_event(agent_id, "FAILURE_REPORT_FOUND", {"file": report.name, "lines": lines})
            except Exception as e:
                print(f"[{agent_id}] ERROR reading {report.name}: {e}", flush=True)

    print(f"[{agent_id}] SCAN COMPLETE: Found {len(failures)} issues", flush=True)
    log_event(agent_id, "SCAN_COMPLETE", {"total_issues": len(failures)})

    if failures:
        print(f"[{agent_id}] *** {len(failures)} FAILURES DETECTED ***", flush=True)
        for i, failure in enumerate(failures, 1):
            print(f"[{agent_id}]   {i}. {failure.get('type')}: {failure.get('file')}", flush=True)
    else:
        print(f"[{agent_id}] NO FAILURES DETECTED", flush=True)

    print(f"[{agent_id}] Monitoring system...", flush=True)
    for i in range(5):
        time.sleep(1)
        print(f"[{agent_id}] Still monitoring... ({(i+1)*1}s elapsed)", flush=True)

    print(f"[{agent_id}] Task complete. Shutting down.", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
