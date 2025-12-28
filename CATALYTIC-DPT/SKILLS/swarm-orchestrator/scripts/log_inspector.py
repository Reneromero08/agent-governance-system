#!/usr/bin/env python3
import sys, os, json, time, re
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

def log_event(agent_id, event_type, data):
    print(f"[{agent_id}] {event_type}: {json.dumps(data)}")
    sys.stdout.flush()

def main():
    agent_id = "log-inspector"
    print(f"[{agent_id}] Agent starting...", flush=True)
    print(f"[{agent_id}] INITIALIZING LOG INSPECTION...", flush=True)
    log_event(agent_id, "INIT", {"status": "inspecting_logs"})

    findings = {"error_files": [], "error_patterns": {}, "suspicious_activity": []}

    swarm_runs_dir = REPO_ROOT / "CATALYTIC-DPT" / "SKILLS" / "swarm-orchestrator" / "runs"
    if swarm_runs_dir.exists():
        print(f"[{agent_id}] SCANNING SWARM LOGS: {swarm_runs_dir}", flush=True)
        log_event(agent_id, "SCANNING_SWARM", {"dir": str(swarm_runs_dir)})

        for run_dir in swarm_runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            for log_file in run_dir.glob("*.log"):
                try:
                    content = log_file.read_text(errors='ignore')
                    if content and any(w in content.lower() for w in ['error', 'fail', 'exception']):
                        print(f"[{agent_id}] FOUND LOG WITH ERRORS: {log_file.name}", flush=True)
                        findings["error_files"].append({"file": log_file.name, "run": run_dir.name, "size": len(content)})
                except Exception as e:
                    print(f"[{agent_id}] ERROR reading {log_file.name}: {e}", flush=True)

    test_patterns = {
        "FAIL": r"FAIL|failed|failure",
        "ERROR": r"ERROR|error|exception",
        "TIMEOUT": r"timeout|Timeout|TIMEOUT",
        "ASSERTION": r"AssertionError|assert|assertion",
        "IMPORT": r"ImportError|ModuleNotFoundError|cannot import",
    }

    test_files = [
        REPO_ROOT / "TEST_FAILURES_REPORT.md",
        REPO_ROOT / "test_results.txt",
        REPO_ROOT / "test_phase7.txt",
        REPO_ROOT / "test_phase8_output.txt",
    ]

    for test_file in test_files:
        if test_file.exists():
            print(f"[{agent_id}] INSPECTING: {test_file.name}", flush=True)
            try:
                content = test_file.read_text(errors='ignore')
                for pattern_name, pattern in test_patterns.items():
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        count = len(matches)
                        print(f"[{agent_id}]   {pattern_name}: {count} occurrences", flush=True)
                        if pattern_name not in findings["error_patterns"]:
                            findings["error_patterns"][pattern_name] = []
                        findings["error_patterns"][pattern_name].append({"file": test_file.name, "count": count, "sample": matches[0] if matches else None})

                lines = content.split('\n')
                for i, line in enumerate(lines):
                    if any(keyword in line.lower() for keyword in ['critical', 'fatal', 'panic', 'corrupted']):
                        print(f"[{agent_id}] SUSPICIOUS: Line {i+1}: {line[:80]}", flush=True)
                        findings["suspicious_activity"].append({"file": test_file.name, "line": i + 1, "content": line[:100]})

            except Exception as e:
                print(f"[{agent_id}] ERROR inspecting {test_file.name}: {e}", flush=True)

    print(f"[{agent_id}] LOG INSPECTION COMPLETE", flush=True)
    print(f"[{agent_id}] Findings: {len(findings['error_files'])} error files, {len(findings['error_patterns'])} error patterns", flush=True)
    log_event(agent_id, "INSPECTION_COMPLETE", findings)

    print(f"[{agent_id}] *** LOG INSPECTION SUMMARY ***", flush=True)
    print(f"[{agent_id}] Error files: {len(findings['error_files'])}", flush=True)
    print(f"[{agent_id}] Error patterns: {len(findings['error_patterns'])}", flush=True)
    print(f"[{agent_id}] Suspicious activities: {len(findings['suspicious_activity'])}", flush=True)

    print(f"[{agent_id}] Monitoring...", flush=True)
    for i in range(5):
        time.sleep(1)
        print(f"[{agent_id}] Running... ({(i+1)*1}s elapsed)", flush=True)

    print(f"[{agent_id}] Inspection complete. Shutting down.", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
