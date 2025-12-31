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
    agent_id = "test-analyzer"
    print(f"[{agent_id}] Agent starting...", flush=True)
    print(f"[{agent_id}] INITIALIZING TEST ANALYSIS...", flush=True)
    log_event(agent_id, "INIT", {"status": "analyzing"})

    analysis = {"total_failures": 0, "categories": {}, "critical": [], "warnings": []}

    failure_report = REPO_ROOT / "TEST_FAILURES_REPORT.md"
    if failure_report.exists():
        print(f"[{agent_id}] ANALYZING: {failure_report.name}", flush=True)
        try:
            content = failure_report.read_text()
            lines = content.split('\n')

            failure_count = len([l for l in lines if l.strip().startswith('- [ ]')])
            analysis["total_failures"] = failure_count
            print(f"[{agent_id}] Found {failure_count} pending fixes", flush=True)
            log_event(agent_id, "FAILURES_FOUND", {"count": failure_count})

            current_category = None
            for line in lines:
                if line.startswith('## '):
                    current_category = line.replace('## ', '').strip()
                    analysis["categories"][current_category] = 0
                elif current_category and line.strip().startswith('- [ ]'):
                    analysis["categories"][current_category] += 1
                    if 'critical' in line.lower() or 'error' in line.lower():
                        analysis["critical"].append(line.strip())

            print(f"[{agent_id}] Categories: {json.dumps(analysis['categories'])}", flush=True)

            if analysis["critical"]:
                print(f"[{agent_id}] *** {len(analysis['critical'])} CRITICAL ISSUES ***", flush=True)
                for issue in analysis["critical"][:5]:
                    print(f"[{agent_id}]   - {issue}", flush=True)

        except Exception as e:
            print(f"[{agent_id}] ERROR analyzing report: {e}", flush=True)

    test_outputs = [REPO_ROOT / "test_results.txt", REPO_ROOT / "test_phase7.txt", REPO_ROOT / "test_phase8_output.txt"]
    for test_file in test_outputs:
        if test_file.exists():
            print(f"[{agent_id}] ANALYZING: {test_file.name}", flush=True)
            try:
                content = test_file.read_text(errors='ignore')
                errors = len(re.findall(r'ERROR|FAIL|Exception', content, re.IGNORECASE))
                warnings = len(re.findall(r'WARN|WARNING', content, re.IGNORECASE))
                if errors > 0:
                    print(f"[{agent_id}]   {errors} errors, {warnings} warnings", flush=True)
                    analysis["warnings"].append({"file": test_file.name, "errors": errors, "warnings": warnings})
            except Exception as e:
                print(f"[{agent_id}] ERROR reading {test_file.name}: {e}", flush=True)

    print(f"[{agent_id}] ANALYSIS COMPLETE", flush=True)
    print(f"[{agent_id}] Summary: {json.dumps(analysis)}", flush=True)
    log_event(agent_id, "ANALYSIS_COMPLETE", analysis)

    print(f"[{agent_id}] Monitoring...", flush=True)
    for i in range(5):
        time.sleep(1)
        print(f"[{agent_id}] Monitoring... ({(i+1)*1}s elapsed)", flush=True)

    print(f"[{agent_id}] Analysis complete. Shutting down.", flush=True)
    sys.exit(0)

if __name__ == "__main__":
    main()
