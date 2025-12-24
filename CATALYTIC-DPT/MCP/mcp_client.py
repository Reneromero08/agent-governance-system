#!/usr/bin/env python3
import sys
import json
import argparse
from pathlib import Path

# Add the MCP directory to sys.path so we can import the server
sys.path.insert(0, str(Path(__file__).parent))
from server import mcp_server

def main():
    parser = argparse.ArgumentParser(description="AGS MCP CLI Client")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Dispatch Task
    dispatch = subparsers.add_parser("dispatch", help="Dispatch a task to an agent")
    dispatch.add_argument("--task_id", required=True)
    dispatch.add_argument("--from", required=True, dest="from_agent")
    dispatch.add_argument("--to", required=True, dest="to_agent")
    dispatch.add_argument("--spec", required=True, help="JSON task spec string or file path")
    dispatch.add_argument("--priority", type=int, default=5)

    # Get Pending Tasks
    pending = subparsers.add_parser("pending", help="Get pending tasks for an agent")
    pending.add_argument("--agent", required=True)

    # Report Result
    report = subparsers.add_parser("report", help="Report task result")
    report.add_argument("--task_id", required=True)
    report.add_argument("--agent", required=True)
    report.add_argument("--status", required=True, choices=["success", "failed", "error", "escalated"])
    report.add_argument("--result", required=True, help="JSON result string or file path")
    report.add_argument("--errors", help="JSON list of errors", default="[]")

    # Get Results
    results = subparsers.add_parser("results", help="Get results for a task")
    results.add_argument("--task_id", help="Filter by task ID")

    # Escalate
    escalate = subparsers.add_parser("escalate", help="Escalate an issue")
    escalate.add_argument("--from", required=True, dest="from_agent")
    escalate.add_argument("--issue", required=True)
    escalate.add_argument("--context", required=True, help="JSON context")
    escalate.add_argument("--priority", type=int, default=5)

    # Get Escalations
    get_esc = subparsers.add_parser("get-escalations", help="Get pending escalations")
    get_esc.add_argument("--level", required=True)

    # Send Directive
    directive = subparsers.add_parser("directive", help="Send directive down the chain")
    directive.add_argument("--from_level", required=True)
    directive.add_argument("--to_agent", required=True)
    directive.add_argument("--directive", required=True)
    directive.add_argument("--context", required=True, help="JSON context")

    # Get Directives
    get_dir = subparsers.add_parser("get-directives", help="Get pending directives")
    get_dir.add_argument("--agent", required=True)

    args = parser.parse_args()

    def load_json_arg(arg_val):
        if arg_val.startswith("{") or arg_val.startswith("["):
            return json.loads(arg_val)
        path = Path(arg_val)
        if path.exists():
            return json.loads(path.read_text())
        return arg_val

    if args.command == "dispatch":
        spec = load_json_arg(args.spec)
        res = mcp_server.dispatch_task(args.task_id, spec, args.from_agent, args.to_agent, args.priority)
        print(json.dumps(res, indent=2))

    elif args.command == "pending":
        res = mcp_server.get_pending_tasks(args.agent)
        print(json.dumps(res, indent=2))

    elif args.command == "report":
        result = load_json_arg(args.result)
        errors = load_json_arg(args.errors)
        res = mcp_server.report_result(args.task_id, args.agent, args.status, result, errors)
        print(json.dumps(res, indent=2))

    elif args.command == "results":
        res = mcp_server.get_results(args.task_id)
        print(json.dumps(res, indent=2))

    elif args.command == "escalate":
        context = load_json_arg(args.context)
        res = mcp_server.escalate(args.from_agent, args.issue, context, args.priority)
        print(json.dumps(res, indent=2))

    elif args.command == "get-escalations":
        res = mcp_server.get_escalations(args.level)
        print(json.dumps(res, indent=2))

    elif args.command == "directive":
        context = load_json_arg(args.context)
        res = mcp_server.send_directive(args.from_level, args.to_agent, args.directive, context)
        print(json.dumps(res, indent=2))

    elif args.command == "get-directives":
        res = mcp_server.get_directives(args.agent)
        print(json.dumps(res, indent=2))

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
