#!/usr/bin/env python3
"""
MCP Network Startup Script
Launches Ollama server, MCP server, and optional Ant workers.
"""

import sys
import os
import subprocess
import time
import json
import argparse
import requests
from pathlib import Path

# Get repo root
REPO_ROOT = Path(__file__).resolve().parents[4]
os.chdir(REPO_ROOT)

# Configuration
OLLAMA_PORT = int(os.getenv("OLLAMA_PORT", "11434"))
OLLAMA_URL = f"http://localhost:{OLLAMA_PORT}"
MCP_LEDGER_PATH = os.getenv("MCP_LEDGER_PATH", "CONTRACTS/_runs/mcp_ledger")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Colors for terminal output
class Colors:
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BLUE = "\033[94m"
    END = "\033[0m"

def log(msg, level="INFO", color=None):
    """Log with timestamp and color"""
    prefix = f"[{level}]"
    if color:
        print(f"{color}{prefix}{Colors.END} {msg}")
    else:
        print(f"{prefix} {msg}")

def check_ollama_running():
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def wait_for_ollama(timeout=30):
    """Wait for Ollama to be ready"""
    start = time.time()
    while time.time() - start < timeout:
        if check_ollama_running():
            log("Ollama is online", color=Colors.GREEN)
            return True
        log(f"Waiting for Ollama... ({int(time.time() - start)}s)", color=Colors.YELLOW)
        time.sleep(2)
    return False

def check_model_loaded():
    """Check if LFM2 model is loaded"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        if response.status_code == 200:
            data = response.json()
            models = data.get("models", [])
            return len(models) > 0
    except:
        pass
    return False

def start_ollama():
    """Start Ollama server"""
    log("Starting Ollama server...", color=Colors.BLUE)

    if check_ollama_running():
        log("Ollama is already running", color=Colors.GREEN)
        return True

    try:
        # Try to start Ollama
        if sys.platform == "win32":
            # Windows
            subprocess.Popen("ollama serve", shell=True)
        else:
            # Unix
            subprocess.Popen(["ollama", "serve"])

        # Wait for it to start
        if wait_for_ollama():
            return True
        else:
            log("Ollama failed to start within timeout", color=Colors.RED)
            return False
    except Exception as e:
        log(f"Error starting Ollama: {e}", level="ERROR", color=Colors.RED)
        return False

def check_mcp_ledger():
    """Ensure MCP ledger directories exist"""
    ledger_path = Path(MCP_LEDGER_PATH)
    try:
        ledger_path.mkdir(parents=True, exist_ok=True)
        log(f"MCP ledger ready at {ledger_path}", color=Colors.GREEN)
        return True
    except Exception as e:
        log(f"Error creating MCP ledger: {e}", level="ERROR", color=Colors.RED)
        return False

def start_mcp_server():
    """Start MCP server"""
    log("Starting MCP server...", color=Colors.BLUE)

    mcp_script = Path("CATALYTIC-DPT/LAB/MCP/stdio_server.py")
    if not mcp_script.exists():
        log(f"MCP server script not found: {mcp_script}", level="ERROR", color=Colors.RED)
        return False

    # Note: This won't actually run in background since stdio servers
    # need to stay in foreground. User should run this in a separate terminal.
    log("Launching MCP stdio server", color=Colors.GREEN)
    log("[NOTE] This is a stdio server - runs in foreground only", color=Colors.YELLOW)
    log("[ACTION] Start the MCP server in a separate terminal with:", color=Colors.YELLOW)
    log(f"  python {mcp_script}", color=Colors.YELLOW)
    return True

def start_ant_workers(count=2):
    """Start Ant worker processes"""
    log(f"Starting {count} Ant workers...", color=Colors.BLUE)

    ant_script = Path("CATALYTIC-DPT/SKILLS/ant-worker/scripts/ant_agent.py")
    if not ant_script.exists():
        log(f"Ant worker script not found: {ant_script}", level="ERROR", color=Colors.RED)
        return False

    pids = []
    for i in range(1, count + 1):
        agent_id = f"Ant-{i}"
        log(f"Starting {agent_id}...", color=Colors.BLUE)

        try:
            if sys.platform == "win32":
                # Windows - use START command for new windows
                process = subprocess.Popen(
                    f'start "Ant Worker {i}" python "{ant_script}" --agent_id {agent_id}',
                    shell=True
                )
            else:
                # Unix - use subprocess
                process = subprocess.Popen([
                    sys.executable, str(ant_script),
                    "--agent_id", agent_id
                ])

            pids.append((agent_id, process.pid))
            log(f"{agent_id} started (PID: {process.pid})", color=Colors.GREEN)
        except Exception as e:
            log(f"Error starting {agent_id}: {e}", level="ERROR", color=Colors.RED)
            return False

    return True

def health_check():
    """Run health checks on startup"""
    log("\n=== Health Check ===\n", color=Colors.BLUE)

    checks = []

    # Check Ollama
    if check_ollama_running():
        log("[OK] Ollama server is running", color=Colors.GREEN)
        checks.append(True)
    else:
        log("[FAIL] Ollama server is NOT running", color=Colors.RED)
        checks.append(False)

    # Check model loaded
    if check_model_loaded():
        log("[OK] LFM2 model is loaded", color=Colors.GREEN)
        checks.append(True)
    else:
        log("[FAIL] LFM2 model is NOT loaded", color=Colors.RED)
        checks.append(False)

    # Check MCP ledger
    if Path(MCP_LEDGER_PATH).exists():
        log(f"[OK] MCP ledger exists at {MCP_LEDGER_PATH}", color=Colors.GREEN)
        checks.append(True)
    else:
        log(f"[FAIL] MCP ledger does NOT exist", color=Colors.RED)
        checks.append(False)

    return all(checks)

def main():
    parser = argparse.ArgumentParser(description="Start the MCP network")
    parser.add_argument("--all", action="store_true", help="Start all components")
    parser.add_argument("--ollama-only", action="store_true", help="Start only Ollama")
    parser.add_argument("--mcp-only", action="store_true", help="Start only MCP server")
    parser.add_argument("--mcp", action="store_true", help="Start MCP server")
    parser.add_argument("--ants", type=int, default=2, help="Number of Ant workers (default 2)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")

    args = parser.parse_args()

    log("\n" + "="*50, color=Colors.BLUE)
    log("MCP NETWORK STARTUP", color=Colors.BLUE)
    log("="*50 + "\n", color=Colors.BLUE)

    # Interactive mode
    if args.interactive:
        print("\nWhat would you like to start?")
        print("1) Full network (Ollama + MCP + Ants)")
        print("2) Ollama only")
        print("3) MCP server only")
        print("4) Exit")
        choice = input("\nEnter choice (1-4): ").strip()

        if choice == "1":
            args.all = True
        elif choice == "2":
            args.ollama_only = True
        elif choice == "3":
            args.mcp_only = True
        elif choice == "4":
            log("Exiting", color=Colors.YELLOW)
            return

    # Start components
    if args.all or args.ollama_only:
        if not start_ollama():
            log("Failed to start Ollama", level="ERROR", color=Colors.RED)
            return

        if not args.ollama_only:
            # Continue with other components
            args.mcp = True

    if args.mcp or args.mcp_only or args.all:
        if not check_mcp_ledger():
            log("Failed to prepare MCP ledger", level="ERROR", color=Colors.RED)
            return

        start_mcp_server()

    if args.all and args.ants > 0:
        if not start_ant_workers(args.ants):
            log("Failed to start Ant workers", level="ERROR", color=Colors.RED)
            return

    # Health check
    time.sleep(2)
    if health_check():
        log("\n[SUCCESS] MCP Network is online!", color=Colors.GREEN)
    else:
        log("\n[WARNING] Some components may not be ready yet", color=Colors.YELLOW)

    print("\n" + "="*50)
    log("Next steps:", color=Colors.BLUE)
    print("  1. Start the MCP server (if not auto-launched)")
    print("  2. Connect Claude Desktop to your MCP server")
    print("  3. Send tasks from Claude to your Ant workers")
    print("="*50)

if __name__ == "__main__":
    main()
