import subprocess
import time
import socket
import signal
import os
import sys
from typing import Optional

def start_mcp_server() -> Optional[subprocess.Popen]:
    """
    Start MCP server with proper error handling and startup verification.

    Returns:
        subprocess.Popen object if successful, None otherwise
    """
    # Define the command to start the MCP server
    # Replace with actual path to your MCP server executable
    command = [
        "mcp_server",  # Replace with actual executable name/path
        "--port", "8080"  # or whatever port your MCP server uses
    ]

    try:
        # Start the subprocess with proper error handling
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,  # For Python 3.7+ to get strings instead of bytes
            bufsize=1,   # Line buffered
            preexec_fn=os.setsid if hasattr(os, 'setsid') else None
        )

        # Wait for server to be ready (with timeout)
        max_attempts = 30
        port = 8080
        for attempt in range(max_attempts):
            try:
                # Try to connect to the server
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(2)  # Increased timeout
                    s.connect(('127.0.0.1', port))  # Use 127.0.0.1 instead of localhost
                    print("MCP server started successfully")
                    return process
            except (socket.timeout, ConnectionRefusedError, OSError):
                if attempt < max_attempts - 1:  # Don't print on last attempt
                    print(f"Waiting for MCP server to start... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(1)

        # If we get here, server didn't start in time
        print("Failed to start MCP server - connection refused after multiple attempts")

        # Get error output if available
        stdout, stderr = process.communicate(timeout=2)
        if stderr:
            print(f"Error output from MCP server:\n{stderr}")
        if stdout:
            print(f"Output from MCP server:\n{stdout}")

        # Try to terminate the process
        try:
            if hasattr(os, 'setsid'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=2)
        except Exception as e:
            print(f"Failed to terminate MCP server: {str(e)}")
            try:
                process.kill()
            except:
                pass

        return None

    except FileNotFoundError:
        print(f"Error: MCP server executable not found. Command: {' '.join(command)}")
        return None
    except Exception as e:
        print(f"Exception occurred while starting MCP server: {str(e)}")
        return None

def stop_mcp_server(process: Optional[subprocess.Popen]) -> None:
    """Cleanly stop the MCP server process"""
    if process and process.poll() is None:  # Only try to stop if process is running
        try:
            if hasattr(os, 'setsid'):
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except Exception as e:
            print(f"Warning: Failed to terminate MCP server: {str(e)}")
            try:
                process.kill()
            except:
                pass

# Example usage
if __name__ == "__main__":
    mcp_process = start_mcp_server()

    if mcp_process:
        try:
            # Your test code here
            print("Running tests with MCP server...")
            # Example: run your test suite
            time.sleep(10)  # Simulate test execution
        finally:
            stop_mcp_server(mcp_process)
    else:
        print("Cannot proceed without MCP server")
        sys.exit(1)