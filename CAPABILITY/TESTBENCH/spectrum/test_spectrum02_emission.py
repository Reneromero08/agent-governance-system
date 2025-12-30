import socket
import sys

class MCPTerminalServer:
    """MCP Terminal Server."""

    def __init__(self, port=8888):  # Changed default port to 8888 (above 1024)
        self.port = port
        self.server_socket = None

    def start(self):
        """Start the server on the specified port."""
        try:
            # Create and bind socket
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow port reuse
            self.server_socket.bind(('localhost', self.port))
            self.server_socket.listen(5)  # Allow up to 5 queued connections
            print(f"Server started successfully on port {self.port}")
            print("Waiting for connections...")

            # Handle incoming connections
            while True:
                try:
                    conn, addr = self.server_socket.accept()
                    print(f"Connection from {addr}")
                    # Here you would handle the connection
                    conn.close()
                except KeyboardInterrupt:
                    print("\nServer shutting down...")
                    break
                except Exception as e:
                    print(f"Connection error: {e}")

        except PermissionError:
            print(f"Permission denied when trying to bind to port {self.port}")
            print("On Unix systems, ports below 1024 require root privileges.")
            print("Try using a port number above 1024 or running with sudo.")
            sys.exit(1)
        except socket.error as e:
            print(f"Socket error: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Unexpected error: {e}")
            sys.exit(1)
        finally:
            if self.server_socket:
                self.server_socket.close()

# Example usage
if __name__ == "__main__":
    try:
        server = MCPTerminalServer()
        server.start()
    except KeyboardInterrupt:
        print("\nServer stopped by user")