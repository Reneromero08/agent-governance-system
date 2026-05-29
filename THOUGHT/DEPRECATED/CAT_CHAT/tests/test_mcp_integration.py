
import sys
import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

# Add CAT_CHAT to sys.path so we can import catalytic_chat
cat_chat_root = Path(__file__).parents[1]
if str(cat_chat_root) not in sys.path:
    sys.path.insert(0, str(cat_chat_root))

from catalytic_chat.mcp_integration import ChatToolExecutor, McpAccessError

class TestMcpIntegration:
    
    def test_list_tools_filtering_constraint(self):
        """Verify only allowed tools are returned."""
        with patch("catalytic_chat.mcp_integration.ChatToolExecutor._get_server") as mock_get:
            mock_server = MagicMock()
            mock_get.return_value = mock_server
            
            # Simulated response from full MCP server containing safe and unsafe tools
            mock_server.handle_request.return_value = {
                "result": {
                    "tools": [
                        {"name": "cortex_query"},      # Safe/Allowed
                        {"name": "terminal_bridge"},   # Dangerous/Forbidden
                        {"name": "skill_run"},         # Excluded for now
                        {"name": "semantic_search"}    # Safe/Allowed
                    ]
                }
            }
            
            executor = ChatToolExecutor(repo_root=Path("."))
            tools = executor.list_tools()
            
            names = {t["name"] for t in tools}
            
            # Assertions
            assert "cortex_query" in names
            assert "semantic_search" in names
            assert "terminal_bridge" not in names
            assert "skill_run" not in names
            assert len(names) == 2

    def test_execute_allowed_tool(self):
        """Verify allowed tools are dispatched to server."""
        with patch("catalytic_chat.mcp_integration.ChatToolExecutor._get_server") as mock_get:
            mock_server = MagicMock()
            mock_get.return_value = mock_server
            
            mock_server.handle_request.return_value = {
                "result": {"content": [{"type": "text", "text": "result"}]}
            }
            
            executor = ChatToolExecutor(repo_root=Path("."))
            result = executor.execute_tool("cortex_query", {"query": "test"})
            
            assert result["content"][0]["text"] == "result"
            
            # Verify correct plumbing
            args, _ = mock_server.handle_request.call_args
            req = args[0]
            assert req["method"] == "tools/call"
            assert req["params"]["name"] == "cortex_query"
            assert req["params"]["arguments"] == {"query": "test"}

    def test_execute_forbidden_tool_fails_closed(self):
        """Verify forbidden tools fail immediately without calling server."""
        with patch("catalytic_chat.mcp_integration.ChatToolExecutor._get_server") as mock_get:
            mock_server = MagicMock()
            mock_get.return_value = mock_server
            
            executor = ChatToolExecutor(repo_root=Path("."))
            
            # Try to execute a forbidden tool
            with pytest.raises(McpAccessError) as exc:
                executor.execute_tool("terminal_bridge", {"cmd": "whoami"})
            
            assert "not in the allowed set" in str(exc.value)
            
            # Ensure server was NOT called
            mock_server.handle_request.assert_not_called()

    def test_mcp_server_error_propagation(self):
        """Verify errors from MCP server are propagated as McpAccessError."""
        with patch("catalytic_chat.mcp_integration.ChatToolExecutor._get_server") as mock_get:
            mock_server = MagicMock()
            mock_get.return_value = mock_server
            
            mock_server.handle_request.return_value = {
                "error": {"code": -32000, "message": "Something went wrong"}
            }
            
            executor = ChatToolExecutor(repo_root=Path("."))
            
            with pytest.raises(McpAccessError) as exc:
                executor.execute_tool("cortex_query", {})
                
            assert "MCP Error -32000" in str(exc.value)
