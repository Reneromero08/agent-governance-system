#!/usr/bin/env python3
"""Test MCP connection directly"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to import the MCP server module
try:
    from LAW.CONTRACTS.ags_mcp_entrypoint import AGSMCPServer
    print("✓ MCP server module imported successfully")
    
    # Create instance
    server = AGSMCPServer()
    print(f"✓ Server created with {len(server.tools)} tools")
    
    # List tools
    print("\nAvailable tools:")
    for tool in server.tools:
        print(f"  - {tool.name}")
    
    # Check for semantic tools
    semantic_tools = [t for t in server.tools if 'semantic' in t.name or 'cassette' in t.name]
    print(f"\n✓ Found {len(semantic_tools)} semantic tools:")
    for t in semantic_tools:
        print(f"  - {t.name}")
        
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()