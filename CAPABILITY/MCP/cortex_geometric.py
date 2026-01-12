#!/usr/bin/env python3
"""
Cortex Geometric MCP Server

Minimal MCP server exposing GeometricCassetteNetwork for context retrieval.
Designed to keep LIL_Q pure while providing cassette network access.

Usage:
  # As MCP server (stdio)
  python CAPABILITY/MCP/cortex_geometric.py

  # Direct Python usage (no MCP needed)
  from CAPABILITY.MCP.cortex_geometric import retrieve
  context = retrieve("What is authentication?", k=5)

The formula: E = <psi|phi>
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any

# Add paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "network"))

# Lazy network loading
_network = None


def _get_network():
    """Lazy load the cassette network."""
    global _network
    if _network is None:
        try:
            from geometric_cassette import GeometricCassetteNetwork
            _network = GeometricCassetteNetwork.from_config()
        except Exception as e:
            print(f"[WARN] Could not load network: {e}", file=sys.stderr)
            return None
    return _network


# =============================================================================
# Direct Python API (no MCP needed)
# =============================================================================

def retrieve(query: str, k: int = 5, threshold: float = 0.3) -> List[str]:
    """
    Retrieve context from cassette network.

    Args:
        query: Search query
        k: Number of results
        threshold: E threshold for filtering

    Returns:
        List of context strings (just the content, no metadata bloat)

    Example:
        >>> from CAPABILITY.MCP.cortex_geometric import retrieve
        >>> context = retrieve("authentication", k=5)
        >>> print(context[0])
    """
    network = _get_network()
    if network is None:
        return []

    try:
        # Initialize query state
        query_state = network.reasoner.initialize(query)

        # Query merged across all cassettes
        results = network.query_merged(query_state, k * 2)

        # Filter by E threshold and return just content
        context = []
        for r in results:
            if r.get('E', 0) >= threshold:
                content = r.get('content', '')
                if content and len(content.strip()) > 0:
                    context.append(content)
                    if len(context) >= k:
                        break

        return context
    except Exception as e:
        print(f"[WARN] Retrieve error: {e}", file=sys.stderr)
        return []


def retrieve_with_scores(query: str, k: int = 5, threshold: float = 0.3) -> List[Dict]:
    """
    Retrieve with E scores and cassette info.

    Returns:
        List of dicts with 'content', 'E', 'cassette_id'
    """
    network = _get_network()
    if network is None:
        return []

    try:
        query_state = network.reasoner.initialize(query)
        results = network.query_merged(query_state, k * 2)

        scored = []
        for r in results:
            if r.get('E', 0) >= threshold:
                scored.append({
                    'content': r.get('content', ''),
                    'E': r.get('E', 0),
                    'cassette_id': r.get('cassette_id', 'unknown')
                })
                if len(scored) >= k:
                    break

        return scored
    except Exception as e:
        print(f"[WARN] Retrieve error: {e}", file=sys.stderr)
        return []


def status() -> Dict[str, Any]:
    """Get network status."""
    network = _get_network()
    if network is None:
        return {'available': False, 'error': 'Network not loaded'}

    try:
        stats = network.get_network_stats()
        return {
            'available': True,
            'cassettes': list(network.cassettes.keys()),
            'total_documents': stats.get('total_documents', 0),
            'total_geometric_ops': stats.get('total_geometric_ops', 0)
        }
    except Exception as e:
        return {'available': False, 'error': str(e)}


# =============================================================================
# MCP Protocol (stdio JSON-RPC)
# =============================================================================

def handle_request(request: Dict) -> Dict:
    """Handle MCP JSON-RPC request."""
    method = request.get('method', '')
    params = request.get('params', {})
    req_id = request.get('id')

    if method == 'initialize':
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'result': {
                'protocolVersion': '0.1.0',
                'serverInfo': {
                    'name': 'cortex-geometric',
                    'version': '1.0.0'
                },
                'capabilities': {
                    'tools': {}
                }
            }
        }

    elif method == 'tools/list':
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'result': {
                'tools': [
                    {
                        'name': 'retrieve',
                        'description': 'Retrieve context from cassette network using E-gating',
                        'inputSchema': {
                            'type': 'object',
                            'properties': {
                                'query': {'type': 'string', 'description': 'Search query'},
                                'k': {'type': 'integer', 'default': 5, 'description': 'Number of results'},
                                'threshold': {'type': 'number', 'default': 0.3, 'description': 'E threshold'}
                            },
                            'required': ['query']
                        }
                    },
                    {
                        'name': 'status',
                        'description': 'Get cassette network status',
                        'inputSchema': {'type': 'object', 'properties': {}}
                    }
                ]
            }
        }

    elif method == 'tools/call':
        tool_name = params.get('name', '')
        args = params.get('arguments', {})

        if tool_name == 'retrieve':
            results = retrieve(
                query=args.get('query', ''),
                k=args.get('k', 5),
                threshold=args.get('threshold', 0.3)
            )
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'content': [{'type': 'text', 'text': json.dumps(results)}]
                }
            }

        elif tool_name == 'status':
            result = status()
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'result': {
                    'content': [{'type': 'text', 'text': json.dumps(result)}]
                }
            }

        else:
            return {
                'jsonrpc': '2.0',
                'id': req_id,
                'error': {'code': -32601, 'message': f'Unknown tool: {tool_name}'}
            }

    elif method == 'notifications/initialized':
        return None  # No response for notifications

    else:
        return {
            'jsonrpc': '2.0',
            'id': req_id,
            'error': {'code': -32601, 'message': f'Method not found: {method}'}
        }


def run_stdio():
    """Run MCP server on stdio."""
    print("[cortex-geometric] Starting MCP server...", file=sys.stderr)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
            response = handle_request(request)
            if response is not None:
                print(json.dumps(response), flush=True)
        except json.JSONDecodeError as e:
            error = {
                'jsonrpc': '2.0',
                'id': None,
                'error': {'code': -32700, 'message': f'Parse error: {e}'}
            }
            print(json.dumps(error), flush=True)
        except Exception as e:
            error = {
                'jsonrpc': '2.0',
                'id': None,
                'error': {'code': -32603, 'message': f'Internal error: {e}'}
            }
            print(json.dumps(error), flush=True)


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Cortex Geometric MCP Server')
    parser.add_argument('--test', action='store_true', help='Test retrieve function')
    parser.add_argument('--query', type=str, help='Test query')
    parser.add_argument('--status', action='store_true', help='Show network status')
    args = parser.parse_args()

    if args.status:
        print(json.dumps(status(), indent=2))
    elif args.test or args.query:
        query = args.query or "authentication"
        print(f"Query: {query}")
        results = retrieve_with_scores(query, k=5)
        for i, r in enumerate(results, 1):
            print(f"{i}. [E={r['E']:.3f}] [{r['cassette_id']}] {r['content'][:80]}...")
    else:
        run_stdio()
