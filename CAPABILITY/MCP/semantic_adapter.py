#!/usr/bin/env python3
"""
Semantic Search MCP Adapter

Connects the CORTEX semantic tools to the AGS MCP server.
Provides vector-based semantic search capabilities via MCP tools.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add CORTEX to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORTEX_ROOT = PROJECT_ROOT / "NAVIGATION" / "CORTEX"
sys.path.insert(0, str(CORTEX_ROOT))
sys.path.insert(0, str(CORTEX_ROOT / "semantic"))
sys.path.insert(0, str(CORTEX_ROOT / "network"))

try:
    from semantic.semantic_search import SemanticSearch, search_cortex
    from semantic.query import CortexQuery
    SEMANTIC_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Semantic tools not available: {e}", file=sys.stderr)
    SEMANTIC_AVAILABLE = False

try:
    from network.network_hub import SemanticNetworkHub
    from network.generic_cassette import load_cassettes_from_json, create_cassette_from_config
    NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Network tools not available: {e}", file=sys.stderr)
    NETWORK_AVAILABLE = False


class SemanticMCPAdapter:
    """Adapter that provides semantic search tools via MCP."""
    
    def __init__(self):
        self.db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
        self.network_hub = None
        self.semantic_search = None
        self.cortex_query = None
        
    def initialize(self):
        """Initialize semantic tools."""
        if not SEMANTIC_AVAILABLE:
            return {"error": "Semantic tools not available"}
        
        try:
            # Initialize semantic search
            if self.db_path.exists():
                self.semantic_search = SemanticSearch(self.db_path)
                self.cortex_query = CortexQuery(self.db_path)
            
            # Initialize cassette network if available
            if NETWORK_AVAILABLE:
                self.network_hub = SemanticNetworkHub()
                self._load_cassettes_from_config()
            
            return {
                "status": "initialized",
                "db_exists": self.db_path.exists(),
                "embeddings_count": self.get_embedding_stats().get("total_embeddings", 0) if self.semantic_search else 0,
                "cassettes_registered": len(self.network_hub.cassettes) if self.network_hub else 0
            }
        except Exception as e:
            return {"error": f"Initialization failed: {str(e)}"}
    
    def _load_cassettes_from_config(self):
        """Load cassettes from JSON configuration using generic cassette system."""
        config_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "network" / "cassettes.json"
        if not config_path.exists():
            print(f"[WARNING] Cassette config not found at {config_path}", file=sys.stderr)
            return
        
        try:
            # Use generic cassette loader with project root
            cassettes = load_cassettes_from_json(config_path, PROJECT_ROOT)
            
            for cassette in cassettes:
                self.network_hub.register_cassette(cassette)
                print(f"[INFO] Loaded cassette: {cassette.cassette_id} ({cassette.description})")
        
        except Exception as e:
            print(f"[ERROR] Failed to load cassette config: {e}", file=sys.stderr)
    
    def semantic_search_tool(self, args: Dict) -> Dict:
        """MCP tool: Semantic search using vector embeddings."""
        if not SEMANTIC_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Semantic tools not available"}],
                "isError": True
            }
        
        try:
            query = args.get("query", "")
            top_k = int(args.get("limit", 10))
            min_similarity = float(args.get("min_similarity", 0.0))
            
            if not self.semantic_search:
                self.semantic_search = SemanticSearch(self.db_path)
            
            results = self.semantic_search.search(query, top_k=top_k, min_similarity=min_similarity)
            
            # Format results for MCP
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "hash": result.hash,
                    "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "similarity": result.similarity,
                    "file_path": result.file_path,
                    "section_name": result.section_name
                })
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "results": formatted_results,
                        "count": len(formatted_results)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Semantic search error: {str(e)}"}],
                "isError": True
            }
    
    def cassette_network_query(self, args: Dict) -> Dict:
        """MCP tool: Query the cassette network."""
        if not NETWORK_AVAILABLE or not self.network_hub:
            return {
                "content": [{"type": "text", "text": "Cassette network not available"}],
                "isError": True
            }
        
        try:
            query = args.get("query", "")
            top_k = int(args.get("limit", 10))
            capability = args.get("capability")
            
            if capability:
                results = self.network_hub.query_by_capability(query, capability, top_k)
            else:
                results = self.network_hub.query_all(query, top_k)
            
            # Flatten results
            all_results = []
            for cassette_id, cassette_results in results.items():
                for result in cassette_results:
                    result["cassette_id"] = cassette_id
                    all_results.append(result)
            
            # Sort by score if available
            all_results.sort(key=lambda x: x.get("score", 0), reverse=True)
            
            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "capability": capability,
                        "results": all_results[:top_k],
                        "cassettes_queried": len(results)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Cassette network error: {str(e)}"}],
                "isError": True
            }
    
    def get_embedding_stats(self) -> Dict:
        """Get statistics about vector embeddings."""
        if not SEMANTIC_AVAILABLE or not self.semantic_search:
            return {"error": "Semantic search not initialized"}
        
        try:
            return self.semantic_search.get_stats()
        except Exception as e:
            return {"error": str(e)}
    
    def get_network_status(self) -> Dict:
        """Get cassette network status."""
        if not NETWORK_AVAILABLE or not self.network_hub:
            return {"error": "Network hub not initialized"}
        
        try:
            return self.network_hub.get_network_status()
        except Exception as e:
            return {"error": str(e)}


# Test the adapter
if __name__ == "__main__":
    adapter = SemanticMCPAdapter()
    print("Initializing Semantic MCP Adapter...")
    init_result = adapter.initialize()
    print(f"Initialization: {json.dumps(init_result, indent=2)}")
    
    if "error" not in init_result:
        # Test semantic search
        print("\n--- Testing semantic search ---")
        search_result = adapter.semantic_search_tool({
            "query": "agent governance",
            "limit": 3
        })
        if search_result.get("isError"):
            print(f"Search error: {search_result['content'][0]['text']}")
        else:
            result_data = json.loads(search_result['content'][0]['text'])
            print(f"Search successful: {len(result_data['results'])} results")
            for i, result in enumerate(result_data['results'], 1):
                print(f"  {i}. {result.get('file_path', 'Unknown')} (similarity: {result['similarity']:.3f})")
        
        # Test network status
        print("\n--- Testing network status ---")
        status = adapter.get_network_status()
        print(f"Network status: {json.dumps(status, indent=2)}")