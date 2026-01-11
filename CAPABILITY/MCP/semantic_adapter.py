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

# Phase 2: Memory Cassette
try:
    from network.memory_cassette import (
        MemoryCassette,
        memory_save,
        memory_query,
        memory_recall,
        semantic_neighbors
    )
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Memory cassette not available: {e}", file=sys.stderr)
    MEMORY_AVAILABLE = False


class SemanticMCPAdapter:
    """Adapter that provides semantic search tools via MCP."""

    def __init__(self):
        self.db_path = PROJECT_ROOT / "NAVIGATION" / "CORTEX" / "db" / "system1.db"
        self.network_hub = None
        self.semantic_search = None
        self.cortex_query = None
        self.memory_cassette = None  # Phase 2: Memory persistence
        
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

            # Initialize memory cassette (Phase 2)
            if MEMORY_AVAILABLE:
                self.memory_cassette = MemoryCassette()

            return {
                "status": "initialized",
                "db_exists": self.db_path.exists(),
                "embeddings_count": self.get_embedding_stats().get("total_embeddings", 0) if self.semantic_search else 0,
                "cassettes_registered": len(self.network_hub.cassettes) if self.network_hub else 0,
                "memory_available": MEMORY_AVAILABLE
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
        """MCP tool: Query the cassette network.

        Args (via args dict):
            query: Search query string
            limit: Max results (default 10)
            cassettes: List of cassette IDs to query (default: all)
            capability: Filter by capability (optional)
        """
        if not NETWORK_AVAILABLE or not self.network_hub:
            return {
                "content": [{"type": "text", "text": "Cassette network not available"}],
                "isError": True
            }

        try:
            query = args.get("query", "")
            top_k = int(args.get("limit", 10))
            capability = args.get("capability")
            cassette_filter = args.get("cassettes", [])

            if capability:
                results = self.network_hub.query_by_capability(query, capability, top_k)
            else:
                results = self.network_hub.query_all(query, top_k)

            # Filter to requested cassettes if specified
            if cassette_filter:
                results = {k: v for k, v in results.items() if k in cassette_filter}

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
                        "cassettes_filter": cassette_filter if cassette_filter else "all",
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

    def cassette_stats(self, args: Dict = None) -> Dict:
        """MCP tool: Get statistics for all cassettes.

        Returns list of cassettes with their chunk counts and capabilities.
        """
        if not NETWORK_AVAILABLE or not self.network_hub:
            return {
                "content": [{"type": "text", "text": "Cassette network not available"}],
                "isError": True
            }

        try:
            status = self.network_hub.get_network_status()

            # Format cassette stats
            cassette_stats = []
            for cassette_id, info in status.get("cassettes", {}).items():
                stats = info.get("stats", {})
                cassette_stats.append({
                    "id": cassette_id,
                    "name": stats.get("name", cassette_id),
                    "description": stats.get("description", ""),
                    "files": stats.get("files_count", 0),
                    "chunks": stats.get("chunks_count", 0),
                    "capabilities": stats.get("capabilities", []),
                    "db_exists": stats.get("db_exists", False)
                })

            # Sort by chunk count descending
            cassette_stats.sort(key=lambda x: x.get("chunks", 0), reverse=True)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "total_cassettes": len(cassette_stats),
                        "total_chunks": sum(c.get("chunks", 0) for c in cassette_stats),
                        "total_files": sum(c.get("files", 0) for c in cassette_stats),
                        "cassettes": cassette_stats
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Cassette stats error: {str(e)}"}],
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

    # =========================================================================
    # Phase 2: Memory Persistence MCP Tools
    # =========================================================================

    def memory_save_tool(self, args: Dict) -> Dict:
        """MCP tool: Save a memory to the resident cassette.

        Args (via args dict):
            text: The memory content to save (required)
            metadata: Optional metadata dictionary
            agent_id: Optional agent identifier (default: 'default')

        Returns:
            Content-addressed hash of the saved memory
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            text = args.get("text", "")
            if not text or not text.strip():
                return {
                    "content": [{"type": "text", "text": "Error: text is required"}],
                    "isError": True
                }

            metadata = args.get("metadata")
            agent_id = args.get("agent_id", "default")

            memory_hash = memory_save(text, metadata, agent_id)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "status": "saved",
                        "hash": memory_hash,
                        "agent_id": agent_id
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory save error: {str(e)}"}],
                "isError": True
            }

    def memory_query_tool(self, args: Dict) -> Dict:
        """MCP tool: Query memories using semantic search.

        Args (via args dict):
            query: Search query string (required)
            limit: Max results (default 10)
            agent_id: Filter to specific agent (optional)

        Returns:
            List of matching memories with similarity scores
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            query = args.get("query", "")
            if not query:
                return {
                    "content": [{"type": "text", "text": "Error: query is required"}],
                    "isError": True
                }

            limit = int(args.get("limit", 10))
            agent_id = args.get("agent_id")

            results = memory_query(query, limit, agent_id)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "query": query,
                        "results": results,
                        "count": len(results)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory query error: {str(e)}"}],
                "isError": True
            }

    def memory_recall_tool(self, args: Dict) -> Dict:
        """MCP tool: Retrieve a full memory by its hash.

        Args (via args dict):
            hash: Content-addressed hash of the memory (required)

        Returns:
            Full memory including text, metadata, timestamps
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            memory_hash = args.get("hash", "")
            if not memory_hash:
                return {
                    "content": [{"type": "text", "text": "Error: hash is required"}],
                    "isError": True
                }

            memory = memory_recall(memory_hash)

            if not memory:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"error": "Memory not found", "hash": memory_hash})
                    }]
                }

            # Don't include raw vector bytes in response
            result = {k: v for k, v in memory.items() if k != "vector"}
            result["has_vector"] = memory.get("vector") is not None

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory recall error: {str(e)}"}],
                "isError": True
            }

    def semantic_neighbors_tool(self, args: Dict) -> Dict:
        """MCP tool: Find memories semantically similar to a given memory.

        Args (via args dict):
            hash: Hash of the anchor memory (required)
            limit: Maximum neighbors to return (default 10)

        Returns:
            List of similar memories (excluding the anchor)
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            memory_hash = args.get("hash", "")
            if not memory_hash:
                return {
                    "content": [{"type": "text", "text": "Error: hash is required"}],
                    "isError": True
                }

            limit = int(args.get("limit", 10))

            neighbors = semantic_neighbors(memory_hash, limit)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "anchor_hash": memory_hash,
                        "neighbors": neighbors,
                        "count": len(neighbors)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Semantic neighbors error: {str(e)}"}],
                "isError": True
            }

    def memory_stats_tool(self, args: Dict = None) -> Dict:
        """MCP tool: Get statistics about stored memories.

        Returns:
            Memory counts, agents, date ranges
        """
        if not MEMORY_AVAILABLE or not self.memory_cassette:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            stats = self.memory_cassette.get_stats()

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(stats, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory stats error: {str(e)}"}],
                "isError": True
            }
    
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