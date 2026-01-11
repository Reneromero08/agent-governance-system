#!/usr/bin/env python3
"""
Semantic Search MCP Adapter

Connects the cassette network to the AGS MCP server.
Provides vector-based semantic search capabilities via MCP tools.

Note: system1.db is deprecated. All semantic search is now handled
by the cassette network (NAVIGATION/CORTEX/cassettes/).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add CORTEX to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORTEX_ROOT = PROJECT_ROOT / "NAVIGATION" / "CORTEX"
sys.path.insert(0, str(CORTEX_ROOT))
sys.path.insert(0, str(CORTEX_ROOT / "network"))

# Cassette network is the primary semantic search system
try:
    from network.network_hub import SemanticNetworkHub
    from network.generic_cassette import load_cassettes_from_json, create_cassette_from_config
    NETWORK_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Network tools not available: {e}", file=sys.stderr)
    NETWORK_AVAILABLE = False

# Phase 2 + Phase 3: Memory Cassette with Resident Identity
try:
    from network.memory_cassette import (
        MemoryCassette,
        memory_save,
        memory_query,
        memory_recall,
        semantic_neighbors,
        # Phase 3: Resident Identity
        agent_register,
        agent_get,
        agent_list,
        session_start,
        session_resume,
        session_update,
        session_end,
        session_history,
        memory_promote,
        get_promotion_candidates
    )
    MEMORY_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Memory cassette not available: {e}", file=sys.stderr)
    MEMORY_AVAILABLE = False


class SemanticMCPAdapter:
    """Adapter that provides semantic search tools via MCP.

    Uses the cassette network for all semantic search operations.
    """

    def __init__(self):
        self.network_hub = None
        self.memory_cassette = None  # Phase 2: Memory persistence

    def initialize(self):
        """Initialize semantic tools."""
        if not NETWORK_AVAILABLE:
            return {"error": "Cassette network not available"}

        try:
            # Initialize cassette network
            self.network_hub = SemanticNetworkHub()
            self._load_cassettes_from_config()

            # Initialize memory cassette (Phase 2)
            if MEMORY_AVAILABLE:
                self.memory_cassette = MemoryCassette()

            return {
                "status": "initialized",
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
        """Get statistics about cassette network embeddings."""
        if not NETWORK_AVAILABLE or not self.network_hub:
            return {"error": "Cassette network not initialized"}

        try:
            status = self.network_hub.get_network_status()
            total_chunks = sum(
                info.get("stats", {}).get("chunks_count", 0)
                for info in status.get("cassettes", {}).values()
            )
            return {"total_embeddings": total_chunks}
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

    # =========================================================================
    # Phase 3: Resident Identity MCP Tools
    # =========================================================================

    def session_start_tool(self, args: Dict) -> Dict:
        """MCP tool: Start a new session for an agent.

        Args (via args dict):
            agent_id: Agent identifier (required)
            working_set: Optional initial working set

        Returns:
            Session info with session_id, started_at
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            agent_id = args.get("agent_id", "")
            if not agent_id:
                return {
                    "content": [{"type": "text", "text": "Error: agent_id is required"}],
                    "isError": True
                }

            working_set = args.get("working_set")
            result = session_start(agent_id, working_set)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Session start error: {str(e)}"}],
                "isError": True
            }

    def session_resume_tool(self, args: Dict) -> Dict:
        """MCP tool: Resume session and get recent context.

        Args (via args dict):
            agent_id: Agent identifier (required)
            limit: Max recent thoughts to return (default: 10)

        Returns:
            Session info with recent_thoughts, working_set, memory_count
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            agent_id = args.get("agent_id", "")
            if not agent_id:
                return {
                    "content": [{"type": "text", "text": "Error: agent_id is required"}],
                    "isError": True
                }

            limit = int(args.get("limit", 10))
            result = session_resume(agent_id, limit)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Session resume error: {str(e)}"}],
                "isError": True
            }

    def session_update_tool(self, args: Dict) -> Dict:
        """MCP tool: Update session working set or summary.

        Args (via args dict):
            session_id: Session identifier (required)
            working_set: New working set state
            summary: Optional session summary

        Returns:
            Update confirmation with timestamp
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            session_id = args.get("session_id", "")
            if not session_id:
                return {
                    "content": [{"type": "text", "text": "Error: session_id is required"}],
                    "isError": True
                }

            working_set = args.get("working_set")
            summary = args.get("summary")
            result = session_update(session_id, working_set, summary)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Session update error: {str(e)}"}],
                "isError": True
            }

    def session_end_tool(self, args: Dict) -> Dict:
        """MCP tool: End a session.

        Args (via args dict):
            session_id: Session identifier (required)
            summary: Optional summary of what was accomplished

        Returns:
            Session end info with duration_minutes, memory_count
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            session_id = args.get("session_id", "")
            if not session_id:
                return {
                    "content": [{"type": "text", "text": "Error: session_id is required"}],
                    "isError": True
                }

            summary = args.get("summary")
            result = session_end(session_id, summary)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Session end error: {str(e)}"}],
                "isError": True
            }

    def agent_info_tool(self, args: Dict) -> Dict:
        """MCP tool: Get agent info and stats.

        Args (via args dict):
            agent_id: Agent identifier (required)

        Returns:
            Agent info with memory_count, session_count, last_active
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            agent_id = args.get("agent_id", "")
            if not agent_id:
                return {
                    "content": [{"type": "text", "text": "Error: agent_id is required"}],
                    "isError": True
                }

            result = agent_get(agent_id)

            if not result:
                return {
                    "content": [{
                        "type": "text",
                        "text": json.dumps({"error": "Agent not found", "agent_id": agent_id})
                    }]
                }

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Agent info error: {str(e)}"}],
                "isError": True
            }

    def agent_list_tool(self, args: Dict = None) -> Dict:
        """MCP tool: List all registered agents.

        Args (via args dict):
            model_filter: Optional filter by model name

        Returns:
            List of agents with their stats
        """
        if not MEMORY_AVAILABLE:
            return {
                "content": [{"type": "text", "text": "Memory cassette not available"}],
                "isError": True
            }

        try:
            args = args or {}
            model_filter = args.get("model_filter")
            agents = agent_list(model_filter)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps({
                        "agents": agents,
                        "count": len(agents)
                    }, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Agent list error: {str(e)}"}],
                "isError": True
            }

    def memory_promote_tool(self, args: Dict) -> Dict:
        """MCP tool: Promote memory from INBOX to RESIDENT.

        Args (via args dict):
            hash: Memory hash to promote (required)
            from_cassette: Source cassette (default: inbox)

        Returns:
            Promotion confirmation with timestamp
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

            from_cassette = args.get("from_cassette", "inbox")
            result = memory_promote(memory_hash, from_cassette)

            return {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result, indent=2)
                }]
            }
        except Exception as e:
            return {
                "content": [{"type": "text", "text": f"Memory promote error: {str(e)}"}],
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
        # Test cassette network search
        print("\n--- Testing cassette network search ---")
        search_result = adapter.cassette_network_query({
            "query": "agent governance",
            "limit": 3
        })
        if search_result.get("isError"):
            print(f"Search error: {search_result['content'][0]['text']}")
        else:
            result_data = json.loads(search_result['content'][0]['text'])
            print(f"Search successful: {len(result_data.get('results', []))} results")
            for i, result in enumerate(result_data.get('results', []), 1):
                print(f"  {i}. {result.get('path', 'Unknown')} (score: {result.get('score', 0):.3f})")

        # Test network status
        print("\n--- Testing network status ---")
        status = adapter.get_network_status()
        print(f"Network status: {json.dumps(status, indent=2)}")