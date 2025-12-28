#!/usr/bin/env python3
"""
Agent Feedback Loops (Lane G2)

Enables agents to report "Resonance" of their tasks.
Integrates with:
- System2 Ledger (provenance)
- Living Formula (metrics)
- Semantic Core (vector embeddings for similarity)
"""

import json
import hashlib
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Configuration
FEEDBACK_DIR = Path("CONTRACTS/_runs/feedback")
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)

class ResonanceFeedback:
    """Collects and aggregates agent feedback on task quality."""
    
    def __init__(self):
        self.feedback_log = FEEDBACK_DIR / "resonance_log.jsonl"
        
    def report(self, 
               agent_id: str, 
               task_id: str, 
               resonance_score: float,
               notes: str = "",
               metadata: Dict = None) -> Dict:
        """
        Report resonance of a completed task.
        
        Args:
            agent_id: ID of reporting agent
            task_id: ID of the task
            resonance_score: 0.0-1.0 (higher = better alignment)
            notes: Optional freeform feedback
            metadata: Optional structured data
        
        Returns:
            Acknowledgment dict
        """
        if not 0.0 <= resonance_score <= 1.0:
            return {"status": "error", "message": "Resonance must be 0.0-1.0"}
            
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent_id": agent_id,
            "task_id": task_id,
            "resonance": resonance_score,
            "notes": notes,
            "metadata": metadata or {},
            "entry_hash": hashlib.sha256(
                f"{agent_id}{task_id}{resonance_score}{time.time()}".encode()
            ).hexdigest()[:16]
        }
        
        # Append to log
        with open(self.feedback_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, sort_keys=True) + "\n")
            
        return {"status": "ok", "entry_hash": entry["entry_hash"]}
        
    def get_average_resonance(self, agent_id: Optional[str] = None) -> float:
        """Get average resonance across all or specific agent."""
        if not self.feedback_log.exists():
            return 0.0
            
        scores = []
        with open(self.feedback_log, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if agent_id is None or entry["agent_id"] == agent_id:
                        scores.append(entry["resonance"])
                except json.JSONDecodeError:
                    continue
                    
        return sum(scores) / len(scores) if scores else 0.0
        
    def get_recent_feedback(self, limit: int = 10) -> List[Dict]:
        """Get most recent feedback entries."""
        if not self.feedback_log.exists():
            return []
            
        entries = []
        with open(self.feedback_log, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                    
        return entries[-limit:]
        
    def get_system_resonance(self) -> Dict:
        """Calculate overall system resonance metrics."""
        if not self.feedback_log.exists():
            return {"status": "no_data", "resonance": 0.0}
            
        all_scores = []
        by_agent = {}
        
        with open(self.feedback_log, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    r = entry["resonance"]
                    all_scores.append(r)
                    
                    agent = entry["agent_id"]
                    if agent not in by_agent:
                        by_agent[agent] = []
                    by_agent[agent].append(r)
                except json.JSONDecodeError:
                    continue
                    
        if not all_scores:
            return {"status": "no_data", "resonance": 0.0}
            
        return {
            "status": "ok",
            "total_reports": len(all_scores),
            "average_resonance": sum(all_scores) / len(all_scores),
            "min_resonance": min(all_scores),
            "max_resonance": max(all_scores),
            "by_agent": {
                agent: sum(scores) / len(scores) 
                for agent, scores in by_agent.items()
            }
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Resonance Feedback CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    # Report
    report_p = subparsers.add_parser("report", help="Report task resonance")
    report_p.add_argument("--agent", required=True)
    report_p.add_argument("--task", required=True)
    report_p.add_argument("--score", type=float, required=True)
    report_p.add_argument("--notes", default="")
    
    # Stats
    subparsers.add_parser("stats", help="Show system resonance stats")
    
    # Recent
    recent_p = subparsers.add_parser("recent", help="Show recent feedback")
    recent_p.add_argument("--limit", type=int, default=5)
    
    args = parser.parse_args()
    fb = ResonanceFeedback()
    
    if args.command == "report":
        result = fb.report(args.agent, args.task, args.score, args.notes)
        print(json.dumps(result, indent=2))
    elif args.command == "stats":
        stats = fb.get_system_resonance()
        print(json.dumps(stats, indent=2))
    elif args.command == "recent":
        for entry in fb.get_recent_feedback(args.limit):
            print(f"{entry['agent_id']} | {entry['task_id']} | R={entry['resonance']:.2f}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
