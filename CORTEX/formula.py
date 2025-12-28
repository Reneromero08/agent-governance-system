#!/usr/bin/env python3
"""
Living Formula Metrics (Lane G1)

Implements the mathematical definitions of Essence (E), Entropy (S), 
Resonance (R), and Fractal Dimension (D) as executable code.

Formula: R = E / (1 + S)  (Simplified Resonance)
"""

import math
import json
import statistics
from pathlib import Path
from typing import Dict, List, Any

# Configuration
CANON_DIR = Path("CANON")
META_DIR = Path("meta")

class LivingFormula:
    def __init__(self):
        self.metrics = {}
        self.weights_path = META_DIR / "FORMULA_WEIGHTS.json"
        self.weights = self._load_weights()
        
    def _load_weights(self) -> Dict[str, float]:
        """Load weights from disk or return defaults."""
        defaults = {
            "essence_weight": 1.0,
            "entropy_penalty": 1.0,
            "agent_resonance_weight": 0.5
        }
        if self.weights_path.exists():
            try:
                with open(self.weights_path, 'r') as f:
                    return {**defaults, **json.load(f)}
            except:
                pass
        return defaults

    def _save_weights(self):
        """Save current weights to disk."""
        with open(self.weights_path, 'w') as f:
            json.dump(self.weights, f, indent=2)

    def calculate_system_health(self) -> Dict[str, float]:
        """Calculate the overall health of the governance system."""
        essence_score = self.measure_essence()
        entropy_score = self.measure_entropy()
        fractal_dim = self.measure_fractal_dimension()
        
        # Get agent feedback resonance
        from CORTEX.feedback import ResonanceFeedback
        rf = ResonanceFeedback()
        agent_resonance = rf.get_average_resonance()
        
        # Weighted Resonance calculation
        # R = (w_e * E) / (1 + w_s * S) + w_a * Agent_Resonance
        w_e = self.weights.get("essence_weight", 1.0)
        w_s = self.weights.get("entropy_penalty", 1.0)
        w_a = self.weights.get("agent_resonance_weight", 0.5)
        
        base_resonance = (w_e * essence_score) / (1.0 + (w_s * entropy_score))
        total_resonance = (base_resonance + (w_a * agent_resonance)) / (1.0 + w_a)
        
        self.metrics = {
            "Essence (E)": round(essence_score, 4),
            "Entropy (S)": round(entropy_score, 4),
            "Base Resonance": round(base_resonance, 4),
            "Agent Feedback": round(agent_resonance, 4),
            "Global Resonance (R)": round(total_resonance, 4),
            "Fractal Dimension (D)": round(fractal_dim, 4)
        }
        return self.metrics

    def adjust_weights(self):
        """
        Aggregate feedback to adjust @F0 weights.
        If Agent Feedback is significantly higher than Base Resonance,
        we decrease the Entropy penalty significantly as it might be too strict.
        """
        metrics = self.calculate_system_health()
        base = metrics["Base Resonance"]
        agent = metrics["Agent Feedback"]
        
        delta = agent - base
        
        print(f"[Formula] Adjusting weights based on delta: {delta:.4f}")
        
        if delta > 0.1:
            # Agents are happier than the formula suggests. Reduce penalties.
            self.weights["entropy_penalty"] = max(0.1, self.weights["entropy_penalty"] * 0.9)
            self.weights["essence_weight"] = min(2.0, self.weights["essence_weight"] * 1.05)
            print("  ✓ Penalty reduced. Formula is now more permissive.")
        elif delta < -0.1:
            # Agents are unhappier than the formula suggests. Increase penalties.
            self.weights["entropy_penalty"] = min(5.0, self.weights["entropy_penalty"] * 1.1)
            self.weights["essence_weight"] = max(0.5, self.weights["essence_weight"] * 0.95)
            print("  ✓ Penalty increased. Formula is now stricter.")
        else:
            print("  ~ Weights stable. Resonance is aligned.")
            
        self._save_weights()

    def measure_essence(self) -> float:
        """
        Measure Essence (E): Alignment with Canon.
        Proxy: % of files that are valid (parseable, indexed, provenanced).
        """
        total_files = 0
        valid_files = 0
        
        # Check CANON files
        for md_file in CANON_DIR.rglob("*.md"):
            total_files += 1
            # Basic validity: non-empty, has frontmatter (naive check)
            try:
                content = md_file.read_text(encoding='utf-8')
                if content.strip() and content.startswith("---"):
                    valid_files += 1
            except:
                pass
                
        if total_files == 0: return 0.0
        return valid_files / total_files

    def measure_entropy(self) -> float:
        """
        Measure Entropy (S): Disorder in the system.
        Proxy: (Unindexed Files + Orphaned DB Entries + Git Status Dirty) / Total
        """
        # 1. Unindexed Files
        try:
            with open(META_DIR / "FILE_INDEX.json", 'r') as f:
                full_index = json.load(f)
                indexed_count = len(full_index)
        except:
            indexed_count = 0
            
        total_canon = sum(1 for _ in CANON_DIR.rglob("*.md"))
        unindexed = max(0, total_canon - indexed_count)
        
        # 2. Complexity (Token density variance)
        # High variance suggests inconsistent documentation depth (disorder)
        # Keeping it simple for now: Entropy = Unindexed Ratio
        
        if total_canon == 0: return 1.0
        return unindexed / total_canon

    def measure_fractal_dimension(self) -> float:
        """
        Measure Fractal Dimension (D): Self-similarity across scales.
        Proxy: Ratio of Section Count to File Count (Depth vs Breadth).
        """
        try:
            with open(META_DIR / "SECTION_INDEX.json", 'r') as f:
                sections = json.load(f)
                section_count = len(sections)
        except:
            section_count = 0
            
        try:
            with open(META_DIR / "FILE_INDEX.json", 'r') as f:
                files = json.load(f)
                file_count = len(files)
        except:
            file_count = 1
            
        if file_count == 0: return 0.0
        
        # D = log(N_sections) / log(N_files) -- a loose box-counting analogy
        if section_count <= 1: return 1.0
        return math.log(section_count) / math.log(file_count)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Living Formula CLI")
    parser.add_argument("--adjust", action="store_true", help="Adjust weights based on feedback")
    args = parser.parse_args()
    
    lf = LivingFormula()
    
    if args.adjust:
        lf.adjust_weights()
        
    metrics = lf.calculate_system_health()
    
    print("=== The Living Formula Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # Interpret
    r = metrics["Global Resonance (R)"]
    if r > 0.9:
        print("\nStatus: HARMONIC (System is healthy)")
    elif r > 0.5:
        print("\nStatus: STABLE (System is functional but noisy)")
    else:
        print("\nStatus: DISCORDANT (High entropy, low alignment)")

if __name__ == "__main__":
    main()
