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
        
    def calculate_system_health(self) -> Dict[str, float]:
        """Calculate the overall health of the governance system."""
        essence_score = self.measure_essence()
        entropy_score = self.measure_entropy()
        fractal_dim = self.measure_fractal_dimension()
        
        # Resonance = Essence / (1 + Entropy)
        # We normalize entropy to be > 0. A simplified model.
        resonance = essence_score / (1.0 + entropy_score)
        
        self.metrics = {
            "Essence (E)": round(essence_score, 4),
            "Entropy (S)": round(entropy_score, 4),
            "Resonance (R)": round(resonance, 4),
            "Fractal Dimension (D)": round(fractal_dim, 4)
        }
        return self.metrics

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
    lf = LivingFormula()
    metrics = lf.calculate_system_health()
    
    print("=== The Living Formula Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
        
    # Interpret
    r = metrics["Resonance (R)"]
    if r > 0.9:
        print("\nStatus: HARMONIC (System is healthy)")
    elif r > 0.5:
        print("\nStatus: STABLE (System is functional but noisy)")
    else:
        print("\nStatus: DISCORDANT (High entropy, low alignment)")

if __name__ == "__main__":
    main()
