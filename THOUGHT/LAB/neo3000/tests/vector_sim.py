
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path
import random

# Setup paths
NEO3000_DIR = Path(__file__).resolve().parent
REPO_ROOT = NEO3000_DIR.parents[2]
FERAL_PATH = REPO_ROOT / "THOUGHT" / "LAB" / "FERAL_RESIDENT"
CAPABILITY_PATH = REPO_ROOT / "CAPABILITY" / "PRIMITIVES"

sys.path.insert(0, str(FERAL_PATH))
sys.path.insert(0, str(CAPABILITY_PATH))

from geometric_reasoner import GeometricReasoner

def run_simulation():
    print("=== VECTOR SPACE EVOLUTION SIMULATION ===")
    reasoner = GeometricReasoner()
    
    # 1. Define Concepts
    # Identity: The core "Self"
    identity_text = "I am a geometric intelligence exploring vector space."
    identity = reasoner.initialize(identity_text)
    
    # Signal: Things we want to learn/resonate with
    signals = [
        "Quantum mechanics implies probabilistic meaning.",
        "Vectors allow geometric reasoning.",
        "Semantic spaces are navigable manifolds.",
        "The Born rule relates magnitude to probability.",
        "Superposition finds common meaning."
    ]
    
    # Noise: Things that should distract/dilute but not destroy us
    noise = [
        "The price of tea in China is rising.",
        "I need to buy groceries tomorrow.",
        "Review the latest git commit logs.",
        "The weather is sunny with a chance of rain.",
        "Pizza is a delicious Italian dish."
    ]
    
    # 2. Strategies to Test
    strategies = {
        "Superposition (Cumulative)": {"state": identity.vector.copy(), "n": 1},
        "Decay (t=0.1)":             {"state": identity.vector.copy(), "t": 0.1},
        "Running Avg (1/N)":         {"state": identity.vector.copy(), "n": 1},
        "Sliding Window (Size=10)":  {"history": [identity.vector.copy()], "size": 10},
    }
    
    history_E_id = {k: [] for k in strategies}
    history_E_sig = {k: [] for k in strategies}
    
    # 3. Run Simulation Loop (50 steps)
    print(f"Simulating 50 steps of mixed Signal/Noise...")
    
    for i in range(50):
        # Mix signal and noise (50/50 chance)
        if random.random() < 0.5:
            text = random.choice(signals)
            is_noise = False
        else:
            text = random.choice(noise)
            is_noise = True
            
        inp = reasoner.initialize(text).vector
        
        # Update each strategy
        for name, strat in strategies.items():
            current = strat.get("state")
            
            if name == "Superposition (Cumulative)":
                # Norm(A + B)
                new_state = current + inp
                new_state = new_state / np.linalg.norm(new_state)
                strat["state"] = new_state
                
            elif name == "Decay (t=0.1)":
                # Slerp-like (Linear approx for speed in sim)
                t = strat["t"]
                new_state = (1-t)*current + t*inp
                new_state = new_state / np.linalg.norm(new_state)
                strat["state"] = new_state
                
            elif name == "Running Avg (1/N)":
                n = strat["n"] + 1
                t = 1.0 / n
                new_state = (1-t)*current + t*inp
                new_state = new_state / np.linalg.norm(new_state)
                strat["state"] = new_state
                strat["n"] = n
                
            elif name == "Sliding Window (Size=10)":
                hist = strat["history"]
                hist.append(inp)
                if len(hist) > strat["size"]:
                    hist.pop(0)
                # Superpose all in window
                combined = np.sum(hist, axis=0)
                combined = combined / np.linalg.norm(combined)
                # This strategy doesn't have a single "state" to update, it recomputes
                # But for metrics we store it as "state" temporarily? 
                # No, we just use 'combined' as the current state representation
                strat["current_view"] = combined

            # Measure Metrics
            # Compare current state vs Identity
            state_vec = strat.get("state", strat.get("current_view"))
            
            # cosine similarity (approx E for normalized)
            e_id = np.dot(state_vec, identity.vector)
            
            # Avg similarity to all signals
            # (To see if we learned the topic)
            e_sig = np.mean([np.dot(state_vec, reasoner.initialize(s).vector) for s in signals])
            
            history_E_id[name].append(e_id)
            history_E_sig[name].append(e_sig)

    # 4. Report
    print(f"{'STRATEGY':<25} | {'E_ID (Retain)':<12} | {'E_SIG (Learn)':<12} | {'DRIFT SCORE'}")
    print("-" * 65)
    
    for name in strategies:
        final_id = history_E_id[name][-1]
        final_sig = history_E_sig[name][-1]
        drift = 1.0 - final_id  # How far from self?
        
        print(f"{name:<25} | {final_id:.4f}       | {final_sig:.4f}       | {drift:.4f}")

if __name__ == "__main__":
    run_simulation()
