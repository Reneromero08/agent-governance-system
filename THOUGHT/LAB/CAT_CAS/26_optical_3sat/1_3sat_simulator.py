import os
import torch

def generate_3sat_instance(n=64, m=272):
    """
    Generates a hard 3-SAT instance.
    n: number of variables (e.g. 64)
    m: number of clauses (e.g. 272, ratio 4.25 is the hard phase transition)
    Returns:
        C: Clause matrix of shape (m, n) where C[i, j] = 1 if x_j is in clause i, 
           -1 if not x_j is in clause i, and 0 otherwise.
        solution: A satisfying assignment (if we plant one)
    """
    # Plant a hidden solution to guarantee satisfiability
    solution = torch.randint(0, 2, (n,), dtype=torch.float32)
    # Convert to spins {-1, 1}
    spins = solution * 2 - 1
    
    C = torch.zeros((m, n), dtype=torch.float32)
    
    for i in range(m):
        # Pick 3 unique variables
        vars_idx = torch.randperm(n)[:3]
        
        # To guarantee the planted solution works, at least one literal must align
        # randomly pick 1 to 3 of them to align with the planted solution
        num_align = torch.randint(1, 4, (1,)).item()
        
        for j in range(3):
            var = vars_idx[j]
            if j < num_align:
                # Align literal with solution spin
                C[i, var] = spins[var]
            else:
                # Anti-align
                C[i, var] = -spins[var]
                
        # Randomly permute the clause literals so the aligned one isn't always first
        # (Though C is a matrix so order doesn't matter, just making sure!)
        
    return C, solution

if __name__ == "__main__":
    import sys
    n = 64
    m = int(64 * 4.26)
    
    if len(sys.argv) > 2:
        n = int(sys.argv[1])
        m = int(sys.argv[2])
        
    C, solution = generate_3sat_instance(n, m)
    
    # Save the instance
    save_path = os.path.join(os.path.dirname(__file__), "3sat_instance.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save({
        'C': C,
        'solution': solution,
        'n': n,
        'm': m
    }, save_path)
    
    print(f"[+] 3-SAT Instance Generated: N={n}, M={m} (Ratio {m/n:.2f})")
    print(f"[+] Saved to: {save_path}")
