# Exp 42.15: Full 100-Epoch Regeneration Instructions
# Run this as a background/long-running job (~5 hours)
# Created 2026-06-02

# Step 1: Navigate to Rust project
cd "D:\CCC 2.0\AI\agent-governance-system\THOUGHT\LAB\CAT_CAS\6_frontier_phases\42_computational_event_horizon\ULTRA\exp_15_quantum_gravity_unification\rust"

# Step 2: Backup existing CSV (if any)
Copy-Item telemetry_42_15_unification.csv telemetry_42_15_unification.csv.bak_$(Get-Date -Format yyyyMMdd_HHmmss)

# Step 3: Run 100-epoch simulation (~3 min/epoch, ~5 hours total)
cargo run --release 2>&1 | Tee-Object -FilePath regen_output_$(Get-Date -Format yyyyMMdd_HHmmss).txt

# Step 4: Verify row count
python -c "import pandas as pd; df=pd.read_csv('telemetry_42_15_unification.csv'); print(f'Rows: {len(df)}')"

# Step 5: Run analysis
python unification_proof.py

# Step 6: Check correlations
python -c "
import pandas as pd; from scipy.stats import pearsonr
df=pd.read_csv('telemetry_42_15_unification.csv')
q=df['QuantumCollisions']; g=df['GravityShift']; r=df['RiemannDrift']
for lab,x,y in [('Q-G',q,g),('G-R',g,r),('Q-R',q,r)]:
    rv,pv=pearsonr(x,y)
    print(f'{lab}: r={rv:.4f} p={pv:.2e} |r|={abs(rv):.4f}')
"
