# ORACLE

Visualizer lab for the CAT_CAS topological-oracle experiments. Standalone tooling
(a FastAPI backend + ES6 frontend), promoted out of `CAT_CAS/` to its own lab
under `THOUGHT/LAB/` because it is an application, not an experiment.

## What it does

Renders the 1D–5D topological halting/Chern/Weyl/axion/Floquet oracles by loading
their source modules directly from the CAT_CAS lab:

| Engine | CAT_CAS source experiment |
|--------|---------------------------|
| `visualizer/engine/oracle_1d.py` | `5_topological_proofs/35_topological_halting_oracle/35_2_nonhermitian_oracle/` |
| `visualizer/engine/oracle_2d.py` | `5_topological_proofs/37_2d_chern_oracle/` |
| `visualizer/engine/oracle_3d.py` | `5_topological_proofs/38_3d_weyl_oracle/` |
| `visualizer/engine/oracle_4d.py` | `5_topological_proofs/39_4d_axion_oracle/` |
| `visualizer/engine/oracle_5d.py` | `5_topological_proofs/40_5d_floquet_oracle/` |

The engines locate CAT_CAS as a sibling lab (`<THOUGHT/LAB>/CAT_CAS`), so this lab
stays runnable regardless of where it sits, as long as CAT_CAS is alongside it.

## Run

```
cd visualizer && pip install -r requirements.txt && python launch_server.py
```

See [ORACLE_MASTER_REPORT.md](ORACLE_MASTER_REPORT.md) for the synthesis across all
oracle instances and [visualizer/ROADMAP.md](visualizer/ROADMAP.md) for status.
