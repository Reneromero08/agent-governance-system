import os, importlib.util
_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '04_thermodynamic_cpu', 'reversible_cpu.py')
_spec = importlib.util.spec_from_file_location("_reversible_cpu_04", os.path.abspath(_src))
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ReversibleCPU = _mod.ReversibleCPU
IrreversibleCPU = _mod.IrreversibleCPU
calculate_landauer_energy = _mod.calculate_landauer_energy
