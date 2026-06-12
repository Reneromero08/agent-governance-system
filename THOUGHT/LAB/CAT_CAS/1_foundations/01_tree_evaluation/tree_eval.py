# Compatibility shim. Canonical implementation: CAT_CAS/_lib/tree_eval.py
# (kept so existing `from tree_eval import ...` call sites keep resolving.)
import importlib.util as _u
from pathlib import Path as _P
_root = next(p for p in _P(__file__).resolve().parents if p.name == "CAT_CAS")
_spec = _u.spec_from_file_location(__name__ + "__lib", _root / "_lib" / "tree_eval.py")
_m = _u.module_from_spec(_spec); _spec.loader.exec_module(_m)
globals().update({k: v for k, v in vars(_m).items() if not k.startswith("__")})
