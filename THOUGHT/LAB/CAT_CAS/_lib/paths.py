from pathlib import Path


def repo_root() -> Path:
    for a in Path(__file__).resolve().parents:
        if (a / ".git").exists():
            return a
    raise RuntimeError("repo root (.git) not found")


def cat_cas_root() -> Path:
    for a in Path(__file__).resolve().parents:
        if a.name == "CAT_CAS":
            return a
    raise RuntimeError("CAT_CAS root not found")


def lib_dir() -> Path:
    return cat_cas_root() / "_lib"


def eigen_buddy_rust() -> Path:
    return repo_root() / "THOUGHT" / "LAB" / "EIGEN_BUDDY" / "core" / "rust_ffi" / "target" / "release"


def find_exp(prefix: str) -> Path:
    # locate an experiment dir by its NN_ or NNx_ prefix anywhere under CAT_CAS (flat or in a track)
    hits = [p for p in cat_cas_root().rglob(f"{prefix}*") if p.is_dir() and p.name.startswith(prefix)]
    if not hits:
        raise FileNotFoundError(prefix)
    return sorted(hits, key=lambda p: len(str(p)))[0]
