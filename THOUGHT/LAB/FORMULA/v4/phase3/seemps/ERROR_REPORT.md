# Phase 3f: SeeMPS — Installation & Cython Extension Status

Date: 2026-05-16 | Status: **BLOCKED — Cython extension compiled but hangs on simplify_mps**

---

## What Works

- `pip install seemps` succeeds — wheel builds from source on Windows
- `import seemps` succeeds — module loads cleanly
- `seemps.state.to_mps(vector)` works — creates MPS array from vector
- `seemps.state.CanonicalMPS(data)` constructs a canonical MPS
- All non-cython modules (`analysis`, `evolution`, `state`, `operators`, etc.) import fine

## What's Broken

- **`simplify_mps(mps)`** hangs indefinitely with no output and no error.
  The Cython extension (`seemps.cython.core`) is compiled but the function
  enters an infinite loop or deadlock in the Cython code.

## Diagnosis

The first `pip install` compiled the Cython extension with whatever toolchain
was available (likely MSVC Build Tools 2022, found installed at
`C:\Program Files\Microsoft Visual Studio\2022\BuildTools`).

The second `pip install --force-reinstall` used the cached wheel from the
first build — it did not recompile. The wheel is `seemps-3.0.0-cp311-cp311-win_amd64.whl`
(236KB). The corrupted function persisted.

The Cython extension file is at:
`.venv\Lib\site-packages\seemps\cython\core.cp311-win_amd64.pyd`

## Root Cause Hypotheses

1. **MSVC toolchain misconfiguration.** The build tools are installed but may
   not have been invoked correctly by pip's build isolation. The `vcvars`
   environment variables (including INCLUDE and LIB paths) may not have been
   set.

2. **ABI mismatch.** The wheel was built against numpy 1.26, but the second
   install upgraded numpy to 2.4.5 before downgrading back. The Cython
   extension may link against the numpy ABI from build time.

3. **SeeMPS v3.0.0 is unstable.** The package was released Jan 2026 and may
   have Windows-specific bugs in the Cython layer. The `simplify_mps`
   function calls into `truncate` / `canonicalize` Cython routines that may
   not be tested on Windows.

## What's Needed to Fix

1. **Clean rebuild in MSVC Developer Command Prompt:**
   ```powershell
   # Launch Developer Command Prompt for VS 2022
   # Then: pip install --no-cache-dir --force-reinstall seemps
   ```
   This ensures vcvars sets INCLUDE/LIB paths correctly.

2. **Alternative: WSL with Python 3.12.** Install Python 3.12 on WSL,
   then `pip install seemps` with gcc compilation. This avoids the MSVC
   toolchain entirely.

3. **Alternative: Use quimb for MPS operations.** quimb is pure Python
   and provides `tensor_split`, `tensor_1d_compress`, and MPS decomposition
   without Cython dependencies. The TT decomposition math is identical.

4. **Alternative: Contact SeeMPS maintainer.** The hang may be a known bug
   (GitHub issues at `github.com/juanjosegarciaripoll/seemps2`).

## Status

The experiment cannot proceed until the Cython hang is resolved. The iterative
SVD approach (numpy-only) is mathematically equivalent to MPS compression but
is not using the SeeMPS library as specified.
