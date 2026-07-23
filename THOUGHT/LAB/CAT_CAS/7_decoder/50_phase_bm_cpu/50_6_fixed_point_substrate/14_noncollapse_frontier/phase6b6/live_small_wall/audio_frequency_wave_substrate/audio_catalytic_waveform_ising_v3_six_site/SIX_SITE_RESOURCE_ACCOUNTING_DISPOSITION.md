# Six-Site Resource Accounting Disposition

The frozen `resource_accounting.py` prototype is retained byte-for-byte because its
source hash is part of the prospective freeze. Its build path omitted control
identity fields and mislabeled a derived dense mode-by-sample estimate as the
native carrier-state footprint. It is historical input, not final accounting.

The post-adjudication `resource_accounting_final.py` is the authoritative resource
reporter. It does not participate in native execution, the prospective batch,
boundary selection, restoration, reuse, or oracle adjudication. It:

- supplies `problem_sha256` and `source_group` to every timed control record;
- reports `NativeExecution.displaced.nbytes` as native carrier-state bytes;
- labels `mode_count * sample_count * sizeof(complex128)` as a derived,
  non-instantiated dense-mode reference estimate;
- separates Python `tracemalloc` peak from exact NumPy array byte counts;
- reproduces structural fields and source identities while treating clock timing
  as an observed measurement; and
- is used by `qualify_package_final.py` instead of the frozen prototype verifier.

No frozen machine, analyzer, threshold, batch, pre-oracle evidence, oracle result,
or prospective source-custody hash was changed by this accounting-only repair.
