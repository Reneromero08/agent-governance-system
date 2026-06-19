# Legacy L4A `HoloRecord` Scaffold

The files `holo_record.h` and `holo_record.c` are retained for provenance of the first Class B carrier scaffold.

They are **not** the canonical `.holo` schema.

The canonical executable object is `HoloObject`, defined by:

- `holo_geometry.h/.c`
- `holo_invariant_family.h/.c`
- `holo_physical_mapping.h/.c`
- `holo_observability_design.h/.c`
- `holo_semantic_integrity.h/.c`
- `HOLO_SCHEMA.md`

The legacy scaffold has known limitations:

- placeholder path/substrate fields;
- serialization-time boundary timestamp;
- heuristic rather than complete forbidden-field validation;
- no geometric relation basis;
- no owned reversible path history;
- no invariant-family recomputation;
- no strict lifecycle reader;
- no physical mapping or observability contract.

It must not be imported into new runtime code, cited as schema version 1.4, or used to authorize a physical-memory claim. Historical Class B material may refer to it only as `LEGACY_L4A_CARRIER_SCAFFOLD`.
