# LAB Prototypes

This directory contains experimental prototypes for Catalytic Computing concepts.

## F2: Catalytic Scratch Layer

**Purpose**: Enable destructive operations in an isolated scratch space with guaranteed byte-identical restoration.

**Key Features**:
- Creates isolated copy of source directory
- Allows arbitrary mutations in scratch space
- Verifies original directory remains untouched
- Proves restoration via hash verification

**Usage**:
```bash
python scratch_layer_poc.py
```

**Workflow**: See `SCRATCH_WORKFLOW.md` for detailed process.

---

## F3: Catalytic Context Compression (CAS)

**Purpose**: Minimize token cost of LITE packs while enabling full reconstruction via Content-Addressed Storage.

**Key Features**:
- Deterministic manifest generation
- Deduplication via content hashing
- Byte-identical reconstruction
- Safety caps (max files, max size)

**Usage**:
```bash
# Build CAS from source directory
python f3_cas_prototype.py build --src ./my_dir --out ./pack

# Reconstruct from CAS
python f3_cas_prototype.py reconstruct --pack ./pack --dst ./restored

# Verify byte-identity
python f3_cas_prototype.py verify --src ./my_dir --dst ./restored
```

**Exit Codes**:
- 0: Success
- 2: Verify mismatch
- 3: Unsafe path
- 4: Bounds exceeded

---

## Testing

Run pytest on the test suites:
```bash
pytest test_f3_cas_prototype.py -v
pytest test_scratch_layer.py -v
```
